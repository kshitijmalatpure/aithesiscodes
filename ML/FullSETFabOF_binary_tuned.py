"""
fabOF_v8_4_0_TUNED_MULTI_TRAIT_BINARY.py
========================================
Runs a tuned FabOF-style pipeline for four *binary* traits
(A_WOB, A_GRB, A_COB, A_TTB) using best hyperparameters from
comprehensive_fabOF_binary_tuning/best_model_config.json.

Key features
------------
- Uses 10-fold files: fold_{n}/fold{n}trb.csv and fold{n}teb.csv
- Preprocess (impute + scale) per best params
- Feature selection: KBest = 50 (f_regression), fixed
- Train RandomForestRegressor with OOB and derive FABOF borders for 0/1 rounding
- Save per-fold predictions and aggregated predictions per trait + master file
- Enhanced confusion matrices ONLY (per fold + aggregated), with bigger font (~14)
- Comprehensive SHAP (overall only) per trait (threshold mean|SHAP| >= 0.01):
    * Composite figure: LEFT = bar (mean|SHAP|), RIGHT = beeswarm
    * One waterfall plot per trait (top-impact sample)
    * Dependence plots for every feature above threshold
    * Export shap_values.csv and shap_importance.csv
- Aggregate summary CSV with rows: WOB, GRB, COB, TTB, AGGREGATE
    * Metrics: Precision, Recall, Accuracy, MAE, MSE, R2, F1-macro, QWK
    * Include mean and std over folds; AGGREGATE = average of trait means/stds
- Two composite confusion matrices at ROOT:
    * composite_confusion_1x4.png (1×4)
    * composite_confusion_2x2.png (2×2)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix
)

# Future-proof RMSE
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse
    def rmse(y_true, y_pred): return _mse(y_true, y_pred, squared=False)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LABELS = [0, 1]                         # Binary classes
K_BEST = 50                             # Fixed FS
SHAP_THRESHOLD = 0.01                   # |SHAP| threshold for importance
CM_ANNOT_FONTSIZE = 14                  # Bigger CM numbers

BEST_CONFIG_PATH = "comprehensive_fabOF_binary_tuning/best_model_config.json"
RESULTS_ROOT_NAME = "fabOF_results_v8.4.0_TUNED_MULTI_TRAIT_BINARY"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tuned_run_{ts}.log"
    logger = logging.getLogger(f"FabOF_Tuned_Binary_{ts}")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def load_best_params(base_dir: Path, logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    path = base_dir / BEST_CONFIG_PATH
    if not path.exists():
        logger.warning(f"Best config not found at {path}. Using empty dict; defaults will be applied.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    logger.info(f"Loaded best params from {path}")
    return cfg


def get_scaler(name: str):
    if name == 'standard': return StandardScaler()
    if name == 'robust':   return RobustScaler()
    if name == 'minmax':   return MinMaxScaler()
    return None  # 'none' or unrecognized


def prepare_dirs(base_dir: Path) -> Dict[str, Path]:
    root = base_dir / RESULTS_ROOT_NAME
    dirs = {
        "root": root,
        "logs": root / "01_logs",
        "configs": root / "02_configs",
        "predictions": root / "03_predictions",
        "master_agg": root / "04_master_aggregated",
        "cm": root / "05_confusion_matrices_enhanced",
        "fi": root / "06_feature_importance",
        "shap_root": root / "07_shap",
        "shap_tables": root / "07_shap" / "tables",
        "shap_composite": root / "07_shap" / "composite_plots",
        "shap_waterfalls": root / "07_shap" / "waterfalls",
        "shap_dep": root / "07_shap" / "dependence_plots",
        "summaries": root / "08_summaries",
        "reports": root / "09_reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_session_config(dirs: Dict[str, Path], meta: Dict[str, Any]):
    out = {"timestamp": datetime.now().isoformat(), **meta}
    with open(dirs["configs"] / "session_config.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_fold(base_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_path = base_dir / f"fold_{fold}" / f"fold{fold}trb.csv"
    te_path = base_dir / f"fold_{fold}" / f"fold{fold}teb.csv"
    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)
    tr, te = tr.align(te, axis=1, fill_value=np.nan)
    return tr, te


def preprocess_and_fs(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    imputation_strategy: str,
    scaling_method: str,
    k_best: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, Optional[Any], SelectKBest]:
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    scaler = get_scaler(scaling_method)
    if scaler is not None:
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_te_s = scaler.transform(X_te_imp)
    else:
        X_tr_s = X_tr_imp
        X_te_s = X_te_imp

    k = min(k_best, X_tr_s.shape[1])
    selector = SelectKBest(f_regression, k=k)
    X_tr_fs = selector.fit_transform(X_tr_s, y_tr.astype(float))
    X_te_fs = selector.transform(X_te_s)
    mask = selector.get_support()
    return X_tr_fs, X_te_fs, mask, imputer, scaler, selector


def fit_fabof_rf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    rf_params: Dict[str, Any]
) -> Tuple[RandomForestRegressor, np.ndarray, np.ndarray]:
    """General FABOF borders for arbitrary label set; here binary [0,1]."""
    clean = {k: v for k, v in rf_params.items()
             if k in ['n_estimators', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']}
    rf = RandomForestRegressor(oob_score=True, **clean)
    rf.fit(X_tr, y_tr.astype(float))

    classes_sorted = np.array(sorted(np.unique(y_tr)))
    inner_classes = classes_sorted[:-1]
    oob = rf.oob_prediction_
    if inner_classes.size > 0:
        pi = np.array([(y_tr <= c).mean() for c in inner_classes])
        borders_inner = np.quantile(oob, pi)
        borders = np.concatenate([[classes_sorted[0]], borders_inner, [classes_sorted[-1]]])
    else:
        # degenerate (shouldn't happen in stratified folds)
        borders = np.array([classes_sorted[0], classes_sorted[-1]])
    return rf, borders, classes_sorted


def predict_fabof(rf: RandomForestRegressor, borders: np.ndarray, classes_sorted: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_raw = rf.predict(X)
    # Map by borders → class indices
    idx = np.searchsorted(borders[1:-1], y_raw)
    y_ord = classes_sorted[idx]
    return y_ord, y_raw


def compute_metrics(y_true_ord, y_pred_ord, y_pred_raw) -> Dict[str, float]:
    return {
        "Precision": precision_score(y_true_ord, y_pred_ord, labels=LABELS, average="macro", zero_division=0),
        "Recall": recall_score(y_true_ord, y_pred_ord, labels=LABELS, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y_true_ord, y_pred_ord),
        "MAE": float(np.mean(np.abs(y_true_ord - y_pred_raw))),
        "MSE": float(np.mean((y_true_ord - y_pred_raw) ** 2)),
        "R2": float(1.0 - np.sum((y_true_ord - y_pred_raw) ** 2) / np.sum((y_true_ord - np.mean(y_true_ord)) ** 2)) if np.var(y_true_ord) > 0 else 0.0,
        "F1-macro": f1_score(y_true_ord, y_pred_ord, labels=LABELS, average="macro", zero_division=0),
        "QWK": cohen_kappa_score(y_true_ord, y_pred_ord, labels=LABELS, weights="quadratic"),
    }


def plot_enhanced_cm(cm: np.ndarray, labels: List[int], title: str, save_path: Path, vmax: Optional[int] = None):
    plt.figure(figsize=(10, 8))
    if vmax is None:
        vmax = np.max(cm)
    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count'}, vmax=vmax,
        square=True, linewidths=0.5, linecolor='white'
    )
    for t in ax.texts:
        t.set_fontsize(CM_ANNOT_FONTSIZE)

    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=0); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def composite_confusions(root_dir: Path, agg_cms: Dict[str, np.ndarray]):
    """Create two composite figures (1x4 and 2x2) from aggregated CMs of traits."""
    if not agg_cms:
        return
    traits_order = ["WOB", "GRB", "COB", "TTB"]
    labels = LABELS
    global_vmax = max(np.max(agg_cms[k]) for k in agg_cms if k in traits_order)

    # 1x4
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for idx, trait in enumerate(traits_order):
        cm = agg_cms.get(trait)
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            vmax=global_vmax, square=True, ax=ax,
            cbar=(idx == 3)
        )
        for t in ax.texts:
            t.set_fontsize(CM_ANNOT_FONTSIZE)
        ax.set_title(f"{trait} (Aggregated)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True' if idx == 0 else '')
    plt.tight_layout()
    fig.savefig(root_dir / "composite_confusion_1x4.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, trait in enumerate(traits_order):
        r, c = idx // 2, idx % 2
        cm = agg_cms.get(trait)
        ax = axes[r, c]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            vmax=global_vmax, square=True, ax=ax,
            cbar=(idx == 3)
        )
        for t in ax.texts:
            t.set_fontsize(CM_ANNOT_FONTSIZE)
        ax.set_title(f"{trait} (Aggregated)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    fig.savefig(root_dir / "composite_confusion_2x2.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def shap_comprehensive_outputs(
    trait_code: str,
    rf_model: RandomForestRegressor,
    X_stack: np.ndarray,
    feature_names: List[str],
    shap_dirs: Dict[str, Path],
    logger: logging.Logger
):
    """Composite (bar-left + beeswarm-right), waterfall, dependence plots, and tables."""
    if X_stack.size == 0:
        logger.warning(f"[{trait_code}] SHAP: no data to analyze.")
        return

    # TreeExplainer
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(X_stack)
    except Exception as e:
        logger.warning(f"[{trait_code}] SHAP explainer failed: {e}")
        return

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs,
        "Mean_SHAP": np.mean(shap_vals, axis=0),
        "Std_SHAP": np.std(shap_vals, axis=0)
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    keep_mask = shap_importance["Mean_Abs_SHAP"] >= SHAP_THRESHOLD
    kept = shap_importance[keep_mask].copy()

    # Save tables
    shap_tables_dir = shap_dirs["tables"]
    shap_importance.to_csv(shap_tables_dir / f"{trait_code}_shap_importance.csv", index=False)
    pd.DataFrame(shap_vals, columns=feature_names).to_csv(shap_tables_dir / f"{trait_code}_shap_values.csv", index=False)

    # Composite: bar-left + beeswarm-right
    try:
        if kept.empty:
            kept = shap_importance.head(20)
        kept_feats = kept["Feature"].tolist()
        kept_idx = [feature_names.index(f) for f in kept_feats]
        shap_vals_kept = shap_vals[:, kept_idx]
        X_kept = X_stack[:, kept_idx]

        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])

        # LEFT: barh
        ax1 = fig.add_subplot(gs[0, 0])
        kept_sorted = kept.sort_values("Mean_Abs_SHAP", ascending=True)
        ax1.barh(kept_sorted["Feature"], kept_sorted["Mean_Abs_SHAP"])
        ax1.set_xlabel("mean(|SHAP|)")
        ax1.set_title(f"{trait_code}: SHAP importance (threshold ≥ {SHAP_THRESHOLD})")
        ax1.margins(y=0.02)

        # RIGHT: beeswarm
        ax2 = fig.add_subplot(gs[0, 1]); plt.sca(ax2)
        try:
            shap.plots.beeswarm(
                shap.Explanation(values=shap_vals_kept, data=X_kept, feature_names=kept_feats),
                show=False, max_display=len(kept_feats)
            )
        except Exception:
            shap.summary_plot(shap_vals_kept, X_kept, feature_names=kept_feats, show=False)
        ax2.set_title(f"{trait_code}: SHAP beeswarm")

        plt.tight_layout()
        fig.savefig(shap_dirs["composite"] / f"{trait_code}_shap_composite.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"[{trait_code}] SHAP composite failed: {e}")

    # Waterfall (top-impact sample)
    try:
        sums = np.abs(shap_vals).sum(axis=1)
        idx = int(np.argmax(sums))
        explanation = shap.Explanation(
            values=shap_vals[idx],
            base_values=getattr(explainer, "expected_value",
                                np.mean(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0),
            data=X_stack[idx],
            feature_names=feature_names
        )
        try:
            shap.plots.waterfall(explanation, show=False)
        except Exception:
            shap.waterfall_plot(explanation, show=False)
        plt.title(f"{trait_code}: SHAP waterfall (sample {idx})")
        plt.tight_layout()
        plt.savefig(shap_dirs["waterfalls"] / f"{trait_code}_shap_waterfall.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
    except Exception as e:
        logger.warning(f"[{trait_code}] SHAP waterfall failed: {e}")

    # Dependence plots for all kept features
    dep_dir = shap_dirs["dep"] / trait_code
    dep_dir.mkdir(parents=True, exist_ok=True)
    kept_feats = kept["Feature"].tolist()
    for f in kept_feats:
        try:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(f, shap_vals, X_stack, feature_names=feature_names, show=False)
            plt.title(f"{trait_code}: dependence — {f}")
            plt.tight_layout()
            plt.savefig(dep_dir / f"{trait_code}_dep_{f}.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
        except Exception as e:
            logger.warning(f"[{trait_code}] Dependence plot failed for {f}: {e}")


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------
class FabOFTunedRunnerBinary:
    def __init__(self, base_dir: str, name_col: str,
                 score_cols: List[str] = None):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        # Internal (dataset) column names:
        self.score_cols = score_cols or ['A_WOB', 'A_GRB', 'A_COB', 'A_TTB']
        # Output-friendly short codes (strip A_ and trailing B)
        self.trait_map = {sc: sc.replace("A_", "").replace("B", "B") for sc in self.score_cols}
        # Ensure mapping is WOB/GRB/COB/TTB
        self.trait_map = {sc: sc.replace("A_", "") for sc in self.score_cols}

        self.dirs = prepare_dirs(self.base_dir)
        self.logger = setup_logging(self.dirs["logs"])
        save_session_config(self.dirs, {
            "base_dir": str(self.base_dir),
            "score_cols": self.score_cols,
            "labels": LABELS,
            "k_best": K_BEST,
            "shap_threshold": SHAP_THRESHOLD,
            "best_config_path": BEST_CONFIG_PATH
        })
        self.best_params = load_best_params(self.base_dir, self.logger)

    def _trait_params(self, score_col: str) -> Dict[str, Any]:
        defaults = {
            'imputation_strategy': 'median',
            'scaling_method': 'standard',
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42
        }
        return {**defaults, **self.best_params.get(score_col, {})}

    def _trait_dirs(self, trait_short: str) -> Dict[str, Path]:
        pred_dir = self.dirs["predictions"] / trait_short
        cm_dir = self.dirs["cm"] / trait_short
        pred_dir.mkdir(parents=True, exist_ok=True)
        cm_dir.mkdir(parents=True, exist_ok=True)
        return {"pred": pred_dir, "cm": cm_dir}

    def run(self):
        self.logger.info("="*60)
        self.logger.info("Starting FabOF Tuned Multi-Trait Runner (Binary)")
        self.logger.info("="*60)

        master_rows = []
        agg_cms_by_trait: Dict[str, np.ndarray] = {}
        metrics_by_trait: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for sc in self.score_cols:
            trait = self.trait_map.get(sc, sc)
            self.logger.info(f"\n--- Trait: {trait} ({sc}) ---")
            p = self._trait_params(sc)
            self.logger.info(f"Params: {p}")
            tdirs = self._trait_dirs(trait)

            agg_cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
            all_preds_ord, all_true_ord, all_files, all_folds = [], [], [], []

            shap_stack_vals = []
            shap_stack_X = []
            selected_feature_names = None
            last_rf_model = None

            for fold in range(1, 11):
                try:
                    tr, te = load_fold(self.base_dir, fold)
                    feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]
                    X_tr, X_te = tr[feats].values, te[feats].values
                    y_tr, y_te = tr[sc].values.astype(int), te[sc].values.astype(int)

                    X_tr_fs, X_te_fs, mask, imputer, scaler, selector = preprocess_and_fs(
                        X_tr, X_te, y_tr,
                        p.get('imputation_strategy', 'median'),
                        p.get('scaling_method', 'standard'),
                        K_BEST
                    )
                    selected_feature_names = np.array(feats)[mask].tolist()

                    rf, borders, classes_sorted = fit_fabof_rf(X_tr_fs, y_tr, p)
                    y_pred_ord, y_pred_raw = predict_fabof(rf, borders, classes_sorted, X_te_fs)

                    met = compute_metrics(y_te, y_pred_ord, y_pred_raw)
                    met.update({"Fold": fold, "Trait": trait})
                    metrics_by_trait[trait].append(met)

                    cm = confusion_matrix(y_te, y_pred_ord, labels=LABELS)
                    agg_cm += cm
                    plot_enhanced_cm(
                        cm, LABELS,
                        title=f"{trait} - Fold {fold} (Enhanced)",
                        save_path=tdirs["cm"] / f"{trait}_fold{fold}_cm_enhanced.png"
                    )

                    fold_pred = pd.DataFrame({
                        "Inputfile": te[self.name_col],
                        "True_Value": y_te,
                        "Predicted_Value": y_pred_ord,
                        "Raw_Prediction": y_pred_raw,
                        "Fold": fold,
                        "Trait": trait
                    })
                    fold_pred.to_csv(tdirs["pred"] / f"{trait}_fold{fold}_predictions.csv", index=False)

                    all_preds_ord.extend(y_pred_ord.tolist())
                    all_true_ord.extend(y_te.tolist())
                    all_files.extend(te[self.name_col].tolist())
                    all_folds.extend([fold] * len(y_te))

                    try:
                        X_te_ready = X_te_fs
                        explainer = shap.TreeExplainer(rf)
                        sv = explainer.shap_values(X_te_ready)
                        if isinstance(sv, list):
                            sv = sv[0]
                        shap_stack_vals.append(sv)
                        shap_stack_X.append(X_te_ready)
                        last_rf_model = rf
                    except Exception as e:
                        self.logger.warning(f"[{trait}] SHAP fold {fold} failed: {e}")

                except Exception as e:
                    self.logger.warning(f"[{trait}] Fold {fold} failed: {e}")
                    continue

            if all_preds_ord:
                agg_df = pd.DataFrame({
                    "Inputfile": all_files,
                    "True_Value": all_true_ord,
                    "Predicted_Value": all_preds_ord,
                    "Fold": all_folds,
                    "Trait": [trait] * len(all_files)
                })
                agg_df.to_csv(self.dirs["predictions"] / trait / f"{trait}_aggregated_predictions.csv", index=False)

            plot_enhanced_cm(
                agg_cm, LABELS,
                title=f"{trait} - Aggregated (10 folds)",
                save_path=self.dirs["cm"] / trait / f"{trait}_aggregated_cm_enhanced.png"
            )
            agg_cms_by_trait[trait] = agg_cm

            if shap_stack_vals and shap_stack_X and selected_feature_names is not None and last_rf_model is not None:
                X_stack = np.vstack(shap_stack_X)
                shap_vals_stack = np.vstack(shap_stack_vals)
                shap_dirs = {
                    "tables": self.dirs["shap_tables"],
                    "composite": self.dirs["shap_composite"],
                    "waterfalls": self.dirs["shap_waterfalls"],
                    "dep": self.dirs["shap_dep"]
                }
                shap_comprehensive_outputs(
                    trait, last_rf_model, X_stack, selected_feature_names, shap_dirs, self.logger
                )

            if all_preds_ord:
                acc = (np.array(all_preds_ord) == np.array(all_true_ord)).mean()
                master_rows.append({"Trait": trait, "Samples": len(all_preds_ord), "Simple_Accuracy": float(acc)})

        composite_confusions(self.dirs["root"], agg_cms_by_trait)

        if master_rows:
            master_df = pd.DataFrame(master_rows)
            master_df.to_csv(self.dirs["master_agg"] / "MASTER_aggregated_predictions_summary.csv", index=False)

        try:
            all_trait_files = []
            for trait in ['WOB', 'GRB', 'COB', 'TTB']:
                p = self.dirs["predictions"] / trait / f"{trait}_aggregated_predictions.csv"
                if p.exists():
                    all_trait_files.append(pd.read_csv(p))
            if all_trait_files:
                master_pred = pd.concat(all_trait_files, ignore_index=True)
                master_pred.to_csv(self.dirs["master_agg"] / "MASTER_aggregated_predictions_all_traits.csv", index=False)
        except Exception as e:
            self.logger.warning(f"Failed to build master aggregated predictions: {e}")

        self._save_aggregate_summary(metrics_by_trait)

        with open(self.dirs["reports"] / "final_report.md", "w", encoding="utf-8") as f:
            f.write("# FabOF Tuned Multi-Trait Run (Binary)\n\n")
            f.write(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Base dir: {self.base_dir}\n")
            f.write(f"- Output: {self.dirs['root']}\n")
            f.write(f"- Traits: {', '.join([self.trait_map.get(s, s) for s in self.score_cols])}\n")
            f.write(f"- Labels: {LABELS}\n")
            f.write(f"- KBest: {K_BEST}\n")
            f.write(f"- SHAP threshold: {SHAP_THRESHOLD}\n")

        self.logger.info("\nAll done.")
        self.logger.info(f"Results saved to: {self.dirs['root']}")

    def _save_aggregate_summary(self, metrics_by_trait: Dict[str, List[Dict[str, float]]]):
        """
        Build AGGREGATE_metrics_summary.csv with rows WOB, GRB, COB, TTB, AGGREGATE and
        columns: mean/std for Precision, Recall, Accuracy, MAE, MSE, R2, F1-macro, QWK.
        """
        metrics_list = ["Precision", "Recall", "Accuracy", "MAE", "MSE", "R2", "F1-macro", "QWK"]

        rows = []
        trait_means = {m: [] for m in metrics_list}
        trait_stds = {m: [] for m in metrics_list}

        for trait in ["WOB", "GRB", "COB", "TTB"]:
            fold_metrics = metrics_by_trait.get(trait, [])
            if not fold_metrics:
                mean_row = {m: 0.0 for m in metrics_list}
                std_row = {m + "_std": 0.0 for m in metrics_list}
            else:
                df = pd.DataFrame(fold_metrics)
                mean_row = {m: float(df[m].mean()) for m in metrics_list}
                std_row = {m + "_std": float(df[m].std()) for m in metrics_list}

            for m in metrics_list:
                trait_means[m].append(mean_row[m])
                trait_stds[m].append(std_row[m + "_std"])

            row = {"Trait": trait, **mean_row, **std_row}
            rows.append(row)

        agg_means = {m: float(np.mean(trait_means[m])) for m in metrics_list}
        agg_stds = {m + "_std": float(np.mean(trait_stds[m])) for m in metrics_list}
        rows.append({"Trait": "AGGREGATE", **agg_means, **agg_stds})

        out = pd.DataFrame(rows, columns=["Trait"] + [m for m in metrics_list] + [m + "_std" for m in metrics_list])
        out.to_csv(self.dirs["summaries"] / "AGGREGATE_metrics_summary.csv", index=False)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # >>> Adjust these two to your environment <<<
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = "Inputfile"

    runner = FabOFTunedRunnerBinary(
        base_dir=base_dir,
        name_col=name_col,
        score_cols=['A_WOB', 'A_GRB', 'A_COB', 'A_TTB']  # binary traits
    )
    try:
        runner.run()
        print("="*60)
        print("TUNED MULTI-TRAIT RUN (BINARY) COMPLETED")
        print("="*60)
        print(f"Results saved to: {runner.dirs['root']}")
    except Exception as e:
        print(f"Tuned run failed: {e}")
