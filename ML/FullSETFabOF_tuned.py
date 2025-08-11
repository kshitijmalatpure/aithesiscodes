"""
fabOF_v8_4_0_TUNED_MULTI_TRAIT.py
=================================
(Updated) Now uses HARD-CODED BEST_PARAMS and matches the hard-coded
preprocessing order:
  - impute for FS -> apply FS
  - then impute + scale inside model fit/predict
Everything else (outputs, SHAP bundle, composites, metrics) unchanged.

NOTE: Aligned with Script 2 for:
  - KBest behavior (can select ALL features; f_regression)
  - y-dtype handling (no int cast pre-fit)
  - SHAP computation (per fold -> stack across folds)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix
)

try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse
    def rmse(y_true, y_pred): return _mse(y_true, y_pred, squared=False)

# ---------------------------------------------------------------------
# Configuration (unchanged constants)
# ---------------------------------------------------------------------
LABELS = [1, 2, 3, 4]
SHAP_THRESHOLD = 0.01
CM_ANNOT_FONTSIZE = 18
RESULTS_ROOT_NAME = "fabOF_results_v8.4.0_TUNED_MULTI_TRAIT_JSON_FS"

# ---------------------------------------------------------------------
# HARD-CODED BEST PARAMS — paste from best_params_snippet.txt here
# ---------------------------------------------------------------------
BEST_PARAMS = {
    'A_WO': {
        'scaling_method': 'robust',
        'random_state': 42,
        'n_estimators': 100,
        'min_samples_split': 7,
        'min_samples_leaf': 5,
        'max_features': None,
        'max_depth': 5,
        'kbest_k': 150,
        'imputation_strategy': 'most_frequent',
        'feature_selection_method': 'kbest',
        'bootstrap': True
    },
    'A_GR': {
        'scaling_method': 'none',
        'random_state': 42,
        'n_estimators': 800,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'max_features': 'sqrt',
        'max_depth': None,
        'kbest_k': 150,
        'imputation_strategy': 'median',
        'feature_selection_method': 'kbest',
        'bootstrap': True
    },
    'A_CO': {
        'scaling_method': 'minmax',
        'random_state': 42,
        'n_estimators': 50,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 0.7,
        'max_depth': 15,
        'kbest_k': 75,
        'imputation_strategy': 'median',
        'feature_selection_method': 'kbest',
        'bootstrap': True
    },
    'A_TT': {
        'scaling_method': 'minmax',
        'random_state': 42,
        'n_estimators': 800,
        'min_samples_split': 7,
        'min_samples_leaf': 2,
        'max_features': 0.3,
        'max_depth': 5,
        'kbest_k': 100,
        'imputation_strategy': 'most_frequent',
        'feature_selection_method': 'kbest',
        'bootstrap': True
    }
}

# ---------------------------------------------------------------------
# Utilities (minor changes only where needed)
# ---------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tuned_run_{ts}.log"
    logger = logging.getLogger(f"FabOF_Tuned_{ts}")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def get_scaler(name: str):
    if name == 'standard': return StandardScaler()
    if name == 'robust':   return RobustScaler()
    if name == 'minmax':   return MinMaxScaler()
    return None

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
    tr_path = base_dir / f"fold_{fold}" / f"fold{fold}tr.csv"
    te_path = base_dir / f"fold_{fold}" / f"fold{fold}te.csv"
    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)
    tr, te = tr.align(te, axis=1, fill_value=np.nan)
    return tr, te

def _select_kbest(score_func_name: str):
    s = str(score_func_name).lower()
    if s in ["f_regression", "f", "freg"]: return f_regression
    if s in ["mutual_info", "mi", "mutual_info_regression"]: return mutual_info_regression
    return f_regression

# ----- FS step: IMPUTE for FS → run FS (no scaling here)
def apply_feature_selection_for_fs(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    imputation_strategy: str,
    fs_cfg: Optional[Dict[str, Any]],
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Script-2 parity for KBest:
      - Impute using provided strategy
      - If fs_cfg['type'] == 'kbest', allow selecting ALL features (k up to n_features)
      - Always use f_regression for KBest (like Script 2)
      - If k is 'all' or non-numeric, keep all features (identity mask)
    """
    n_features = X_tr.shape[1]
    identity_mask = np.ones(n_features, dtype=bool)

    # impute for FS
    fs_imputer = SimpleImputer(strategy=imputation_strategy)
    X_tr_imp = fs_imputer.fit_transform(X_tr)
    X_te_imp = fs_imputer.transform(X_te)

    if not fs_cfg or str(fs_cfg.get("type", "none")).lower() in ["none", "false", "off"]:
        return X_tr_imp, X_te_imp, identity_mask

    method = str(fs_cfg.get("type", "kbest")).lower()

    if method == "kbest":
        k_param = fs_cfg.get("k", "all")

        # parse requested k (Script 2 behavior)
        if isinstance(k_param, (int, np.integer)):
            k_req = int(k_param)
        elif isinstance(k_param, str) and k_param.isdigit():
            k_req = int(k_param)
        else:
            # 'all' or anything non-numeric -> keep ALL features
            return X_tr_imp, X_te_imp, identity_mask

        # clamp to [1, n_features] (allow selecting all features)
        n_feat = X_tr_imp.shape[1]
        k = min(max(1, k_req), n_feat)

        # Always use f_regression to mirror Script 2
        selector = SelectKBest(score_func=f_regression, k=k)
        X_tr_fs = selector.fit_transform(X_tr_imp, y_tr)
        X_te_fs = selector.transform(X_te_imp)
        mask = selector.get_support()
        if mask.sum() == 0:
            logger.warning("KBest selected 0 features; reverting to identity.")
            return X_tr_imp, X_te_imp, identity_mask
        return X_tr_fs, X_te_fs, mask

    if method == "lasso":
        # keep existing behavior; not used in current BEST_PARAMS
        use_cv = bool(fs_cfg.get("use_cv", True))
        if use_cv:
            cv = int(fs_cfg.get("cv", 5))
            model = LassoCV(cv=cv, random_state=fs_cfg.get("random_state", 42), max_iter=10000)
        else:
            alpha = float(fs_cfg.get("alpha", 0.001))
            model = Lasso(alpha=alpha, random_state=fs_cfg.get("random_state", 42), max_iter=10000)
        model.fit(X_tr_imp, y_tr.astype(float))
        coefs = getattr(model, "coef_", None)
        if coefs is None:
            logger.warning("Lasso produced no coefficients; skipping FS.")
            return X_tr_imp, X_te_imp, identity_mask
        mask = np.abs(coefs) > 1e-12
        if not mask.any():
            logger.warning("Lasso selected 0 features; keeping top-1 by |coef|.")
            idx = int(np.argmax(np.abs(coefs)))
            mask = np.zeros_like(coefs, dtype=bool); mask[idx] = True
        return X_tr_imp[:, mask], X_te_imp[:, mask], mask

    logger.warning(f"Unrecognized fs.type='{method}'. Skipping FS.")
    return X_tr_imp, X_te_imp, identity_mask

# ----- Model fit: IMPUTE + SCALE inside
def fit_fabof_rf_with_internal_preproc(
    X_tr_for_fs: np.ndarray,
    y_tr: np.ndarray,
    rf_params: Dict[str, Any],
    imputation_strategy: str,
    scaling_method: str
) -> Tuple[RandomForestRegressor, np.ndarray, Optional[StandardScaler], SimpleImputer]:
    model_imputer = SimpleImputer(strategy=imputation_strategy)
    X_tr_imp = model_imputer.fit_transform(X_tr_for_fs)

    scaler = get_scaler(scaling_method)
    if scaler is not None:
        X_tr_final = scaler.fit_transform(X_tr_imp)
    else:
        X_tr_final = X_tr_imp

    clean = {k: v for k, v in rf_params.items()
             if k in ['n_estimators', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']}
    rf = RandomForestRegressor(oob_score=True, **clean)
    rf.fit(X_tr_final, y_tr.astype(float))

    kmax = int(np.max(y_tr))
    oob = rf.oob_prediction_
    pi = np.array([(y_tr <= c).mean() for c in range(1, kmax)])
    borders_inner = np.quantile(oob, pi) if len(pi) else np.array([])
    borders = np.concatenate([[1.0], borders_inner, [float(kmax)]])
    return rf, borders, scaler, model_imputer

# ----- Predict: apply the SAME internal imputer+scaler
def predict_fabof_with_internal_preproc(
    rf: RandomForestRegressor,
    borders: np.ndarray,
    X_te_for_fs: np.ndarray,
    scaler: Optional[StandardScaler],
    model_imputer: SimpleImputer
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_te_imp = model_imputer.transform(X_te_for_fs)
    if scaler is not None:
        X_te_final = scaler.transform(X_te_imp)
    else:
        X_te_final = X_te_imp
    y_raw = rf.predict(X_te_final)
    y_ord = np.searchsorted(borders[1:-1], y_raw) + 1
    return y_ord, y_raw, X_te_final  # return final matrix for SHAP

def compute_metrics(y_true_ord, y_pred_ord, y_pred_raw) -> Dict[str, float]:
    return {
        "Precision": precision_score(y_true_ord, y_pred_ord, labels=LABELS, average="macro", zero_division=0),
        "Recall": recall_score(y_true_ord, y_pred_ord, labels=LABELS, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y_true_ord, y_pred_ord),
        "MAE": float(np.mean(np.abs(y_true_ord - y_pred_raw))),
        "MSE": float(np.mean((y_true_ord - y_pred_raw) ** 2)),
        "R2": float(1.0 - np.sum((y_true_ord - y_pred_raw) ** 2) / np.sum((y_true_ord - np.mean(y_true_ord)) ** 2)),
        "F1-macro": f1_score(y_true_ord, y_pred_ord, labels=LABELS, average="macro", zero_division=0),
        "QWK": cohen_kappa_score(y_true_ord, y_pred_ord, labels=LABELS, weights="quadratic"),
    }

def plot_enhanced_cm(cm: np.ndarray, labels: List[int], title: str, save_path: Path, vmax: Optional[int] = None):
    plt.figure(figsize=(10, 8))
    if vmax is None: vmax = np.max(cm)
    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count'}, vmax=vmax,
        square=True, linewidths=0.5, linecolor='white'
    )
    for t in ax.texts: t.set_fontsize(CM_ANNOT_FONTSIZE)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=0); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def shap_comprehensive_outputs(
    trait_code: str,
    rf_model: RandomForestRegressor,
    X_stack_final: np.ndarray,  # scaled matrix
    feature_names: List[str],
    shap_dirs: Dict[str, Path],
    logger: logging.Logger,
    precomputed_shap: Optional[np.ndarray] = None
):
    """Now accepts precomputed_shap (stacked across folds) to match Script 2."""
    if X_stack_final.size == 0:
        logger.warning(f"[{trait_code}] SHAP: no data to analyze.")
        return
    try:
        if precomputed_shap is None:
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer.shap_values(X_stack_final)
            # Backward compat: RFRegressor sometimes returns array, sometimes list
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
        else:
            shap_vals = precomputed_shap
            # we still create an explainer only to get expected_value for waterfall aesthetics
            explainer = shap.TreeExplainer(rf_model)
    except Exception as e:
        logger.warning(f"[{trait_code}] SHAP explainer failed: {e}")
        return

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs,
        "Mean_SHAP": np.mean(shap_vals, axis=0),
        "Std_SHAP": np.std(shap_vals, axis=0)
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    keep_mask = shap_importance["Mean_Abs_SHAP"] >= SHAP_THRESHOLD
    kept = shap_importance[keep_mask].copy()
    shap_tables_dir = shap_dirs["tables"]
    shap_importance.to_csv(shap_tables_dir / f"{trait_code}_shap_importance.csv", index=False)
    pd.DataFrame(shap_vals, columns=feature_names).to_csv(shap_tables_dir / f"{trait_code}_shap_values.csv", index=False)

    try:
        if kept.empty: kept = shap_importance.head(20)
        kept_feats = kept["Feature"].tolist()
        kept_idx = [feature_names.index(f) for f in kept_feats]
        shap_vals_kept = shap_vals[:, kept_idx]
        X_kept = X_stack_final[:, kept_idx]

        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
        ax1 = fig.add_subplot(gs[0, 0])
        kept_sorted = kept.sort_values("Mean_Abs_SHAP", ascending=True)
        ax1.barh(kept_sorted["Feature"], kept_sorted["Mean_Abs_SHAP"])
        ax1.set_xlabel("mean(|SHAP|)")
        ax1.set_title(f"{trait_code}: SHAP importance (threshold ≥ {SHAP_THRESHOLD})")
        ax1.margins(y=0.02)
        ax2 = fig.add_subplot(gs[0, 1])
        plt.sca(ax2)
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

    # Waterfall (single sample) — purely visual; model expected_value from last fold is fine
    try:
        sums = np.abs(shap_vals).sum(axis=1)
        idx = int(np.argmax(sums))
        explanation = shap.Explanation(
            values=shap_vals[idx],
            base_values=getattr(explainer, "expected_value",
                                np.mean(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0),
            data=X_stack_final[idx],
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

    # Dependence plots for kept features
    dep_dir = shap_dirs["dep"] / trait_code
    dep_dir.mkdir(parents=True, exist_ok=True)
    for f in kept_feats:
        try:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(f, shap_vals, X_stack_final, feature_names=feature_names, show=False)
            plt.title(f"{trait_code}: dependence — {f}")
            plt.tight_layout()
            plt.savefig(dep_dir / f"{trait_code}_dep_{f}.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
        except Exception as e:
            logger.warning(f"[{trait_code}] Dependence plot failed for {f}: {e}")

# ---------------------------------------------------------------------
# Main runner (hard-coded params + new order)
# ---------------------------------------------------------------------
class FabOFTunedRunner:
    def __init__(self, base_dir: str, name_col: str,
                 score_cols: List[str] = None):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = score_cols or ['A_WO', 'A_GR', 'A_CO', 'A_TT']
        self.trait_map = {'A_WO': 'WO', 'A_GR': 'GR', 'A_CO': 'CO', 'A_TT': 'TT'}

        self.dirs = prepare_dirs(self.base_dir)
        self.logger = setup_logging(self.dirs["logs"])

        save_session_config(self.dirs, {
            "base_dir": str(self.base_dir),
            "score_cols": self.score_cols,
            "labels": LABELS,
            "shap_threshold": SHAP_THRESHOLD,
            "config_source": "hardcoded"
        })

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
            'random_state': 42,
        }
        # Map hard-coded dict into runner's expected 'fs' block
        base = BEST_PARAMS.get(score_col, {})
        fs_method = str(base.get('feature_selection_method', 'none')).lower()
        fs_block = {'type': 'none'}
        if fs_method == 'kbest':
            if 'kbest_k' in base:
                fs_block = {'type': 'kbest', 'k': base['kbest_k'], 'score_func': 'f_regression'}
            else:
                fs_block = {'type': 'none'}
        elif fs_method == 'lasso':
            fs_block = {'type': 'lasso', 'use_cv': True, 'cv': 5}

        merged = {
            **defaults,
            **{k: v for k, v in base.items() if k in defaults},
            'fs': fs_block
        }
        return merged

    def _trait_dirs(self, trait_short: str) -> Dict[str, Path]:
        pred_dir = self.dirs["predictions"] / trait_short
        cm_dir = self.dirs["cm"] / trait_short
        pred_dir.mkdir(parents=True, exist_ok=True)
        cm_dir.mkdir(parents=True, exist_ok=True)
        return {"pred": pred_dir, "cm": cm_dir}

    def run(self):
        self.logger.info("="*60)
        self.logger.info("Starting FabOF Tuned Multi-Trait Runner (HARDCODED params + hard-coded preprocessing order)")
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

            # SHAP parity with Script 2
            X_stack_final_list = []
            shap_vals_list = []
            selected_feature_names = None
            last_rf_model = None

            for fold in range(1, 10 + 1):
                try:
                    tr, te = load_fold(self.base_dir, fold)
                    feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]

                    X_tr, X_te = tr[feats].values, te[feats].values
                    # Script-2 parity: do NOT cast to int here
                    y_tr, y_te = tr[sc].values, te[sc].values

                    # FS stage: impute → FS (no scaling here)
                    X_tr_fs, X_te_fs, mask = apply_feature_selection_for_fs(
                        X_tr, X_te, y_tr,
                        p.get('imputation_strategy', 'median'),
                        p.get('fs', None),
                        self.logger
                    )
                    if selected_feature_names is None:
                        selected_feature_names = np.array(feats)[mask].tolist()

                    # Model stage: impute + scale inside fit / predict
                    rf, borders, scaler, model_imputer = fit_fabof_rf_with_internal_preproc(
                        X_tr_fs, y_tr, p,
                        p.get('imputation_strategy', 'median'),
                        p.get('scaling_method', 'standard')
                    )
                    y_pred_ord, y_pred_raw, X_te_final = predict_fabof_with_internal_preproc(
                        rf, borders, X_te_fs, scaler, model_imputer
                    )

                    # Per-fold SHAP (Script 2 behavior)
                    try:
                        explainer = shap.TreeExplainer(rf)
                        shap_vals_fold = explainer.shap_values(X_te_final)
                        if isinstance(shap_vals_fold, list):
                            shap_vals_fold = shap_vals_fold[0]
                        shap_vals_list.append(shap_vals_fold)
                        X_stack_final_list.append(X_te_final)
                    except Exception as shap_e:
                        self.logger.warning(f"[{trait}] SHAP fold {fold} failed: {shap_e}")

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

                    last_rf_model = rf
                    all_files.extend(te[self.name_col].tolist())
                    all_true_ord.extend(y_te.tolist())
                    all_preds_ord.extend(y_pred_ord.tolist())
                    all_folds.extend([fold]*len(y_te))

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

            # SHAP: stack per-fold shap + X like Script 2
            if shap_vals_list and selected_feature_names is not None and last_rf_model is not None:
                try:
                    X_stack_final = np.vstack(X_stack_final_list)
                    shap_stacked = np.vstack(shap_vals_list)
                    shap_dirs = {
                        "tables": self.dirs["shap_tables"],
                        "composite": self.dirs["shap_composite"],
                        "waterfalls": self.dirs["shap_waterfalls"],
                        "dep": self.dirs["shap_dep"]
                    }
                    shap_comprehensive_outputs(
                        trait, last_rf_model, X_stack_final, selected_feature_names,
                        shap_dirs, self.logger, precomputed_shap=shap_stacked
                    )
                except Exception as e:
                    self.logger.warning(f"[{trait}] SHAP stacking failed: {e}")

            if all_preds_ord:
                acc = (np.array(all_preds_ord) == np.array(all_true_ord)).mean()
                master_rows.append({"Trait": trait, "Samples": len(all_preds_ord), "Simple_Accuracy": float(acc)})

        # No change: composites and summaries
        composite_confusions(self.dirs["root"], agg_cms_by_trait)

        if master_rows:
            master_df = pd.DataFrame(master_rows)
            master_df.to_csv(self.dirs["master_agg"] / "MASTER_aggregated_predictions_summary.csv", index=False)

        try:
            all_trait_files = []
            for trait in ['WO', 'GR', 'CO', 'TT']:
                pth = self.dirs["predictions"] / trait / f"{trait}_aggregated_predictions.csv"
                if pth.exists():
                    all_trait_files.append(pd.read_csv(pth))
            if all_trait_files:
                master_pred = pd.concat(all_trait_files, ignore_index=True)
                master_pred.to_csv(self.dirs["master_agg"] / "MASTER_aggregated_predictions_all_traits.csv", index=False)
        except Exception as e:
            self.logger.warning(f"Failed to build master aggregated predictions: {e}")

        self._save_aggregate_summary(metrics_by_trait)

        with open(self.dirs["reports"] / "final_report.md", "w", encoding="utf-8") as f:
            f.write("# FabOF Tuned Multi-Trait Run (HARDCODED params)\n\n")
            f.write(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Base dir: {self.base_dir}\n")
            f.write(f"- Output: {self.dirs['root']}\n")
            f.write(f"- Traits: {', '.join([self.trait_map.get(s, s) for s in self.score_cols])}\n")
            f.write(f"- Labels: {LABELS}\n")
            f.write(f"- SHAP threshold: {SHAP_THRESHOLD}\n")
            f.write(f"- Config source: hardcoded\n")

        self.logger.info("\nAll done.")
        self.logger.info(f"Results saved to: {self.dirs['root']}")

    def _save_aggregate_summary(self, metrics_by_trait: Dict[str, List[Dict[str, float]]]):
        metrics_list = ["Precision", "Recall", "Accuracy", "MAE", "MSE", "R2", "F1-macro", "QWK"]
        rows = []
        trait_means = {m: [] for m in metrics_list}
        trait_stds = {m: [] for m in metrics_list}

        for trait in ["WO", "GR", "CO", "TT"]:
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
            rows.append({"Trait": trait, **mean_row, **std_row})

        agg_means = {m: float(np.mean(trait_means[m])) for m in metrics_list}
        agg_stds = {m + "_std": float(np.mean(trait_stds[m])) for m in metrics_list}
        rows.append({"Trait": "AGGREGATE", **agg_means, **agg_stds})

        out_cols = ["Trait"] + [m for m in metrics_list] + [m + "_std" for m in metrics_list]
        pd.DataFrame(rows, columns=out_cols).to_csv(self.dirs["summaries"] / "AGGREGATE_metrics_summary.csv", index=False)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = "Inputfile"

    runner = FabOFTunedRunner(base_dir=base_dir, name_col=name_col,
                              score_cols=['A_WO', 'A_GR', 'A_CO', 'A_TT'])
    try:
        runner.run()
        print("="*60)
        print("TUNED MULTI-TRAIT RUN (HARDCODED) COMPLETED")
        print("="*60)
        print(f"Results saved to: {runner.dirs['root']}")
    except Exception as e:
        print(f"Tuned run failed: {e}")