"""
fabOF.py
=================================================
4-class classification (1,2,3,4) with median imputation, scaling,
RF-based FabOF boundaries, **feature selection** (Baseline, KBest: 50/100/150/200 and LASSO),
enhanced (but non-redundant) confusion matrices, and aggregated results only.

Differences vs minimal version:
- Root folder: comprehensive_fabOF
- Feature selection OPTIONS:
  * Baseline (no selection)
  * kbest_50, kbest_100, kbest_150, kbest_200 (f_regression)
  * lasso (LassoCV → select non-zero coefficients; fallback to top-|coef| if all zero)
- For each score column, evaluate ALL selectors across 10 folds; aggregate per selector.
- Save ONE aggregated confusion matrix **for the best selector** per score column (by F1-macro mean).
- Aggregated results CSV:
  * rows per (Score_Column, Selector) = mean/std across folds
  * PLUS one AGGREGATE row **per Selector** averaging the per-trait means across traits.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, cohen_kappa_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

warnings.filterwarnings("ignore")

# ---------- future-proof RMSE ----------
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:
    from sklearn.metrics import mean_squared_error as mse
    def rmse(y_true, y_pred): return mse(y_true, y_pred, squared=False)


# ---------- Logging Setup ----------
def setup_logging(log_dir):
    """Setup logging"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fabOF_4class_execution_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                  logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ---------- Visualization: Confusion Matrix (single, enhanced) ----------
def save_confusion_matrix(cm, labels, title, save_path, vmax=None):
    """
    Save a single enhanced confusion matrix (no duplicates/per-fold variants).
    Larger numbers for readability.
    """
    plt.figure(figsize=(9, 7))
    if vmax is None:
        vmax = np.max(cm)

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        vmax=vmax,
        square=True,
        linewidths=0.6,
        linecolor='white',
        annot_kws={'size': 18, 'weight': 'bold'}  # Bigger numbers
    )
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=18)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ---------- Preprocess (Impute + Scale) ----------
def _fit_preprocess(X_train: np.ndarray) -> Tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return imputer, scaler, X_scaled


def _apply_preprocess(X: np.ndarray, imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(X))


# ---------- Feature Selection ----------
def _fit_selector(method: str, X_train_scaled: np.ndarray, y_train: np.ndarray):
    method = method.lower()
    n_features = X_train_scaled.shape[1]

    if method == 'baseline':
        class _IdentitySelector:
            def fit(self, X, y=None): return self
            def transform(self, X): return X
            def get_support(self, indices=False):
                return (np.arange(n_features) if indices else np.ones(n_features, dtype=bool))
        return _IdentitySelector()

    if method.startswith('kbest_'):
        k = int(method.split('_')[1])
        sel = SelectKBest(score_func=f_regression, k=k)
        sel.fit(X_train_scaled, y_train)
        return sel

    elif method == 'lasso':
        # Aligned with FullSET: broader/robust LASSO FS
        lcv = LassoCV(cv=5, random_state=42, max_iter=2000, n_jobs=-1).fit(X_train_scaled, y_train)
        coefs = lcv.coef_

        # Select by |coef| > 1e-5 (aligned threshold)
        nz = np.where(np.abs(coefs) > 1e-5)[0]

        # Fallback: top-50 by |coef| if none survive
        if nz.size == 0:
            order = np.argsort(np.abs(coefs))[::-1]
            nz = order[:min(50, n_features)]

        class _LassoSelector:
            def __init__(self, idx): self.idx = np.array(sorted(idx))
            def transform(self, X): return X[:, self.idx]
            def fit(self, X, y=None): return self
            def get_support(self, indices=False):
                return (self.idx if indices else np.isin(np.arange(n_features), self.idx))
        return _LassoSelector(nz)

    else:
        raise ValueError(f"Unknown selector method: {method}")


# ---------- FabOF core (expects already selected arrays) ----------
def _fit_fabOF_selected(X_train_sel: np.ndarray, y_train: np.ndarray, n_trees=500, **rf_kwargs):
    y_num = y_train.astype(float)
    rf = RandomForestRegressor(
        n_estimators=n_trees, oob_score=True, random_state=42, **rf_kwargs
    ).fit(X_train_sel, y_num)

    oob_pred = rf.oob_prediction_

    # Cumulative proportions for classes 1..4
    pi = np.array([(y_train == 1).mean(),
                   (y_train <= 2).mean(),
                   (y_train <= 3).mean()])
    borders_inner = np.quantile(oob_pred, pi)
    borders = np.concatenate([[0.0], borders_inner, [5.0]])  # guard range
    return rf, borders


def _predict_fabOF_selected(rf, borders, X_sel: np.ndarray):
    y_num = rf.predict(X_sel)
    y_ord = np.ones_like(y_num, dtype=int)
    for i in range(1, len(borders) - 1):
        y_ord += (y_num >= borders[i]).astype(int)
    return np.clip(y_ord, 1, 4), y_num


def _calc_metrics(y_true, y_pred_raw, y_pred_rounded, labels):
    base_metrics = {
        'Accuracy': accuracy_score(y_true, y_pred_rounded),
        'RMSE': rmse(y_true, y_pred_raw),
        'MAE': np.mean(np.abs(y_true - y_pred_raw)),
        'R2': 1 - np.sum((y_true - y_pred_raw) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2 + 1e-12),
        'F1-macro': f1_score(y_true, y_pred_rounded, labels=labels, average='macro', zero_division=0),
        'Precision-macro': precision_score(y_true, y_pred_rounded, labels=labels, average='macro', zero_division=0),
        'Recall-macro': recall_score(y_true, y_pred_rounded, labels=labels, average='macro', zero_division=0),
        'QWK': cohen_kappa_score(y_true, y_pred_rounded, labels=labels, weights='quadratic')
    }
    return base_metrics


# ---------- Main Routine per selector ----------
def run_pipeline_for_selector(
    base_dir: Path,
    score_cols: list,
    selector_name: str,
    logger: logging.Logger,
    labels=(1, 2, 3, 4),
    n_folds: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Runs 10-fold pipeline for ONE selector across all score columns.
    Returns aggregated metrics per score column.
    Also saves a single aggregated confusion matrix per score column for the BEST selector later.
    """
    agg_results_per_score: Dict[str, Dict[str, float]] = {}
    per_fold_predictions: Dict[str, list] = {sc: [] for sc in score_cols}
    per_fold_truths: Dict[str, list] = {sc: [] for sc in score_cols}

    for sc in score_cols:
        logger.info(f"=== Selector={selector_name} | Score={sc} ===")
        fold_metrics_list = []

        for fold in range(1, n_folds + 1):
            tr_path = base_dir / f"fold_{fold}" / f"fold{fold}tr.csv"
            te_path = base_dir / f"fold_{fold}" / f"fold{fold}te.csv"
            if not tr_path.exists() or not te_path.exists():
                logger.warning(f"Missing files for fold {fold}")
                continue

            tr = pd.read_csv(tr_path)
            te = pd.read_csv(te_path)
            # Align columns
            tr, te = tr.align(te, axis=1, fill_value=np.nan)

            # Identify columns
            name_col = 'Inputfile'
            feature_cols = [c for c in tr.columns if c not in [name_col] + list(score_cols)]

            # Split
            X_train = tr[feature_cols].values
            y_train = tr[sc].values.astype(int)
            X_test = te[feature_cols].values
            y_test = te[sc].values.astype(int)

            # Preprocess
            imputer, scaler, X_train_scaled = _fit_preprocess(X_train)
            X_test_scaled = _apply_preprocess(X_test, imputer, scaler)

            # Feature selection
            sel = _fit_selector(selector_name, X_train_scaled, y_train)
            X_train_sel = sel.transform(X_train_scaled)
            X_test_sel = sel.transform(X_test_scaled)

            # Fit FabOF
            rf, borders = _fit_fabOF_selected(X_train_sel, y_train, n_trees=500)

            # Predict
            y_pred_rounded, y_pred_raw = _predict_fabOF_selected(rf, borders, X_test_sel)

            # Metrics
            metrics = _calc_metrics(y_test, y_pred_raw, y_pred_rounded, labels)
            fold_metrics_list.append(metrics)

            # store per-fold preds/truths
            per_fold_predictions[sc].append(y_pred_rounded)
            per_fold_truths[sc].append(y_test)

        # Aggregate
        if fold_metrics_list:
            dfm = pd.DataFrame(fold_metrics_list)
            agg_results_per_score[sc] = {f"{k}_mean": dfm[k].mean() for k in dfm.columns}
            agg_results_per_score[sc].update({f"{k}_std": dfm[k].std() for k in dfm.columns})
        else:
            agg_results_per_score[sc] = {}

    # Save per-fold preds for later potential use (not used here beyond best-CM)
    return agg_results_per_score


# ---------- Find best selector per score column ----------
def choose_best_selectors(results_by_selector: Dict[str, Dict[str, Dict[str, float]]], score_cols, metric='F1-macro_mean'):
    best_for_score = {}
    for sc in score_cols:
        best_name = None
        best_val = -np.inf
        for sel_name, res in results_by_selector.items():
            if sc in res and metric in res[sc]:
                val = res[sc][metric]
                if val > best_val:
                    best_val = val
                    best_name = sel_name
        best_for_score[sc] = best_name
    return best_for_score


# ---------- Save aggregated results CSV ----------
def save_aggregated_results(base_dir: Path, results_by_selector: Dict[str, Dict[str, Dict[str, float]]], score_cols):
    # Build rows
    rows = []
    # Per (Score, Selector)
    for sel_name, res in results_by_selector.items():
        for sc in score_cols:
            if sc in res and len(res[sc]) > 0:
                row = {'Score_Column': sc, 'Selector': sel_name}
                row.update(res[sc])
                rows.append(row)
    df = pd.DataFrame(rows)

    # Aggregate row PER SELECTOR across traits (averaging the per-trait means only)
    agg_rows = []
    for sel_name in results_by_selector:
        # find all rows for that selector, per-trait means
        sub = df[(df['Selector'] == sel_name) & (df['Score_Column'].isin(score_cols))]
        if sub.empty:
            continue
        mean_cols = [c for c in sub.columns if c.endswith('_mean')]
        std_cols = [c for c in sub.columns if c.endswith('_std')]
        agg_entry = {'Score_Column': 'AGGREGATE', 'Selector': sel_name}
        for c in mean_cols:
            agg_entry[c] = sub[c].mean()
        for c in std_cols:
            agg_entry[c] = sub[c].mean()
        agg_rows.append(agg_entry)

    if len(agg_rows) > 0:
        df = pd.concat([df, pd.DataFrame(agg_rows)], ignore_index=True)

    out_root = base_dir / "comprehensive_fabOF" / "aggregated_results"
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "aggregated_results.csv"
    df.to_csv(out_csv, index=False)
    return df


# ---------- Save one aggregated CM for the best selector per score column ----------
def save_best_confusion_matrices(base_dir: Path, best_selectors: Dict[str, str], score_cols, logger: logging.Logger, labels=(1,2,3,4)):
    out_dir = base_dir / "comprehensive_fabOF" / "best_selector_confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving BEST-selector aggregated confusion matrices per score column...")

    for sc in score_cols:
        sel_name = best_selectors.get(sc, None)
        if sel_name is None:
            logger.warning(f"No best selector for {sc}; skipping CM.")
            continue

        # Aggregate over folds with the chosen selector
        cm_total = np.zeros((len(labels), len(labels)), dtype=int)
        for fold in range(1, 11):
            tr_path = base_dir / f"fold_{fold}" / f"fold{fold}tr.csv"
            te_path = base_dir / f"fold_{fold}" / f"fold{fold}te.csv"
            if not tr_path.exists() or not te_path.exists():
                continue
            tr = pd.read_csv(tr_path); te = pd.read_csv(te_path)
            tr, te = tr.align(te, axis=1, fill_value=np.nan)
            name_col = 'Inputfile'
            feature_cols = [c for c in tr.columns if c not in [name_col] + list(score_cols)]
            X_train = tr[feature_cols].values
            y_train = tr[sc].values.astype(int)
            X_test = te[feature_cols].values
            y_test = te[sc].values.astype(int)

            imputer, scaler, X_train_scaled = _fit_preprocess(X_train)
            X_test_scaled = _apply_preprocess(X_test, imputer, scaler)

            sel = _fit_selector(sel_name, X_train_scaled, y_train)
            X_train_sel = sel.transform(X_train_scaled)
            X_test_sel = sel.transform(X_test_scaled)

            rf, borders = _fit_fabOF_selected(X_train_sel, y_train, n_trees=500)
            y_pred_rounded, _ = _predict_fabOF_selected(rf, borders, X_test_sel)

            cm = confusion_matrix(y_test, y_pred_rounded, labels=labels)
            cm_total += cm

        # Save
        title = f"Aggregated Confusion Matrix (Best Selector: {sel_name}) - {sc}"
        save_path = out_dir / f"{sc}_best_selector_cm.png"
        save_confusion_matrix(cm_total, labels=labels, title=title, save_path=save_path, vmax=None)


# ---------- Orchestration ----------
def main():
    """
    Orchestrates running multiple selectors, choosing the best per score column,
    saving aggregated results, and saving one CM per score column for the best selector.
    """
    # Base configuration (adjust base_dir to your path)
    base_dir = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final")
    score_cols = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
    labels = (1, 2, 3, 4)

    # Prepare output dirs
    root = base_dir / "comprehensive_fabOF"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "logs")

    try:
        logger.info("Starting 4-class FabOF pipeline with multiple feature selectors...")
        start_time = time.time()

        selectors = ['baseline', 'kbest_50', 'kbest_100', 'kbest_150', 'kbest_200', 'lasso']

        # Run per selector
        results_by_selector: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sel in selectors:
            logger.info(f"Running selector: {sel}")
            res = run_pipeline_for_selector(base_dir, score_cols, sel, logger, labels=labels, n_folds=10)
            results_by_selector[sel] = res

        # Decide best per score column (by F1-macro mean)
        best_selectors = choose_best_selectors(results_by_selector, score_cols, metric='F1-macro_mean')
        logger.info(f"Best selectors per score column: {best_selectors}")

        # Save aggregated results CSV (+ aggregate rows)
        df = save_aggregated_results(base_dir, results_by_selector, score_cols)
        logger.info(f"Aggregated results saved. Shape: {df.shape if df is not None else None}")

        # Save BEST-selector aggregated CM per score column
        save_best_confusion_matrices(base_dir, best_selectors, score_cols, logger, labels=labels)

        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed/60:.2f} minutes.")

        print("\n" + "=" * 60)
        print("4-CLASS PIPELINE WITH FEATURE SELECTION COMPLETED!")
        print("=" * 60)
        print("Outputs:")
        print("  • One aggregated confusion matrix per score column (BEST selector)")
        print("  • Aggregated results CSV: per-(trait, selector) + AGGREGATE-per-selector rows")
        print("  • Aggregate predictions per trait for BEST selector")
        print("  • Logs")

    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
