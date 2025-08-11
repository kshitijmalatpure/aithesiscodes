"""
fabOF_v8_4_0_binary_with_feature_selection.py
=============================================
Binary classification (0,1) with median imputation, scaling,
RF-based FabOF boundary, **feature selection** (Baseline, KBest: 50/100/150/200 and LASSO),
enhanced (but non-redundant) confusion matrices, and aggregated results only.

Differences vs 4-class version:
- Labels are binary: 0,1 (with auto-fallback mapping {1,2}->0, {3,4}->1 if needed).
- Single FabOF boundary derived from OOB predictions at P(y==0).
- Score columns renamed to A_WOB, A_GRB, A_COB, A_TTB.
- Outputs remain: one aggregated CM for best selector per trait, aggregated results CSV, aggregate predictions CSV.

Aggregated results CSV:
- Per (Trait, Selector): mean/std across folds.
- PLUS: one AGGREGATE row PER selector, averaging the per-trait means across traits.
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
    log_file = log_dir / f"fabOF_binary_execution_{timestamp}.log"
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
        lcv = LassoCV(cv=5, n_alphas=100, random_state=42, n_jobs=None).fit(X_train_scaled, y_train)
        coefs = lcv.coef_
        nz = np.where(np.abs(coefs) > 1e-8)[0]
        if nz.size == 0:
            order = np.argsort(np.abs(coefs))[::-1]
            nz = order[:min(100, n_features)]
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
def _fit_fabOF_selected_binary(X_train_sel: np.ndarray, y_train_bin: np.ndarray, n_trees=500, **rf_kwargs):
    """
    Train RF regressor on binary labels (0/1) and derive a single boundary from OOB predictions
    at the empirical proportion of class 0.
    """
    y_num = y_train_bin.astype(float)
    rf = RandomForestRegressor(
        n_estimators=n_trees, oob_score=True, random_state=42, **rf_kwargs
    ).fit(X_train_sel, y_num)

    oob_pred = rf.oob_prediction_

    # Single boundary at P(y==0)
    p0 = float((y_train_bin == 0).mean())
    boundary = np.quantile(oob_pred, p0) if 0.0 < p0 < 1.0 else 0.5

    # Guard rails for prediction-to-class conversion
    borders = np.array([-1.0, boundary, 2.0])
    return rf, borders


def _predict_fabOF_selected_binary(rf, borders, X_sel: np.ndarray):
    """
    Predict RF continuous scores and convert to binary labels via the learned boundary.
    """
    y_num = rf.predict(X_sel)
    y_bin = np.zeros_like(y_num, dtype=int)
    # Single inner boundary -> 1 if >= boundary, else 0
    y_bin += (y_num >= borders[1]).astype(int)
    return np.clip(y_bin, 0, 1), y_num


def _calc_metrics(y_true, y_pred_raw, y_pred_bin, labels):
    base_metrics = {
        'Accuracy': accuracy_score(y_true, y_pred_bin),
        'RMSE': rmse(y_true, y_pred_raw),
        'MAE': np.mean(np.abs(y_true - y_pred_raw)),
        'R2': 1 - np.sum((y_true - y_pred_raw) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
              if np.var(y_true) > 0 else 0.0,
        'F1_macro': f1_score(y_true, y_pred_bin, labels=labels, average='macro', zero_division=0),
        'Precision_macro': precision_score(y_true, y_pred_bin, labels=labels, average='macro', zero_division=0),
        'Recall_macro': recall_score(y_true, y_pred_bin, labels=labels, average='macro', zero_division=0),
        'QWK': cohen_kappa_score(y_true, y_pred_bin, labels=labels, weights='quadratic')
    }
    return base_metrics


class FabOFV840BinaryWithFS:
    """
    Binary pipeline with feature selection:
    - Selectors: Baseline, kbest_50/100/150/200, lasso
    - Aggregated metrics per (Score_Column, Selector)
    - Best selector per score column (by F1_macro mean) → single CM + aggregate predictions
    - Auto-fallback mapping {1,2}->0, {3,4}->1 if non-binary labels are encountered.
    """
    def __init__(self, base_dir, name_col, score_cols, labels=(0,1),
                 selectors=("baseline","kbest_50","kbest_100","kbest_150","kbest_200","lasso")):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = score_cols
        self.labels = np.array(labels, dtype=int)
        self.selectors = tuple(selectors)

        # Root folder (kept same name)
        self.root = self.base_dir / "comprehensive_fabOF_binary"
        self.root.mkdir(exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.root / 'logs')

        # Folders (trimmed)
        self.dirs = {
            'logs': self.root / 'logs',
            'predictions': self.root / 'predictions',
            'cm': self.root / 'confusion_matrices',
            'results': self.root  # aggregated CSV saved at root
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Log init
        self.logger.info("=" * 60)
        self.logger.info("FabOF v8.4.0 BINARY Classification WITH Feature Selection")
        self.logger.info("=" * 60)
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Results directory: {self.root}")
        self.logger.info(f"Name column: {self.name_col}")
        self.logger.info(f"Score columns: {self.score_cols}")
        self.logger.info(f"Labels: {self.labels.tolist()}")
        self.logger.info(f"Selectors: {self.selectors}")
        self.logger.info("Using median imputation + standard scaling before FS")

    @staticmethod
    def _maybe_binarize(y: np.ndarray, logger: logging.Logger, sc: str, fold: int):
        """If labels are 1..4, map {1,2}->0 and {3,4}->1; if already binary (0/1), return as-is."""
        uniq = np.unique(y[~pd.isna(y)])
        if np.all(np.isin(uniq, [0,1])):
            return y.astype(int)
        if np.all(np.isin(uniq, [1,2,3,4])):
            logger.info(f"    Detected 1..4 labels in fold {fold}, column {sc}; mapping {{1,2}}->0, {{3,4}}->1.")
            y_map = np.where(y <= 2, 0, 1).astype(int)
            return y_map
        logger.warning(f"    Unexpected label set {uniq} in fold {fold}, column {sc}; "
                       f"this run expects 0/1 (or 1..4 for auto-map).")
        return y.astype(int)

    def run(self):
        self.logger.info("Starting BINARY pipeline with feature selection...")
        pipeline_start_time = time.time()

        total_operations = len(self.score_cols) * 10 * len(self.selectors)
        current_operation = 0

        # store per-fold metrics for aggregation
        all_rows = []

        for sc_idx, sc in enumerate(self.score_cols, 1):
            self.logger.info(f"[{sc_idx}/{len(self.score_cols)}] Score Column: {sc}")

            # track results per selector to pick the best one
            selector_fold_metrics: Dict[str, list] = {m: [] for m in self.selectors}
            selector_cms: Dict[str, np.ndarray] = {m: np.zeros((len(self.labels), len(self.labels)), dtype=int) for m in self.selectors}

            # For saving predictions only for BEST selector
            selector_predictions: Dict[str, Dict[str, list]] = {
                m: {"fold": [], "true": [], "pred": []} for m in self.selectors
            }

            for fold in range(1, 11):
                for method in self.selectors:
                    current_operation += 1
                    progress = (current_operation / total_operations) * 100
                    self.logger.info(f"  Fold {fold}/10 | {method:>10s} - Progress: {progress:.1f}%")
                    try:
                        tr_path = self.base_dir / f"fold_{fold}" / f"fold{fold}trb.csv"
                        te_path = self.base_dir / f"fold_{fold}" / f"fold{fold}teb.csv"
                        if not tr_path.exists() or not te_path.exists():
                            self.logger.error(f"    Missing files for fold {fold}: {tr_path} or {te_path}")
                            continue

                        tr = pd.read_csv(tr_path)
                        te = pd.read_csv(te_path)
                        tr, te = tr.align(te, axis=1, fill_value=np.nan)

                        feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]
                        X_tr, X_te = tr[feats].values, te[feats].values

                        y_tr_raw, y_te_raw = tr[sc].values, te[sc].values
                        y_tr = self._maybe_binarize(y_tr_raw, self.logger, sc, fold)
                        y_te = self._maybe_binarize(y_te_raw, self.logger, sc, fold)

                        # Validate binary labels
                        if not np.all(np.isin(y_tr, [0,1])) or not np.all(np.isin(y_te, [0,1])):
                            self.logger.warning(f"    Non-binary labels in fold {fold}, column {sc}")
                            continue

                        # Preprocess (fit on train only)
                        imputer, scaler, X_tr_scaled = _fit_preprocess(X_tr)
                        X_te_scaled = _apply_preprocess(X_te, imputer, scaler)

                        # Feature selection (fit on train only)
                        sel = _fit_selector(method, X_tr_scaled, y_tr)
                        X_tr_sel = sel.transform(X_tr_scaled)
                        X_te_sel = sel.transform(X_te_scaled)

                        # FabOF training/prediction on selected features (binary)
                        rf, borders = _fit_fabOF_selected_binary(X_tr_sel, y_tr)
                        y_pred_bin, y_pred_raw = _predict_fabOF_selected_binary(rf, borders, X_te_sel)

                        # metrics
                        met = _calc_metrics(y_te, y_pred_raw, y_pred_bin, self.labels)
                        met.update({'Score_Column': sc, 'Fold': fold, 'Selector': method})
                        all_rows.append(met)
                        selector_fold_metrics[method].append(met)

                        # confusion matrix accumulation
                        fold_cm = confusion_matrix(y_te, y_pred_bin, labels=self.labels)
                        selector_cms[method] += fold_cm

                        # predictions (for best-selector export later)
                        selector_predictions[method]["fold"].extend([fold] * len(y_te))
                        selector_predictions[method]["true"].extend(y_te.tolist())
                        selector_predictions[method]["pred"].extend(y_pred_bin.tolist())

                    except Exception as e:
                        self.logger.error(f"    Error in fold {fold} ({method}): {str(e)}")
                        import traceback
                        self.logger.debug(traceback.format_exc())
                        continue

            # Pick best selector by mean F1_macro
            best_selector = None
            best_f1 = -np.inf
            for method, rows in selector_fold_metrics.items():
                if not rows:
                    continue
                f1_mean = np.mean([r['F1_macro'] for r in rows])
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    best_selector = method

            if best_selector is None:
                self.logger.warning(f"No valid results for {sc}; skipping CM/predictions.")
                continue

            self.logger.info(f"Best selector for {sc}: {best_selector} (F1_macro={best_f1:.4f})")

            # Save single (aggregated) confusion matrix for THIS score column using best selector
            title = f"Aggregated Confusion Matrix - {sc} (10 folds) | {best_selector}"
            cm_path = self.dirs['cm'] / f"{sc}_confusion_matrix_{best_selector}.png"
            save_confusion_matrix(selector_cms[best_selector], self.labels, title, cm_path)

            # Save aggregate predictions for BEST selector only
            pred_pack = selector_predictions[best_selector]
            if pred_pack["pred"]:
                aggregate_predictions_df = pd.DataFrame({
                    'Fold': pred_pack["fold"],
                    'True_Value': pred_pack["true"],
                    'Predicted_Value': pred_pack["pred"],
                })
                aggregate_predictions_df['Absolute_Error'] = (aggregate_predictions_df['True_Value'] - aggregate_predictions_df['Predicted_Value']).abs()
                aggregate_predictions_df['Squared_Error'] = (aggregate_predictions_df['True_Value'] - aggregate_predictions_df['Predicted_Value']) ** 2
                aggregate_predictions_df.to_csv(
                    self.dirs['predictions'] / f"{sc}_aggregate_predictions_{best_selector}.csv",
                    index=False
                )

        # Write aggregated (mean/std across folds) metrics ONLY
        self._save_aggregated_results(all_rows)

        pipeline_time = time.time() - pipeline_start_time
        self.logger.info("=" * 60)
        self.logger.info("FabOF BINARY WITH FS Pipeline COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total execution time: {pipeline_time:.2f}s ({pipeline_time/60:.1f} minutes)")
        self.logger.info(f"Results saved to: {self.root}")

    def _save_aggregated_results(self, all_rows):
        """
        Save a single CSV with aggregated metrics (mean/std across folds)
        Rows per (Score_Column, Selector) and AGGREGATE rows per Selector
        (averaging per-trait means across traits). No global ALL-selector row.
        """
        if not all_rows:
            self.logger.warning("No metrics collected; skipping aggregated results CSV.")
            return

        df = pd.DataFrame(all_rows)
        metrics_cols = ['Accuracy', 'RMSE', 'MAE', 'R2', 'F1_macro', 'Precision_macro', 'Recall_macro', 'QWK']

        rows = []

        # 1) Per-(Trait, Selector): mean/std across folds
        for (sc, sel), d in df.groupby(['Score_Column', 'Selector']):
            agg = {f'{m}_mean': d[m].mean() for m in metrics_cols}
            agg.update({f'{m}_std': d[m].std(ddof=0) for m in metrics_cols})
            agg['Score_Column'] = sc
            agg['Selector'] = sel
            rows.append(agg)

        # 2) Aggregate per Selector across Traits:
        #    First compute per-trait means per selector (across folds), then average across traits.
        trait_list = list(self.score_cols)
        df_trait_means = (
            df[df['Score_Column'].isin(trait_list)]
            .groupby(['Score_Column', 'Selector'])[metrics_cols]
            .mean()
            .reset_index()
        )

        for sel, d in df_trait_means.groupby('Selector'):
            agg_sel = {f'{m}_mean': d[m].mean() for m in metrics_cols}
            agg_sel.update({f'{m}_std': d[m].std(ddof=0) for m in metrics_cols})  # variability across traits
            agg_sel['Score_Column'] = 'AGGREGATE'
            agg_sel['Selector'] = sel
            rows.append(agg_sel)

        out_df = pd.DataFrame(rows)
        out_path = self.dirs['results'] / "all_models_aggregated_results.csv"
        out_df.to_csv(out_path, index=False)
        self.logger.info(f"Aggregated results saved: {out_path}")


# ----------------------------------------------------------
if __name__ == "__main__":
    # Binary configuration
    base_dir = r"C:\\Research\\AI Folder\\Thesis\\Data\\data_CTO_Kshitij\\Main\\10-fold-final"
    name_col = 'Inputfile'
    # Renamed binary traits
    score_cols = ['A_WOB', 'A_GRB', 'A_COB', 'A_TTB']
    labels = [0, 1]

    selectors = ("baseline","kbest_50","kbest_100","kbest_150","kbest_200","lasso")

    try:
        pipeline = FabOFV840BinaryWithFS(base_dir, name_col, score_cols, labels, selectors)
        pipeline.run()

        print("\n" + "=" * 60)
        print("BINARY PIPELINE WITH FEATURE SELECTION COMPLETED!")
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
