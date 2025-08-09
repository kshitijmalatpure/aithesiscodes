"""
fabOF_v8_4_0_binary_minimal.py
==============================
Binary classification (0/1) with median imputation, RF-based FabOF boundary,
single (aggregated) confusion matrix per score column with bigger numbers,
and aggregated results only (means/std over 10 folds).

Mirrors the 4-class minimal pipeline, adapted to binary:
- Root folder: comprehensive_fabOF_binary
- No feature selection, SHAP, or feature importance
- Reads *b* folds: fold{n}trb.csv / fold{n}teb.csv
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, cohen_kappa_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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


# ---------- Confusion Matrix (single, enhanced) ----------
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


# ---------- FabOF helpers (binary) ----------
def _fit_fabOF_binary(X_train, y_train, n_trees=500, **rf_kwargs):
    """
    Train RF regressor and derive a single decision border from OOB predictions
    using the prevalence-based quantile for 0/1 mapping.

    Border is set at quantile q = P(y==0) of the OOB predictions.
    """
    # Median imputation + scaling
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    y_num = y_train.astype(float)
    rf = RandomForestRegressor(
        n_estimators=n_trees, oob_score=True, random_state=42, **rf_kwargs
    ).fit(X_train_scaled, y_num)

    oob_pred = rf.oob_prediction_

    # Prevalence-based threshold (FabOF-style)
    p0 = (y_train == 0).mean()
    # Guard against degenerate folds (all 0 or all 1)
    if p0 == 0.0 or p0 == 1.0:
        # Fallback to 0.5 if a single-class fold sneaks in
        border = 0.5
    else:
        border = float(np.quantile(oob_pred, p0))

    # Borders array for reuse with 4-class-style mapper
    borders = np.array([-1.0, border, 2.0])  # guard range
    return rf, borders, scaler, imputer


def _predict_fabOF_binary(rf, borders, X, scaler, imputer):
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    y_num = rf.predict(X_scaled)

    # Map to classes 0/1 via learned border
    y_bin = (y_num >= borders[1]).astype(int)
    return y_bin, y_num


def _calc_metrics_binary(y_true, y_pred_raw, y_pred_bin, labels):
    """Regression+classification metrics for binary 0/1"""
    # Handle constant y_true for R2
    if np.var(y_true) > 0:
        r2_val = 1 - np.sum((y_true - y_pred_raw) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    else:
        r2_val = 0.0

    base_metrics = {
        'Accuracy': accuracy_score(y_true, y_pred_bin),
        'RMSE': rmse(y_true, y_pred_raw),
        'MAE': np.mean(np.abs(y_true - y_pred_raw)),
        'R2': r2_val,
        'F1_macro': f1_score(y_true, y_pred_bin, labels=labels, average='macro', zero_division=0),
        'Precision_macro': precision_score(y_true, y_pred_bin, labels=labels, average='macro', zero_division=0),
        'Recall_macro': recall_score(y_true, y_pred_bin, labels=labels, average='macro', zero_division=0),
        'QWK': cohen_kappa_score(y_true, y_pred_bin, labels=labels, weights='quadratic')
    }
    return base_metrics


class FabOFV840BinaryMinimal:
    """
    Minimal binary (0/1) pipeline per your constraints:
    - No feature selection
    - No SHAP
    - No feature importance
    - Only aggregated results and one CM per score column
    """
    def __init__(self, base_dir, name_col, score_cols, labels):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = score_cols
        self.labels = np.array(labels)

        # Root folder renamed
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
        self.logger.info("FabOF v8.4.0 Binary Classification (Minimal)")
        self.logger.info("=" * 60)
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Results directory: {self.root}")
        self.logger.info(f"Name column: {self.name_col}")
        self.logger.info(f"Score columns: {self.score_cols}")
        self.logger.info(f"Labels: {self.labels}")
        self.logger.info("Using median imputation")

    def run(self):
        """Main execution with aggregated outputs only"""
        self.logger.info("Starting pipeline...")
        pipeline_start_time = time.time()

        total_operations = len(self.score_cols) * 10
        current_operation = 0

        # store per-fold metrics to aggregate later
        all_rows = []
        global_vmax = 1  # for consistent CM scaling if desired

        for sc_idx, sc in enumerate(self.score_cols, 1):
            self.logger.info(f"[{sc_idx}/{len(self.score_cols)}] Score Column: {sc}")

            cms = np.zeros((len(self.labels), len(self.labels)), dtype=int)
            all_predictions = []
            all_true_values = []
            all_fold_info = []

            for fold in range(1, 11):
                current_operation += 1
                progress = (current_operation / total_operations) * 100
                self.logger.info(f"  Fold {fold}/10 - Progress: {progress:.1f}%")

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
                    y_tr, y_te = tr[sc].values, te[sc].values

                    # Ensure binary labels 0/1
                    if not np.all(np.isin(y_tr, [0, 1])) or not np.all(np.isin(y_te, [0, 1])):
                        self.logger.warning(f"    Non-binary labels in fold {fold}, column {sc}")
                        continue

                    # Degenerate fold guard (all zeros or all ones in train)
                    if len(np.unique(y_tr)) < 2:
                        self.logger.warning(f"    Single-class training fold for {sc}, fold {fold}; skipping.")
                        continue

                    # No feature selection; fit FabOF
                    rf, borders, scaler, imputer = _fit_fabOF_binary(X_tr, y_tr)
                    y_pred_bin, y_pred_raw = _predict_fabOF_binary(rf, borders, X_te, scaler, imputer)

                    # predictions for export
                    all_predictions.extend(y_pred_bin.tolist())
                    all_true_values.extend(y_te.tolist())
                    all_fold_info.extend([fold] * len(y_te))

                    # metrics (per fold, for later aggregation)
                    met = _calc_metrics_binary(y_te, y_pred_raw, y_pred_bin, self.labels)
                    met.update({'Score_Column': sc, 'Fold': fold})
                    all_rows.append(met)

                    # aggregate confusion matrix
                    fold_cm = confusion_matrix(y_te, y_pred_bin, labels=self.labels)
                    cms += fold_cm
                    if cms.max() > global_vmax:
                        global_vmax = int(cms.max())

                except Exception as e:
                    self.logger.error(f"    Error in fold {fold}: {str(e)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    continue

            # Save single (aggregated) confusion matrix for this score column
            title = f"Aggregated Confusion Matrix - {sc} (10 folds)"
            cm_path = self.dirs['cm'] / f"{sc}_confusion_matrix.png"
            save_confusion_matrix(cms, self.labels, title, cm_path, vmax=global_vmax)

            # Save aggregate predictions (optional; unchanged requirement)
            if all_predictions and all_true_values and all_fold_info:
                aggregate_predictions_df = pd.DataFrame({
                    'Fold': all_fold_info,
                    'True_Value': all_true_values,
                    'Predicted_Value': all_predictions,
                    'Absolute_Error': [abs(t - p) for t, p in zip(all_true_values, all_predictions)],
                    'Squared_Error': [(t - p) ** 2 for t, p in zip(all_true_values, all_predictions)]
                })
                aggregate_predictions_df.to_csv(
                    self.dirs['predictions'] / f"{sc}_aggregate_predictions.csv",
                    index=False
                )

        # Write aggregated (mean/std across folds) metrics ONLY
        self._save_aggregated_results(all_rows)

        pipeline_time = time.time() - pipeline_start_time
        self.logger.info("=" * 60)
        self.logger.info("FabOF Binary Minimal Pipeline COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total execution time: {pipeline_time:.2f}s ({pipeline_time/60:.1f} minutes)")
        self.logger.info(f"Results saved to: {self.root}")

    def _save_aggregated_results(self, all_rows):
        """
        Save a single CSV with aggregated metrics (mean/std across folds)
        One row per Score_Column, plus an overall AGGREGATE row.
        """
        if not all_rows:
            self.logger.warning("No metrics collected; skipping aggregated results CSV.")
            return

        df = pd.DataFrame(all_rows)
        metrics_cols = ['Accuracy', 'RMSE', 'MAE', 'R2', 'F1_macro', 'Precision_macro', 'Recall_macro', 'QWK']

        # Per-score-column aggregation
        rows = []
        for sc in sorted(df['Score_Column'].unique()):
            d = df[df['Score_Column'] == sc]
            agg = {f'{m}_mean': d[m].mean() for m in metrics_cols}
            agg.update({f'{m}_std': d[m].std(ddof=0) for m in metrics_cols})
            agg['Score_Column'] = sc
            rows.append(agg)

        # Overall aggregate across all score columns
        agg_all = {f'{m}_mean': df[m].mean() for m in metrics_cols}
        agg_all.update({f'{m}_std': df[m].std(ddof=0) for m in metrics_cols})
        agg_all['Score_Column'] = 'AGGREGATE'
        rows.append(agg_all)

        out_df = pd.DataFrame(rows)
        out_path = self.dirs['results'] / "all_models_aggregated_results_binary.csv"
        out_df.to_csv(out_path, index=False)
        self.logger.info(f"Aggregated results saved: {out_path}")


# ----------------------------------------------------------
if __name__ == "__main__":
    # Binary configuration
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = 'Inputfile'
    score_cols = ['A_WOB', 'A_GRB', 'A_COB', 'A_TTB']  # binary target columns
    labels = [0, 1]

    try:
        pipeline = FabOFV840BinaryMinimal(base_dir, name_col, score_cols, labels)
        pipeline.run()

        print("\n" + "=" * 60)
        print("MINIMAL BINARY CLASSIFICATION PIPELINE COMPLETED!")
        print("=" * 60)
        print("Outputs:")
        print("  • One aggregated confusion matrix per score column (bigger numbers)")
        print("  • Aggregated results CSV (mean/std over folds) incl. per trait + overall")
        print("  • Aggregate predictions per trait")
        print("  • Logs")

    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
