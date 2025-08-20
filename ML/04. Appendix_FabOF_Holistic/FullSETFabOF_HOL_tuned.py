"""
fabOF_v8_4_0_tuned_pipeline_FIXED_WITH_BEESWARM_A_HOL.py
========================================================
Single-trait version (A_HOL) with non-integer ordinal labels.
SMOTE removed.
Fix for sklearn classification metrics: encode ordinal float labels to
integer indices for F1/Precision/Recall/QWK/CM while keeping continuous
labels for RMSE/MAE and saving outputs.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import shap
import warnings
import os
import math

warnings.filterwarnings("ignore")

# ---------- future-proof RMSE ----------
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:
    from sklearn.metrics import mean_squared_error as mse
    def rmse(y_true, y_pred): return mse(y_true, y_pred, squared=False)

# Central dictionary for all best hyperparameters (single trait)
BEST_PARAMS = {
    'A_HOL': {
        'random_state': 42,
        'n_estimators': 800,
        'min_samples_split': 7,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'max_depth': 16,
        'lasso_scaler': 'standard',
        'lasso_min_features': 100,
        'lasso_max_iter': 4000,
        'lasso_cv': 3,
        'lasso_coef_threshold': 1e-06,
        'imputation_strategy': 'most_frequent',
        'bootstrap': True
    }
}

# ---------- Logging Setup ----------
def setup_logging(log_dir):
    """Setup comprehensive logging"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fabOF_execution_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# ---------- Enhanced Confusion Matrix Functions ----------
def create_enhanced_confusion_matrix(cm, labels, title, save_path, method_name="", vmax=None):
    plt.figure(figsize=(10, 8))
    if vmax is None: vmax = np.max(cm)
    _ = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count'}, vmax=vmax,
        square=True, linewidths=0.5, linecolor='white'
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return cm

def save_confusion_matrix_data(cm, labels, save_path, method_name, score_col):
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'True_Label'; cm_df.columns.name = 'Predicted_Label'
    csv_path = save_path.parent / f"{save_path.stem}_data.csv"; cm_df.to_csv(csv_path)
    stats = calculate_cm_statistics(cm, labels); stats_df = pd.DataFrame([stats])
    stats_df['Method'] = method_name; stats_df['Score_Column'] = score_col
    stats_path = save_path.parent / f"{save_path.stem}_statistics.csv"; stats_df.to_csv(stats_path, index=False)
    return cm_df, stats_df

def calculate_cm_statistics(cm, labels):
    total = np.sum(cm); correct = np.trace(cm); accuracy = correct / total if total > 0 else 0
    precision_per_class, recall_per_class, f1_per_class = [], [], []
    for i, _ in enumerate(labels):
        tp = cm[i, i]; fp = np.sum(cm[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0; precision_per_class.append(precision)
        fn = np.sum(cm[i, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0; recall_per_class.append(recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0; f1_per_class.append(f1)
    stats = {
        'Total_Predictions': total,
        'Correct_Predictions': correct,
        'Overall_Accuracy': accuracy,
        'Macro_Precision': np.mean(precision_per_class),
        'Macro_Recall': np.mean(recall_per_class),
        'Macro_F1': np.mean(f1_per_class)
    }
    for i, label in enumerate(labels):
        stats[f'Precision_Class_{label}'] = precision_per_class[i]
        stats[f'Recall_Class_{label}'] = recall_per_class[i]
        stats[f'F1_Class_{label}'] = f1_per_class[i]
    return stats

# ---------- SHAP Beeswarm Plot Generator ----------
def create_shap_beeswarm_plots(shap_values, X_data, feature_names, fs_name, sc, save_dir,
                              importance_threshold=0.005, features_per_page=15, logger=None):
    if logger:
        logger.info(f"  Creating SHAP beeswarm plots for {sc} (threshold: {importance_threshold})")
    try:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Feature_Index': range(len(feature_names))
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        significant_features = importance_df[importance_df['Mean_Abs_SHAP'] >= importance_threshold]
        if len(significant_features) == 0:
            if logger: logger.warning(f"    No features above threshold {importance_threshold} for {sc}")
            return
        if logger: logger.info(f"    Found {len(significant_features)} features above threshold for {sc}")
        n_pages = math.ceil(len(significant_features) / features_per_page)
        for page in range(n_pages):
            start_idx = page * features_per_page
            end_idx = min(start_idx + features_per_page, len(significant_features))
            page_features = significant_features.iloc[start_idx:end_idx]
            feature_indices = page_features['Feature_Index'].values
            page_shap_values = shap_values[:, feature_indices]
            page_X_data = X_data[:, feature_indices]
            page_feature_names = page_features['Feature'].values
            plt.figure(figsize=(12, max(8, len(page_features) * 0.4 + 2)))
            try:
                shap.plots.beeswarm(
                    shap.Explanation(
                        values=page_shap_values,
                        data=page_X_data,
                        feature_names=page_feature_names
                    ),
                    max_display=len(page_features),
                    show=False
                )
                if n_pages == 1:
                    title = f'SHAP Beeswarm Plot - {fs_name} - {sc}\n(Features with importance ≥ {importance_threshold})'
                    filename = f"{fs_name}_{sc}_shap_beeswarm_important_features.png"
                else:
                    title = f'SHAP Beeswarm Plot - {fs_name} - {sc} (Page {page + 1}/{n_pages})\n(Features with importance ≥ {importance_threshold})'
                    filename = f"{fs_name}_{sc}_shap_beeswarm_important_features_page_{page + 1}.png"
                plt.title(title, fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                save_path = save_dir / filename
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                if logger: logger.info(f"    Saved beeswarm plot page {page + 1}/{n_pages}: {filename}")
            except Exception as plot_error:
                if logger: logger.warning(f"    Failed to create beeswarm plot page {page + 1}: {str(plot_error)}")
                plt.close(); continue
        summary_path = save_dir / f"{fs_name}_{sc}_significant_features_summary.csv"
        significant_features.to_csv(summary_path, index=False)
        if logger: logger.info(f"    Saved significant features summary: {summary_path.name}")
    except Exception as e:
        if logger: logger.error(f"    Error creating beeswarm plots for {sc}: {str(e)}")

# ---------- Label utilities ----------
def encode_to_index(y_values, classes, tol=1e-6, logger=None):
    """
    Map float labels to integer indices [0..K-1] robustly (handles float fp noise).
    """
    y_values = np.asarray(y_values, dtype=float)
    classes = np.asarray(classes, dtype=float)
    idxs = np.empty_like(y_values, dtype=int)
    for i, v in enumerate(y_values):
        j = int(np.argmin(np.abs(classes - v)))
        if logger and abs(classes[j] - v) > tol:
            logger.warning(f"      Label value {v} mapped to nearest class {classes[j]} (Δ={abs(classes[j]-v):.6f})")
        idxs[i] = j
    return idxs

# ---------- Helper Functions ----------
def _fit_fabOF(X_train, y_train, params, classes):
    """
    Fit FABOF with arbitrary ordered class values (floats).
    'classes' should be sorted ascending list/array of all possible labels.
    """
    # 1. Imputation
    imputer = SimpleImputer(strategy=params['imputation_strategy'])
    X_train_imputed = imputer.fit_transform(X_train)

    # 2. Scaling
    if params['scaling_method'] == 'standard':
        scaler = StandardScaler()
    elif params['scaling_method'] == 'minmax':
        scaler = MinMaxScaler()
    elif params['scaling_method'] == 'robust':
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train_imputed)
    else:
        X_train_scaled = X_train_imputed

    # 3. RF regressor
    rf_params = {
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'max_features': params['max_features'],
        'bootstrap': params['bootstrap'],
        'random_state': params['random_state'],
        'oob_score': True
    }
    y_num = y_train.astype(float)
    rf = RandomForestRegressor(**rf_params).fit(X_train_scaled, y_num)

    # FABOF borders via OOB quantiles at cumulative probs up to each class (except last)
    oob_pred = rf.oob_prediction_
    classes = np.array(classes, dtype=float)
    inner_classes = classes[:-1]
    pi = np.array([(y_train <= c).mean() for c in inner_classes])
    borders_inner = np.quantile(oob_pred, pi)
    borders = np.concatenate([[classes[0]], borders_inner, [classes[-1]]])

    return rf, borders, scaler, imputer, rf.feature_importances_, classes

def _predict_fabOF(rf, borders, classes, X, scaler, imputer):
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed) if scaler else X_imputed
    y_num = rf.predict(X_scaled)
    # Map numeric preds to nearest ordinal class by borders
    idx = np.searchsorted(borders[1:-1], y_num)
    y_ord = classes[idx]
    return y_ord

def _calc_metrics(y_true_numeric, y_pred_raw, y_true_idx, y_pred_idx, labels_idx):
    """Classification metrics computed on integer indices; regression metrics on numeric targets."""
    return {
        'Accuracy': accuracy_score(y_true_idx, y_pred_idx),
        'RMSE': rmse(y_true_numeric, y_pred_raw),
        'MAE': np.mean(np.abs(y_true_numeric - y_pred_raw)),
        'R2': 1 - np.sum((y_true_numeric - y_pred_raw) ** 2) / np.sum((y_true_numeric - np.mean(y_true_numeric)) ** 2),
        'F1_macro': f1_score(y_true_idx, y_pred_idx, labels=labels_idx, average='macro', zero_division=0),
        'Precision_macro': precision_score(y_true_idx, y_pred_idx, labels=labels_idx, average='macro', zero_division=0),
        'Recall_macro': recall_score(y_true_idx, y_pred_idx, labels=labels_idx, average='macro', zero_division=0),
        'QWK': cohen_kappa_score(y_true_idx, y_pred_idx, labels=labels_idx, weights='quadratic')
    }

class FabOFV840Enhanced:
    def __init__(self, base_dir, name_col, score_cols, labels, importance_threshold=0.005, features_per_page=15):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = score_cols
        # ensure ascending order for ordinal labels
        self.labels = np.array(sorted(map(float, labels)))
        self.importance_threshold = importance_threshold
        self.features_per_page = features_per_page
        self.root = self.base_dir / "fabOF_results_v8.4.0_TUNED_PIPELINE_FIXED_A_HOL"
        self.root.mkdir(exist_ok=True)

        self.logger = setup_logging(self.root / 'logs')

        self.dirs = {
            'logs': self.root / 'logs',
            'predictions': self.root / 'predictions',
            'aggregated_predictions': self.root / 'aggregated_predictions',
            'feature_importance': self.root / 'feature_importance',
            'shap_analysis': self.root / 'shap_analysis',
            'shap_per_fold': self.root / 'shap_per_fold',
            'shap_beeswarm': self.root / 'shap_beeswarm_plots',
            'agg_cm': self.root / 'aggregate_confusion_matrices',
            'enhanced_cm': self.root / 'enhanced_confusion_matrices',
            'fs_tracking': self.root / 'feature_selection_tracking'
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.all_confusion_matrices = {}

        self.logger.info("="*60)
        self.logger.info("FabOF v8.4.0 - FIXED TUNED PIPELINE (Single trait: A_HOL)")
        self.logger.info("="*60)
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Results directory: {self.root}")
        self.logger.info(f"Using pre-tuned hyperparameters for score columns: {self.score_cols}")
        self.logger.info(f"Ordered Labels: {self.labels.tolist()}")
        self.logger.info(f"Beeswarm importance threshold: {self.importance_threshold}")
        self.logger.info(f"Features per beeswarm page: {self.features_per_page}")

    def _select_features(self, X_tr, y_tr, X_te, params):
        method_name = params['feature_selection_method']
        self.logger.debug(f"    Applying feature selection: {method_name}")

        start_time = time.time()
        imputer = SimpleImputer(strategy=params['imputation_strategy'])
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        mask = np.ones(X_tr_imp.shape[1], dtype=bool)

        if method_name == 'kbest':
            k = min(params['kbest_k'], X_tr_imp.shape[1])
            selector = SelectKBest(f_regression, k=k)
            X_tr_fs = selector.fit_transform(X_tr_imp, y_tr)
            X_te_fs = selector.transform(X_te_imp)
            mask = selector.get_support()
            self.logger.debug(f"    KBest selected {k} features from {X_tr_imp.shape[1]}")
        else:
            X_tr_fs, X_te_fs = X_tr_imp, X_te_imp
            self.logger.warning(f"    FS method '{method_name}' not fully implemented. Using all features.")

        elapsed = time.time() - start_time
        self.logger.debug(f"    Feature selection completed in {elapsed:.2f}s")
        return X_tr_fs, X_te_fs, mask, imputer

    def run(self):
        """Main execution for single score column (A_HOL)."""
        self.logger.info("Starting FabOF FIXED TUNED pipeline execution...")
        pipeline_start_time = time.time()

        total_operations = len(self.score_cols) * 10
        current_operation = 0

        rows = []
        fs_name = "Tuned_KBest"

        for sc_idx, sc in enumerate(self.score_cols, 1):
            self.logger.info(f"\n[{sc_idx}/{len(self.score_cols)}] Processing Score Column: {sc}")
            sc_start_time = time.time()

            try:
                params = BEST_PARAMS[sc]
                self.logger.info(f"  > Loaded parameters: {params}")
            except KeyError:
                self.logger.error(f"  !!! No parameters found for '{sc}' in BEST_PARAMS. Skipping. !!!")
                continue

            cms = np.zeros((len(self.labels), len(self.labels)), dtype=int)
            rf_imp, feature_masks, fold_cms = [], [], []
            all_shap_values, all_X_te_fs = [], []
            all_predictions, all_true_values, all_fold_info = [], [], []
            all_feature_names = None
            all_sample_names = []

            label_indices = np.arange(len(self.labels))

            for fold in range(1, 11):
                current_operation += 1
                progress = (current_operation / total_operations) * 100
                self.logger.info(f"    [{fold}/10] Processing Fold {fold} - Progress: {progress:.1f}%")

                try:
                    tr_path = self.base_dir / f"fold_{fold}" / f"fold{fold}trh.csv"
                    te_path = self.base_dir / f"fold_{fold}" / f"fold{fold}teh.csv"
                    if not tr_path.exists() or not te_path.exists():
                        self.logger.error(f"    Missing files for fold {fold}")
                        continue

                    tr, te = pd.read_csv(tr_path), pd.read_csv(te_path)
                    tr, te = tr.align(te, axis=1, fill_value=np.nan)

                    feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]
                    X_tr, X_te = tr[feats].values, te[feats].values
                    y_tr, y_te = tr[sc].values.astype(float), te[sc].values.astype(float)

                    X_tr_fs, X_te_fs, mask, _ = self._select_features(X_tr, y_tr, X_te, params)

                    rf, borders, scaler, model_imputer, imp, classes = _fit_fabOF(X_tr_fs, y_tr, params, self.labels)

                    # Predictions
                    y_pred_rounded = _predict_fabOF(rf, borders, classes, X_te_fs, scaler, model_imputer)
                    y_pred_raw = rf.predict(scaler.transform(model_imputer.transform(X_te_fs)) if scaler else model_imputer.transform(X_te_fs))

                    # Encode to indices for classification metrics
                    y_te_idx = encode_to_index(y_te, self.labels, logger=self.logger)
                    y_pred_idx = encode_to_index(y_pred_rounded, self.labels, logger=self.logger)

                    # Save fold predictions (keep original label values)
                    fold_predictions_df = pd.DataFrame({
                        'Inputfile': te[self.name_col],
                        'True_Value': y_te,
                        'Predicted_Value': y_pred_rounded,
                        'Raw_Prediction': y_pred_raw,
                        'Fold': fold
                    })
                    fold_predictions_df.to_csv(self.dirs['predictions'] / f"{fs_name}_{sc}_fold{fold}_predictions.csv", index=False)

                    # Store for aggregated predictions
                    all_predictions.extend(y_pred_rounded.tolist())
                    all_true_values.extend(y_te.tolist())
                    all_fold_info.extend([fold] * len(y_te))
                    all_sample_names.extend(te[self.name_col].tolist())

                    # Metrics (indices for classification metrics)
                    met = _calc_metrics(y_te, y_pred_raw, y_te_idx, y_pred_idx, label_indices)
                    met.update({'FS_Method': fs_name, 'Score_Column': sc, 'Fold': fold})
                    rows.append(met)

                    # Confusion matrix (compute on indices; display original labels)
                    fold_cm = confusion_matrix(y_te_idx, y_pred_idx, labels=label_indices)
                    fold_cms.append(fold_cm)
                    cms += fold_cm

                    # Feature importance
                    rf_imp.append(imp)
                    feature_masks.append(mask)

                    selected_feature_names = np.array(feats)[mask]
                    if all_feature_names is None:
                        all_feature_names = selected_feature_names

                    # SHAP
                    self.logger.debug("    Starting SHAP analysis...")
                    try:
                        explainer = shap.TreeExplainer(rf)
                        X_te_fs_processed = scaler.transform(model_imputer.transform(X_te_fs)) if scaler else model_imputer.transform(X_te_fs)
                        shap_values = explainer.shap_values(X_te_fs_processed)
                        all_shap_values.append(shap_values)
                        all_X_te_fs.append(X_te_fs_processed)

                        if len(shap_values) > 0:
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(shap_values, X_te_fs_processed, feature_names=selected_feature_names, show=False)
                            plt.title(f'SHAP Summary - {sc} - Fold {fold}')
                            plt.tight_layout()
                            plt.savefig(self.dirs['shap_per_fold'] / f"{fs_name}_{sc}_fold{fold}_shap_summary.png",
                                        dpi=300, bbox_inches='tight')
                            plt.close()

                            if len(X_te_fs_processed) > 0:
                                plt.figure(figsize=(10, 8))
                                shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                                     base_values=explainer.expected_value,
                                                                     data=X_te_fs_processed[0],
                                                                     feature_names=selected_feature_names),
                                                    show=False)
                                plt.title(f'SHAP Waterfall - {sc} - Fold {fold} - Sample 1')
                                plt.tight_layout()
                                plt.savefig(self.dirs['shap_per_fold'] / f"{fs_name}_{sc}_fold{fold}_shap_waterfall.png",
                                            dpi=300, bbox_inches='tight')
                                plt.close()
                        self.logger.debug(f"    SHAP analysis completed for fold {fold}")
                    except Exception as shap_error:
                        self.logger.warning(f"    SHAP analysis failed for fold {fold}: {str(shap_error)}")
                        all_shap_values.append(None)
                        all_X_te_fs.append(None)

                except Exception as e:
                    self.logger.error(f"    Error in fold {fold} for {sc}: {str(e)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    continue

            # Aggregated predictions for this score column
            if all_predictions:
                aggregated_predictions_df = pd.DataFrame({
                    'Inputfile': all_sample_names,
                    'True_Value': all_true_values,
                    'Predicted_Value': all_predictions,
                    'Fold': all_fold_info,
                    'Score_Column': [sc] * len(all_predictions),
                    'Method': [fs_name] * len(all_predictions)
                })
                agg_pred_file = self.dirs['aggregated_predictions'] / f"{fs_name}_{sc}_aggregated_predictions.csv"
                aggregated_predictions_df.to_csv(agg_pred_file, index=False)
                self.logger.info(f"  > Saved aggregated predictions to: {agg_pred_file}")

            self.logger.info(f"  Saving results for {fs_name} - {sc}...")
            try:
                self._save_combination_results(
                    fs_name, sc, rows, cms, rf_imp, feature_masks,
                    all_feature_names, all_shap_values, all_X_te_fs,
                    all_predictions, all_true_values, all_fold_info,
                    fold_cms
                )
                key = f"{fs_name}_{sc}"
                self.all_confusion_matrices[key] = cms
                sc_time = time.time() - sc_start_time
                self.logger.info(f"  Score column {sc} completed in {sc_time:.2f}s")
            except Exception as e:
                self.logger.error(f"  Error saving results for {fs_name} - {sc}: {str(e)}")

        # Master aggregated predictions (single trait, still useful)
        self._save_master_aggregated_predictions()
        self._save_all_confusion_matrices()

        self.logger.info("\nSaving global summary...")
        pd.DataFrame(rows).to_csv(self.root / "tuned_model_aggregated_results.csv", index=False)

        pipeline_time = time.time() - pipeline_start_time
        self.logger.info("="*60)
        self.logger.info("FabOF FIXED TUNED Pipeline Execution COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Total execution time: {pipeline_time:.2f}s ({pipeline_time/60:.1f} minutes)")
        self.logger.info(f"Results saved to: {self.root}")

    def _save_master_aggregated_predictions(self):
        """Combine all score column predictions into a master file"""
        try:
            all_files = list(self.dirs['aggregated_predictions'].glob("*_aggregated_predictions.csv"))
            if not all_files:
                self.logger.warning("No aggregated prediction files found to combine")
                return
            master_df_list = [pd.read_csv(file) for file in all_files]
            if master_df_list:
                master_df = pd.concat(master_df_list, ignore_index=True)
                master_file = self.dirs['aggregated_predictions'] / "MASTER_aggregated_predictions_all_scores.csv"
                master_df.to_csv(master_file, index=False)
                self.logger.info(f"  > Saved master aggregated predictions to: {master_file}")
                summary_stats = master_df.groupby(['Score_Column', 'Method']).agg({
                    'True_Value': 'count',
                    'Predicted_Value': lambda x: (x == master_df.loc[x.index, 'True_Value']).mean()
                }).rename(columns={'True_Value': 'Total_Samples', 'Predicted_Value': 'Accuracy'})
                summary_file = self.dirs['aggregated_predictions'] / "MASTER_aggregated_predictions_summary.csv"
                summary_stats.to_csv(summary_file)
                self.logger.info(f"  > Saved master predictions summary to: {summary_file}")
        except Exception as e:
            self.logger.error(f"Error creating master aggregated predictions: {str(e)}")

    def _save_combination_results(self, fs_name, sc, rows, cms, rf_imp, feature_masks,
                                  all_feature_names, all_shap_values, all_X_te_fs,
                                  all_predictions, all_true_values, all_fold_info, fold_cms):
        """Enhanced to properly save SHAP analysis with beeswarm plots"""

        # Fold-level metrics
        df_fold = pd.DataFrame([r for r in rows if (r['FS_Method'] == fs_name) and (r['Score_Column'] == sc)])
        if not df_fold.empty:
            metrics_cols = ['Accuracy', 'RMSE', 'MAE', 'R2', 'F1_macro', 'Precision_macro', 'Recall_macro', 'QWK']
            agg_stats = df_fold[metrics_cols].agg(['mean', 'std', 'min', 'max']).T.reset_index().rename(columns={'index': 'Metric'})
            agg_stats.to_csv(self.dirs['feature_importance'] / f"{fs_name}_{sc}_aggregated_metrics.csv", index=False)

        # Feature importance
        if rf_imp and all_feature_names is not None:
            avg_importance = np.mean(rf_imp, axis=0)
            importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': avg_importance}).sort_values('Importance', ascending=False)
            importance_df.to_csv(self.dirs['feature_importance'] / f"{fs_name}_{sc}_feature_importance.csv", index=False)
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 20 Feature Importance - {fs_name} - {sc}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.dirs['feature_importance'] / f"{fs_name}_{sc}_feature_importance.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

        # SHAP analysis with beeswarm plots
        if all_shap_values and any(sv is not None for sv in all_shap_values):
            try:
                self.logger.info(f"  Creating comprehensive SHAP analysis for {sc}...")
                valid_shap_values = [sv for sv in all_shap_values if sv is not None]
                valid_X_te_fs = [x for x in all_X_te_fs if x is not None]
                if valid_shap_values and valid_X_te_fs and all_feature_names is not None:
                    combined_shap_values = np.vstack(valid_shap_values)
                    combined_X_te_fs = np.vstack(valid_X_te_fs)

                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(combined_shap_values, combined_X_te_fs,
                                      feature_names=all_feature_names, show=False, max_display=20)
                    plt.title(f'SHAP Summary Plot - {fs_name} - {sc} (All Folds)')
                    plt.tight_layout()
                    plt.savefig(self.dirs['shap_analysis'] / f"{fs_name}_{sc}_shap_summary_all_folds.png",
                                dpi=300, bbox_inches='tight')
                    plt.close()

                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(combined_shap_values, combined_X_te_fs,
                                      feature_names=all_feature_names, plot_type="bar",
                                      show=False, max_display=20)
                    plt.title(f'SHAP Feature Importance - {fs_name} - {sc} (All Folds)')
                    plt.tight_layout()
                    plt.savefig(self.dirs['shap_analysis'] / f"{fs_name}_{sc}_shap_bar_plot.png",
                                dpi=300, bbox_inches='tight')
                    plt.close()

                    self.logger.info(f"  Creating SHAP beeswarm plots for {sc}...")
                    create_shap_beeswarm_plots(
                        shap_values=combined_shap_values,
                        X_data=combined_X_te_fs,
                        feature_names=all_feature_names,
                        fs_name=fs_name,
                        sc=sc,
                        save_dir=self.dirs['shap_beeswarm'],
                        importance_threshold=self.importance_threshold,
                        features_per_page=self.features_per_page,
                        logger=self.logger
                    )

                    mean_abs_shap = np.mean(np.abs(combined_shap_values), axis=0)
                    top_5_indices = np.argsort(mean_abs_shap)[-5:]
                    for feature_idx in top_5_indices:
                        try:
                            plt.figure(figsize=(10, 6))
                            shap.dependence_plot(feature_idx, combined_shap_values, combined_X_te_fs,
                                                 feature_names=all_feature_names, show=False)
                            plt.title(f'SHAP Dependence Plot - {all_feature_names[feature_idx]} - {fs_name} - {sc}')
                            plt.tight_layout()
                            plt.savefig(self.dirs['shap_analysis'] / f"{fs_name}_{sc}_shap_dependence_{all_feature_names[feature_idx]}.png",
                                        dpi=300, bbox_inches='tight')
                            plt.close()
                        except Exception as dep_error:
                            self.logger.warning(f"    Could not create dependence plot for feature {feature_idx}: {str(dep_error)}")

                    shap_df = pd.DataFrame(combined_shap_values, columns=all_feature_names)
                    shap_df.to_csv(self.dirs['shap_analysis'] / f"{fs_name}_{sc}_shap_values.csv", index=False)
                    shap_importance = pd.DataFrame({
                        'Feature': all_feature_names,
                        'Mean_Abs_SHAP': mean_abs_shap,
                        'Mean_SHAP': np.mean(combined_shap_values, axis=0),
                        'Std_SHAP': np.std(combined_shap_values, axis=0)
                    }).sort_values('Mean_Abs_SHAP', ascending=False)
                    shap_importance.to_csv(self.dirs['shap_analysis'] / f"{fs_name}_{sc}_shap_importance.csv", index=False)
                    self.logger.info(f"  > SHAP analysis completed for {sc}")
            except Exception as e:
                self.logger.error(f"  Error in SHAP analysis for {sc}: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())

        # Confusion matrices (use original labels for ticks)
        if len(fold_cms) > 0:
            for fold_idx, fold_cm in enumerate(fold_cms, 1):
                title = f"Confusion Matrix - {fs_name} - {sc} - Fold {fold_idx}"
                save_path = self.dirs['enhanced_cm'] / f"{fs_name}_{sc}_fold{fold_idx}_cm.png"
                create_enhanced_confusion_matrix(fold_cm, self.labels, title, save_path, fs_name)
            title = f"Aggregated Confusion Matrix - {fs_name} - {sc}\n(Sum of 10 Folds)"
            save_path = self.dirs['agg_cm'] / f"{fs_name}_{sc}_aggregated_cm.png"
            create_enhanced_confusion_matrix(cms, self.labels, title, save_path, fs_name)
            save_confusion_matrix_data(cms, self.labels, save_path, fs_name, sc)

    def _save_all_confusion_matrices(self):
        """Save comprehensive confusion matrix comparison"""
        self.logger.info("Saving comprehensive confusion matrix comparison...")
        if not self.all_confusion_matrices:
            return
        global_vmax = max(np.max(cm) for cm in self.all_confusion_matrices.values())
        for key, cm in self.all_confusion_matrices.items():
            method, score = key.split('_', 1)
            title = f"CM: {method} - {score}\n(Summed over 10 Folds)"
            save_path = self.dirs['enhanced_cm'] / f"{key}_enhanced_confusion_matrix.png"
            create_enhanced_confusion_matrix(cm, self.labels, title, save_path, method, vmax=global_vmax)
            save_confusion_matrix_data(cm, self.labels, save_path, method, score)

        n_matrices = len(self.all_confusion_matrices)
        if n_matrices > 1:
            fig, axes = plt.subplots(1, n_matrices, figsize=(6*n_matrices, 5))
            axes = [axes] if n_matrices == 1 else axes
            for idx, (key, cm) in enumerate(self.all_confusion_matrices.items()):
                method, score = key.split('_', 1)
                title = f"CM: {method} - {score}\n(Summed over 10 Folds)"
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=self.labels, yticklabels=self.labels,
                            vmax=global_vmax, square=True, ax=axes[idx],
                            cbar=idx == n_matrices-1)
                axes[idx].set_title(title, fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('Predicted Label', fontsize=9)
                axes[idx].set_ylabel('True Label' if idx == 0 else '', fontsize=9)
            plt.tight_layout()
            plt.savefig(self.dirs['enhanced_cm'] / "all_confusion_matrices_comparison.png",
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        all_stats = []
        for key, cm in self.all_confusion_matrices.items():
            method, score = key.split('_', 1)
            stats = calculate_cm_statistics(cm, self.labels)
            stats['Method'] = method
            stats['Score_Column'] = score
            stats['Matrix_Key'] = key
            all_stats.append(stats)
        if all_stats:
            pd.DataFrame(all_stats).to_csv(self.dirs['enhanced_cm'] / "all_confusion_matrices_summary.csv", index=False)

# ----------------------------------------------------------
if __name__ == "__main__":
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = 'Inputfile'
    score_cols = ['A_HOL']

    # Provided possible scores for A_HOL (unsorted list); we will sort in-class init
    A_HOL_scores = [
        81.25, 68.75, 75, 56.25, 50, 62.5, 43.75, 87.5, 100, 31.25, 93.75, 37.5, 25
    ]

    # Beeswarm plot configuration
    importance_threshold = 0.005
    features_per_page = 15

    try:
        pipeline = FabOFV840Enhanced(
            base_dir=base_dir,
            name_col=name_col,
            score_cols=score_cols,
            labels=A_HOL_scores,
            importance_threshold=importance_threshold,
            features_per_page=features_per_page
        )
        pipeline.run()
        print("\n" + "="*60)
        print("FIXED TUNED PIPELINE (A_HOL, no SMOTE) WITH BEESWARM PLOTS COMPLETED!")
        print("="*60)
        print(f"Results saved to: {pipeline.root}")
        print("\nGenerated outputs:")
        print("✓ Individual fold predictions")
        print("✓ Aggregated predictions for A_HOL")
        print("✓ Master aggregated predictions file")
        print("✓ SHAP analysis (summary, dependence, feature importance)")
        print("✓ SHAP beeswarm plots for important features")
        print("✓ Feature importance analysis")
        print("✓ Confusion matrices (individual and aggregated)")
        print("✓ Comprehensive metrics and statistics")
        print(f"\nBeeswarm plots configuration:")
        print(f"  - Importance threshold: {importance_threshold}")
        print(f"  - Features per page: {features_per_page}")
        print(f"  - Plots saved in: {pipeline.dirs['shap_beeswarm']}")
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
