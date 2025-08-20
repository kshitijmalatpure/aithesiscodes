"""
fabOF_v8_4_0_FS_SHAP_A_HOL_enhanced_CM.py
=========================================
Single-trait pipeline for A_HOL with non-integer ordinal labels.
- No tuned hyperparameters (fixed RF, standard preprocessing).
- Multiple feature selection styles: Baseline (all), KBest (50/100/150/200), LASSO.
- Enhanced confusion matrices and SHAP (per-fold + aggregate when feasible).
- Metrics: classification metrics computed on integer-encoded indices of the float labels.
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
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import shap
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
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fabOF_AHOL_execution_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ---------- Enhanced Confusion Matrix Utilities ----------
def create_enhanced_confusion_matrix(cm, labels_for_ticks, title, save_path, method_name="", vmax=None):
    plt.figure(figsize=(10, 8))
    if vmax is None:
        vmax = np.max(cm)
    _ = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels_for_ticks, yticklabels=labels_for_ticks,
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

def save_confusion_matrix_data(cm, labels_for_ticks, save_path, method_name, score_col):
    cm_df = pd.DataFrame(cm, index=labels_for_ticks, columns=labels_for_ticks)
    cm_df.index.name = 'True_Label'; cm_df.columns.name = 'Predicted_Label'
    csv_path = save_path.parent / f"{save_path.stem}_data.csv"
    cm_df.to_csv(csv_path)

    stats = calculate_cm_statistics(cm, labels_for_ticks)
    stats_df = pd.DataFrame([stats])
    stats_df['Method'] = method_name
    stats_df['Score_Column'] = score_col
    stats_path = save_path.parent / f"{save_path.stem}_statistics.csv"
    stats_df.to_csv(stats_path, index=False)

    return cm_df, stats_df

def calculate_cm_statistics(cm, labels_for_ticks):
    total = np.sum(cm); correct = np.trace(cm); accuracy = correct / total if total > 0 else 0
    precision_per_class, recall_per_class, f1_per_class = [], [], []
    for i, _ in enumerate(labels_for_ticks):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class.append(precision)
        fn = np.sum(cm[i, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class.append(recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1)

    stats = {
        'Total_Predictions': total,
        'Correct_Predictions': correct,
        'Overall_Accuracy': accuracy,
        'Macro_Precision': np.mean(precision_per_class),
        'Macro_Recall': np.mean(recall_per_class),
        'Macro_F1': np.mean(f1_per_class)
    }
    for i, lab in enumerate(labels_for_ticks):
        stats[f'Precision_Class_{lab}'] = precision_per_class[i]
        stats[f'Recall_Class_{lab}'] = recall_per_class[i]
        stats[f'F1_Class_{lab}'] = f1_per_class[i]
    return stats


# ---------- Label utilities (map floats <-> indices) ----------
def encode_to_index(y_values, classes, tol=1e-6, logger=None):
    y_values = np.asarray(y_values, dtype=float)
    classes = np.asarray(classes, dtype=float)
    idxs = np.empty_like(y_values, dtype=int)
    for i, v in enumerate(y_values):
        j = int(np.argmin(np.abs(classes - v)))
        if logger and abs(classes[j] - v) > tol:
            logger.warning(f"      Label {v} mapped to nearest class {classes[j]} (Δ={abs(classes[j]-v):.6f})")
        idxs[i] = j
    return idxs


# ---------- FabOF helpers (arbitrary float-ordered labels) ----------
def _fit_fabOF(X_train, y_train, n_trees=500, random_state=42):
    # Impute + scale (fixed, not tuned)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    rf = RandomForestRegressor(
        n_estimators=n_trees, oob_score=True, random_state=random_state, n_jobs=-1
    ).fit(X_train_scaled, y_train.astype(float))

    # Compute ordinal borders via OOB quantiles at cumulative probs between unique class values
    oob_pred = rf.oob_prediction_
    unique_sorted = np.unique(y_train.astype(float))
    inner_classes = unique_sorted[:-1]
    pi = np.array([(y_train <= c).mean() for c in inner_classes])
    borders_inner = np.quantile(oob_pred, pi)
    borders = np.concatenate([[unique_sorted[0]], borders_inner, [unique_sorted[-1]]])

    return rf, borders, scaler, imputer, rf.feature_importances_, unique_sorted

def _predict_fabOF(rf, borders, classes, X, scaler, imputer):
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    y_num = rf.predict(X_scaled)
    idx = np.searchsorted(borders[1:-1], y_num)
    return classes[idx]

def _calc_metrics(y_true_numeric, y_pred_raw, y_true_idx, y_pred_idx, labels_idx):
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


class FabOFV840_AHOL_FS:
    def __init__(self, base_dir, name_col, score_col, labels):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = [score_col]
        self.labels = np.array(sorted(map(float, labels)))  # enforce order
        self.root = self.base_dir / "fabOF_results_v8.4.0_AHOL_FS_SHAP_Enhanced"
        self.root.mkdir(exist_ok=True)

        self.logger = setup_logging(self.root / 'logs')

        # Feature-selection configs (no hyperparameters to tune)
        self.fs_methods = {
            'Baseline': {'method': 'all'},           # all features
            'KBest_50': {'method': 'kbest', 'k': 50},
            'KBest_100': {'method': 'kbest', 'k': 100},
            'KBest_150': {'method': 'kbest', 'k': 150},
            'KBest_200': {'method': 'kbest', 'k': 200},
            'LASSO': {'method': 'lasso'}             # LassoCV with standardization
        }

        self.dirs = {
            'logs': self.root / 'logs',
            'predictions': self.root / 'predictions',
            'feature_importance': self.root / 'feature_importance',
            'shap_analysis': self.root / 'shap_analysis',
            'shap_per_fold': self.root / 'shap_per_fold',
            'agg_cm': self.root / 'aggregate_confusion_matrices',
            'enhanced_cm': self.root / 'enhanced_confusion_matrices',
            'fs_tracking': self.root / 'feature_selection_tracking'
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.all_confusion_matrices = {}

        self.logger.info("="*60)
        self.logger.info("FabOF v8.4.0 - A_HOL Feature-Selection Pipeline (no SMOTE, no tuned HPs)")
        self.logger.info("="*60)
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Results directory: {self.root}")
        self.logger.info(f"Score column: {self.score_cols}")
        self.logger.info(f"Ordered labels: {self.labels.tolist()}")
        self.logger.info(f"FS methods: {list(self.fs_methods.keys())}")

    # -------- Feature Selection --------
    def _select_features(self, X_tr, y_tr, X_te, cfg):
        method_name = cfg['method']
        self.logger.debug(f"    Applying feature selection: {method_name}")

        start_time = time.time()
        imputer = SimpleImputer(strategy='median')
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        mask = np.ones(X_tr_imp.shape[1], dtype=bool)

        if method_name == 'all':
            X_tr_fs, X_te_fs = X_tr_imp, X_te_imp
            self.logger.debug(f"    Using all {X_tr_fs.shape[1]} features")

        elif method_name == 'kbest':
            k = min(cfg['k'], X_tr_imp.shape[1])
            selector = SelectKBest(f_regression, k=k)
            X_tr_fs = selector.fit_transform(X_tr_imp, y_tr)
            X_te_fs = selector.transform(X_te_imp)
            mask = selector.get_support()
            self.logger.debug(f"    KBest selected {k} / {X_tr_imp.shape[1]}")

        else:  # LASSO with standardization
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_imp)
            X_te_s = scaler.transform(X_te_imp)
            lasso = LassoCV(cv=5, random_state=42, max_iter=4000).fit(X_tr_s, y_tr)
            mask = np.abs(lasso.coef_) > 1e-5
            if mask.sum() == 0:
                # fallback: take top 50 by |coef|
                top = min(50, X_tr_s.shape[1])
                mask[np.argsort(np.abs(lasso.coef_))[-top:]] = True
                self.logger.warning("    LASSO selected 0 features; falling back to top-50 by |coef|.")
            X_tr_fs, X_te_fs = X_tr_s[:, mask], X_te_s[:, mask]
            self.logger.debug(f"    LASSO selected {mask.sum()} / {X_tr_imp.shape[1]}")

        elapsed = time.time() - start_time
        self.logger.debug(f"    Feature selection completed in {elapsed:.2f}s")
        return X_tr_fs, X_te_fs, mask, imputer

    # -------- Main run --------
    def run(self):
        self.logger.info("Starting execution...")
        pipeline_start = time.time()

        total_operations = len(self.fs_methods) * len(self.score_cols) * 10
        current_op = 0
        rows = []

        for fs_idx, (fs_name, cfg) in enumerate(self.fs_methods.items(), 1):
            self.logger.info(f"\n[{fs_idx}/{len(self.fs_methods)}] FS Method: {fs_name}")
            fs_start = time.time()

            for sc_idx, sc in enumerate(self.score_cols, 1):
                self.logger.info(f"  [{sc_idx}/{len(self.score_cols)}] Score Column: {sc}")
                sc_start = time.time()

                cms = np.zeros((len(self.labels), len(self.labels)), dtype=int)
                rf_imp_list = []
                feature_masks = []
                fold_cms = []

                # SHAP aggregation buffers (only if shapes consistent)
                all_shap_values = []
                all_X_te_fs = []
                shap_feature_names_ref = None
                shap_feature_names_match = True

                # Aggregated predictions
                all_predictions = []
                all_true_values = []
                all_fold_info = []

                for fold in range(1, 11):
                    current_op += 1
                    progress = (current_op / total_operations) * 100
                    self.logger.info(f"    [{fold}/10] Fold {fold} - Progress: {progress:.1f}%")

                    try:
                        tr_path = self.base_dir / f"fold_{fold}" / f"fold{fold}trh.csv"
                        te_path = self.base_dir / f"fold_{fold}" / f"fold{fold}teh.csv"
                        if not tr_path.exists() or not te_path.exists():
                            self.logger.error(f"    Missing files for fold {fold}")
                            continue

                        tr = pd.read_csv(tr_path)
                        te = pd.read_csv(te_path)
                        tr, te = tr.align(te, axis=1, fill_value=np.nan)

                        feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]
                        X_tr, X_te = tr[feats].values, te[feats].values
                        y_tr, y_te = tr[sc].values.astype(float), te[sc].values.astype(float)

                        X_tr_fs, X_te_fs, mask, _ = self._select_features(X_tr, y_tr, X_te, cfg)

                        rf, borders, scaler, model_imputer, imp, classes = _fit_fabOF(X_tr_fs, y_tr)

                        # Predictions
                        y_pred_rounded = _predict_fabOF(rf, borders, classes, X_te_fs, scaler, model_imputer)
                        y_pred_raw = rf.predict(scaler.transform(model_imputer.transform(X_te_fs)))

                        # Encode for classification metrics
                        y_te_idx = encode_to_index(y_te, self.labels, logger=self.logger)
                        y_pred_idx = encode_to_index(y_pred_rounded, self.labels, logger=self.logger)
                        label_indices = np.arange(len(self.labels))

                        # Save fold predictions
                        fold_pred_df = pd.DataFrame({
                            'Inputfile': te[self.name_col],
                            'True_Value': y_te,
                            'Predicted_Value': y_pred_rounded,
                            'Raw_Prediction': y_pred_raw,
                            'Fold': fold
                        })
                        fold_pred_df.to_csv(self.dirs['predictions'] / f"{fs_name}_{sc}_fold{fold}_predictions.csv", index=False)

                        # Aggregate buffers
                        all_predictions.extend(y_pred_rounded.tolist())
                        all_true_values.extend(y_te.tolist())
                        all_fold_info.extend([fold] * len(y_te))

                        # Metrics
                        met = _calc_metrics(y_te, y_pred_raw, y_te_idx, y_pred_idx, label_indices)
                        met.update({'FS_Method': fs_name, 'Score_Column': sc, 'Fold': fold})
                        rows.append(met)

                        # Confusion matrix (on indices; show float ticks)
                        fold_cm = confusion_matrix(y_te_idx, y_pred_idx, labels=label_indices)
                        fold_cms.append(fold_cm)
                        cms += fold_cm

                        # RF importances for selected subset
                        rf_imp_list.append(imp)
                        feature_masks.append(mask)

                        selected_feature_names = np.array(feats)[mask]

                        # SHAP per fold
                        try:
                            explainer = shap.TreeExplainer(rf)
                            X_te_fs_processed = scaler.transform(model_imputer.transform(X_te_fs))
                            shap_values = explainer.shap_values(X_te_fs_processed)  # (n_samples, n_features)
                            # Save per-fold SHAP CSV + plots
                            fold_shap_dir = self.dirs['shap_per_fold'] / f"{fs_name}_{sc}"
                            fold_shap_dir.mkdir(parents=True, exist_ok=True)

                            pd.DataFrame(shap_values, columns=selected_feature_names).to_csv(
                                fold_shap_dir / f"fold{fold}_shap_values.csv", index=False)

                            plt.figure(figsize=(12, 8))
                            shap.summary_plot(shap_values, X_te_fs_processed,
                                              feature_names=selected_feature_names, show=False)
                            plt.title(f"SHAP Summary - {fs_name} - {sc} - Fold {fold}")
                            plt.tight_layout()
                            plt.savefig(fold_shap_dir / f"fold{fold}_shap_summary.png", dpi=300, bbox_inches='tight')
                            plt.close()

                            plt.figure(figsize=(12, 8))
                            shap.summary_plot(shap_values, X_te_fs_processed,
                                              feature_names=selected_feature_names, plot_type="bar", show=False)
                            plt.title(f"SHAP Feature Importance - {fs_name} - {sc} - Fold {fold}")
                            plt.tight_layout()
                            plt.savefig(fold_shap_dir / f"fold{fold}_shap_bar.png", dpi=300, bbox_inches='tight')
                            plt.close()

                            for idx in range(min(3, len(X_te_fs_processed))):
                                plt.figure(figsize=(10, 6))
                                shap.waterfall_plot(
                                    shap.Explanation(values=shap_values[idx],
                                                     base_values=getattr(explainer, "expected_value", 0),
                                                     data=X_te_fs_processed[idx],
                                                     feature_names=selected_feature_names),
                                    show=False
                                )
                                plt.title(f"SHAP Waterfall - {fs_name} - {sc} - Fold {fold} - Sample {idx+1}")
                                plt.tight_layout()
                                plt.savefig(fold_shap_dir / f"fold{fold}_waterfall_sample{idx+1}.png",
                                            dpi=300, bbox_inches='tight')
                                plt.close()

                            # For aggregate SHAP, only stack if feature names match across folds
                            if shap_feature_names_ref is None:
                                shap_feature_names_ref = selected_feature_names
                            else:
                                if not np.array_equal(shap_feature_names_ref, selected_feature_names):
                                    shap_feature_names_match = False
                            if shap_feature_names_match:
                                all_shap_values.append(shap_values)
                                all_X_te_fs.append(X_te_fs_processed)

                        except Exception as shap_err:
                            self.logger.warning(f"    SHAP (fold {fold}) failed: {shap_err}")

                    except Exception as e:
                        self.logger.error(f"    Error in fold {fold}: {str(e)}")
                        import traceback
                        self.logger.debug(traceback.format_exc())
                        continue

                # ----- Save results for this FS + score -----
                self.logger.info(f"  Saving results for {fs_name} - {sc}...")
                try:
                    self._save_combination_results(
                        fs_name, sc, rows, cms, rf_imp_list, feature_masks,
                        shap_feature_names_ref, all_shap_values, all_X_te_fs,
                        all_predictions, all_true_values, all_fold_info,
                        fold_cms
                    )
                    key = f"{fs_name}_{sc}"
                    self.all_confusion_matrices[key] = cms
                    self.logger.info(f"  Finished {fs_name} - {sc} in {time.time()-sc_start:.2f}s")
                except Exception as e:
                    self.logger.error(f"  Error saving results for {fs_name} - {sc}: {str(e)}")

            self.logger.info(f"FS method {fs_name} completed in {time.time()-fs_start:.2f}s")

        # ----- Comparison across all confusion matrices -----
        self._save_all_confusion_matrices()

        # Global summary
        self.logger.info("\nSaving global summary...")
        try:
            pd.DataFrame(rows).to_csv(self.root / "all_models_aggregated_results.csv", index=False)
            self.logger.info(f"Global results saved: {len(rows)} rows")
        except Exception as e:
            self.logger.error(f"Error saving global summary: {str(e)}")

        self.logger.info("="*60)
        self.logger.info("Pipeline COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Results saved to: {self.root}")

    # ----- Save helpers -----
    def _save_all_confusion_matrices(self):
        self.logger.info("Saving comprehensive confusion matrix comparison...")
        if not self.all_confusion_matrices:
            return
        global_vmax = max(np.max(cm) for cm in self.all_confusion_matrices.values())
        for key, cm in self.all_confusion_matrices.items():
            method, score = key.split('_', 1)
            title = f"CM: {method} - {score}\n(Summed over 10 Folds)"
            path = self.dirs['enhanced_cm'] / f"{key}_enhanced_confusion_matrix.png"
            create_enhanced_confusion_matrix(cm, self.labels, title, path, method, vmax=global_vmax)
            save_confusion_matrix_data(cm, self.labels, path, method, score)

        # Optional combined comparison grid (kept simple)
        n = len(self.all_confusion_matrices)
        if n > 1:
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
            axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])
            for ax in axes[len(self.all_confusion_matrices):]:
                ax.set_visible(False)

            for ax, (key, cm) in zip(axes, self.all_confusion_matrices.items()):
                method, score = key.split('_', 1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=self.labels, yticklabels=self.labels,
                            vmax=global_vmax, square=True, ax=ax)
                ax.set_title(f"{method} - {score} (10 folds)", fontsize=10, fontweight='bold')
                ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            plt.tight_layout()
            plt.savefig(self.dirs['enhanced_cm'] / "all_confusion_matrices_comparison.png",
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        # Summary CSV
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

    def _save_combination_results(self, fs_name, sc, rows, cms, rf_imp_list, feature_masks,
                                  feat_names_ref, all_shap_values, all_X_te_fs,
                                  all_predictions, all_true_values, all_fold_info, fold_cms):

        # 1) Metrics (aggregate across folds for this FS+score)
        df_fold = pd.DataFrame([r for r in rows if (r['FS_Method'] == fs_name) and (r['Score_Column'] == sc)])
        if not df_fold.empty:
            metrics_cols = ['Accuracy', 'RMSE', 'MAE', 'R2', 'F1_macro', 'Precision_macro', 'Recall_macro', 'QWK']
            agg_stats = df_fold[metrics_cols].agg(['mean', 'std', 'min', 'max']).T.reset_index().rename(columns={'index': 'Metric'})
            agg_stats.to_csv(self.dirs['feature_importance'] / f"{fs_name}_{sc}_aggregated_metrics.csv", index=False)

        # 2) Enhanced CM (aggregate + per-fold)
        title = f"Aggregated Confusion Matrix - {fs_name} - {sc}\n(Sum of 10 Folds)"
        save_path = self.dirs['agg_cm'] / f"{fs_name}_{sc}_aggregated_cm.png"
        create_enhanced_confusion_matrix(cms, self.labels, title, save_path, fs_name)
        save_confusion_matrix_data(cms, self.labels, save_path, fs_name, sc)

        fold_cm_dir = self.dirs['enhanced_cm'] / f"{fs_name}_{sc}_individual_folds"
        fold_cm_dir.mkdir(parents=True, exist_ok=True)
        for fold_idx, fold_cm in enumerate(fold_cms, 1):
            fold_title = f"CM: {fs_name} - {sc} - Fold {fold_idx}"
            fold_path = fold_cm_dir / f"fold_{fold_idx}_confusion_matrix.png"
            create_enhanced_confusion_matrix(fold_cm, self.labels, fold_title, fold_path, fs_name)

        # 3) RF importance (mean only if same length across folds)
        try:
            same_len = len(set(len(arr) for arr in rf_imp_list)) == 1
            if rf_imp_list and same_len and feature_masks:
                imp_mean = np.mean(np.stack(rf_imp_list), axis=0)
                # Use feature names from the first fold's mask
                first_fold_file = self.base_dir / "fold_1" / "fold1tr.csv"
                if first_fold_file.exists():
                    feats_all = [c for c in pd.read_csv(first_fold_file).columns
                                 if c not in [self.name_col] + self.score_cols]
                    mask0 = feature_masks[0]
                    feat_names = np.array(feats_all)[mask0]
                    rf_df = pd.DataFrame({'Feature': feat_names, 'Importance': imp_mean}).sort_values('Importance', ascending=False)
                    rf_df.to_csv(self.dirs['feature_importance'] / f"{fs_name}_{sc}_rf_importance.csv", index=False)
        except Exception as e:
            self.logger.warning(f"    RF importance aggregation skipped: {e}")

        # 4) FS prevalence (how often each original feature was selected)
        try:
            if feature_masks:
                prevalence = np.sum(np.stack(feature_masks), axis=0)
                first_fold_file = self.base_dir / "fold_1" / "fold1tr.csv"
                if first_fold_file.exists():
                    feats_all = [c for c in pd.read_csv(first_fold_file).columns
                                 if c not in [self.name_col] + self.score_cols]
                    prev_df = pd.DataFrame({'Feature': feats_all, 'Prevalence': prevalence})
                    prev_df = prev_df.sort_values('Prevalence', ascending=False)
                    prev_df.to_csv(self.dirs['fs_tracking'] / f"{fs_name}_{sc}_feature_selection.csv", index=False)
        except Exception as e:
            self.logger.warning(f"    FS prevalence save skipped: {e}")

        # 5) Aggregate SHAP (only if all folds used the same selected features)
        try:
            if all_shap_values and feat_names_ref is not None:
                shapes_equal = len(set(tuple(v.shape[1:] for v in all_shap_values))) == 1
                if shapes_equal:
                    shap_values_agg = np.vstack(all_shap_values)
                    X_te_fs_agg = np.vstack(all_X_te_fs)

                    # CSV
                    pd.DataFrame(shap_values_agg, columns=feat_names_ref).to_csv(
                        self.dirs['shap_analysis'] / f"{fs_name}_{sc}_aggregate_shap_values.csv", index=False
                    )

                    # Summary
                    plt.figure(figsize=(14, 10))
                    shap.summary_plot(shap_values_agg, X_te_fs_agg, feature_names=feat_names_ref, show=False)
                    plt.title(f"Aggregate SHAP Summary - {fs_name} - {sc}")
                    plt.tight_layout()
                    plt.savefig(self.dirs['shap_analysis'] / f"{fs_name}_{sc}_aggregate_shap_summary.png",
                                dpi=300, bbox_inches='tight')
                    plt.close()

                    # Bar
                    plt.figure(figsize=(14, 10))
                    shap.summary_plot(shap_values_agg, X_te_fs_agg, feature_names=feat_names_ref,
                                      plot_type="bar", show=False)
                    plt.title(f"Aggregate SHAP Feature Importance - {fs_name} - {sc}")
                    plt.tight_layout()
                    plt.savefig(self.dirs['shap_analysis'] / f"{fs_name}_{sc}_aggregate_shap_bar.png",
                                dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    self.logger.info("    Skipping aggregate SHAP (selected feature sets differ across folds).")
        except Exception as e:
            self.logger.warning(f"    Aggregate SHAP failed: {e}")

        # 6) Aggregate predictions CSV
        if all_predictions and all_true_values and all_fold_info:
            agg_pred_df = pd.DataFrame({
                'Fold': all_fold_info,
                'True_Value': all_true_values,
                'Predicted_Value': all_predictions,
                'Absolute_Error': [abs(t - p) for t, p in zip(all_true_values, all_predictions)],
                'Squared_Error': [(t - p) ** 2 for t, p in zip(all_true_values, all_predictions)]
            })
            agg_pred_df.to_csv(self.dirs['predictions'] / f"{fs_name}_{sc}_aggregate_predictions.csv", index=False)


# ----------------------------------------------------------
if __name__ == "__main__":
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = 'Inputfile'
    score_col = 'A_HOL'

    # Provided possible scores for A_HOL (will be sorted inside the class)
    A_HOL_scores = [
        81.25, 68.75, 75, 56.25, 50, 62.5, 43.75, 87.5, 100, 31.25, 93.75, 37.5, 25
    ]

    try:
        pipeline = FabOFV840_AHOL_FS(
            base_dir=base_dir,
            name_col=name_col,
            score_col=score_col,
            labels=A_HOL_scores
        )
        pipeline.run()

        print("\n" + "="*60)
        print("A_HOL FEATURE-SELECTION PIPELINE COMPLETED!")
        print("="*60)
        print(f"Results saved to: {pipeline.root}")
        print("Outputs include:")
        print("  • Per-fold & aggregated predictions")
        print("  • Enhanced confusion matrices (per-fold + aggregate)")
        print("  • Metrics CSVs (Accuracy, RMSE, MAE, R2, F1/Precision/Recall macro, QWK)")
        print("  • SHAP per fold; aggregate SHAP (when selected features align across folds)")
        print("  • RF importance (when aggregatable) and FS prevalence tracking")

    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
