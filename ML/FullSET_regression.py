import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import defaultdict
from sklearn.base import clone

# Import Regressors
# --- MODIFIED: Removed DecisionTreeRegressor from imports ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV

# Import SMOTE from imbalanced-learn
# If you don't have it, run: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# ML Libraries
from sklearn.metrics import (
    mean_squared_error, r2_score, f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, cohen_kappa_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer  # <-- Added for median imputation

# Suppress warnings
warnings.filterwarnings("ignore")


class ComprehensiveMLPipeline:
    def __init__(self, base_dir, name_column, score_columns, labels):
        self.base_dir = Path(base_dir)
        self.name_column = name_column
        self.score_columns = score_columns
        self.labels = labels
        self.results = defaultdict(lambda: defaultdict(list))
        self.feature_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.confusion_matrices = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: np.zeros((len(labels), len(labels)), dtype=int))))

        # --- MODIFIED: Removed 'DecisionTree' from the list of models to be trained ---
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'SVR': SVR(kernel='linear'),
            'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='rmse', n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'Ridge': RidgeCV(cv=5),
            'ElasticNet': ElasticNetCV(cv=5, random_state=42, n_jobs=-1)
        }

        # Feature selection uses f_regression
        self.feature_selections = {
            'Baseline': None,
            'KBest_50': SelectKBest(f_regression, k=50),
            'KBest_100': SelectKBest(f_regression, k=100),
            'KBest_150': SelectKBest(f_regression, k=150),
            'KBest_200': SelectKBest(f_regression, k=200),
            'LASSO': 'lasso'
        }

    def calculate_metrics(self, y_true, y_pred_raw, y_pred_rounded):
        """Calculate all required metrics for both regression and classification."""
        metrics = {}
        y_true, y_pred_raw, y_pred_rounded = y_true.flatten(), y_pred_raw.flatten(), y_pred_rounded.flatten()

        # Regression metrics
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred_raw))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred_raw)
        metrics['R2'] = r2_score(y_true, y_pred_raw)

        # Classification metrics
        metrics['Accuracy'] = accuracy_score(y_true, y_pred_rounded)

        # Check if more than one class is present in the true labels to avoid errors
        unique_labels_in_y_true = np.unique(y_true)
        if len(unique_labels_in_y_true) < 2:
            print(f"    Warning: Only one class present in y_true. Setting F1, Precision, Recall, and QWK to 0.0.")
            metrics['F1_macro'] = 0.0
            metrics['Precision_macro'] = 0.0
            metrics['Recall_macro'] = 0.0
            metrics['QWK'] = 0.0
        else:
            # If multiple classes are present, calculate metrics as normal.
            metrics['F1_macro'] = f1_score(y_true, y_pred_rounded, labels=self.labels, average='macro', zero_division=0)
            metrics['Precision_macro'] = precision_score(y_true, y_pred_rounded, labels=self.labels, average='macro',
                                                         zero_division=0)
            metrics['Recall_macro'] = recall_score(y_true, y_pred_rounded, labels=self.labels, average='macro',
                                                   zero_division=0)
            metrics['QWK'] = cohen_kappa_score(y_true, y_pred_rounded, labels=self.labels, weights='quadratic')

        return metrics

    def apply_lasso_feature_selection(self, X_train, y_train_col, X_test):
        """Apply LASSO feature selection for a single target column."""
        try:
            lasso = LassoCV(cv=5, random_state=42, max_iter=2000, n_jobs=-1)
            lasso.fit(X_train, y_train_col)
            selected_features = np.abs(lasso.coef_) > 1e-5
            if np.sum(selected_features) == 0:
                top_indices = np.argsort(np.abs(lasso.coef_))[-50:]
                selected_features = np.zeros_like(lasso.coef_, dtype=bool)
                selected_features[top_indices] = True
            return X_train[:, selected_features], X_test[:, selected_features], selected_features
        except Exception as e:
            print(f"    LASSO selection failed: {e}")
            return X_train, X_test, np.ones(X_train.shape[1], dtype=bool)

    def run_single_experiment(self, fold_num, model_name, feature_selection_name):
        """Run a single experiment configuration, iterating over each target."""
        try:
            fold_dir = self.base_dir / f"fold_{fold_num}"
            train_path = fold_dir / f"fold{fold_num}tr.csv"
            test_path = fold_dir / f"fold{fold_num}te.csv"
            if not train_path.exists() or not test_path.exists(): return None

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_cols = train_df.columns
            test_df = test_df.reindex(columns=train_cols, fill_value=0)

            feature_cols = [col for col in train_df.columns if col not in [self.name_column] + self.score_columns]
            X_train, X_test = train_df[feature_cols].values, test_df[feature_cols].values
            y_train, y_test = train_df[self.score_columns].values, test_df[self.score_columns].values

            # --- Median imputation (fit on train, apply to test) ---
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            fold_results = {}
            aggregate_metrics_for_fold = []

            for i, score_col in enumerate(self.score_columns):
                y_train_col, y_test_col = y_train[:, i], y_test[:, i]

                if feature_selection_name == 'Baseline':
                    X_train_selected, X_test_selected = X_train_scaled, X_test_scaled
                elif feature_selection_name == 'LASSO':
                    X_train_selected, X_test_selected, _ = self.apply_lasso_feature_selection(X_train_scaled,
                                                                                              y_train_col,
                                                                                              X_test_scaled)
                else:  # KBest
                    k = int(feature_selection_name.split('_')[1])
                    selector = SelectKBest(f_regression, k=min(k, X_train_scaled.shape[1]))
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train_col)
                    X_test_selected = selector.transform(X_test_scaled)

                X_train_resampled, y_train_resampled = X_train_selected, y_train_col
                try:
                    smote = SMOTE(random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train_col)
                except Exception as e:
                    print(f"    Could not apply SMOTE for {score_col}: {e}. Using original data.")

                model_clone = clone(self.models[model_name])
                model_clone.fit(X_train_resampled, y_train_resampled)
                y_pred_raw = model_clone.predict(X_test_selected)

                y_pred_rounded = np.round(y_pred_raw).astype(int)
                y_pred_rounded = np.clip(y_pred_rounded, min(self.labels), max(self.labels))

                metrics = self.calculate_metrics(y_test_col, y_pred_raw, y_pred_rounded)
                fold_results[score_col] = metrics
                aggregate_metrics_for_fold.append(metrics)
                self.confusion_matrices[model_name][feature_selection_name][score_col] += confusion_matrix(y_test_col,
                                                                                                           y_pred_rounded,
                                                                                                           labels=self.labels)

            if aggregate_metrics_for_fold:
                agg_df = pd.DataFrame(aggregate_metrics_for_fold)
                fold_results['AGGREGATE'] = agg_df.mean().to_dict()
            return fold_results

        except Exception as e:
            print(f"Error in experiment {model_name}-{feature_selection_name}-fold{fold_num}: {e}")
            import traceback;
            traceback.print_exc()
            return None

    def run_all_experiments(self):
        """Run all combinations of models and feature selection methods"""
        total_experiments = len(self.models) * len(self.feature_selections) * 10
        current_experiment = 0
        for model_name in self.models.keys():
            for feature_selection_name in self.feature_selections.keys():
                print(f"\nRunning {model_name} with {feature_selection_name}...")
                fold_results = []
                for fold_num in range(1, 11):
                    current_experiment += 1
                    print(f"  Fold {fold_num}/10 (Experiment {current_experiment}/{total_experiments})")
                    result = self.run_single_experiment(fold_num, model_name, feature_selection_name)
                    if result: fold_results.append(result)
                self.results[model_name][feature_selection_name] = fold_results

    def aggregate_results(self):
        aggregated = defaultdict(lambda: defaultdict(dict))
        score_cols_to_process = self.score_columns + ['AGGREGATE']
        for model_name, fs_results in self.results.items():
            for fs_name, fold_results_list in fs_results.items():
                if not fold_results_list: continue
                for score_col in score_cols_to_process:
                    metrics_across_folds = defaultdict(list)
                    for fold_result in fold_results_list:
                        if fold_result and score_col in fold_result:
                            for metric, value in fold_result[score_col].items():
                                metrics_across_folds[metric].append(value)
                    agg_metrics = {f'{metric}_mean': np.mean(values) for metric, values in metrics_across_folds.items()
                                   if values}
                    agg_metrics.update(
                        {f'{metric}_std': np.std(values) for metric, values in metrics_across_folds.items() if values})
                    aggregated[model_name][fs_name][score_col] = agg_metrics
        return aggregated

    def save_results(self):
        results_dir = self.base_dir / "comprehensive_results_regression"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving results to {results_dir}...")
        aggregated = self.aggregate_results()
        results_rows = []
        for model, fs_data in aggregated.items():
            for fs, score_data in fs_data.items():
                for score, metrics in score_data.items():
                    row = {'Model': model, 'FeatureSelection': fs, 'ScoreColumn': score, **metrics}
                    results_rows.append(row)
        if results_rows:
            pd.DataFrame(results_rows).to_csv(results_dir / "aggregated_results_regression.csv", index=False)
        cm_dir = results_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        print("  Saving confusion matrices...")
        for m, fs_data in self.confusion_matrices.items():
            for fs, score_data in fs_data.items():
                for col, cm in score_data.items():
                    if cm.sum() > 0:
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels,
                                    yticklabels=self.labels)
                        plt.title(f"CM: {m} - {fs} - {col}\n(Summed over 10 Folds)")
                        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
                        plt.tight_layout()
                        plt.savefig(cm_dir / f"{m}_{fs}_{col}_cm.png", dpi=300)
                        plt.close()
        print("Results saved successfully.")

    def create_visualizations(self):
        """
        Create comprehensive visualizations for both regression and classification metrics,
        saving them into separate, organized folders.
        """
        results_dir = self.base_dir / "comprehensive_results_regression"
        viz_dir = results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        results_file = results_dir / "aggregated_results_regression.csv"

        if not results_file.exists():
            print("Aggregated results file not found. Skipping visualizations.")
            return

        results_df = pd.read_csv(results_file)
        if results_df.empty:
            print("No results data available for visualization.")
            return

        print("\nCreating comprehensive visualizations...")

        metric_groups = {
            'regression_metrics': {
                'R2': {'higher_is_better': True},
                'RMSE': {'higher_is_better': False},
                'MAE': {'higher_is_better': False}
            },
            'classification_metrics': {
                'Accuracy': {'higher_is_better': True},
                'F1_macro': {'higher_is_better': True},
                'QWK': {'higher_is_better': True},
                'Precision_macro': {'higher_is_better': True},
                'Recall_macro': {'higher_is_better': True}
            }
        }

        score_cols_to_plot = results_df['ScoreColumn'].unique()

        for group_name, metrics_to_plot in metric_groups.items():
            group_viz_dir = viz_dir / group_name
            group_viz_dir.mkdir(exist_ok=True)
            print(f"  Generating plots for: {group_name}")

            for score_col in score_cols_to_plot:
                score_df = results_df[results_df['ScoreColumn'] == score_col]
                if score_df.empty:
                    continue

                score_col_viz_dir = group_viz_dir / score_col
                score_col_viz_dir.mkdir(exist_ok=True)

                for metric, properties in metrics_to_plot.items():
                    mean_col_name = f'{metric}_mean'
                    if mean_col_name not in score_df.columns:
                        continue

                    try:
                        pivot = score_df.pivot_table(index='Model', columns='FeatureSelection', values=mean_col_name)
                        if pivot.empty:
                            continue

                        plt.figure(figsize=(14, 8))
                        cmap = 'viridis' if properties['higher_is_better'] else 'viridis_r'
                        sns.heatmap(pivot, annot=True, cmap=cmap, fmt='.3f', linewidths=.5, cbar=True)
                        plt.title(f'Mean {metric} for {score_col}', fontsize=16)
                        plt.xlabel('Feature Selection Method', fontsize=12)
                        plt.ylabel('Model', fontsize=12)
                        plt.tight_layout()

                        plot_path = score_col_viz_dir / f"heatmap_{metric.lower()}.png"
                        plt.savefig(plot_path, dpi=300)
                        plt.close()

                    except Exception as e:
                        print(f"    Could not generate heatmap for {score_col} - {metric}: {e}")

        print("Visualizations created successfully.")

    def generate_report(self):
        results_dir = self.base_dir / "comprehensive_results_regression"
        results_file = results_dir / "aggregated_results_regression.csv"
        if not results_file.exists(): return
        results_df = pd.read_csv(results_file)
        if results_df.empty: return
        print("\nGenerating final summary report...")
        report_path = results_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE REGRESSION & CLASSIFICATION PIPELINE REPORT\n" + "=" * 60 + "\n\n")
            agg_df = results_df[results_df['ScoreColumn'] == 'AGGREGATE'].copy()
            if not agg_df.empty:
                f.write("--- OVERALL RECOMMENDATIONS (Based on Aggregate Performance) ---\n\n")
                f.write("--- Top Performers: REGRESSION METRICS ---\n\n")
                if 'R2_mean' in agg_df.columns:
                    best_r2 = agg_df.loc[agg_df['R2_mean'].idxmax()]
                    f.write("Best for R² (Higher is Better):\n")
                    f.write(f"  - Combination: {best_r2['Model']} / {best_r2['FeatureSelection']}\n")
                    f.write(f"  - R² Score:    {best_r2['R2_mean']:.4f} ± {best_r2['R2_std']:.4f}\n\n")
                if 'RMSE_mean' in agg_df.columns:
                    best_rmse = agg_df.loc[agg_df['RMSE_mean'].idxmin()]
                    f.write("Best for RMSE (Lower is Better):\n")
                    f.write(f"  - Combination: {best_rmse['Model']} / {best_rmse['FeatureSelection']}\n")
                    f.write(f"  - RMSE Score:  {best_rmse['RMSE_mean']:.4f} ± {best_rmse['RMSE_std']:.4f}\n\n")
                f.write("--- Top Performers: CLASSIFICATION METRICS (from rounded results) ---\n\n")
                if 'Accuracy_mean' in agg_df.columns:
                    best_acc = agg_df.loc[agg_df['Accuracy_mean'].idxmax()]
                    f.write("Best for Accuracy (Higher is Better):\n")
                    f.write(f"  - Combination: {best_acc['Model']} / {best_acc['FeatureSelection']}\n")
                    f.write(f"  - Accuracy:    {best_acc['Accuracy_mean']:.4f} ± {best_acc['Accuracy_std']:.4f}\n\n")
                if 'F1_macro_mean' in agg_df.columns:
                    best_f1 = agg_df.loc[agg_df['F1_macro_mean'].idxmax()]
                    f.write("Best for F1 Score (Macro):\n")
                    f.write(f"  - Combination: {best_f1['Model']} / {best_f1['FeatureSelection']}\n")
                    f.write(f"  - F1 Score:    {best_f1['F1_macro_mean']:.4f} ± {best_f1['F1_macro_std']:.4f}\n\n")
                if 'QWK_mean' in agg_df.columns:
                    best_qwk = agg_df.loc[agg_df['QWK_mean'].idxmax()]
                    f.write("Best for Quadratic Weighted Kappa (QWK):\n")
                    f.write(f"  - Combination: {best_qwk['Model']} / {best_qwk['FeatureSelection']}\n")
                    f.write(f"  - QWK Score:   {best_qwk['QWK_mean']:.4f} ± {best_qwk['QWK_std']:.4f}\n\n")
        print(f"Comprehensive report generated: {report_path}")


def main():
    base_dir = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final")
    name_column = 'Inputfile'
    score_columns = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
    labels = [1,2,3,4]

    pipeline = ComprehensiveMLPipeline(base_dir, name_column, score_columns, labels)

    print("Starting comprehensive REGRESSION pipeline with SMOTE...")
    pipeline.run_all_experiments()
    pipeline.save_results()
    pipeline.create_visualizations()
    pipeline.generate_report()

    print("\nPipeline completed successfully!")
    print(f"Check the '{pipeline.base_dir / 'comprehensive_results_regression'}' directory for all outputs.")


if __name__ == "__main__":
    main()
