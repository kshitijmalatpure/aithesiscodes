import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from collections import defaultdict
from sklearn.base import clone

# --- NEW: Import SMOTE from imbalanced-learn ---
# If you don't have it, run: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# ML Libraries
from sklearn.metrics import (
    mean_squared_error, r2_score, f1_score, accuracy_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, cohen_kappa_score, mean_absolute_error
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
import xgboost as xgb

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

        self.models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'GaussianNB': GaussianNB(),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(random_state=42, probability=True),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
        }

        self.feature_selections = {
            'Baseline': None,
            'KBest_50': SelectKBest(f_classif, k=50),
            'KBest_100': SelectKBest(f_classif, k=100),
            'KBest_150': SelectKBest(f_classif, k=150),
            'KBest_200': SelectKBest(f_classif, k=200),
            'PCA_50': PCA(n_components=50, random_state=42),
            'PCA_100': PCA(n_components=100, random_state=42),
            'PCA_150': PCA(n_components=150, random_state=42),
            'PCA_200': PCA(n_components=200, random_state=42),
            'LASSO': 'lasso'
        }

    def calculate_metrics(self, y_true, y_pred):
        """Calculate all required metrics"""
        metrics = {}
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['F1'] = f1_score(y_true, y_pred, labels=self.labels, average='macro', zero_division=0)
        metrics['Precision'] = precision_score(y_true, y_pred, labels=self.labels, average='macro', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, labels=self.labels, average='macro', zero_division=0)
        metrics['QWK'] = cohen_kappa_score(y_true, y_pred, labels=self.labels, weights='quadratic')
        return metrics

    def apply_lasso_feature_selection(self, X_train, y_train_col, X_test):
        """Apply LASSO feature selection for a single target column"""
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

    # In your ComprehensiveMLPipeline class (the ML script)

    def run_single_experiment(self, fold_num, model_name, feature_selection_name):
        """Run a single experiment configuration, iterating over each target."""
        try:
            fold_dir = self.base_dir / f"fold_{fold_num}"
            train_path = fold_dir / f"fold{fold_num}trb.csv"
            test_path = fold_dir / f"fold{fold_num}teb.csv"

            if not train_path.exists() or not test_path.exists():
                print(f"Missing files for fold {fold_num}")
                return None

            train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
            feature_cols = [col for col in train_df.columns if col not in [self.name_column] + self.score_columns]
            X_train, X_test = train_df[feature_cols].values, test_df[feature_cols].values
            y_train, y_test = train_df[self.score_columns].values, test_df[self.score_columns].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            fold_results = {}
            aggregate_metrics_for_fold = []

            for i, score_col in enumerate(self.score_columns):
                print(f"    Target: {score_col}")
                y_train_col, y_test_col = y_train[:, i], y_test[:, i]
                y_train_col_adj = y_train_col

                if feature_selection_name == 'Baseline':
                    X_train_selected, X_test_selected = X_train_scaled, X_test_scaled
                    used_features = feature_cols
                elif feature_selection_name.startswith('PCA'):
                    if i == 0:
                        n_components = int(feature_selection_name.split('_')[1])
                        pca = PCA(n_components=min(n_components, X_train_scaled.shape[1]), random_state=42)
                        self.X_train_pca = pca.fit_transform(X_train_scaled)
                        self.X_test_pca = pca.transform(X_test_scaled)
                    X_train_selected, X_test_selected = self.X_train_pca, self.X_test_pca
                    used_features = [f'PC_{j}' for j in range(X_train_selected.shape[1])]
                elif feature_selection_name == 'LASSO':
                    X_train_selected, X_test_selected, support = self.apply_lasso_feature_selection(X_train_scaled,
                                                                                                    y_train_col_adj,
                                                                                                    X_test_scaled)
                    used_features = [feature_cols[j] for j, s in enumerate(support) if s]
                else:  # KBest methods
                    k = int(feature_selection_name.split('_')[1])
                    selector = SelectKBest(f_classif, k=min(k, X_train_scaled.shape[1]))
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train_col_adj)
                    X_test_selected = selector.transform(X_test_scaled)
                    support = selector.get_support()
                    used_features = [feature_cols[j] for j, s in enumerate(support) if s]

                all_used_indices = {feature_cols.index(f) for f in used_features if not f.startswith('PC_')}
                unused_features = [f for i, f in enumerate(feature_cols) if i not in all_used_indices]

                self.feature_usage[model_name][feature_selection_name][score_col].append({
                    'fold': fold_num, 'used_features': used_features, 'unused_features': unused_features,
                    'n_features': len(used_features)
                })

                # --- NEW: APPLY SMOTE TO THE TRAINING DATA ---
                # SMOTE is applied only to the training set to prevent data leakage and
                # to create a balanced dataset for the model to learn from.
                # The test set remains unchanged for a realistic evaluation.
                smote = SMOTE(random_state=42, n_jobs=-1)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train_col_adj)
                # --- END OF NEW CODE ---

                model_clone = clone(self.models[model_name])

                # --- MODIFIED: Train the model on the RESAMPLED data ---
                model_clone.fit(X_train_resampled, y_train_resampled)
                # --- END OF MODIFICATION ---

                # Prediction is correctly done on the original, unmodified test set
                y_pred = model_clone.predict(X_test_selected)

                metrics = self.calculate_metrics(y_test_col, y_pred)
                fold_results[score_col] = metrics
                aggregate_metrics_for_fold.append(metrics)
                self.confusion_matrices[model_name][feature_selection_name][score_col] += confusion_matrix(y_test_col,
                                                                                                           y_pred,
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

    # --- The rest of the functions (aggregate_results, save_results, create_visualizations, generate_report) ---
    # --- do not need to be changed. They will automatically handle the results from the SMOTE-trained models. ---

    def aggregate_results(self):
        # ... (no changes needed) ...
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
        # ... (no changes needed) ...
        results_dir = self.base_dir / "comprehensive_results_smote_binary"  # Changed folder name for clarity
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
            pd.DataFrame(results_rows).to_csv(results_dir / "aggregated_results.csv", index=False)
        cm_dir = results_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        for m, fs_data in self.confusion_matrices.items():
            for fs, score_data in fs_data.items():
                for col, cm in score_data.items():
                    if cm.sum() > 0:
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels,
                                    yticklabels=self.labels)
                        plt.title(f"CM: {m} - {fs} - {col}\n(Summed over 10 Folds)")
                        plt.ylabel('True Label');
                        plt.xlabel('Predicted Label')
                        plt.tight_layout()
                        plt.savefig(cm_dir / f"{m}_{fs}_{col}_cm.png", dpi=300)
                        plt.close()
        print("Results saved successfully.")

    def create_visualizations(self):
        # ... (no changes needed) ...
        results_dir = self.base_dir / "comprehensive_results_smote_binary"
        viz_dir = results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        results_file = results_dir / "aggregated_results.csv"
        if not results_file.exists(): return
        results_df = pd.read_csv(results_file)
        if results_df.empty: return
        print("\nCreating visualizations...")
        score_cols_to_plot = self.score_columns + ['AGGREGATE']
        metrics_to_plot = ['Accuracy', 'F1', 'Precision', 'Recall', 'QWK']
        for score_col in score_cols_to_plot:
            score_df = results_df[results_df['ScoreColumn'] == score_col]
            if score_df.empty: continue
            score_viz_dir = viz_dir / score_col
            score_viz_dir.mkdir(exist_ok=True)
            for metric in metrics_to_plot:
                if f'{metric}_mean' not in score_df.columns: continue
                try:
                    pivot = score_df.pivot_table(index='Model', columns='FeatureSelection', values=f'{metric}_mean')
                    if pivot.empty: continue
                    plt.figure(figsize=(14, 7))
                    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
                    plt.title(f'Mean {metric} for {score_col}')
                    plt.tight_layout()
                    plt.savefig(score_viz_dir / f"heatmap_{metric.lower()}.png", dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"    Could not generate heatmap for {score_col} - {metric}: {e}")
        print("Visualizations created.")


def main():
    base_dir = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final-processed")
    name_column = 'Inputfile'
    score_columns = ['A_WOB', 'A_GRB', 'A_COB', 'A_TTB']
    labels = [0,1]

    pipeline = ComprehensiveMLPipeline(base_dir, name_column, score_columns, labels)

    print("Starting comprehensive CLASSIFICATION pipeline with SMOTE...")
    pipeline.run_all_experiments()
    pipeline.save_results()
    pipeline.create_visualizations()
    # pipeline.generate_report() # generate_report function is not included in the provided script

    print("\nPipeline completed successfully!")
    print(f"Check the '{pipeline.base_dir / 'comprehensive_results_smote_binary'}' directory for all outputs.")


if __name__ == "__main__":
    main()