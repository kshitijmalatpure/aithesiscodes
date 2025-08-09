# Python Script: Translation of the R fabOF Pipeline

# ── 1. Libraries ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from collections import defaultdict

# Scikit-learn for ML tasks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, \
    mean_absolute_error, r2_score, cohen_kappa_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV

# rpy2 for bridging to R and using fabOF
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Rpy2 Setup: Activate pandas conversion and import R libraries ---
pandas2ri.activate()
R_BASE = importr('base')
R_FABOF = importr('fabOF')


# ── 2. FabOF Wrapper Class ────────────────────────────────────────────────────
class FabOFWrapper:
    """A Python wrapper for the R fabOF package."""

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """
        Trains the fabOF model.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series or np.array): Target labels.
        """
        # Create a combined DataFrame for R's formula interface
        r_df = pd.concat([X, pd.Series(y, name='target', index=X.index)], axis=1)

        # Ensure the target is an ordered factor in R
        r_df['target'] = r_df['target'].astype('category')

        # Convert to R DataFrame and set the target as an ordered factor
        with localconverter(ro.default_converter + pandas2ri.converter) as cv:
            r_dataframe = ro.conversion.py2rpy(r_df)
            r_dataframe = R_BASE.transform(r_dataframe, target=R_BASE.factor(r_dataframe.rx2('target'), ordered=True))

        # Define the R formula
        formula = ro.Formula('target ~ .')

        # Call the R fabOF function
        self.model = R_FABOF.fabOF(formula, data=r_dataframe)
        return self

    def predict(self, X):
        """
        Makes predictions using the trained fabOF model.

        Args:
            X (pd.DataFrame): Test feature data.

        Returns:
            np.array: Predicted labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        # Convert test data to an R DataFrame
        with localconverter(ro.default_converter + pandas2ri.converter) as cv:
            r_test_df = ro.conversion.py2rpy(X)

        # Use the generic R predict function
        r_predict = ro.r['predict']
        predictions = r_predict(self.model, newdata=r_test_df)

        # Predictions are in a list, extract the 'predictions' element
        y_pred_r = predictions.rx2('predictions')

        # Convert R factor back to a numpy array of integers
        with localconverter(ro.default_converter + pandas2ri.converter) as cv:
            y_pred_py = ro.conversion.rpy2py(y_pred_r)

        return y_pred_py.astype(int)


# ── 3. Pipeline Class ────────────────────────────────────────────────────────
class FabOFPipeline:
    def __init__(self, base_dir, name_col, score_cols, labels):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = score_cols
        self.labels = labels
        self.results_list = []  # Store results as a list of dicts

        self.results_dir = self.base_dir / "fabOF_only_results_Python"
        self.cm_dir = self.results_dir / "confusion_matrices"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir.mkdir(exist_ok=True)

        self.fs_opts = {
            'Baseline': {'method': 'all', 'k': -1},
            'KBest_50': {'method': 'kbest', 'k': 50},
            'KBest_100': {'method': 'kbest', 'k': 100},
            'KBest_150': {'method': 'kbest', 'k': 150},
            'KBest_200': {'method': 'kbest', 'k': 200},
            'LASSO_FS': {'method': 'lasso', 'k': -1}
        }

    def calculate_metrics(self, y_true, y_pred):
        if len(np.unique(y_true)) < 2:
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'RMSE': mean_squared_error(y_true, y_pred, squared=False),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred),
                'F1_macro': 0.0, 'Precision_macro': 0.0, 'Recall_macro': 0.0, 'QWK': 0.0
            }

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'F1_macro': f1_score(y_true, y_pred, labels=self.labels, average='macro', zero_division=0),
            'Precision_macro': precision_score(y_true, y_pred, labels=self.labels, average='macro', zero_division=0),
            'Recall_macro': recall_score(y_true, y_pred, labels=self.labels, average='macro', zero_division=0),
            'QWK': cohen_kappa_score(y_true, y_pred, labels=self.labels, weights='quadratic')
        }

    def run_fold(self, fold, fs_name, fs_cfg):
        try:
            train_path = self.base_dir / f"fold_{fold}" / f"fold{fold}trb.csv"
            test_path = self.base_dir / f"fold_{fold}" / f"fold{fold}teb.csv"
            tr = pd.read_csv(train_path)
            te = pd.read_csv(test_path)
        except FileNotFoundError:
            return None

        feats = [c for c in tr.columns if c not in self.score_cols + [self.name_col]]
        if not feats: return None

        fold_out = {}
        for sc in self.score_cols:
            if sc not in tr.columns or sc not in te.columns: continue

            tr_clean = tr.dropna(subset=[sc])
            te_clean = te.dropna(subset=[sc])

            X_tr = tr_clean[feats].copy()
            y_tr = tr_clean[sc].copy()
            X_te = te_clean[feats].copy()
            y_te = te_clean[sc].copy()

            # Impute medians
            for col in X_tr.columns:
                if X_tr[col].isnull().any():
                    median_val = X_tr[col].median()
                    X_tr[col].fillna(median_val, inplace=True)
                    X_te[col].fillna(median_val, inplace=True)

            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            X_tr_df = pd.DataFrame(X_tr_scaled, columns=X_tr.columns)
            X_te_df = pd.DataFrame(X_te_scaled, columns=X_te.columns)

            # Feature Selection
            if fs_cfg['method'] == 'kbest' and fs_cfg['k'] > 0 and X_tr_df.shape[1] > fs_cfg['k']:
                correlations = X_tr_df.corrwith(y_tr).abs().sort_values(ascending=False)
                keep_cols = correlations.head(fs_cfg['k']).index
                X_tr_df = X_tr_df[keep_cols]
                X_te_df = X_te_df[keep_cols]
            elif fs_cfg['method'] == 'lasso':
                # Use LogisticRegressionCV for classification feature selection with L1 penalty
                l1_selector = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='liblinear', random_state=42).fit(
                    X_tr_df, y_tr)
                coefs = l1_selector.coef_[0]
                keep_cols = X_tr_df.columns[np.abs(coefs) > 1e-5]
                if len(keep_cols) > 0:
                    X_tr_df = X_tr_df[keep_cols]
                    X_te_df = X_te_df[keep_cols]

            # Model Training and Prediction
            model = FabOFWrapper()
            try:
                model.fit(X_tr_df, y_tr)
                y_pr = model.predict(X_te_df)

                # Metrics
                cm = confusion_matrix(y_te, y_pr, labels=self.labels)
                cm_df = pd.DataFrame(cm, index=self.labels, columns=self.labels)
                cm_df.to_csv(self.cm_dir / f"{fs_name}_fold{fold}_{sc}.csv")

                metrics = self.calculate_metrics(y_te, y_pr)
                fold_out[sc] = metrics
            except Exception as e:
                print(f"  > Error during model fit/predict for {sc}: {e}")
                continue

        # Aggregate across score columns
        if not fold_out: return None
        agg_metrics = pd.DataFrame(fold_out).T.mean().to_dict()
        fold_out['AGGREGATE'] = agg_metrics

        return fold_out

    def run(self):
        for fs_name, fs_cfg in self.fs_opts.items():
            print(f"Running {fs_name}...")
            fold_count = 0
            for f in range(1, 11):
                res = self.run_fold(f, fs_name, fs_cfg)
                if res:
                    fold_count += 1
                    # Flatten results and append to the master list
                    for score_col, metrics in res.items():
                        for metric_name, value in metrics.items():
                            self.results_list.append({
                                'FeatureSelection': fs_name,
                                'Fold': f,
                                'ScoreColumn': score_col,
                                'Metric': metric_name,
                                'Value': value
                            })
            print(f"{fs_name} – {fold_count}/10 folds completed")

    def save_results(self):
        if not self.results_list:
            print("No results to save.")
            return

        results_df = pd.DataFrame(self.results_list)
        results_df.to_csv(self.results_dir / "all_results_long.csv", index=False)

        # Aggregate results
        agg_df = results_df.groupby(['FeatureSelection', 'ScoreColumn', 'Metric'])['Value'].agg(
            ['mean', 'std']).reset_index()
        agg_df.rename(columns={'mean': 'Mean', 'std': 'SD'}, inplace=True)

        agg_df.to_csv(self.results_dir / "aggregated_results_long.csv", index=False)
        print(f"Saved aggregated results to {self.results_dir}")


# ── 4. main() helper ───────────────────────────────────────────────────────
def main():
    base_dir = "C:/Research/AI Folder/Thesis/Data/data_CTO_Kshitij/Main/10-fold-final"
    name_col = "Inputfile"
    score_cols = ["A_WOB", "A_GRB", "A_COB", "A_TTB"]
    labels = [0, 1]

    pipe = FabOFPipeline(base_dir, name_col, score_cols, labels)
    print(f"\n=== Running pipeline with {len(pipe.fs_opts)} feature-selection modes ===\n")
    pipe.run()
    pipe.save_results()
    print(f"\nPipeline finished. Outputs written to {pipe.results_dir}\n")


# Run the script
if __name__ == "__main__":
    main()