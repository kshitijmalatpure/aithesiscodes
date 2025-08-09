import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from datetime import datetime
from tqdm import tqdm

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional libraries
try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


class SimpleRegressionPipeline:
    def __init__(self, base_path):
        """Initialize the simplified regression pipeline."""
        self.base_path = Path(base_path)
        self.results_dir = self.base_path / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Simple model collection
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=1.0, random_state=42),
            "ElasticNet": ElasticNet(alpha=1.0, random_state=42),
            "SVR_RBF": SVR(kernel='rbf', C=1.0, gamma='scale'),
            "SVR_Linear": SVR(kernel='linear', C=1.0),
            "SVR_Poly": SVR(kernel='poly', C=1.0, degree=3, gamma='scale')
        }

        # Add external libraries if available
        if LIGHTGBM_AVAILABLE:
            self.models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        if XGBOOST_AVAILABLE:
            self.models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        # Feature selection methods
        self.feature_methods = {
            "AllFeatures": None,
            "Top50_FRegression": ("f_regression", 50),
            "Top100_FRegression": ("f_regression", 100),
            "Top50_MutualInfo": ("mutual_info", 50),
            "Top100_MutualInfo": ("mutual_info", 100)
        }

        self.results = []
        self.folds = list(range(1, 11))  # fold_1 to fold_10

    def preview_data_structure(self):
        """Preview the data structure to understand file organization."""
        print("\n🔍 PREVIEWING DATA STRUCTURE")
        print("=" * 50)

        for fold in self.folds:
            fold_dir = self.base_path / f"fold_{fold}"
            if fold_dir.exists():
                # Look for the specific files with fold number
                train_file = fold_dir / f"fold{fold}trh.csv"
                test_file = fold_dir / f"fold{fold}teh.csv"

                print(f"\n📁 fold_{fold}:")
                if train_file.exists():
                    file_size = train_file.stat().st_size / 1024  # Size in KB
                    print(f"   ✅ fold{fold}trh.csv (TRAIN) - {file_size:.1f} KB")
                else:
                    print(f"   ❌ fold{fold}trh.csv (TRAIN) - NOT FOUND")

                if test_file.exists():
                    file_size = test_file.stat().st_size / 1024  # Size in KB
                    print(f"   ✅ fold{fold}teh.csv (TEST)  - {file_size:.1f} KB")
                else:
                    print(f"   ❌ fold{fold}teh.csv (TEST)  - NOT FOUND")
            else:
                print(f"\n❌ fold_{fold}: Directory not found")

        print(f"\n💡 Expected file structure:")
        print(f"   - Each fold directory should contain:")
        print(f"     • fold[N]trh.csv (training data)")
        print(f"     • fold[N]teh.csv (test data)")
        print(f"   - Where [N] is the fold number (1-10)")

    def _load_fold_data(self, fold):
        """Load training and test data for a given fold using the specific naming convention."""
        fold_dir = self.base_path / f"fold_{fold}"

        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

        # Use the specific file naming convention with fold number
        train_file = fold_dir / f"fold{fold}trh.csv"
        test_file = fold_dir / f"fold{fold}teh.csv"

        # Check if both files exist
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")

        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        print(f"   📂 Fold {fold}: Loading fold{fold}trh.csv (train) and fold{fold}teh.csv (test)")

        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
        except Exception as e:
            raise ValueError(f"Error reading CSV files in fold {fold}: {e}")

        # Skip first column (filenames) and use second column as target
        X_train = df_train.iloc[:, 2:]  # Skip first two columns (filename, target)
        y_train = df_train.iloc[:, 1]  # Second column is target
        X_test = df_test.iloc[:, 2:]  # Skip first two columns (filename, target)
        y_test = df_test.iloc[:, 1]  # Second column is target

        # Basic validation
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature count mismatch in fold {fold}: "
                             f"train={X_train.shape[1]}, test={X_test.shape[1]}")

        print(f"      📊 Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"      📊 Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")

        return X_train, y_train, X_test, y_test

    def _scale_data(self, X_train, X_test):
        """Scale the features."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def _select_features(self, method, X_train, y_train, X_test):
        """Apply feature selection."""
        if method is None:
            return X_train, X_test

        selector_type, k = method
        k = min(k, X_train.shape[1])

        if selector_type == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=k)
        elif selector_type == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        return X_train_sel, X_test_sel

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }

    def run(self):
        """Execute the simplified pipeline."""
        print("🚀 STARTING SIMPLIFIED REGRESSION PIPELINE")
        print("=" * 60)
        print(f"📁 Base path: {self.base_path}")
        print(f"📊 Models: {list(self.models.keys())}")
        print(f"🔧 Feature methods: {list(self.feature_methods.keys())}")
        print(f"📁 Folds: {self.folds}")
        print(f"📄 Expected files: fold[N]trh.csv (train), fold[N]teh.csv (test)")
        print("=" * 60)

        total_experiments = len(self.models) * len(self.feature_methods) * len(self.folds)

        with tqdm(total=total_experiments, desc="Processing") as pbar:
            for fold in self.folds:
                try:
                    # Load data
                    X_train, y_train, X_test, y_test = self._load_fold_data(fold)

                    # Scale data
                    X_train_scaled, X_test_scaled = self._scale_data(X_train, X_test)

                    for model_name, model in self.models.items():
                        for method_name, method in self.feature_methods.items():
                            try:
                                pbar.set_description(f"Fold {fold}: {model_name} + {method_name}")

                                # Feature selection
                                X_train_sel, X_test_sel = self._select_features(
                                    method, X_train_scaled, y_train, X_test_scaled
                                )

                                # Train model
                                model_copy = model.__class__(**model.get_params())
                                model_copy.fit(X_train_sel, y_train)

                                # Predict
                                y_pred = model_copy.predict(X_test_sel)

                                # Calculate metrics
                                metrics = self._calculate_metrics(y_test, y_pred)

                                # Store results
                                result = {
                                    "fold": fold,
                                    "model": model_name,
                                    "feature_method": method_name,
                                    "n_features": X_train_sel.shape[1],
                                    **metrics
                                }
                                self.results.append(result)

                            except Exception as e:
                                print(f"\n❌ Error: Fold {fold}, {model_name}, {method_name}: {e}")

                            pbar.update(1)

                except Exception as e:
                    print(f"\n❌ Error loading fold {fold}: {e}")
                    # Skip all experiments for this fold
                    pbar.update(len(self.models) * len(self.feature_methods))
                    continue

        self._save_results()
        self._print_summary()

    def _save_results(self):
        """Save results to CSV."""
        if not self.results:
            print("❌ No results to save.")
            return

        # Detailed results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.results_dir / "detailed_results.csv", index=False)

        # Summary results (averaged across folds)
        summary = results_df.groupby(["model", "feature_method"]).agg({
            "R2": ["mean", "std"],
            "RMSE": ["mean", "std"],
            "MAE": ["mean", "std"],
            "n_features": "mean"
        }).round(4)

        summary.columns = ["_".join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(self.results_dir / "summary_results.csv", index=False)

        print(f"✅ Results saved to: {self.results_dir}")

    def _print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("❌ No results to summarize.")
            return

        results_df = pd.DataFrame(self.results)

        print("\n🎯 RESULTS SUMMARY")
        print("=" * 50)

        # Best combination by R2
        avg_results = results_df.groupby(["model", "feature_method"])["R2"].mean()
        best_combo = avg_results.idxmax()
        best_r2 = avg_results.max()

        print(f"🏆 Best Combination (by R²):")
        print(f"   Model: {best_combo[0]}")
        print(f"   Feature Method: {best_combo[1]}")
        print(f"   Average R²: {best_r2:.4f}")

        # Top 5 combinations
        print(f"\n📊 Top 5 Combinations:")
        top_5 = avg_results.nlargest(5)
        for i, ((model, method), r2) in enumerate(top_5.items(), 1):
            print(f"   {i}. {model} + {method}: {r2:.4f}")

    def get_results_dataframe(self):
        """Return results as DataFrame for further analysis."""
        return pd.DataFrame(self.results) if self.results else None


def main():
    """Main function."""
    print("🔬 SIMPLIFIED REGRESSION PIPELINE")
    print("=" * 40)

    # Get data path
    data_path = input("Enter the path to your data directory (or press Enter for current directory): ").strip()
    if not data_path:
        data_path = "."

    # Check if path exists
    if not Path(data_path).exists():
        print(f"❌ Path does not exist: {data_path}")
        return

    # Check for fold directories
    base_path = Path(data_path)
    fold_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("fold_")]

    if not fold_dirs:
        print(f"❌ No fold directories found in {data_path}")
        print("Expected directories: fold_1, fold_2, ..., fold_10")
        return

    print(f"✅ Found {len(fold_dirs)} fold directories")

    # Preview the data structure
    pipeline_preview = SimpleRegressionPipeline(data_path)
    pipeline_preview.preview_data_structure()

    # Validate that all expected files exist
    missing_files = []
    for fold in range(1, 11):
        fold_dir = base_path / f"fold_{fold}"
        if fold_dir.exists():
            train_file = fold_dir / f"fold{fold}trh.csv"
            test_file = fold_dir / f"fold{fold}teh.csv"

            if not train_file.exists():
                missing_files.append(f"fold_{fold}/fold{fold}trh.csv")
            if not test_file.exists():
                missing_files.append(f"fold_{fold}/fold{fold}teh.csv")

    if missing_files:
        print(f"\n❌ Missing files detected:")
        for file in missing_files:
            print(f"   • {file}")
        print(f"\nPlease ensure all required files are present before proceeding.")
        return

    # Confirm and run
    print(f"\n" + "=" * 50)
    print(f"✅ All required files found!")
    for fold in range(1, 11):
        fold_dir = base_path / f"fold_{fold}"
        if fold_dir.exists():
            print(f"   fold_{fold}: fold{fold}trh.csv, fold{fold}teh.csv")

    confirm = input("\nProceed with pipeline execution? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Pipeline cancelled.")
        return

    try:
        pipeline = SimpleRegressionPipeline(data_path)
        pipeline.run()

        # Optionally return results for further analysis
        results_df = pipeline.get_results_dataframe()
        if results_df is not None:
            print(f"\n📈 Total experiments completed: {len(results_df)}")

    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()