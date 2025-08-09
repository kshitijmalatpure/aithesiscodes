import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
from typing import Tuple, Any

# Scikit-learn Models and Tools
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the pipeline."""
    FOLDS_BASE_DIR = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final-processed")
    OUTPUT_DIR = Path(
        r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final-processed\holistic_prediction_results_comprehensive")

    TARGET_COLUMNS = ["A_HOL"]
    ID_COLUMN = "Inputfile"
    RANDOM_STATE = 42

    # --- Experiment Grid: Models ---
    MODELS = {
        'RandomForest': RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                                              random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                                                      random_state=RANDOM_STATE),
        'Ridge': RidgeCV(alphas=np.logspace(-6, 6, 13)),
        'Lasso': LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1),
        'ElasticNet': ElasticNetCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1),
        'SVR_RBF': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'SVR_Linear': SVR(kernel='linear', C=1.0),
        'SVR_Poly': SVR(kernel='poly', degree=3, C=1.0)
    }

    # --- Experiment Grid: Feature Selection ---
    FEATURE_SELECTIONS = {
        'Baseline': None,  # Using all features
        'K_50': 50,
        'K_100': 100,
        'K_150': 150,
        'K_200': 200
    }

    def create_output_dirs(self):
        """Create all necessary output directories."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / 'diagnostic_plots').mkdir(exist_ok=True)
        (self.OUTPUT_DIR / 'summary_plots').mkdir(exist_ok=True)
        (self.OUTPUT_DIR / 'reports').mkdir(exist_ok=True)
        logger.info(f"Output directory structure created/verified: {self.OUTPUT_DIR}")


# --- Data Handling Functions ---
def load_and_validate_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        if df.empty: raise ValueError("Dataset is empty.")
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"Error loading {data_path}: {e}")


def prepare_features_targets(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.Series]:
    if config.ID_COLUMN not in df.columns or config.TARGET_COLUMNS[0] not in df.columns:
        raise ValueError("ID or Target column not found.")

    feature_cols = [col for col in df.columns if col not in [config.ID_COLUMN] + config.TARGET_COLUMNS]
    if not feature_cols: raise ValueError("No feature columns found.")

    X = df[feature_cols]
    y = df[config.TARGET_COLUMNS[0]]
    return X, y


# --- DETAILED VISUALIZATION FUNCTIONS ---

def create_deep_diagnostic_grid(model: Any, model_name: str, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray,
                                save_path: Path):
    """Generates the comprehensive 3x3 diagnostic grid for a trained model."""
    logger.info(f"    Generating 3x3 diagnostic grid...")
    fig = plt.figure(figsize=(24, 18))
    plt.suptitle(f"{model_name}: Deep Dive Diagnostic Analysis", fontsize=24, y=1.02)

    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    correlation, _ = stats.pearsonr(y, y_pred)

    # 1. Actual vs. Predicted
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(y, y_pred, alpha=0.5, color='green', edgecolors='k', s=40)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    sns.regplot(x=y, y=y_pred, scatter=False, color='darkorange', ax=ax1, line_kws={'label': 'Trend Line'})
    ax1.set_title(f"Actual vs. Predicted\nR² = {r2:.4f}, RMSE = {rmse:.2f}", fontsize=14)
    ax1.set_xlabel("Actual Holistic Score", fontsize=12)
    ax1.set_ylabel("Predicted Holistic Score", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    textstr = f'MAE = {mae:.2f}\nCorrelation = {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    ax2 = plt.subplot(3, 3, 2)
    top_n = 15
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(top_n)
        sns.barplot(x=importances.values, y=importances.index, ax=ax2, palette='viridis')
        ax2.set_title(f'Top {top_n} Feature Importance', fontsize=14)
        ax2.set_xlabel("Gini Importance", fontsize=12)
    elif hasattr(model, 'coef_'):
        flat_coefs = model.coef_.flatten()
        if len(flat_coefs) == X.shape[1]:
            coefs = pd.Series(flat_coefs, index=X.columns)
            coefs_to_plot = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(top_n)
            sns.barplot(x=coefs_to_plot.values, y=coefs_to_plot.index, ax=ax2, palette='vlag')
            ax2.set_title(f'Top {top_n} Coefficients (by magnitude)', fontsize=14)
            ax2.set_xlabel("Coefficient Value", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "Direct feature coefficients\nnot applicable for this kernel.", ha='center', va='center', fontsize=12)
            ax2.set_title('Feature Importance', fontsize=14)
    else:
        ax2.text(0.5, 0.5, "Importance/Coefficients not available", ha='center', va='center')
        ax2.set_title('Feature Importance', fontsize=14)
    ax2.tick_params(axis='y', labelsize=10)

    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(y_pred, residuals, alpha=0.5, color='red', edgecolors='k', s=40)
    ax3.axhline(0, color='k', linestyle='--', lw=2)
    ax3.set_title("Residuals vs. Predicted", fontsize=14)
    ax3.set_xlabel("Predicted Holistic Score", fontsize=12)
    ax3.set_ylabel("Residuals", fontsize=12)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 3, 4)
    sns.histplot(residuals, kde=True, ax=ax4, color='purple', bins=50)
    ax4.set_title("Distribution of Residuals", fontsize=14)
    ax4.set_xlabel("Residual Value", fontsize=12)
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    stats.probplot(residuals, dist="norm", plot=ax5)
    ax5.set_title("Q-Q Plot of Residuals", fontsize=14)
    ax5.get_lines()[0].set_markerfacecolor('blue')
    ax5.get_lines()[0].set_alpha(0.5)
    ax5.get_lines()[1].set_color('red')

    ax6 = plt.subplot(3, 3, 6)
    quintiles = pd.qcut(y, 5, labels=[f"Quintile {i + 1}" for i in range(5)], duplicates='drop')
    sns.scatterplot(x=y, y=y_pred, hue=quintiles, ax=ax6, alpha=0.7, s=40, palette='deep')
    ax6.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, alpha=0.7)
    ax6.set_title("Predictions by Value Range", fontsize=14)
    ax6.set_xlabel("Actual Holistic Score", fontsize=12)
    ax6.set_ylabel("Predicted Holistic Score", fontsize=12)

    ax7 = plt.subplot(3, 3, 7)
    sns.scatterplot(x=y_pred, y=abs(residuals), ax=ax7, alpha=0.5, color='orange', edgecolors='k', s=40)
    sns.regplot(x=y_pred, y=abs(residuals), scatter=False, color='red', ax=ax7, line_kws={'label': 'Error Trend'})
    ax7.set_title("Absolute Error vs. Predicted Value", fontsize=14)
    ax7.set_xlabel("Predicted Holistic Score", fontsize=12)
    ax7.set_ylabel("Absolute Error", fontsize=12)
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    sns.histplot(y, color="blue", label='Actual', kde=True, stat="density", linewidth=0, ax=ax8, bins=50)
    sns.histplot(y_pred, color="green", label='Predicted', kde=True, stat="density", linewidth=0, ax=ax8, bins=50)
    ax8.set_title("Distribution Comparison", fontsize=14)
    ax8.legend()

    ax9 = plt.subplot(3, 3, 9)
    if model_name in ['RandomForest', 'GradientBoosting']:
        tree_depths = [est.get_depth() for est in model.estimators_]
        sns.histplot(tree_depths, ax=ax9, color='brown', bins=20)
        ax9.set_title(f'Tree Depth Distribution (Avg: {np.mean(tree_depths):.1f})', fontsize=14)
        ax9.set_xlabel("Tree Depth", fontsize=12)
    else:
        ax9.text(0.5, 0.5, "No model-specific plot available.", ha='center', va='center')
        ax9.set_title("Model-Specific Diagnostic", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def create_executive_scatter_plot(model_name: str, y: pd.Series, y_pred: np.ndarray, n_features: int, save_path: Path):
    """Generates the polished, information-dense scatter plot."""
    logger.info(f"    Generating executive scatter plot...")
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    correlation, _ = stats.pearsonr(y, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(y, y_pred, c=residuals, cmap='RdYlBu_r', alpha=0.6, edgecolors='k', linewidth=0.5, s=50)

    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    z = np.polyfit(y, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y, p(y), 'navy', lw=2.5, label='Trend Line')
    std_residuals = residuals.std()
    ax.fill_between(sorted(y), p(sorted(y)) - std_residuals, p(sorted(y)) + std_residuals,
                    color='blue', alpha=0.15, label=f'±1 Std Dev ({std_residuals:.2f})')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title(f'{model_name}: Prediction Analysis', fontsize=18)
    ax.set_xlabel('Actual Holistic Score', fontsize=14)
    ax.set_ylabel('Predicted Holistic Score', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    textstr = (f'Performance:\n'
               f'• R² Score: {r2:.4f}\n'
               f'• RMSE: {rmse:.2f}\n'
               f'• MAE: {mae:.2f}\n'
               f'• Correlation: {correlation:.4f}\n\n'
               f'Data Info:\n'
               f'• Sample Size: {len(y)}\n'
               f'• Features Used: {n_features}')
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def create_summary_visualizations(agg_df: pd.DataFrame, config: Config):
    """Creates bar charts summarizing the best performing combinations."""
    logger.info("Generating summary visualizations of top performers...")
    metrics = {'R²': 'r2', 'RMSE': 'rmse', 'MAE': 'mae'}

    for metric_name, col_name in metrics.items():
        is_higher_better = metric_name == 'R²'
        top_10 = agg_df.sort_values(by=col_name, ascending=not is_higher_better).head(10)

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=col_name, y='combination', data=top_10, palette='mako')
        plt.title(f'Top 10 Performers by {metric_name}', fontsize=16)
        plt.xlabel(metric_name, fontsize=12)
        plt.ylabel('Model and Feature Selection', fontsize=12)

        for p in ax.patches:
            width = p.get_width()
            ax.text(width, p.get_y() + p.get_height() / 2, f'{width:.4f}', va='center')

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'summary_plots' / f'best_performers_{metric_name}.png', dpi=200)
        plt.close()


def create_final_report(agg_df: pd.DataFrame, config: Config):
    """Creates a final summary text report."""
    logger.info("Generating final text report...")
    report_path = config.OUTPUT_DIR / 'reports' / 'comprehensive_summary_report.txt'

    best_r2 = agg_df.sort_values(by='r2', ascending=False).iloc[0]
    best_rmse = agg_df.sort_values(by='rmse', ascending=True).iloc[0]
    best_mae = agg_df.sort_values(by='mae', ascending=True).iloc[0]

    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE REGRESSION ANALYSIS REPORT\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Variable: {config.TARGET_COLUMNS[0]}\n\n")
        f.write(f"Models Tested: {', '.join(config.MODELS.keys())}\n")
        f.write(f"Feature Selections Tested: {', '.join(config.FEATURE_SELECTIONS.keys())}\n\n")

        f.write("--- BEST PERFORMING COMBINATIONS (Mean over 10 Folds) ---\n")
        f.write(f"Best R² Score:\n")
        f.write(f"  - Combination: {best_r2['combination']}\n")
        f.write(f"  - R²: {best_r2['r2']:.4f} ± {best_r2['r2_std']:.4f}\n\n")

        f.write(f"Lowest RMSE:\n")
        f.write(f"  - Combination: {best_rmse['combination']}\n")
        f.write(f"  - RMSE: {best_rmse['rmse']:.2f} ± {best_rmse['rmse_std']:.2f}\n\n")

        f.write(f"Lowest MAE:\n")
        f.write(f"  - Combination: {best_mae['combination']}\n")
        f.write(f"  - MAE: {best_mae['mae']:.2f} ± {best_mae['mae_std']:.2f}\n\n")

        f.write("--- FULL RANKED RESULTS ---\n")
        f.write("See 'reports/aggregated_results_ranked.csv' for the full list of all combinations ranked by R².\n")


# --- Main Orchestration Function ---
def run_main_analysis():
    """Main execution function for the comprehensive regression analysis."""
    try:
        config = Config()
        config.create_output_dirs()

        fold_dirs = sorted([d for d in config.FOLDS_BASE_DIR.iterdir() if d.is_dir() and d.name.startswith('fold_')])
        if not fold_dirs:
            raise FileNotFoundError(f"No fold directories found in {config.FOLDS_BASE_DIR}")

        logger.info(f"Found {len(fold_dirs)} folds. Starting comprehensive analysis...")
        all_results = []

        # --- PART 1: CROSS-VALIDATION LOOP ---
        for fold_dir in fold_dirs:
            fold_num = int(fold_dir.name.split('_')[-1])
            logger.info(f"--- Processing Fold {fold_num} ---")
            train_df = load_and_validate_data(str(next(fold_dir.glob('*trh.csv'))))
            test_df = load_and_validate_data(str(next(fold_dir.glob('*teh.csv'))))

            X_train_df, y_train = prepare_features_targets(train_df, config)
            X_test_df, y_test = prepare_features_targets(test_df, config)

            X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)

            # --- SCALING AND ROBUST CLEANUP ---
            scaler = StandardScaler().fit(X_train_df)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train_df), columns=X_train_df.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)

            # --- CRITICAL FIX: Handle NaNs/Infs created by StandardScaler ---
            # Replace any resulting inf values with NaN, then fill all NaNs with 0.
            X_train_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_test_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_train_scaled.fillna(0, inplace=True)
            X_test_scaled.fillna(0, inplace=True)
            # --- END OF FIX ---

            for model_name, model_instance in config.MODELS.items():
                for fs_name, k_value in config.FEATURE_SELECTIONS.items():
                    logger.info(f"  Running {model_name} with {fs_name}...")

                    if k_value is not None:
                        # Ensure k is not larger than the number of features
                        actual_k = min(k_value, X_train_scaled.shape[1])
                        selector = SelectKBest(f_regression, k=actual_k)
                        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                        X_test_selected = selector.transform(X_test_scaled)
                    else:  # Baseline
                        X_train_selected, X_test_selected = X_train_scaled.values, X_test_scaled.values

                    model = model_instance
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)

                    all_results.append({
                        'fold': fold_num,
                        'model': model_name,
                        'fs_method': fs_name,
                        'k': k_value if k_value is not None else X_train_df.shape[1],
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred)
                    })

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(config.OUTPUT_DIR / 'reports' / 'raw_results_all_folds.csv', index=False)

        # --- PART 2: AGGREGATE RESULTS ---
        agg_df = results_df.groupby(['model', 'fs_method', 'k']).agg(
            r2=('r2', 'mean'), r2_std=('r2', 'std'),
            rmse=('rmse', 'mean'), rmse_std=('rmse', 'std'),
            mae=('mae', 'mean'), mae_std=('mae', 'std')
        ).reset_index()
        agg_df['combination'] = agg_df['model'] + '_' + agg_df['fs_method']
        agg_df = agg_df.sort_values('r2', ascending=False)
        agg_df.to_csv(config.OUTPUT_DIR / 'reports' / 'aggregated_results_ranked.csv', index=False)

        # --- PART 3: DETAILED VISUALIZATION LOOP ---
        logger.info("\n--- Generating Detailed Diagnostic Plots for each combination ---")
        full_train_df = pd.concat([load_and_validate_data(str(next(f.glob('*trh.csv')))) for f in fold_dirs])
        X_full, y_full = prepare_features_targets(full_train_df, config)

        full_scaler = StandardScaler().fit(X_full)
        X_full_scaled = pd.DataFrame(full_scaler.transform(X_full), columns=X_full.columns)

        # --- Robust cleanup for the full dataset ---
        X_full_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_full_scaled.fillna(0, inplace=True)

        for _, row in agg_df.iterrows():
            model_name, fs_name, k = row['model'], row['fs_method'], row['k']
            combo_name = row['combination']
            logger.info(f"  Analyzing combination: {combo_name}")

            output_folder = config.OUTPUT_DIR / 'diagnostic_plots' / combo_name
            output_folder.mkdir(parents=True, exist_ok=True)

            model = config.MODELS[model_name]

            if fs_name != 'Baseline':
                actual_k = min(int(k), X_full_scaled.shape[1])
                selector = SelectKBest(f_regression, k=actual_k)
                X_train_sel = pd.DataFrame(selector.fit_transform(X_full_scaled, y_full),
                                           columns=X_full_scaled.columns[selector.get_support()])
                feature_count = X_train_sel.shape[1]
            else:
                X_train_sel = X_full_scaled
                feature_count = X_train_sel.shape[1]

            model.fit(X_train_sel, y_full)
            y_pred_full = model.predict(X_train_sel)

            create_deep_diagnostic_grid(model, combo_name, X_train_sel, y_full, y_pred_full,
                                        output_folder / 'deep_diagnostic_grid.png')
            create_executive_scatter_plot(combo_name, y_full, y_pred_full, feature_count,
                                          output_folder / 'executive_summary_scatter.png')

        # --- PART 4: FINAL REPORTING ---
        create_summary_visualizations(agg_df, config)
        create_final_report(agg_df, config)

        logger.info("\n--- ANALYSIS COMPLETE ---")
        logger.info(f"\nAll results and plots saved to: {config.OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Main analysis pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_main_analysis()