import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess   # add next to other imports
from scipy.optimize import curve_fit
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the pipeline."""
    # Data parameters - MAIN TRAINING DATASET
    DATA_PATH = r"C:\Research\Thesis Set\Malatpure_Kshitij_Attachment_All\Malatpure_Kshitij_Attachment_All\01.COHA (Fundamentals)\COHA(Fiction)\1.worksheet.normalised.csv"

    # TEST DATASETS - UPDATE THESE PATHS TO YOUR ACTUAL DATASET LOCATIONS
    TEST_DATASETS = {
        'Dataset_1': r"C:\Research\Thesis Set\Malatpure_Kshitij_Attachment_All\Malatpure_Kshitij_Attachment_All\02.CLMET\1.CLMET.worksheet.normalised.csv",
        'Dataset_2': r"C:\Research\Thesis Set\Malatpure_Kshitij_Attachment_All\Malatpure_Kshitij_Attachment_All\04.LOB(1960)\1.LOB.worksheet.fiction.normalised.csv",
        'Dataset_3': r"C:\Research\Thesis Set\Malatpure_Kshitij_Attachment_All\Malatpure_Kshitij_Attachment_All\06.FLOB(1990)\1.FLOB.worksheet.fiction.normalised.csv",
        'Dataset_4': r"C:\Research\Thesis Set\Malatpure_Kshitij_Attachment_All\Malatpure_Kshitij_Attachment_All\07.British National Corpus\1.BNC.worksheet.normalised.csv"
    }

    TEXT_COLUMN = "File Name"
    TARGET_COLUMNS = ["Year"]
    ID_COLUMN = "File Name"

    # Your 50 feature columns
    FEATURE_COLUMNS = [
        'of', 'on', 'be', 'not', 'by', 'this', 'which', 'no', 'about', 'these',
        'may', 'upon', 'such', 'should', 'must', 'back', 'just', 'even', 'every',
        'get', 'might', 'without', 'off', 'yet', 'because', 'shall', 'went', 'got',
        'ever', 'going', 'something', 'want', 'around', 'looked', 'whole', 'present',
        'thus', 'nor', 'TRUE', 'used', 'whom', 'anything', 'need', 'toward', 'really',
        'behind', 'help', 'means', 'big', 'hard'
    ]

    # Cross-validation parameters
    N_FOLDS = 10
    RANDOM_STATE = 42
    N_BINS = 5

    # Random Forest parameters
    N_ESTIMATORS = 200
    MAX_DEPTH = None
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    N_JOBS = -1

    # Feature selection
    K_BEST = 50

    palette = {
        "Dataset_1": ("#1f77b4", "o"),  # CLMET
        "Dataset_2": ("#ff7f0e", "s"),  # LOB
        "Dataset_3": ("#2ca02c", "D"),  # FLOB  ← will be on top
        "Dataset_4": ("#d62728", "^"),  # BNC
        "Training": ("lightgray", "o"),  # training cloud (same as before)
        "Aggregate": ("#8c198c", None)  # magenta for aggregate trend/curve
    }

    # Output paths
    OUTPUT_DIR = Path(
        r"C:\Research\Thesis Set\Malatpure_Kshitij_Attachment_All\Malatpure_Kshitij_Attachment_All\01.COHA (Fundamentals)\COHA(Fiction)\year_prediction_results2")

    def create_output_dirs(self):
        """Create output directories if they don't exist."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for external dataset results
        (self.OUTPUT_DIR / 'external_datasets').mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / 'model_comparisons').mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / 'training_analysis').mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {self.OUTPUT_DIR}")


def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate the input data.

    Args:
        data_path: Path to the CSV file

    Returns:
        Loaded and validated DataFrame
    """
    try:
        # Load the data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")

        # Basic validation
        if df.empty:
            raise ValueError("Loaded dataset is empty")

        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate rows")
            df = df.drop_duplicates()
            logger.info(f"Removed duplicates, new shape: {df.shape}")

        logger.info("Data loaded and validated successfully")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def prepare_features_targets(df: pd.DataFrame, config: Config) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and targets from the dataframe.

    Args:
        df: Input dataframe
        config: Configuration object

    Returns:
        Features array, targets array, feature names
    """
    # Use only the specified feature columns
    feature_cols = config.FEATURE_COLUMNS

    # Validate that all feature columns exist in the dataframe
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns in dataset: {missing_cols}")
        # Use only available features
        available_features = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_features)} available features out of {len(feature_cols)}")
        feature_cols = available_features

    # Extract features and targets
    X = df[feature_cols].values
    y = df[config.TARGET_COLUMNS].values.ravel()  # Flatten for single target

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Year range: {y.min()} - {y.max()}")

    return X, y, feature_cols


def validate_year_data(df: pd.DataFrame, config: Config):
    """
    Validate the year column for prediction.

    Args:
        df: Input dataframe
        config: Configuration object
    """
    year_col = config.TARGET_COLUMNS[0]

    if year_col not in df.columns:
        raise ValueError(f"Target column '{year_col}' not found in dataset")

    # Check for missing years
    missing_years = df[year_col].isnull().sum()
    if missing_years > 0:
        logger.warning(f"Found {missing_years} missing years in dataset")

    # Year statistics
    years = df[year_col].dropna()
    logger.info(f"Year statistics:")
    logger.info(f"  Range: {years.min()} - {years.max()}")
    logger.info(f"  Mean: {years.mean():.1f}")
    logger.info(f"  Std: {years.std():.1f}")

    # Check for reasonable year range
    if years.min() < 1500 or years.max() > 2024:
        logger.warning(f"Unusual year range detected: {years.min()} - {years.max()}")


def create_stratified_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Create stratified bins for regression targets.

    Args:
        y: Target values
        n_bins: Number of bins to create

    Returns:
        Bin labels for stratification
    """
    # Create bins based on quantiles
    bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicates if any

    # Create labels
    bin_labels = np.digitize(y, bins) - 1
    bin_labels = np.clip(bin_labels, 0, len(bins) - 2)

    return bin_labels


def perform_cross_validation(df: pd.DataFrame, config: Config) -> Dict:
    """
    Perform stratified cross-validation for year prediction.

    Args:
        df: Input dataframe
        config: Configuration object

    Returns:
        Dictionary containing results for each target
    """
    # Prepare features and targets
    X, y, feature_names = prepare_features_targets(df, config)

    # Remove any rows with missing values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    logger.info(f"Using {len(X)} samples after removing missing values")

    # Create stratification bins
    y_bins = create_stratified_bins(y, config.N_BINS)

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results = {config.TARGET_COLUMNS[0]: {'fold_results': []}}

    fold_num = 1
    for train_idx, validation_idx in skf.split(X, y_bins):
        logger.info(f"Processing fold {fold_num}/{config.N_FOLDS}")

        # Split data
        X_train, X_validation = X[train_idx], X[validation_idx]
        y_train, y_validation = y[train_idx], y[validation_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_validation_scaled = scaler.transform(X_validation)

        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(config.K_BEST, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_validation_selected = selector.transform(X_validation_scaled)

        # Train model
        model = RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )

        model.fit(X_train_selected, y_train)

        # Make predictions
        y_pred = model.predict(X_validation_selected)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
        mae = mean_absolute_error(y_validation, y_pred)
        r2 = r2_score(y_validation, y_pred)

        # Store results
        fold_result = {
            'fold': fold_num,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(y_train),
            'n_validation': len(y_validation)
        }

        results[config.TARGET_COLUMNS[0]]['fold_results'].append(fold_result)

        logger.info(f"Fold {fold_num} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        fold_num += 1

    return results


def train_final_models(df: pd.DataFrame, config: Config) -> Tuple[Dict, Dict, Dict]:
    """
    Train final models on the full training dataset.

    Args:
        df: Training dataframe
        config: Configuration object

    Returns:
        Tuple of (trained models, scalers, feature selectors)
    """
    # Prepare data
    X, y, feature_names = prepare_features_targets(df, config)

    # Remove any rows with missing values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=min(config.K_BEST, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS
    )
    rf_model.fit(X_selected, y)

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_selected, y)

    models = {
        'random_forest': rf_model,
        'linear_regression': lr_model
    }

    scalers = {'scaler': scaler}
    selectors = {'selector': selector}

    # Save trained models
    joblib.dump(models, config.OUTPUT_DIR / 'trained_models.pkl')
    joblib.dump(scalers, config.OUTPUT_DIR / 'scalers.pkl')
    joblib.dump(selectors, config.OUTPUT_DIR / 'selectors.pkl')

    logger.info("Final models trained and saved successfully")

    return models, scalers, selectors


def create_comprehensive_report(evaluation_results: Dict, lr_metrics: pd.DataFrame,
                                rf_metrics: pd.DataFrame, all_results: Dict, config: Config):
    """
    Create a comprehensive HTML and CSV report with all results.

    Args:
        evaluation_results: External evaluation results
        lr_metrics: Linear regression metrics
        rf_metrics: Random forest metrics
        all_results: Cross-validation results
        config: Configuration object
    """
    try:
        # Extract training metrics
        target_name = config.TARGET_COLUMNS[0]
        cv_results = all_results[target_name]['fold_results']

        avg_rmse = np.mean([r['rmse'] for r in cv_results])
        avg_mae = np.mean([r['mae'] for r in cv_results])
        avg_r2 = np.mean([r['r2'] for r in cv_results])

        lr_r2 = float(lr_metrics[lr_metrics['metric'] == 'R²']['value'].iloc[0])
        lr_rmse = float(lr_metrics[lr_metrics['metric'] == 'RMSE']['value'].iloc[0])
        lr_mae = float(lr_metrics[lr_metrics['metric'] == 'MAE']['value'].iloc[0])

        rf_r2 = float(rf_metrics[rf_metrics['metric'] == 'R²']['value'].iloc[0])
        rf_rmse = float(rf_metrics[rf_metrics['metric'] == 'RMSE']['value'].iloc[0])
        rf_mae = float(rf_metrics[rf_metrics['metric'] == 'MAE']['value'].iloc[0])

        # Create comprehensive summary DataFrame
        summary_data = []

        # Training results
        summary_data.append({
            'Dataset': 'Training (10-Fold CV)',
            'Model': 'Random Forest',
            'R²': avg_r2,
            'RMSE': avg_rmse,
            'MAE': avg_mae,
            'N_Samples': len(cv_results) * np.mean([r['n_validation'] for r in cv_results]),
            'Notes': f'{config.N_FOLDS}-fold cross-validation'
        })

        summary_data.append({
            'Dataset': 'Training (Full)',
            'Model': 'Random Forest',
            'R²': rf_r2,
            'RMSE': rf_rmse,
            'MAE': rf_mae,
            'N_Samples': int(rf_metrics[rf_metrics['metric'] == 'Sample_Size']['value'].iloc[0]),
            'Notes': 'Full training dataset'
        })

        summary_data.append({
            'Dataset': 'Training (Full)',
            'Model': 'Linear Regression',
            'R²': lr_r2,
            'RMSE': lr_rmse,
            'MAE': lr_mae,
            'N_Samples': int(lr_metrics[lr_metrics['metric'] == 'Sample_Size']['value'].iloc[0]),
            'Notes': 'Full training dataset'
        })

        # External evaluation results
        for dataset_name, results in evaluation_results.items():
            for model_name, metrics in results.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name.replace('_', ' ').title(),
                    'R²': float(metrics['r2']),
                    'RMSE': float(metrics['rmse']),
                    'MAE': float(metrics['mae']),
                    'N_Samples': int(metrics['n_samples']),
                    'Notes': f"Correlation: {metrics['correlation']:.4f}"
                })

        # Save comprehensive summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(config.OUTPUT_DIR / 'reports' / 'comprehensive_results.csv', index=False)

        # Create detailed HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Year Prediction Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4fd; }}
                .dataset {{ background-color: #fff8e1; }}
                .summary {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Year Prediction Analysis - Comprehensive Report</h1>

            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Training Dataset:</strong> {config.DATA_PATH}</p>
                <p><strong>Training Samples:</strong> {int(lr_metrics[lr_metrics['metric'] == 'Sample_Size']['value'].iloc[0])}</p>
                <p><strong>Features Used:</strong> {len(config.FEATURE_COLUMNS)} word frequency features</p>
                <p><strong>External Datasets:</strong> {len(evaluation_results)}</p>
            </div>

            <h2>Training Performance (Cross-Validation)</h2>
            <table>
                <tr><th>Model</th><th>Method</th><th>R²</th><th>RMSE (years)</th><th>MAE (years)</th></tr>
                <tr class="metric">
                    <td>Random Forest</td>
                    <td>10-Fold CV</td>
                    <td>{avg_r2:.4f}</td>
                    <td>{avg_rmse:.2f}</td>
                    <td>{avg_mae:.2f}</td>
                </tr>
                <tr class="metric">
                    <td>Random Forest</td>
                    <td>Full Dataset</td>
                    <td>{rf_r2:.4f}</td>
                    <td>{rf_rmse:.2f}</td>
                    <td>{rf_mae:.2f}</td>
                </tr>
                <tr class="metric">
                    <td>Linear Regression</td>
                    <td>Full Dataset</td>
                    <td>{lr_r2:.4f}</td>
                    <td>{lr_rmse:.2f}</td>
                    <td>{lr_mae:.2f}</td>
                </tr>
            </table>
        """

        if evaluation_results:
            html_content += f"""
            <h2>External Dataset Evaluation</h2>
            <table>
                <tr><th>Dataset</th><th>Model</th><th>R²</th><th>RMSE (years)</th><th>MAE (years)</th><th>Samples</th><th>Correlation</th></tr>
            """

            for dataset_name, results in evaluation_results.items():
                for model_name, metrics in results.items():
                    html_content += f"""
                    <tr class="dataset">
                        <td>{dataset_name}</td>
                        <td>{model_name.replace('_', ' ').title()}</td>
                        <td>{metrics['r2']:.4f}</td>
                        <td>{metrics['rmse']:.2f}</td>
                        <td>{metrics['mae']:.2f}</td>
                        <td>{metrics['n_samples']}</td>
                        <td>{metrics['correlation']:.4f}</td>
                    </tr>
                    """

            html_content += "</table>"

        html_content += """
            <h2>Model Configuration</h2>
            <ul>
        """

        html_content += f"""
                <li><strong>Random Forest:</strong> {config.N_ESTIMATORS} trees, max_depth={config.MAX_DEPTH}</li>
                <li><strong>Feature Selection:</strong> Top {config.K_BEST} features using F-regression</li>
                <li><strong>Cross-Validation:</strong> {config.N_FOLDS}-fold stratified</li>
                <li><strong>Random State:</strong> {config.RANDOM_STATE}</li>
            </ul>

            <h2>Generated Files</h2>
            <ul>
                <li>comprehensive_results.csv - All results in tabular format</li>
                <li>training_analysis/ - Training visualizations</li>
                <li>model_comparisons/ - Model comparison charts</li>
                <li>external_datasets/ - External evaluation results</li>
            </ul>

        </body>
        </html>
        """

        # Save HTML report
        with open(config.OUTPUT_DIR / 'reports' / 'comprehensive_report.html', 'w') as f:
            f.write(html_content)

        # Create detailed CSV reports for each category

        # 1. Training results detail
        training_detail = []
        for i, fold_result in enumerate(cv_results, 1):
            training_detail.append({
                'Fold': i,
                'RMSE': fold_result['rmse'],
                'MAE': fold_result['mae'],
                'R²': fold_result['r2'],
                'Train_Samples': fold_result['n_train'],
                'Validation_Samples': fold_result['n_validation']
            })

        training_df = pd.DataFrame(training_detail)
        training_df.to_csv(config.OUTPUT_DIR / 'reports' / 'cross_validation_details.csv', index=False)

        # 2. External evaluation details
        if evaluation_results:
            external_detail = []
            for dataset_name, results in evaluation_results.items():
                for model_name, metrics in results.items():
                    external_detail.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'R²': metrics['r2'],
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'Correlation': metrics['correlation'],
                        'P_Value': metrics['p_value'],
                        'N_Samples': metrics['n_samples']
                    })

            external_df = pd.DataFrame(external_detail)
            external_df.to_csv(config.OUTPUT_DIR / 'reports' / 'external_evaluation_details.csv', index=False)

        # 3. Model comparison summary
        comparison_data = [
            {
                'Model': 'Random Forest (CV)',
                'R²': avg_r2,
                'RMSE': avg_rmse,
                'MAE': avg_mae,
                'Type': 'Cross-Validation'
            },
            {
                'Model': 'Random Forest (Full)',
                'R²': rf_r2,
                'RMSE': rf_rmse,
                'MAE': rf_mae,
                'Type': 'Full Training'
            },
            {
                'Model': 'Linear Regression',
                'R²': lr_r2,
                'RMSE': lr_rmse,
                'MAE': lr_mae,
                'Type': 'Full Training'
            }
        ]

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(config.OUTPUT_DIR / 'reports' / 'model_comparison_summary.csv', index=False)

        # Create executive summary text file
        with open(config.OUTPUT_DIR / 'reports' / 'executive_summary.txt', 'w') as f:
            f.write("YEAR PREDICTION ANALYSIS - EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("BEST PERFORMING MODEL:\n")
            f.write("-" * 25 + "\n")
            if rf_r2 > lr_r2:
                f.write(f"Random Forest: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.2f} years\n")
            else:
                f.write(f"Linear Regression: R² = {lr_r2:.4f}, RMSE = {lr_rmse:.2f} years\n")

            f.write(f"\nTRAINING PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Random Forest (10-Fold CV): R² = {avg_r2:.4f}, RMSE = {avg_rmse:.2f}\n")
            f.write(f"Random Forest (Full): R² = {rf_r2:.4f}, RMSE = {rf_rmse:.2f}\n")
            f.write(f"Linear Regression: R² = {lr_r2:.4f}, RMSE = {lr_rmse:.2f}\n")

            if evaluation_results:
                f.write(f"\nEXTERNAL EVALUATION:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Evaluated on {len(evaluation_results)} external datasets\n")

                # Find best external performance
                best_external_r2 = 0
                best_external_dataset = ""
                best_external_model = ""

                for dataset_name, results in evaluation_results.items():
                    for model_name, metrics in results.items():
                        if metrics['r2'] > best_external_r2:
                            best_external_r2 = metrics['r2']
                            best_external_dataset = dataset_name
                            best_external_model = model_name

                f.write(f"Best external performance: {best_external_model} on {best_external_dataset}\n")
                f.write(f"  R² = {best_external_r2:.4f}\n")

            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 15 + "\n")
            f.write("• comprehensive_results.csv - All results\n")
            f.write("• comprehensive_report.html - Detailed HTML report\n")
            f.write("• cross_validation_details.csv - CV fold results\n")
            f.write("• model_comparison_summary.csv - Model comparisons\n")
            if evaluation_results:
                f.write("• external_evaluation_details.csv - External dataset results\n")

        logger.info(f"Comprehensive reports created successfully in {config.OUTPUT_DIR / 'reports'}")

    except Exception as e:
        logger.error(f"Error creating comprehensive report: {e}")
        # Still create a basic summary even if detailed report fails
        try:
            with open(config.OUTPUT_DIR / 'basic_results_summary.txt', 'w') as f:
                f.write("BASIC RESULTS SUMMARY\n")
                f.write("=" * 30 + "\n")
                f.write(f"Error creating detailed report: {e}\n\n")
                f.write("Check the console output and individual result files.\n")
        except:
            pass


def evaluate_external_datasets(models: Dict, scalers: Dict, selectors: Dict, config: Config) -> Dict:
    """
    Evaluate trained models on external datasets.

    Args:
        models: Trained models
        scalers: Fitted scalers
        selectors: Fitted feature selectors
        config: Configuration object

    Returns:
        Dictionary containing evaluation results for each dataset
    """
    evaluation_results = {}

    for dataset_name, dataset_path in config.TEST_DATASETS.items():
        logger.info(f"Evaluating on {dataset_name}: {dataset_path}")

        try:
            # Check if file exists
            if not Path(dataset_path).exists():
                logger.warning(f"Dataset file not found: {dataset_path}. Skipping {dataset_name}")
                continue

            # Load external dataset
            external_df = load_and_validate_data(dataset_path)
            validate_year_data(external_df, config)

            # Prepare external features and targets
            X_external, y_external, _ = prepare_features_targets(external_df, config)

            # Remove missing values
            mask = ~(np.isnan(X_external).any(axis=1) | np.isnan(y_external))
            X_external = X_external[mask]
            y_external = y_external[mask]

            if len(X_external) == 0:
                logger.warning(f"No valid samples in {dataset_name} after cleaning")
                continue

            # Apply same preprocessing as training
            X_external_scaled = scalers['scaler'].transform(X_external)
            X_external_selected = selectors['selector'].transform(X_external_scaled)

            dataset_results = {}

            # Evaluate each model
            for model_name, model in models.items():
                y_pred = model.predict(X_external_selected)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_external, y_pred))
                mae = mean_absolute_error(y_external, y_pred)
                r2 = r2_score(y_external, y_pred)
                correlation, p_value = stats.pearsonr(y_external, y_pred)

                dataset_results[model_name] = {
                    'y_true': y_external,
                    'y_pred': y_pred,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_samples': len(y_external)
                }

                logger.info(f"{dataset_name} - {model_name}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

            evaluation_results[dataset_name] = dataset_results

        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            continue

    return evaluation_results


def create_logistic_prediction_curves(evaluation_results: Dict, config: Config):
    """
    Create logistic-like prediction curves for model comparison across datasets.

    Args:
        evaluation_results: Evaluation results from external datasets
        config: Configuration object
    """
    try:
        # Set up the plotting style
        plt.style.use('default')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Create figure with subplots for each model
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        models = ['random_forest', 'linear_regression']

        for model_idx, model_name in enumerate(models):
            ax = axes[model_idx]

            for dataset_idx, (dataset_name, results) in enumerate(evaluation_results.items()):
                if model_name not in results:
                    continue

                model_results = results[model_name]
                y_true = model_results['y_true']
                y_pred = model_results['y_pred']
                r2 = model_results['r2']

                # Sort data for smooth curve
                sorted_indices = np.argsort(y_true)
                y_true_sorted = y_true[sorted_indices]
                y_pred_sorted = y_pred[sorted_indices]

                # Create prediction accuracy curve (similar to logistic curve)
                # Calculate cumulative accuracy within different error thresholds
                thresholds = np.linspace(0, 50, 100)  # Error thresholds in years
                accuracies = []

                for threshold in thresholds:
                    errors = np.abs(y_true - y_pred)
                    accuracy = np.mean(errors <= threshold)
                    accuracies.append(accuracy)

                # Plot the prediction accuracy curve
                ax.plot(thresholds, accuracies,
                        linewidth=3, alpha=0.8, color=colors[dataset_idx % len(colors)],
                        label=f'{dataset_name} (R²={r2:.3f}, n={len(y_true)})')

            ax.set_xlabel('Error Threshold (years)', fontsize=12)
            ax.set_ylabel('Prediction Accuracy', fontsize=12)
            ax.set_title(f'{model_name.replace("_", " ").title()}\nPrediction Accuracy Curves', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 1)

            # Add reference lines
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Accuracy')
            ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% Accuracy')
            ax.axvline(x=5, color='gray', linestyle=':', alpha=0.5, label='5-year threshold')
            ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5, label='10-year threshold')

        # Create overall comparison plot
        ax = axes[2]

        # Calculate mean performance across datasets for each model
        model_performance = {}
        for model_name in models:
            r2_scores = []
            rmse_scores = []
            for dataset_name, results in evaluation_results.items():
                if model_name in results:
                    r2_scores.append(results[model_name]['r2'])
                    rmse_scores.append(results[model_name]['rmse'])

            model_performance[model_name] = {
                'mean_r2': np.mean(r2_scores) if r2_scores else 0,
                'std_r2': np.std(r2_scores) if r2_scores else 0,
                'mean_rmse': np.mean(rmse_scores) if rmse_scores else 0,
                'std_rmse': np.std(rmse_scores) if rmse_scores else 0
            }

        # Plot model comparison
        model_names = list(model_performance.keys())
        r2_means = [model_performance[m]['mean_r2'] for m in model_names]
        r2_stds = [model_performance[m]['std_r2'] for m in model_names]

        bars = ax.bar(range(len(model_names)), r2_means, yerr=r2_stds,
                      capsize=5, alpha=0.7, color=['lightblue', 'lightgreen'])
        ax.set_ylabel('Average R² Score', fontsize=12)
        ax.set_title('Model Performance Comparison\n(Average across External Datasets)', fontsize=14)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names])
        ax.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar, mean, std in zip(bars, r2_means, r2_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Create detailed metrics heatmap
        ax = axes[3]

        # Prepare data for heatmap
        metrics_data = []
        dataset_labels = []

        for dataset_name, results in evaluation_results.items():
            for model_name in models:
                if model_name in results:
                    metrics_data.append([
                        results[model_name]['r2'],
                        results[model_name]['rmse'],
                        results[model_name]['mae'],
                        results[model_name]['correlation']
                    ])
                    dataset_labels.append(f"{dataset_name}\n{model_name}")

        if metrics_data:
            metrics_array = np.array(metrics_data)

            # Normalize RMSE and MAE for better visualization (lower is better)
            metrics_array[:, 1] = 1 / (1 + metrics_array[:, 1] / 10)  # Normalize RMSE
            metrics_array[:, 2] = 1 / (1 + metrics_array[:, 2] / 10)  # Normalize MAE

            im = ax.imshow(metrics_array.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Performance Score\n(Higher = Better)', fontsize=10)

            # Set labels
            ax.set_xticks(range(len(dataset_labels)))
            ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(4))
            ax.set_yticklabels(['R²', 'RMSE (norm)', 'MAE (norm)', 'Correlation'], fontsize=10)
            ax.set_title('Performance Metrics Heatmap\n(Normalized)', fontsize=14)

            # Add text annotations
            for i in range(len(dataset_labels)):
                for j in range(4):
                    text = ax.text(i, j, f'{metrics_array[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'model_comparisons' / 'logistic_prediction_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Logistic-like prediction curves created successfully")

    except Exception as e:
        logger.error(f"Error creating prediction curves: {e}")


def create_dataset_comparison_visualization(evaluation_results: Dict, config: Config):
    """
    Create comprehensive comparison visualization across datasets.

    Args:
        evaluation_results: Evaluation results from external datasets
        config: Configuration object
    """
    try:
        # Create performance summary table
        summary_data = []

        for dataset_name, results in evaluation_results.items():
            for model_name, metrics in results.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name.replace('_', ' ').title(),
                    'R²': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'Correlation': metrics['correlation'],
                    'N_Samples': metrics['n_samples']
                })

        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        summary_df.to_csv(config.OUTPUT_DIR / 'model_comparisons' / 'dataset_comparison_summary.csv',
                          index=False)

        # Create detailed scatterplots for each dataset
        n_datasets = len(evaluation_results)
        if n_datasets == 0:
            logger.warning("No evaluation results to visualize")
            return

        fig, axes = plt.subplots(2, max(2, (n_datasets + 1) // 2), figsize=(5 * max(2, (n_datasets + 1) // 2), 10))

        if n_datasets == 1:
            axes = [axes]
        elif n_datasets <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        colors = {'random_forest': 'green', 'linear_regression': 'blue'}

        for idx, (dataset_name, results) in enumerate(evaluation_results.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            for model_name, metrics in results.items():
                y_true = metrics['y_true']
                y_pred = metrics['y_pred']
                r2 = metrics['r2']
                rmse = metrics['rmse']

                # Scatter plot
                ax.scatter(y_true, y_pred, alpha=0.6, color=colors.get(model_name, 'gray'),
                           s=50, label=f'{model_name.replace("_", " ").title()}\nR²={r2:.3f}, RMSE={rmse:.1f}')

            # Perfect prediction line
            min_val = min([m['y_true'].min() for m in results.values()])
            max_val = max([m['y_true'].max() for m in results.values()])
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)

            ax.set_xlabel('Actual Year', fontsize=12)
            ax.set_ylabel('Predicted Year', fontsize=12)
            ax.set_title(f'{dataset_name}\nPrediction Comparison', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(evaluation_results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'model_comparisons' / 'dataset_scatterplots.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Dataset comparison visualizations created successfully")

    except Exception as e:
        logger.error(f"Error creating dataset comparison: {e}")


def aggregate_results(all_results: Dict, config: Config) -> pd.DataFrame:
    """
    Aggregate cross-validation results.

    Args:
        all_results: Results from cross-validation
        config: Configuration object

    Returns:
        DataFrame with aggregated results
    """
    aggregated = []

    for target, results in all_results.items():
        fold_results = results['fold_results']

        metrics = ['rmse', 'mae', 'r2']
        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            aggregated.append({
                'target': target,
                'metric': metric,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            })

    return pd.DataFrame(aggregated)


def create_visualizations(all_results: Dict, config: Config, feature_names: List[str]):
    """
    Create visualizations for the results.

    Args:
        all_results: Results from cross-validation
        config: Configuration object
        feature_names: List of feature names
    """
    try:
        # Set up the plotting style
        plt.style.use('default')

        # Create results visualization
        target_name = config.TARGET_COLUMNS[0]
        fold_results = all_results[target_name]['fold_results']

        # Extract metrics
        rmse_values = [r['rmse'] for r in fold_results]
        mae_values = [r['mae'] for r in fold_results]
        r2_values = [r['r2'] for r in fold_results]

        # Create subplots - Updated for 10 folds
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # RMSE plot
        axes[0].bar(range(1, len(rmse_values) + 1), rmse_values, color='steelblue', alpha=0.7)
        axes[0].set_title('RMSE by Fold (10-Fold CV)', fontsize=14)
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('RMSE')
        axes[0].axhline(y=np.mean(rmse_values), color='r', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(rmse_values):.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE plot
        axes[1].bar(range(1, len(mae_values) + 1), mae_values, color='green', alpha=0.7)
        axes[1].set_title('MAE by Fold (10-Fold CV)', fontsize=14)
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('MAE')
        axes[1].axhline(y=np.mean(mae_values), color='r', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(mae_values):.2f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # R² plot
        axes[2].bar(range(1, len(r2_values) + 1), r2_values, color='orange', alpha=0.7)
        axes[2].set_title('R² by Fold (10-Fold CV)', fontsize=14)
        axes[2].set_xlabel('Fold')
        axes[2].set_ylabel('R²')
        axes[2].axhline(y=np.mean(r2_values), color='r', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(r2_values):.4f}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'training_analysis' / 'cross_validation_results_10fold.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        logger.info("10-fold cross-validation visualizations created successfully")

    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


def create_linear_regression_visualization(df: pd.DataFrame, config: Config):
    """
    Create comprehensive linear regression visualizations with scatterplots.

    Args:
        df: Input dataframe
        config: Configuration object
    """
    try:
        # Prepare data
        X, y, feature_names = prepare_features_targets(df, config)

        # Remove any rows with missing values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(config.K_BEST, X_scaled.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)

        # Get selected feature names
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]

        # Train linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_selected, y)
        y_pred = lr_model.predict(X_selected)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Calculate correlation coefficient
        correlation, p_value = stats.pearsonr(y, y_pred)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # Main scatterplot (actual vs predicted)
        ax1 = plt.subplot(2, 3, 1)
        plt.scatter(y, y_pred, alpha=0.6, color='steelblue', s=50)

        # Perfect prediction line (diagonal)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Regression line
        z = np.polyfit(y, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y, p(y), 'orange', linewidth=2, label=f'Regression Line (slope={z[0]:.3f})')

        plt.xlabel('Actual Year', fontsize=12)
        plt.ylabel('Predicted Year', fontsize=12)
        plt.title(f'Linear Regression: Actual vs Predicted Years\nR² = {r2:.4f}, RMSE = {rmse:.2f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add text box with metrics
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nCorrelation = {correlation:.4f}\np-value = {p_value:.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Residuals plot
        ax2 = plt.subplot(2, 3, 2)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green', s=50)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Year', fontsize=12)
        plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
        plt.title('Residuals Plot', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add mean and std of residuals
        textstr_res = f'Mean Residual = {residuals.mean():.3f}\nStd Residual = {residuals.std():.3f}'
        ax2.text(0.05, 0.95, textstr_res, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # Histogram of residuals
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Residuals', fontsize=14)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)

        # Feature importance plot (top 10)
        ax4 = plt.subplot(2, 3, 4)
        feature_importance = np.abs(lr_model.coef_)
        top_10_indices = np.argsort(feature_importance)[-10:]
        top_10_features = [selected_features[i] for i in top_10_indices]
        top_10_importance = feature_importance[top_10_indices]

        plt.barh(range(len(top_10_features)), top_10_importance, color='coral')
        plt.yticks(range(len(top_10_features)), top_10_features)
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.title('Top 10 Feature Importance (Linear Regression)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')

        # Q-Q plot for residuals normality check
        ax5 = plt.subplot(2, 3, 5)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals\n(Normality Check)', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Year distribution comparison
        ax6 = plt.subplot(2, 3, 6)
        plt.hist(y, bins=30, alpha=0.7, label='Actual Years', color='blue', density=True)
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted Years', color='red', density=True)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'training_analysis' / 'linear_regression_visualization.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create a detailed scatterplot with confidence intervals
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot
        residuals = y - y_pred
        scatter = ax.scatter(y, y_pred, alpha=0.6, c=residuals, cmap='RdYlBu_r',
                             s=60, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3,
                label='Perfect Prediction', alpha=0.8)

        # Regression line with confidence interval
        z = np.polyfit(y, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y, p(y), 'orange', linewidth=3, label=f'Regression Line', alpha=0.9)

        # Add confidence bands (±1 std of residuals)
        std_residuals = residuals.std()
        ax.fill_between(sorted(y), p(sorted(y)) - std_residuals, p(sorted(y)) + std_residuals,
                        alpha=0.2, color='orange', label=f'±1 Std ({std_residuals:.1f} years)')

        # Colorbar for residuals
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Residuals (Actual - Predicted)', fontsize=12)

        ax.set_xlabel('Actual Year', fontsize=14)
        ax.set_ylabel('Predicted Year', fontsize=14)
        ax.set_title(f'Linear Regression: Year Prediction Analysis\n'
                     f'R² = {r2:.4f}, RMSE = {rmse:.2f} years, MAE = {mae:.2f} years',
                     fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = (f'Model Performance:\n'
                      f'• R² Score: {r2:.4f}\n'
                      f'• RMSE: {rmse:.2f} years\n'
                      f'• MAE: {mae:.2f} years\n'
                      f'• Correlation: {correlation:.4f}\n'
                      f'• Sample Size: {len(y)}\n'
                      f'• Features Used: {len(selected_features)}')

        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'training_analysis' / 'detailed_linear_regression_scatterplot.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save feature importance data
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'coefficient': lr_model.coef_,
            'abs_coefficient': np.abs(lr_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)

        feature_importance_df.to_csv(
            config.OUTPUT_DIR / 'training_analysis' / 'linear_regression_feature_importance.csv',
            index=False)

        # Save model performance metrics
        metrics_df = pd.DataFrame({
            'metric': ['R²', 'RMSE', 'MAE', 'Correlation', 'P-value', 'Sample_Size', 'Features_Used'],
            'value': [r2, rmse, mae, correlation, p_value, len(y), len(selected_features)]
        })

        metrics_df.to_csv(config.OUTPUT_DIR / 'training_analysis' / 'linear_regression_metrics.csv', index=False)

        logger.info("Linear regression visualizations created successfully")
        logger.info(f"Linear Regression Performance - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        return lr_model, y, y_pred, selected_features, metrics_df

    except Exception as e:
        logger.error(f"Error creating linear regression visualization: {e}")
        raise


def create_random_forest_visualizations(df: pd.DataFrame, config: Config):
    """
    Create comprehensive Random Forest visualizations including feature importance,
    prediction scatter plots, and performance analysis.

    Args:
        df: Input dataframe
        config: Configuration object
    """
    try:
        # Prepare data
        X, y, feature_names = prepare_features_targets(df, config)

        # Remove any rows with missing values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(config.K_BEST, X_scaled.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)

        # Get selected feature names
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]

        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )

        rf_model.fit(X_selected, y)
        y_pred = rf_model.predict(X_selected)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Calculate correlation coefficient
        correlation, p_value = stats.pearsonr(y, y_pred)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))

        # 1. Main scatterplot (actual vs predicted)
        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(y, y_pred, alpha=0.6, color='forestgreen', s=50, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Regression line
        z = np.polyfit(y, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y, p(y), 'orange', linewidth=2, label=f'Trend Line (slope={z[0]:.3f})')

        plt.xlabel('Actual Year', fontsize=12)
        plt.ylabel('Predicted Year', fontsize=12)
        plt.title(f'Random Forest: Actual vs Predicted Years\nR² = {r2:.4f}, RMSE = {rmse:.2f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add metrics text box
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nCorrelation = {correlation:.4f}'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # 2. Feature Importance (top 15)
        ax2 = plt.subplot(3, 3, 2)
        feature_importance = rf_model.feature_importances_
        top_15_indices = np.argsort(feature_importance)[-15:]
        top_15_features = [selected_features[i] for i in top_15_indices]
        top_15_importance = feature_importance[top_15_indices]

        bars = plt.barh(range(len(top_15_features)), top_15_importance, color='darkgreen', alpha=0.7)
        plt.yticks(range(len(top_15_features)), top_15_features, fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top 15 Feature Importance\n(Random Forest)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')

        # Add values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.0001, bar.get_y() + bar.get_height() / 2,
                     f'{width:.4f}', ha='left', va='center', fontsize=8)

        # 3. Residuals plot
        ax3 = plt.subplot(3, 3, 3)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='red', s=50)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Year', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title('Residuals Plot', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add residual statistics
        textstr_res = f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}\nMin: {residuals.min():.1f}\nMax: {residuals.max():.1f}'
        ax3.text(0.05, 0.95, textstr_res, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # 4. Histogram of residuals
        ax4 = plt.subplot(3, 3, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black', density=True)
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Residuals', fontsize=14)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)

        # Add normal distribution overlay
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit (μ={mu:.1f}, σ={sigma:.1f})')
        plt.legend()

        # 5. Q-Q plot
        ax5 = plt.subplot(3, 3, 5)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals\n(Normality Check)', fontsize=14)
        plt.grid(True, alpha=0.3)

        # 6. Prediction vs Actual by Year Range
        ax6 = plt.subplot(3, 3, 6)
        year_ranges = pd.cut(y, bins=5, labels=['Very Early', 'Early', 'Middle', 'Late', 'Very Late'])
        df_plot = pd.DataFrame({'Actual': y, 'Predicted': y_pred, 'Range': year_ranges})

        for i, range_name in enumerate(['Very Early', 'Early', 'Middle', 'Late', 'Very Late']):
            range_data = df_plot[df_plot['Range'] == range_name]
            if len(range_data) > 0:
                plt.scatter(range_data['Actual'], range_data['Predicted'],
                            label=range_name, alpha=0.7, s=60)

        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.8)
        plt.xlabel('Actual Year', fontsize=12)
        plt.ylabel('Predicted Year', fontsize=12)
        plt.title('Predictions by Year Range', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 7. Error distribution by prediction value
        ax7 = plt.subplot(3, 3, 7)
        abs_errors = np.abs(residuals)
        plt.scatter(y_pred, abs_errors, alpha=0.6, color='orange', s=50)
        plt.xlabel('Predicted Year', fontsize=12)
        plt.ylabel('Absolute Error', fontsize=12)
        plt.title('Absolute Error vs Prediction', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add trend line for errors
        z_error = np.polyfit(y_pred, abs_errors, 1)
        p_error = np.poly1d(z_error)
        plt.plot(y_pred, p_error(y_pred), 'red', linewidth=2, alpha=0.8)

        # 8. Year distribution comparison
        ax8 = plt.subplot(3, 3, 8)
        plt.hist(y, bins=30, alpha=0.7, label='Actual Years', color='blue', density=True)
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted Years', color='green', density=True)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 9. Model complexity analysis (tree depths)
        ax9 = plt.subplot(3, 3, 9)
        tree_depths = [tree.get_depth() for tree in rf_model.estimators_]
        plt.hist(tree_depths, bins=20, alpha=0.7, color='brown', edgecolor='black')
        plt.xlabel('Tree Depth', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Tree Depth Distribution\n(Avg: {np.mean(tree_depths):.1f})', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'training_analysis' / 'random_forest_comprehensive_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create detailed scatterplot similar to linear regression
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot with color-coded residuals
        scatter = ax.scatter(y, y_pred, alpha=0.6, c=residuals, cmap='RdYlGn',
                             s=60, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3,
                label='Perfect Prediction', alpha=0.8)

        # Trend line
        ax.plot(y, p(y), 'navy', linewidth=3, label=f'Trend Line', alpha=0.9)

        # Add confidence bands (±1 std of residuals)
        std_residuals = residuals.std()
        ax.fill_between(sorted(y), p(sorted(y)) - std_residuals, p(sorted(y)) + std_residuals,
                        alpha=0.2, color='blue', label=f'±1 Std ({std_residuals:.1f} years)')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Residuals (Actual - Predicted)', fontsize=12)

        ax.set_xlabel('Actual Year', fontsize=14)
        ax.set_ylabel('Predicted Year', fontsize=14)
        ax.set_title(f'Random Forest: Year Prediction Analysis\n'
                     f'R² = {r2:.4f}, RMSE = {rmse:.2f} years, MAE = {mae:.2f} years',
                     fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = (f'Random Forest Performance:\n'
                      f'• R² Score: {r2:.4f}\n'
                      f'• RMSE: {rmse:.2f} years\n'
                      f'• MAE: {mae:.2f} years\n'
                      f'• Correlation: {correlation:.4f}\n'
                      f'• Trees: {config.N_ESTIMATORS}\n'
                      f'• Avg Tree Depth: {np.mean(tree_depths):.1f}\n'
                      f'• Sample Size: {len(y)}\n'
                      f'• Features Used: {len(selected_features)}')

        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'training_analysis' / 'random_forest_detailed_scatterplot.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save feature importance data
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importance_df.to_csv(config.OUTPUT_DIR / 'training_analysis' / 'random_forest_feature_importance.csv',
                                     index=False)

        # Save model performance metrics
        metrics_df = pd.DataFrame({
            'metric': ['R²', 'RMSE', 'MAE', 'Correlation', 'P-value', 'Sample_Size', 'Features_Used', 'Avg_Tree_Depth'],
            'value': [r2, rmse, mae, correlation, p_value, len(y), len(selected_features), np.mean(tree_depths)]
        })

        metrics_df.to_csv(config.OUTPUT_DIR / 'training_analysis' / 'random_forest_metrics.csv', index=False)

        logger.info("Random Forest visualizations created successfully")
        logger.info(f"Random Forest Performance - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        return rf_model, y, y_pred, selected_features, metrics_df

    except Exception as e:
        logger.error(f"Error creating Random Forest visualization: {e}")
        raise


def create_model_comparison_visualization(df: pd.DataFrame, config: Config,
                                          lr_metrics: pd.DataFrame, rf_metrics: pd.DataFrame):
    """
    Create side-by-side comparison of Linear Regression and Random Forest models.

    Args:
        df: Input dataframe
        config: Configuration object
        lr_metrics: Linear Regression metrics
        rf_metrics: Random Forest metrics
    """
    try:
        # Extract key metrics for comparison
        lr_r2 = lr_metrics[lr_metrics['metric'] == 'R²']['value'].iloc[0]
        lr_rmse = lr_metrics[lr_metrics['metric'] == 'RMSE']['value'].iloc[0]
        lr_mae = lr_metrics[lr_metrics['metric'] == 'MAE']['value'].iloc[0]

        rf_r2 = rf_metrics[rf_metrics['metric'] == 'R²']['value'].iloc[0]
        rf_rmse = rf_metrics[rf_metrics['metric'] == 'RMSE']['value'].iloc[0]
        rf_mae = rf_metrics[rf_metrics['metric'] == 'MAE']['value'].iloc[0]

        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        models = ['Linear Regression', 'Random Forest']

        # R² comparison
        r2_values = [lr_r2, rf_r2]
        bars1 = axes[0].bar(models, r2_values, color=['lightblue', 'lightgreen'], alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² Score Comparison', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar, value in zip(bars1, r2_values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # RMSE comparison
        rmse_values = [lr_rmse, rf_rmse]
        bars2 = axes[1].bar(models, rmse_values, color=['lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('RMSE (years)', fontsize=12)
        axes[1].set_title('RMSE Comparison', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars2, rmse_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # MAE comparison
        mae_values = [lr_mae, rf_mae]
        bars3 = axes[2].bar(models, mae_values, color=['lightsalmon', 'lightgreen'], alpha=0.8, edgecolor='black')
        axes[2].set_ylabel('MAE (years)', fontsize=12)
        axes[2].set_title('MAE Comparison', fontsize=14)
        axes[2].grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars3, mae_values):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'training_analysis' / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create summary comparison table
        comparison_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest'],
            'R²': [lr_r2, rf_r2],
            'RMSE': [lr_rmse, rf_rmse],
            'MAE': [lr_mae, rf_mae]
        })

        comparison_df.to_csv(config.OUTPUT_DIR / 'training_analysis' / 'model_comparison.csv', index=False)

        logger.info("Model comparison visualization created successfully")

    except Exception as e:
        logger.error(f"Error creating model comparison: {e}")


def create_feature_correlation_heatmap(df: pd.DataFrame, config: Config):
    """
    Create a correlation heatmap for the top features.

    Args:
        df: Input dataframe
        config: Configuration object
    """
    try:
        # Prepare data
        X, y, feature_names = prepare_features_targets(df, config)

        # Remove any rows with missing values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        # Create DataFrame with features and target
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df['Year'] = y

        # Calculate correlations with target
        correlations = feature_df.corr()['Year'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
        top_features = correlations.head(15).index.tolist()  # Top 15 features

        # Create correlation matrix for top features + target
        correlation_matrix = feature_df[top_features + ['Year']].corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8})

        plt.title('Feature Correlation Heatmap\n(Top 15 Features vs Year)', fontsize=16)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / 'training_analysis' / 'feature_correlation_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Feature correlation heatmap created successfully")

    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")


def run_main_analysis():
    """Main execution function for year prediction with external dataset evaluation."""
    try:
        # Initialize configuration
        config = Config()
        config.create_output_dirs()

        # Load and validate main training data
        df = load_and_validate_data(config.DATA_PATH)
        validate_year_data(df, config)

        # Check that all required columns exist
        required_cols = [config.TEXT_COLUMN] + config.TARGET_COLUMNS + config.FEATURE_COLUMNS
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Training dataset contains {len(df)} documents")
        logger.info(f"Using {len(config.FEATURE_COLUMNS)} feature columns")
        logger.info(f"Performing {config.N_FOLDS}-fold cross-validation")

        # Perform cross-validation on training data
        all_results = perform_cross_validation(df, config)

        # CREATE TRAINING VISUALIZATIONS
        logger.info("Creating linear regression visualizations...")
        lr_model, y_actual_lr, y_predicted_lr, selected_features_lr, lr_metrics = create_linear_regression_visualization(
            df, config)

        logger.info("Creating Random Forest visualizations...")
        rf_model, y_actual_rf, y_predicted_rf, selected_features_rf, rf_metrics = create_random_forest_visualizations(
            df, config)

        logger.info("Creating model comparison visualizations...")
        create_model_comparison_visualization(df, config, lr_metrics, rf_metrics)

        create_feature_correlation_heatmap(df, config)

        # TRAIN FINAL MODELS ON FULL TRAINING DATA
        logger.info("Training final models on full training dataset...")
        models, scalers, selectors = train_final_models(df, config)

        # EVALUATE ON EXTERNAL DATASETS
        logger.info("Evaluating models on external datasets...")
        evaluation_results = evaluate_external_datasets(models, scalers, selectors, config)

        # CREATE COMPARISON VISUALIZATIONS
        if evaluation_results:
            logger.info("Creating logistic-like prediction curves...")
            create_logistic_prediction_curves(evaluation_results, config)

            logger.info("Creating dataset comparison visualizations...")
            create_dataset_comparison_visualization(evaluation_results, config)
        else:
            logger.warning("No external dataset evaluation results available for visualization")

        logger.info("Creating overlay & logistic comparison plots...")
        plot_overlay_and_logistic_curves(
            y_train=y_actual_rf,
            y_train_pred=y_predicted_rf,
            evaluation_results=evaluation_results,
            config=config
        )

        # Process training results
        aggregated_results = aggregate_results(all_results, config)
        aggregated_results.to_csv(config.OUTPUT_DIR / "year_prediction_results_10fold.csv", index=False)

        target_name = config.TARGET_COLUMNS[0]
        fold_results = pd.DataFrame(all_results[target_name]['fold_results'])
        fold_results.to_csv(config.OUTPUT_DIR / f"{target_name}_fold_results_10fold.csv", index=False)

        create_visualizations(all_results, config, config.FEATURE_COLUMNS)

        # CREATE COMPREHENSIVE REPORTS
        logger.info("Creating comprehensive reports...")
        create_comprehensive_report(evaluation_results, lr_metrics, rf_metrics, all_results, config)

        # Print comprehensive results summary
        logger.info("Analysis completed successfully!")

        print(f"\n{'=' * 60}")
        print("TRAINING DATASET RESULTS (10-Fold Cross-Validation)")
        print(f"{'=' * 60}")

        year_results = all_results[target_name]['fold_results']
        avg_rmse = np.mean([r['rmse'] for r in year_results])
        avg_mae = np.mean([r['mae'] for r in year_results])
        avg_r2 = np.mean([r['r2'] for r in year_results])

        print(f"Random Forest Cross-Validation Performance:")
        print(f"  RMSE: {avg_rmse:.2f} years")
        print(f"  MAE: {avg_mae:.2f} years")
        print(f"  R²: {avg_r2:.4f}")

        print(f"\nLinear Regression Performance (Full Dataset):")
        print(f"  R²: {lr_metrics[lr_metrics['metric'] == 'R²']['value'].iloc[0]:.4f}")
        print(f"  RMSE: {lr_metrics[lr_metrics['metric'] == 'RMSE']['value'].iloc[0]:.2f} years")
        print(f"  MAE: {lr_metrics[lr_metrics['metric'] == 'MAE']['value'].iloc[0]:.2f} years")

        print(f"\nRandom Forest Performance (Full Dataset):")
        print(f"  R²: {rf_metrics[rf_metrics['metric'] == 'R²']['value'].iloc[0]:.4f}")
        print(f"  RMSE: {rf_metrics[rf_metrics['metric'] == 'RMSE']['value'].iloc[0]:.2f} years")
        print(f"  MAE: {rf_metrics[rf_metrics['metric'] == 'MAE']['value'].iloc[0]:.2f} years")

        # Print external evaluation results
        if evaluation_results:
            print(f"\n{'=' * 60}")
            print("EXTERNAL DATASET EVALUATION RESULTS")
            print(f"{'=' * 60}")

            for dataset_name, results in evaluation_results.items():
                print(f"\n{dataset_name}:")
                for model_name, metrics in results.items():
                    print(f"  {model_name.replace('_', ' ').title()}:")
                    print(f"    R²: {metrics['r2']:.4f}")
                    print(f"    RMSE: {metrics['rmse']:.2f} years")
                    print(f"    MAE: {metrics['mae']:.2f} years")
                    print(f"    Samples: {metrics['n_samples']}")

        print(f"\n{'=' * 60}")
        print("COMPREHENSIVE REPORTS GENERATED!")
        print(f"{'=' * 60}")
        print(f"Results saved to: {config.OUTPUT_DIR}")
        print("KEY REPORT FILES:")
        print("• reports/comprehensive_report.html - Complete HTML report")
        print("• reports/comprehensive_results.csv - All results in CSV")
        print("• reports/executive_summary.txt - Executive summary")
        print("• reports/cross_validation_details.csv - CV details")
        if evaluation_results:
            print("• reports/external_evaluation_details.csv - External dataset results")
        print("• reports/model_comparison_summary.csv - Model comparisons")
        print(f"{'=' * 60}")

    except Exception as e:
        logger.error(f"Year prediction pipeline failed: {e}")
        raise


def plot_overlay_and_logistic_curves(
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        evaluation_results: Dict,
        config: Config,
        max_threshold: int = 50
    ):
    """
    1. Overlay external-set scatters (with fixed colours / markers) on the
       training scatter and add a LOWESS aggregate trend.
    2. Plot accuracy-vs-error-threshold (“logistic”) curves for training,
       every individual external set, and the external aggregate.

    Saved files:
        • model_comparisons/scatter_with_test_overlay.png
        • model_comparisons/logistic_curves_train_vs_test.png
    """
    # ------------------------------------------------------------------ #
    #                     ── helpers & settings ──                       #
    # ------------------------------------------------------------------ #
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def accuracy_curve(y_true, y_pred, thresholds):
        err = np.abs(y_true - y_pred)
        return np.array([np.mean(err <= t) for t in thresholds])

    palette = config.palette          # shorthand
    ds_order = ["Dataset_1", "Dataset_2", "Dataset_4", "Dataset_3"]  # draw order

    thresholds = np.linspace(0, max_threshold, 100)

    # ------------------------------------------------------------------ #
    # 1⃣   Scatter overlay                                               #
    # ------------------------------------------------------------------ #
    fig_s, ax = plt.subplots(figsize=(10, 8))

    # training cloud
    ax.scatter(y_train, y_train_pred,
               alpha=0.4, s=40,
               color=palette["Training"][0],
               marker=palette["Training"][1],
               label=f"Training ({len(y_train)})")

    ext_y_true_all, ext_y_pred_all = [], []   # for aggregate trend

    for ds_name in ds_order:
        if ds_name not in evaluation_results:
            continue
        if "random_forest" not in evaluation_results[ds_name]:
            continue

        y_true = evaluation_results[ds_name]["random_forest"]["y_true"]
        y_pred = evaluation_results[ds_name]["random_forest"]["y_pred"]

        ext_y_true_all.append(y_true)
        ext_y_pred_all.append(y_pred)

        color, marker = palette[ds_name]
        z = 4 if ds_name == "Dataset_3" else 3     # FLOB on top
        ax.scatter(y_true, y_pred, s=60,
                   color=color, marker=marker,
                   edgecolors='black', linewidth=0.5,
                   alpha=0.75, zorder=z,
                   label=f"{ds_name} ({len(y_true)})")

    # LOWESS aggregate trend
    if ext_y_true_all:
        y_true_cat = np.concatenate(ext_y_true_all)
        y_pred_cat = np.concatenate(ext_y_pred_all)
        lowess_out = lowess(y_pred_cat, y_true_cat,
                            frac=0.25, return_sorted=True)
        ax.plot(lowess_out[:, 0], lowess_out[:, 1],
                color=palette["Aggregate"][0],
                lw=3, label="External aggregate trend")

    # perfect-prediction line
    min_ax = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_ax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_ax, max_ax], [min_ax, max_ax],
            'k--', lw=2, label='Perfect prediction')

    ax.set_xlabel("Actual Year", fontsize=12)
    ax.set_ylabel("Predicted Year", fontsize=12)
    ax.set_title("Random Forest – Training & External Datasets", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig_s.tight_layout()
    fig_s.savefig(
        config.OUTPUT_DIR / 'model_comparisons' /
        'scatter_with_test_overlay.png',
        dpi=300, bbox_inches='tight')
    plt.close(fig_s)

    # ------------------------------------------------------------------ #
    # 2⃣   Logistic / accuracy-threshold curves                           #
    # ------------------------------------------------------------------ #
    fig_l, axl = plt.subplots(figsize=(10, 8))

    # training baseline (black)
    train_acc = accuracy_curve(y_train, y_train_pred, thresholds)
    axl.plot(thresholds, train_acc, lw=3, color='black',
             label=f"Training – {len(y_train)}")
    try:
        popt, _ = curve_fit(logistic, thresholds, train_acc,
                            bounds=([0.8, 0, -10], [1.05, 1, 60]))
        axl.plot(thresholds, logistic(thresholds, *popt),
                 lw=2, color='black', linestyle='--')
    except RuntimeError:
        pass

    # individual external curves
    for ds_name in ds_order:
        if ds_name not in evaluation_results:
            continue
        if "random_forest" not in evaluation_results[ds_name]:
            continue

        y_true = evaluation_results[ds_name]["random_forest"]["y_true"]
        y_pred = evaluation_results[ds_name]["random_forest"]["y_pred"]

        acc = accuracy_curve(y_true, y_pred, thresholds)
        color, marker = palette[ds_name]
        axl.plot(thresholds, acc,
                 lw=2, color=color, marker=marker,
                 markevery=10, markersize=6,
                 label=f"{ds_name} – {len(y_true)}")
        try:
            popt, _ = curve_fit(logistic, thresholds, acc,
                                bounds=([0.8, 0, -10], [1.05, 1, 60]))
            axl.plot(thresholds, logistic(thresholds, *popt),
                     lw=2, color=color, linestyle='--')
        except RuntimeError:
            pass

    # external aggregate curve
    if ext_y_true_all:
        agg_acc = accuracy_curve(y_true_cat, y_pred_cat, thresholds)
        axl.plot(thresholds, agg_acc,
                 lw=3, color=palette["Aggregate"][0],
                 label=f"External aggregate ({len(y_true_cat)})")
        try:
            popt, _ = curve_fit(logistic, thresholds, agg_acc,
                                bounds=([0.8, 0, -10], [1.05, 1, 60]))
            axl.plot(thresholds, logistic(thresholds, *popt),
                     lw=2, color=palette["Aggregate"][0], linestyle='--')
        except RuntimeError:
            pass

    # cosmetics
    axl.set_xlim(0, max_threshold)
    axl.set_ylim(0, 1.05)
    axl.set_xlabel("Error Threshold (years)", fontsize=12)
    axl.set_ylabel("Proportion ≤ Threshold", fontsize=12)
    axl.set_title("Accuracy-vs-Tolerance Curves\n"
                  "(Training & External Sets)", fontsize=14)
    axl.axhline(0.5, ls=':', color='gray')
    axl.axhline(0.8, ls=':', color='gray')
    axl.axvline(5,  ls=':', color='gray')
    axl.axvline(10, ls=':', color='gray')
    axl.grid(alpha=0.3)
    axl.legend(fontsize=9, loc='lower right')

    fig_l.tight_layout()
    fig_l.savefig(
        config.OUTPUT_DIR / 'model_comparisons' /
        'logistic_curves_train_vs_test.png',
        dpi=300, bbox_inches='tight')
    plt.close(fig_l)

    logger.info("Overlay scatter & logistic-curve comparison saved.")

if __name__ == "__main__":
    run_main_analysis()