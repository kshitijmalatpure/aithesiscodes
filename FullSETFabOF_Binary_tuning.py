"""
binary_classification_hyperparameter_tuning_fixed.py
====================================================
Fixed comprehensive hyperparameter tuning framework for Binary Classification
Supports multiple optimization strategies with SelectKBest50 and organized structure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time
import json
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import multiprocessing as mp

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler

# Bayesian optimization (optional - install with: pip install scikit-optimize)
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not available. Bayesian optimization disabled.")

warnings.filterwarnings("ignore")


class BinaryHyperparameterSpace:
    """Define hyperparameter search spaces for binary classification components"""

    @staticmethod
    def get_random_forest_space(search_type='grid'):
        """Get Random Forest hyperparameter space for binary classification"""
        if search_type == 'grid':
            return {
                'n_estimators': [100, 300, 500, 800],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
                'bootstrap': [True],  # Keep True for consistency
                'random_state': [42]
            }
        elif search_type == 'random':
            return {
                'n_estimators': [50, 100, 200, 300, 500, 800, 1000],
                'max_depth': [None, 5, 10, 15, 20, 25, 30, 50],
                'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
                'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10],
                'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
                'bootstrap': [True],
                'random_state': [42]
            }
        elif search_type == 'bayesian':
            return [
                Integer(50, 1000, name='n_estimators'),
                Integer(5, 50, name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Categorical(['sqrt', 'log2', None], name='max_features'),
                Categorical([None, 'balanced', 'balanced_subsample'], name='class_weight')
            ]

    @staticmethod
    def get_feature_selection_space(search_type='grid'):
        """Get feature selection hyperparameter space - focused on SelectKBest50"""
        if search_type in ['grid', 'random']:
            return {
                'kbest_k': [50],  # Fixed to 50 as requested
                'kbest_score_func': ['f_classif', 'chi2', 'mutual_info_classif'],
                'feature_selection_method': ['kbest']  # Focus only on KBest
            }
        elif search_type == 'bayesian':
            return [
                Categorical(['f_classif', 'chi2', 'mutual_info_classif'], name='kbest_score_func')
                # k is fixed at 50
            ]

    @staticmethod
    def get_preprocessing_space(search_type='grid'):
        """Get preprocessing hyperparameter space"""
        if search_type in ['grid', 'random']:
            return {
                'imputation_strategy': ['mean', 'median', 'most_frequent'],
                'scaling_method': ['standard', 'robust', 'minmax', 'none']
            }
        elif search_type == 'bayesian':
            return [
                Categorical(['mean', 'median', 'most_frequent'], name='imputation_strategy'),
                Categorical(['standard', 'robust', 'minmax', 'none'], name='scaling_method')
            ]


def evaluate_single_hyperparameter_set(args):
    """Function for parallel evaluation of a single hyperparameter set"""
    params, score_col, base_dir, name_col, score_cols, folds_to_use = args

    fold_results = []
    fold_details = []

    for fold in folds_to_use:
        try:
            # Load fold data
            tr_path = Path(base_dir) / f"fold_{fold}" / f"fold{fold}trb.csv"
            te_path = Path(base_dir) / f"fold_{fold}" / f"fold{fold}teb.csv"

            if not tr_path.exists() or not te_path.exists():
                continue

            tr = pd.read_csv(tr_path)
            te = pd.read_csv(te_path)
            tr, te = tr.align(te, axis=1, fill_value=np.nan)

            feats = [c for c in tr.columns if c not in [name_col] + score_cols]
            X_tr, X_te = tr[feats].values, te[feats].values
            y_tr, y_te = tr[score_col].values, te[score_col].values

            # Check class distribution
            unique_tr = np.unique(y_tr)
            if len(unique_tr) < 2:
                continue

            # Apply preprocessing
            X_tr_prep, X_te_prep = apply_preprocessing_standalone(X_tr, X_te, params)

            # Apply feature selection (SelectKBest50)
            X_tr_fs, X_te_fs = apply_feature_selection_standalone(
                X_tr_prep, X_te_prep, y_tr, params
            )

            # Fit model
            rf = fit_binary_model_standalone(X_tr_fs, y_tr, params)

            # Make predictions
            y_pred, y_pred_proba = predict_binary_model_standalone(rf, X_te_fs)

            # Calculate metrics
            metrics = calculate_binary_metrics_standalone(y_te, y_pred, y_pred_proba)

            fold_results.append(metrics)

            # Store detailed fold information
            fold_detail = {
                'fold': fold,
                'train_samples': len(y_tr),
                'test_samples': len(y_te),
                'train_positive_ratio': float(np.mean(y_tr)),
                'test_positive_ratio': float(np.mean(y_te)),
                'selected_features': 50,  # Fixed to 50
                **{k: float(v) for k, v in metrics.items()}
            }
            fold_details.append(fold_detail)

        except Exception as e:
            continue

    if not fold_results:
        return {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
            'specificity': 0, 'roc_auc': 0, 'average_precision': 0, 'mcc': 0,
            'n_valid_folds': 0, 'fold_details': []
        }

    # Aggregate results across folds
    aggregated = {}
    for metric in fold_results[0].keys():
        values = [result[metric] for result in fold_results]
        aggregated[metric] = float(np.mean(values))
        aggregated[f'{metric}_std'] = float(np.std(values))

    # Add fold information
    aggregated['n_valid_folds'] = len(fold_results)
    aggregated['fold_details'] = fold_details

    return aggregated


def apply_preprocessing_standalone(X_tr, X_te, params):
    """Standalone preprocessing function for multiprocessing"""
    # Imputation
    imputation_strategy = params.get('imputation_strategy', 'median')
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    # Scaling
    scaling_method = params.get('scaling_method', 'standard')
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:  # none
        scaler = None

    if scaler:
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_te_scaled = scaler.transform(X_te_imp)
    else:
        X_tr_scaled, X_te_scaled = X_tr_imp, X_te_imp

    return X_tr_scaled, X_te_scaled


def apply_feature_selection_standalone(X_tr, X_te, y_tr, params):
    """Standalone feature selection function for multiprocessing"""
    k = 50  # Fixed to 50
    score_func_name = params.get('kbest_score_func', 'f_classif')

    # Map score function names to actual functions
    score_func_map = {
        'f_classif': f_classif,
        'chi2': chi2,
        'mutual_info_classif': mutual_info_classif
    }

    score_func = score_func_map.get(score_func_name, f_classif)

    # Handle chi2 requirement for non-negative features
    if score_func_name == 'chi2':
        # Make features non-negative for chi2
        X_tr_chi2 = X_tr - X_tr.min(axis=0) + 1e-8
        X_te_chi2 = X_te - X_tr.min(axis=0) + 1e-8  # Use training min for consistency
        selector = SelectKBest(score_func, k=k)
        X_tr_fs = selector.fit_transform(X_tr_chi2, y_tr)
        X_te_fs = selector.transform(X_te_chi2)
    else:
        selector = SelectKBest(score_func, k=k)
        X_tr_fs = selector.fit_transform(X_tr, y_tr)
        X_te_fs = selector.transform(X_te)

    return X_tr_fs, X_te_fs


def fit_binary_model_standalone(X_tr, y_tr, rf_params):
    """Standalone model fitting function for multiprocessing"""
    # Remove non-RF parameters
    clean_rf_params = {k: v for k, v in rf_params.items()
                       if k in ['n_estimators', 'max_depth', 'min_samples_split',
                                'min_samples_leaf', 'max_features', 'bootstrap',
                                'class_weight', 'random_state']}

    # Ensure bootstrap=True and add probability estimation
    clean_rf_params['bootstrap'] = True
    clean_rf_params['oob_score'] = True

    # Fit Random Forest Classifier
    rf = RandomForestClassifier(**clean_rf_params)
    rf.fit(X_tr, y_tr)

    return rf


def predict_binary_model_standalone(rf, X_te):
    """Standalone prediction function for multiprocessing"""
    y_pred = rf.predict(X_te)
    y_pred_proba = rf.predict_proba(X_te)[:, 1]  # Probability of positive class
    return y_pred, y_pred_proba


def calculate_binary_metrics_standalone(y_true, y_pred, y_pred_proba):
    """Standalone metrics calculation function for multiprocessing"""
    # Handle edge cases
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'specificity': 0.0,  # Will calculate below
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    # Calculate specificity manually
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['specificity'] = 0.0

    # AUC and AP only if both classes present
    if len(unique_true) > 1 and len(np.unique(y_pred_proba)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
        metrics['average_precision'] = 0.0

    return metrics


class BinaryClassificationTuner:
    """Advanced hyperparameter tuning framework for Binary Classification"""

    def __init__(self, base_dir: str, name_col: str, score_cols: List[str],
                 tuning_config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = score_cols
        self.config = tuning_config

        # Create organized results directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_root = self.base_dir / "hyperparameter_tuning_results"
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.session_dir = self.results_root / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Organized directory structure
        self.dirs = {
            'logs': self.session_dir / '01_logs',
            'configs': self.session_dir / '02_configurations',
            'raw_results': self.session_dir / '03_raw_results',
            'processed_results': self.session_dir / '04_processed_results',
            'visualizations': self.session_dir / '05_visualizations',
            'best_models': self.session_dir / '06_best_models',
            'confusion_matrices': self.session_dir / '07_confusion_matrices',
            'feature_analysis': self.session_dir / '08_feature_analysis',
            'performance_analysis': self.session_dir / '09_performance_analysis',
            'summary_reports': self.session_dir / '10_summary_reports'
        }

        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize tracking
        self.best_params = {}
        self.tuning_history = []
        self.evaluation_results = []

        # Save configuration
        self._save_session_config()

        self.logger.info("=" * 80)
        self.logger.info("BINARY CLASSIFICATION HYPERPARAMETER TUNING FRAMEWORK")
        self.logger.info("=" * 80)
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Session directory: {self.session_dir}")
        self.logger.info(f"Name column: {self.name_col}")
        self.logger.info(f"Binary score columns: {self.score_cols}")
        self.logger.info(f"Feature selection: SelectKBest with k=50")
        self.logger.info(f"File pattern: fold[X]trb.csv / fold[X]teb.csv")
        self.logger.info(f"Tuning configuration: {self.config}")
        self.logger.info("=" * 80)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging with organized structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.dirs['logs'] / f"hyperparameter_tuning_{timestamp}.log"

        # Create custom formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(funcName)-20s - %(message)s'
        )

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Configure logger
        logger = logging.getLogger(f"{__name__}_{id(self)}")  # Unique logger per instance
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()  # Clear existing handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _save_session_config(self):
        """Save session configuration for reproducibility"""
        config_data = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'base_directory': str(self.base_dir),
                'name_column': self.name_col,
                'score_columns': self.score_cols,
                'file_pattern': 'fold[X]trb.csv / fold[X]teb.csv'
            },
            'tuning_config': self.config,
            'feature_selection': {
                'method': 'SelectKBest',
                'k': 50,
                'score_functions': ['f_classif', 'chi2', 'mutual_info_classif']
            },
            'directory_structure': {k: str(v) for k, v in self.dirs.items()}
        }

        with open(self.dirs['configs'] / 'session_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)

    def random_search(self, score_col: str, param_distributions: Dict[str, List],
                      n_iter: int = 50, n_jobs: int = 1, cv_folds: List[int] = None) -> Dict[str, Any]:
        """Perform random search hyperparameter tuning"""
        self.logger.info(f"Starting Random Search for binary classification: {score_col}")
        self.logger.info(f"Number of iterations: {n_iter}")
        self.logger.info(f"Feature selection: SelectKBest with k=50")

        if cv_folds is None:
            cv_folds = list(range(1, 11))

        # Generate random parameter combinations
        param_sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)
        results = []
        detailed_results = []

        start_time = time.time()

        if n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(param_sampler):
                self.logger.info(f"Evaluating parameter set {i + 1}/{n_iter}")
                self.logger.debug(f"Parameters: {params}")

                eval_result = self._evaluate_hyperparameters_sequential(params, score_col, cv_folds)
                fold_details = eval_result.pop('fold_details', [])

                result_entry = {**params, **eval_result}
                results.append(result_entry)

                # Store detailed results
                detailed_entry = {
                    'iteration': i + 1,
                    'params': params,
                    'aggregated_metrics': eval_result,
                    'fold_details': fold_details
                }
                detailed_results.append(detailed_entry)

                # Log progress
                primary_metric = self.config.get('primary_metric', 'f1_score')
                self.logger.info(f"  {primary_metric}: {eval_result.get(primary_metric, 0):.4f} "
                                 f"(±{eval_result.get(f'{primary_metric}_std', 0):.4f})")
        else:
            # Parallel execution
            self.logger.info(f"Using {min(n_jobs, mp.cpu_count())} parallel processes")

            # Prepare arguments for parallel processing
            param_args = [
                (params, score_col, str(self.base_dir), self.name_col, self.score_cols, cv_folds)
                for params in param_sampler
            ]

            # Use ProcessPoolExecutor for parallel execution
            with ProcessPoolExecutor(max_workers=min(n_jobs, mp.cpu_count())) as executor:
                # Submit all jobs
                future_to_params = {
                    executor.submit(evaluate_single_hyperparameter_set, args): args[0]
                    for args in param_args
                }

                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_params), 1):
                    try:
                        params = future_to_params[future]
                        eval_result = future.result()
                        fold_details = eval_result.pop('fold_details', [])

                        result_entry = {**params, **eval_result}
                        results.append(result_entry)

                        # Store detailed results
                        detailed_entry = {
                            'iteration': i,
                            'params': params,
                            'aggregated_metrics': eval_result,
                            'fold_details': fold_details
                        }
                        detailed_results.append(detailed_entry)

                        # Log progress
                        primary_metric = self.config.get('primary_metric', 'f1_score')
                        self.logger.info(f"Completed {i}/{n_iter} - "
                                         f"{primary_metric}: {eval_result.get(primary_metric, 0):.4f}")

                    except Exception as e:
                        self.logger.warning(f"Error in parallel evaluation: {str(e)}")
                        continue

        # Find best parameters
        primary_metric = self.config.get('primary_metric', 'f1_score')
        maximize = self.config.get('maximize_metric', True)

        valid_results = [r for r in results if r.get('n_valid_folds', 0) > 0]
        if not valid_results:
            raise ValueError("No valid results obtained during hyperparameter search")

        best_result = max(valid_results, key=lambda x: x[primary_metric]) if maximize else min(valid_results,
                                                                                               key=lambda x: x[
                                                                                                   primary_metric])

        elapsed_time = time.time() - start_time
        self.logger.info(f"Random Search completed in {elapsed_time:.2f}s")
        self.logger.info(f"Best {primary_metric}: {best_result[primary_metric]:.4f}")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.dirs['raw_results'] / f"{score_col}_random_search_results.csv", index=False)

        # Save detailed results
        with open(self.dirs['raw_results'] / f"{score_col}_detailed_results.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            detailed_json = []
            for entry in detailed_results:
                json_entry = {}
                for key, value in entry.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_entry[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        json_entry[key] = value.tolist()
                    else:
                        json_entry[key] = value
                detailed_json.append(json_entry)
            json.dump(detailed_json, f, indent=2)

        return {
            'best_params': {k: v for k, v in best_result.items()
                            if k not in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                                         'roc_auc', 'average_precision', 'mcc', 'n_valid_folds']
                            and not k.endswith('_std')},
            'best_score': best_result[primary_metric],
            'best_metrics': {k: v for k, v in best_result.items()
                             if k in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                                      'roc_auc', 'average_precision', 'mcc']},
            'all_results': results,
            'detailed_results': detailed_results,
            'search_time': elapsed_time,
            'n_valid_results': len(valid_results)
        }

    def _evaluate_hyperparameters_sequential(self, params: Dict[str, Any], score_col: str,
                                           folds_to_use: List[int] = None) -> Dict[str, float]:
        """Sequential evaluation of hyperparameters for single-process execution"""
        if folds_to_use is None:
            folds_to_use = list(range(1, 11))

        fold_results = []
        fold_details = []

        for fold in folds_to_use:
            try:
                # Load fold data
                tr_path = self.base_dir / f"fold_{fold}" / f"fold{fold}trb.csv"
                te_path = self.base_dir / f"fold_{fold}" / f"fold{fold}teb.csv"

                if not tr_path.exists() or not te_path.exists():
                    self.logger.warning(f"Missing files for fold {fold}: {tr_path} or {te_path}")
                    continue

                tr = pd.read_csv(tr_path)
                te = pd.read_csv(te_path)
                tr, te = tr.align(te, axis=1, fill_value=np.nan)

                feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]
                X_tr, X_te = tr[feats].values, te[feats].values
                y_tr, y_te = tr[score_col].values, te[score_col].values

                # Check class distribution
                unique_tr = np.unique(y_tr)
                if len(unique_tr) < 2:
                    self.logger.warning(f"Fold {fold}: Training set has only one class: {unique_tr}")
                    continue

                # Apply preprocessing
                X_tr_prep, X_te_prep = apply_preprocessing_standalone(X_tr, X_te, params)

                # Apply feature selection (SelectKBest50)
                X_tr_fs, X_te_fs = apply_feature_selection_standalone(
                    X_tr_prep, X_te_prep, y_tr, params
                )

                # Fit model
                rf = fit_binary_model_standalone(X_tr_fs, y_tr, params)

                # Make predictions
                y_pred, y_pred_proba = predict_binary_model_standalone(rf, X_te_fs)

                # Calculate metrics
                metrics = calculate_binary_metrics_standalone(y_te, y_pred, y_pred_proba)

                fold_results.append(metrics)

                # Store detailed fold information
                fold_detail = {
                    'fold': fold,
                    'train_samples': len(y_tr),
                    'test_samples': len(y_te),
                    'train_positive_ratio': float(np.mean(y_tr)),
                    'test_positive_ratio': float(np.mean(y_te)),
                    'selected_features': 50,
                    **{k: float(v) for k, v in metrics.items()}
                }
                fold_details.append(fold_detail)

            except Exception as e:
                self.logger.warning(f"Error in fold {fold}: {str(e)}")
                continue

        if not fold_results:
            return {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
                'specificity': 0, 'roc_auc': 0, 'average_precision': 0, 'mcc': 0,
                'n_valid_folds': 0, 'fold_details': []
            }

        # Aggregate results across folds
        aggregated = {}
        for metric in fold_results[0].keys():
            values = [result[metric] for result in fold_results]
            aggregated[metric] = float(np.mean(values))
            aggregated[f'{metric}_std'] = float(np.std(values))

        # Add fold information
        aggregated['n_valid_folds'] = len(fold_results)
        aggregated['fold_details'] = fold_details

        return aggregated

    def run_comprehensive_binary_tuning(self) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive hyperparameter tuning for all binary score columns"""
        self.logger.info("Starting comprehensive binary classification hyperparameter tuning")

        all_results = {}
        summary_data = []

        for i, score_col in enumerate(self.score_cols, 1):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"[{i}/{len(self.score_cols)}] Tuning hyperparameters for: {score_col}")
            self.logger.info(f"{'=' * 60}")

            try:
                # Get search spaces
                search_type = self.config.get('search_type', 'random')

                # Combine all parameter spaces
                rf_space = BinaryHyperparameterSpace.get_random_forest_space(search_type)
                fs_space = BinaryHyperparameterSpace.get_feature_selection_space(search_type)
                prep_space = BinaryHyperparameterSpace.get_preprocessing_space(search_type)

                param_distributions = {**rf_space, **fs_space, **prep_space}

                # Run random search (can be extended to other methods)
                score_results = self.random_search(
                    score_col, param_distributions,
                    n_iter=self.config.get('n_iter', 50),
                    n_jobs=self.config.get('n_jobs', 1),
                    cv_folds=self.config.get('cv_folds', None)
                )

                all_results[score_col] = score_results
                self.best_params[score_col] = score_results['best_params']

                # Collect summary data
                summary_entry = {
                    'score_column': score_col,
                    'best_f1_score': score_results['best_metrics'].get('f1_score', 0),
                    'best_accuracy': score_results['best_metrics'].get('accuracy', 0),
                    'best_precision': score_results['best_metrics'].get('precision', 0),
                    'best_recall': score_results['best_metrics'].get('recall', 0),
                    'best_roc_auc': score_results['best_metrics'].get('roc_auc', 0),
                    'search_time': score_results['search_time'],
                    'n_valid_results': score_results['n_valid_results']
                }
                summary_data.append(summary_entry)

                self.logger.info(f"Best results for {score_col}:")
                for metric, value in score_results['best_metrics'].items():
                    self.logger.info(f"  {metric}: {value:.4f}")

            except Exception as e:
                self.logger.error(f"Error tuning {score_col}: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                continue

        # Save comprehensive results
        self._save_comprehensive_binary_results(all_results, summary_data)

        return all_results

    def _save_comprehensive_binary_results(self, results: Dict[str, Dict[str, Any]],
                                           summary_data: List[Dict[str, Any]]):
        """Save comprehensive binary tuning results with organized structure"""
        self.logger.info("Saving comprehensive binary classification results...")

        try:
            # 1. Save best parameters summary
            best_params_df = pd.DataFrame([
                {'Score_Column': score_col, **params}
                for score_col, result in results.items()
                for params in [result['best_params']]
            ])
            best_params_df.to_csv(self.dirs['summary_reports'] / "best_parameters_summary.csv", index=False)

            # 2. Save performance summary
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.dirs['summary_reports'] / "performance_summary.csv", index=False)

            # 3. Save detailed results as JSON
            results_json = {}
            for score_col, result in results.items():
                results_json[score_col] = {
                    'best_params': result['best_params'],
                    'best_metrics': result['best_metrics'],
                    'search_time': float(result['search_time']),
                    'n_valid_results': int(result['n_valid_results'])
                }

            with open(self.dirs['summary_reports'] / "comprehensive_results.json", 'w') as f:
                json.dump(results_json, f, indent=2)

            # 4. Create visualizations
            self._create_binary_visualizations(results, summary_data)

            # 5. Generate confusion matrices for best models
            self._generate_best_model_confusion_matrices(results)

            # 6. Create performance analysis
            self._create_performance_analysis(results, summary_data)

            # 7. Generate final report
            self._generate_final_report(results, summary_data)

            self.logger.info(f"All results saved to session directory: {self.session_dir}")

        except Exception as e:
            self.logger.error(f"Error saving comprehensive results: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _create_binary_visualizations(self, results: Dict[str, Dict[str, Any]],
                                      summary_data: List[Dict[str, Any]]):
        """Create comprehensive visualizations for binary classification results"""
        try:
            # Performance comparison across score columns
            plt.figure(figsize=(15, 10))

            metrics = ['best_f1_score', 'best_accuracy', 'best_precision', 'best_recall', 'best_roc_auc']
            metric_labels = ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC']

            x = np.arange(len(self.score_cols))
            width = 0.15

            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = [next(s[metric] for s in summary_data if s['score_column'] == col)
                          for col in self.score_cols]
                plt.bar(x + i * width, values, width, label=label, alpha=0.8)

            plt.xlabel('Binary Score Columns', fontweight='bold', fontsize=12)
            plt.ylabel('Performance Score', fontweight='bold', fontsize=12)
            plt.title('Binary Classification Hyperparameter Tuning Results', fontweight='bold', fontsize=14)
            plt.xticks(x + width * 2, self.score_cols)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.dirs['visualizations'] / "performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Individual metric distributions
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = [s[metric] for s in summary_data]
                axes[i].bar(self.score_cols, values, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'{label} Comparison', fontweight='bold')
                axes[i].set_ylabel(label)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)

                # Add value labels
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

            # Hide the last subplot
            axes[-1].set_visible(False)

            plt.suptitle('Binary Classification Performance Metrics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.dirs['visualizations'] / "detailed_metrics_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Parameter importance heatmap
            self._create_parameter_heatmap(results)

            self.logger.info("Visualizations created successfully")

        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {str(e)}")

    def _create_parameter_heatmap(self, results: Dict[str, Dict[str, Any]]):
        """Create parameter importance heatmap"""
        try:
            # Collect parameter values for all score columns
            param_data = []
            for score_col, result in results.items():
                entry = {'score_column': score_col}
                entry.update(result['best_params'])
                entry.update(result['best_metrics'])
                param_data.append(entry)

            df = pd.DataFrame(param_data)

            # Select numeric parameters for correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            param_cols = [col for col in numeric_cols
                          if col not in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                                         'roc_auc', 'average_precision', 'mcc']]

            if len(param_cols) > 1:
                correlation_data = df[param_cols + ['f1_score']].corr()

                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                            square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
                plt.title('Parameter Correlation Matrix with F1-Score', fontweight='bold', fontsize=14)
                plt.tight_layout()
                plt.savefig(self.dirs['visualizations'] / "parameter_correlation_heatmap.png",
                            dpi=300, bbox_inches='tight')
                plt.close()

                # Save correlation data
                correlation_data.to_csv(self.dirs['feature_analysis'] / "parameter_correlations.csv")

        except Exception as e:
            self.logger.warning(f"Error creating parameter heatmap: {str(e)}")

    def _generate_best_model_confusion_matrices(self, results: Dict[str, Dict[str, Any]]):
        """Generate confusion matrices for best models"""
        self.logger.info("Generating confusion matrices for best models...")

        for score_col, result in results.items():
            try:
                best_params = result['best_params']

                # Evaluate best model on all folds to get predictions
                all_y_true = []
                all_y_pred = []

                for fold in range(1, 11):
                    try:
                        tr_path = self.base_dir / f"fold_{fold}" / f"fold{fold}trb.csv"
                        te_path = self.base_dir / f"fold_{fold}" / f"fold{fold}teb.csv"

                        if not tr_path.exists() or not te_path.exists():
                            continue

                        tr = pd.read_csv(tr_path)
                        te = pd.read_csv(te_path)
                        tr, te = tr.align(te, axis=1, fill_value=np.nan)

                        feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]
                        X_tr, X_te = tr[feats].values, te[feats].values
                        y_tr, y_te = tr[score_col].values, te[score_col].values

                        # Apply best preprocessing and feature selection
                        X_tr_prep, X_te_prep = apply_preprocessing_standalone(X_tr, X_te, best_params)
                        X_tr_fs, X_te_fs = apply_feature_selection_standalone(
                            X_tr_prep, X_te_prep, y_tr, best_params
                        )

                        # Fit and predict with best model
                        rf = fit_binary_model_standalone(X_tr_fs, y_tr, best_params)
                        y_pred, _ = predict_binary_model_standalone(rf, X_te_fs)

                        all_y_true.extend(y_te.tolist())
                        all_y_pred.extend(y_pred.tolist())

                    except Exception as e:
                        self.logger.warning(f"Error in fold {fold} for {score_col}: {str(e)}")
                        continue

                if all_y_true and all_y_pred:
                    # Create confusion matrix
                    cm = confusion_matrix(all_y_true, all_y_pred)

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Negative', 'Positive'],
                                yticklabels=['Negative', 'Positive'],
                                cbar_kws={'label': 'Count'})
                    plt.xlabel('Predicted Label', fontweight='bold')
                    plt.ylabel('True Label', fontweight='bold')
                    plt.title(f'Confusion Matrix - Best Model\n{score_col}', fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(self.dirs['confusion_matrices'] / f"{score_col}_best_model_confusion_matrix.png",
                                dpi=300, bbox_inches='tight')
                    plt.close()

                    # Save confusion matrix data
                    cm_df = pd.DataFrame(cm,
                                         index=['True_Negative', 'True_Positive'],
                                         columns=['Pred_Negative', 'Pred_Positive'])
                    cm_df.to_csv(self.dirs['confusion_matrices'] / f"{score_col}_confusion_matrix_data.csv")

            except Exception as e:
                self.logger.warning(f"Error generating confusion matrix for {score_col}: {str(e)}")

    def _create_performance_analysis(self, results: Dict[str, Dict[str, Any]],
                                     summary_data: List[Dict[str, Any]]):
        """Create detailed performance analysis"""
        try:
            analysis_data = []

            for score_col, result in results.items():
                # Get detailed fold results from the search
                if 'detailed_results' in result and result['detailed_results']:
                    best_iteration = max(result['detailed_results'],
                                         key=lambda x: x['aggregated_metrics'].get('f1_score', 0))

                    fold_details = best_iteration.get('fold_details', [])

                    for fold_detail in fold_details:
                        analysis_entry = {
                            'score_column': score_col,
                            **fold_detail
                        }
                        analysis_data.append(analysis_entry)

            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_csv(self.dirs['performance_analysis'] / "fold_level_performance_analysis.csv",
                                   index=False)

                # Create fold-level performance visualization
                plt.figure(figsize=(15, 10))
                n_cols = len(self.score_cols)
                n_rows = (n_cols + 1) // 2

                for i, score_col in enumerate(self.score_cols):
                    col_data = analysis_df[analysis_df['score_column'] == score_col]
                    if not col_data.empty:
                        plt.subplot(n_rows, 2, i + 1)
                        folds = col_data['fold']
                        f1_scores = col_data['f1_score']
                        plt.plot(folds, f1_scores, 'o-', label='F1-Score', linewidth=2, markersize=6)
                        plt.xlabel('Fold', fontweight='bold')
                        plt.ylabel('F1-Score', fontweight='bold')
                        plt.title(f'{score_col} - Fold Performance', fontweight='bold')
                        plt.grid(True, alpha=0.3)
                        plt.ylim(0, 1)

                plt.suptitle('Fold-Level Performance Analysis', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.dirs['performance_analysis'] / "fold_level_performance.png",
                            dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Error creating performance analysis: {str(e)}")

    def _generate_final_report(self, results: Dict[str, Dict[str, Any]],
                               summary_data: List[Dict[str, Any]]):
        """Generate comprehensive final report"""
        try:
            report_content = []
            report_content.append("# Binary Classification Hyperparameter Tuning Report")
            report_content.append("=" * 60)
            report_content.append("")

            # Session information
            report_content.append("## Session Information")
            report_content.append(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"- **Base Directory**: {self.base_dir}")
            report_content.append(f"- **Score Columns**: {', '.join(self.score_cols)}")
            report_content.append(f"- **Feature Selection**: SelectKBest with k=50")
            report_content.append(f"- **File Pattern**: fold[X]trb.csv / fold[X]teb.csv")
            report_content.append("")

            # Configuration
            report_content.append("## Tuning Configuration")
            for key, value in self.config.items():
                report_content.append(f"- **{key}**: {value}")
            report_content.append("")

            # Results summary
            report_content.append("## Results Summary")
            report_content.append("")

            for i, summary in enumerate(summary_data, 1):
                report_content.append(f"### {i}. {summary['score_column']}")
                report_content.append(f"- **F1-Score**: {summary['best_f1_score']:.4f}")
                report_content.append(f"- **Accuracy**: {summary['best_accuracy']:.4f}")
                report_content.append(f"- **Precision**: {summary['best_precision']:.4f}")
                report_content.append(f"- **Recall**: {summary['best_recall']:.4f}")
                report_content.append(f"- **ROC-AUC**: {summary['best_roc_auc']:.4f}")
                report_content.append(f"- **Search Time**: {summary['search_time']:.2f}s")
                report_content.append(f"- **Valid Results**: {summary['n_valid_results']}")
                report_content.append("")

            # Best parameters
            report_content.append("## Best Parameters by Score Column")
            report_content.append("")

            for score_col, result in results.items():
                report_content.append(f"### {score_col}")
                best_params = result['best_params']
                for param, value in best_params.items():
                    report_content.append(f"- **{param}**: {value}")
                report_content.append("")

            # File structure
            report_content.append("## Generated Files Structure")
            report_content.append("")
            for dir_name, dir_path in self.dirs.items():
                report_content.append(f"- **{dir_name}**: {dir_path.name}/")
                if dir_path.exists():
                    files = list(dir_path.glob("*"))
                    for file in files[:5]:  # Show first 5 files
                        report_content.append(f"  - {file.name}")
                    if len(files) > 5:
                        report_content.append(f"  - ... and {len(files) - 5} more files")
                report_content.append("")

            # Save report
            with open(self.dirs['summary_reports'] / "final_report.md", 'w') as f:
                f.write('\n'.join(report_content))

            # Also save as text
            with open(self.dirs['summary_reports'] / "final_report.txt", 'w') as f:
                f.write('\n'.join(report_content))

            self.logger.info("Final report generated successfully")

        except Exception as e:
            self.logger.warning(f"Error generating final report: {str(e)}")


# Configuration and usage functions
def create_binary_tuning_config(search_type='random', n_iter=50, primary_metric='f1_score',
                                maximize_metric=True, n_jobs=1, cv_folds=None):
    """Create a tuning configuration dictionary for binary classification"""
    return {
        'search_type': search_type,  # 'grid', 'random', or 'bayesian'
        'n_iter': n_iter,  # For random search
        'n_calls': n_iter,  # For bayesian optimization
        'primary_metric': primary_metric,  # Metric to optimize
        'maximize_metric': maximize_metric,  # True to maximize, False to minimize
        'n_jobs': n_jobs,  # Parallel jobs (-1 for all cores)
        'cv_folds': cv_folds  # Specific folds to use (None for all 10)
    }


# Example usage script
if __name__ == "__main__":
    # Configuration for binary classification
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = 'Inputfile'
    score_cols = ['A_WOB', 'A_GRB', 'A_COB', 'A_TTB']  # Binary score columns

    # Create tuning configuration
    tuning_config = create_binary_tuning_config(
        search_type='random',  # Use random search
        n_iter=30,  # 30 random combinations
        primary_metric='f1_score',  # Optimize F1 score for binary classification
        maximize_metric=True,  # Maximize F1
        n_jobs=1,  # Use 1 process to avoid parallel issues initially
        cv_folds=None  # Use all 10 folds
    )

    try:
        print("=" * 80)
        print("BINARY CLASSIFICATION HYPERPARAMETER TUNING FRAMEWORK")
        print("=" * 80)
        print(f"Score columns: {score_cols}")
        print(f"Feature selection: SelectKBest with k=50")
        print(f"File pattern: fold[X]trb.csv / fold[X]teb.csv")
        print("=" * 80)

        # Initialize tuner
        tuner = BinaryClassificationTuner(base_dir, name_col, score_cols, tuning_config)

        # Run comprehensive tuning
        results = tuner.run_comprehensive_binary_tuning()

        print("\n" + "=" * 80)
        print("BINARY CLASSIFICATION HYPERPARAMETER TUNING COMPLETED!")
        print("=" * 80)

        # Display best parameters
        for score_col, best_params in tuner.best_params.items():
            print(f"\nBest parameters for {score_col}:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")

        print(f"\nResults saved to: {tuner.session_dir}")
        print("\nGenerated directories:")
        for dir_name, dir_path in tuner.dirs.items():
            print(f"  {dir_name}: {dir_path}")

    except Exception as e:
        print(f"Binary classification tuning failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("BINARY FRAMEWORK FEATURES:")
    print("=" * 80)
    print("✓ Optimized for binary classification problems")
    print("✓ Fixed SelectKBest with k=50 feature selection")
    print("✓ Comprehensive binary metrics (F1, Precision, Recall, ROC-AUC, etc.)")
    print("✓ Organized directory structure with 10 categories")
    print("✓ Support for trb.csv/teb.csv file naming convention")
    print("✓ Binary-specific Random Forest parameters (class_weight, etc.)")
    print("✓ Detailed confusion matrices and performance analysis")
    print("✓ Multiple score function options for SelectKBest")
    print("✓ Comprehensive reporting and visualization")
    print("✓ JSON and CSV outputs for easy integration")
    print("✓ Fixed parallel processing implementation")
    print("=" * 80)