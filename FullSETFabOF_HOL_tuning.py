"""
ahol_hyperparameter_tuning_lasso.py
===================================
Comprehensive hyperparameter tuning framework for A_HOL (ordinal, float labels)
- LASSO-based feature selection only (with CV)
- RandomForestRegressor trained in FABOF style
- Works with 10-fold files: fold{n}tr.csv / fold{n}te.csv
- Metrics: QWK, Accuracy, F1/Precision/Recall (macro) via index-encoding,
           plus RMSE/MAE/R2 on the raw continuous predictions
- Organized outputs (logs, raw/processed results, visualizations, reports)
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from typing import Dict, List, Tuple, Any, Optional
import multiprocessing as mp

# Scikit-learn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, ParameterSampler

warnings.filterwarnings("ignore")

# ---------- future-proof RMSE ----------
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:
    from sklearn.metrics import mean_squared_error as mse
    def rmse(y_true, y_pred): return mse(y_true, y_pred, squared=False)


# =========================
# Label utilities (ordinal)
# =========================
def encode_to_index(y_values, classes, tol=1e-6):
    y_values = np.asarray(y_values, dtype=float)
    classes = np.asarray(classes, dtype=float)
    idxs = np.empty_like(y_values, dtype=int)
    for i, v in enumerate(y_values):
        j = int(np.argmin(np.abs(classes - v)))
        idxs[i] = j
    return idxs


# ======================================
# FABOF-style model fit & predict (RF)
# ======================================
def fit_fabof_rf(X_tr, y_tr, rf_params, random_state=42):
    """
    Fit RandomForestRegressor and derive ordinal borders from OOB predictions.
    Returns: rf, borders (quantile cutpoints), feature_importances_, classes_sorted
    """
    y_tr = y_tr.astype(float)
    rf_params = dict(rf_params) if rf_params else {}
    rf_params.setdefault('n_estimators', 500)
    rf_params.setdefault('oob_score', True)
    rf_params.setdefault('bootstrap', True)
    rf_params.setdefault('random_state', random_state)
    rf = RandomForestRegressor(**rf_params).fit(X_tr, y_tr)

    # Ordinal borders computed from cumulative proportions up to each class (except last)
    oob_pred = rf.oob_prediction_
    classes_sorted = np.unique(y_tr)
    inner_classes = classes_sorted[:-1]
    pi = np.array([(y_tr <= c).mean() for c in inner_classes])
    borders_inner = np.quantile(oob_pred, pi)
    borders = np.concatenate([[classes_sorted[0]], borders_inner, [classes_sorted[-1]]])
    return rf, borders, rf.feature_importances_, classes_sorted


def predict_fabof_rf(rf, borders, classes_sorted, X_te):
    y_num = rf.predict(X_te)
    idx = np.searchsorted(borders[1:-1], y_num)
    y_ord = classes_sorted[idx]
    return y_ord, y_num


# ======================================
# Metrics for ordinal (indices + numeric)
# ======================================
def ordinal_metrics(y_true_numeric, y_pred_numeric, y_true_idx, y_pred_idx, labels_idx):
    out = {
        'accuracy': accuracy_score(y_true_idx, y_pred_idx),
        'rmse': rmse(y_true_numeric, y_pred_numeric),
        'mae': float(np.mean(np.abs(y_true_numeric - y_pred_numeric))),
        'r2': float(1 - np.sum((y_true_numeric - y_pred_numeric) ** 2) /
                    np.sum((y_true_numeric - np.mean(y_true_numeric)) ** 2))
    }
    out['f1_macro'] = f1_score(y_true_idx, y_pred_idx, labels=labels_idx, average='macro', zero_division=0)
    out['precision_macro'] = precision_score(y_true_idx, y_pred_idx, labels=labels_idx, average='macro', zero_division=0)
    out['recall_macro'] = recall_score(y_true_idx, y_pred_idx, labels=labels_idx, average='macro', zero_division=0)
    out['qwk'] = cohen_kappa_score(y_true_idx, y_pred_idx, labels=labels_idx, weights='quadratic')
    return {k: float(v) for k, v in out.items()}


# ======================================
# Preprocess + LASSO feature selection
# ======================================
def _get_scaler(name: str):
    if name == 'standard':
        return StandardScaler()
    elif name == 'robust':
        return RobustScaler()
    elif name == 'minmax':
        return MinMaxScaler()
    else:
        return StandardScaler()  # sensible default for LASSO


def preprocess_impute(X_tr, X_te, imputation_strategy='median'):
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)
    return X_tr_imp, X_te_imp


def lasso_feature_selection(X_tr_imp, X_te_imp, y_tr, params):
    """
    Returns: X_tr_fs, X_te_fs, mask(bool), n_selected
    LASSO CV is used to pick coefficients; threshold decides selection.
    If none selected, fall back to top-N by |coef|.
    """
    scaler_name = params.get('lasso_scaler', 'standard')
    coef_thresh = float(params.get('lasso_coef_threshold', 1e-5))
    lasso_cv = int(params.get('lasso_cv', 5))
    lasso_max_iter = int(params.get('lasso_max_iter', 4000))
    min_features = int(params.get('lasso_min_features', 50))

    scaler = _get_scaler(scaler_name)
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_te_s = scaler.transform(X_te_imp)

    lasso = LassoCV(cv=lasso_cv, max_iter=lasso_max_iter, random_state=42).fit(X_tr_s, y_tr.astype(float))
    coefs = np.abs(lasso.coef_)
    mask = coefs > coef_thresh

    if mask.sum() == 0:
        # fallback: take top-N by |coef|
        top = min(min_features, X_tr_s.shape[1])
        top_idx = np.argsort(coefs)[-top:]
        mask = np.zeros_like(coefs, dtype=bool)
        mask[top_idx] = True

    X_tr_fs = X_tr_s[:, mask]
    X_te_fs = X_te_s[:, mask]
    return X_tr_fs, X_te_fs, mask, int(mask.sum())


# ======================================
# Search spaces (LASSO + RF + Preproc)
# ======================================
class AHOLHyperparameterSpace:
    @staticmethod
    def rf_space(search_type='random'):
        if search_type == 'grid':
            return {
                'n_estimators': [200, 500, 800],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
                'bootstrap': [True],
                'random_state': [42]
            }
        elif search_type == 'random':
            return {
                'n_estimators': [100, 200, 300, 500, 800, 1000],
                'max_depth': [None, 8, 12, 16, 20, 30],
                'min_samples_split': [2, 3, 5, 7, 10, 15],
                'min_samples_leaf': [1, 2, 3, 4, 5],
                'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
                'bootstrap': [True],
                'random_state': [42]
            }

    @staticmethod
    def lasso_fs_space(search_type='random'):
        # LASSO-only FS knobs
        if search_type in ['grid', 'random']:
            return {
                'imputation_strategy': ['mean', 'median', 'most_frequent'],
                'lasso_scaler': ['standard', 'robust', 'minmax'],
                'lasso_cv': [3, 5, 10],
                'lasso_max_iter': [2000, 4000, 8000],
                'lasso_coef_threshold': [1e-6, 1e-5, 1e-4],
                'lasso_min_features': [25, 50, 100]
            }


# ======================================
# Standalone evaluator (for parallel)
# ======================================
def evaluate_single_param_set(args):
    """
    params, score_col, base_dir, name_col, labels, folds_to_use
    Returns aggregated metrics (mean/std) + n_valid_folds + fold_details
    """
    params, score_col, base_dir, name_col, labels, folds_to_use = args
    base_dir = Path(base_dir)
    labels = np.array(sorted(map(float, labels)))
    label_indices = np.arange(len(labels))

    fold_results = []
    fold_details = []

    for fold in folds_to_use:
        try:
            tr_path = base_dir / f"fold_{fold}" / f"fold{fold}trh.csv"
            te_path = base_dir / f"fold_{fold}" / f"fold{fold}teh.csv"
            if not tr_path.exists() or not te_path.exists():
                continue

            tr = pd.read_csv(tr_path)
            te = pd.read_csv(te_path)
            tr, te = tr.align(te, axis=1, fill_value=np.nan)

            feats = [c for c in tr.columns if c not in [name_col, score_col]]
            X_tr, X_te = tr[feats].values, te[feats].values
            y_tr, y_te = tr[score_col].values.astype(float), te[score_col].values.astype(float)

            # Preprocess → LASSO FS
            X_tr_imp, X_te_imp = preprocess_impute(X_tr, X_te, params.get('imputation_strategy', 'median'))
            X_tr_fs, X_te_fs, mask, n_selected = lasso_feature_selection(X_tr_imp, X_te_imp, y_tr, params)

            if X_tr_fs.shape[1] == 0:
                # skip impossible case
                continue

            # Fit RF (FABOF)
            rf_params = {k: v for k, v in params.items()
                         if k in ['n_estimators', 'max_depth', 'min_samples_split',
                                  'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']}
            rf, borders, _, classes_sorted = fit_fabof_rf(X_tr_fs, y_tr, rf_params)

            # Predict
            y_pred_rounded, y_pred_raw = predict_fabof_rf(rf, borders, classes_sorted, X_te_fs)

            # Metrics (indices for classification metrics)
            y_te_idx = encode_to_index(y_te, labels)
            y_pred_idx = encode_to_index(y_pred_rounded, labels)
            metrics = ordinal_metrics(y_te, y_pred_raw, y_te_idx, y_pred_idx, label_indices)

            fold_results.append(metrics)

            fold_details.append({
                'fold': fold,
                'train_samples': int(len(y_tr)),
                'test_samples': int(len(y_te)),
                'selected_features': int(n_selected),
                **{k: float(v) for k, v in metrics.items()}
            })
        except Exception:
            continue

    if not fold_results:
        zero = {k: 0.0 for k in ['accuracy', 'rmse', 'mae', 'r2', 'f1_macro', 'precision_macro', 'recall_macro', 'qwk']}
        zero.update({'n_valid_folds': 0, 'fold_details': []})
        return zero

    aggregated = {}
    for metric in fold_results[0].keys():
        vals = [fr[metric] for fr in fold_results]
        aggregated[metric] = float(np.mean(vals))
        aggregated[f"{metric}_std"] = float(np.std(vals))
    aggregated['n_valid_folds'] = len(fold_results)
    aggregated['fold_details'] = fold_details
    return aggregated


# ======================================
# Tuner
# ======================================
class AHOLHyperparameterTuner:
    def __init__(self, base_dir: str, name_col: str, score_col: str,
                 labels: List[float], tuning_config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_col = score_col
        self.labels = np.array(sorted(map(float, labels)))
        self.config = tuning_config

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_root = self.base_dir / "ahol_hyperparameter_tuning_results"
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.results_root / f"session_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

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
        for p in self.dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()
        self.best_params = {}
        self._save_session_config()

        self.logger.info("=" * 80)
        self.logger.info("A_HOL HYPERPARAMETER TUNING (LASSO FS ONLY)")
        self.logger.info("=" * 80)
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Session directory: {self.session_dir}")
        self.logger.info(f"Score column: {self.score_col}")
        self.logger.info(f"Labels (ordered): {self.labels.tolist()}")
        self.logger.info(f"File pattern: fold[X]tr.csv / fold[X]te.csv")
        self.logger.info(f"Tuning configuration: {self.config}")
        self.logger.info("=" * 80)

    def _setup_logging(self) -> logging.Logger:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.dirs['logs'] / f"ahol_tuning_{timestamp}.log"

        formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(funcName)-22s - %(message)s')
        fh = logging.FileHandler(log_file, encoding='utf-8'); fh.setLevel(logging.DEBUG); fh.setFormatter(formatter)
        ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger = logging.getLogger(f"{__name__}_{id(self)}")
        logger.setLevel(logging.DEBUG); logger.handlers.clear()
        logger.addHandler(fh); logger.addHandler(ch)
        return logger

    def _save_session_config(self):
        cfg = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'base_directory': str(self.base_dir),
                'name_column': self.name_col,
                'score_column': self.score_col,
                'file_pattern': 'fold[X]tr.csv / fold[X]te.csv'
            },
            'labels': self.labels.tolist(),
            'tuning_config': self.config,
            'feature_selection': {
                'method': 'LASSO',
                'params': ['lasso_scaler', 'lasso_cv', 'lasso_max_iter', 'lasso_coef_threshold', 'lasso_min_features']
            },
            'directory_structure': {k: str(v) for k, v in self.dirs.items()}
        }
        with open(self.dirs['configs'] / 'session_config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

    # ---------------------------
    # Random search (primary mode)
    # ---------------------------
    def random_search(self, n_iter: int = 40, n_jobs: int = 1, cv_folds: Optional[List[int]] = None):
        self.logger.info(f"Starting Random Search ({n_iter} iterations) with LASSO FS")
        if cv_folds is None:
            cv_folds = list(range(1, 10 + 1))

        rf_space = AHOLHyperparameterSpace.rf_space('random')
        fs_space = AHOLHyperparameterSpace.lasso_fs_space('random')
        param_distributions = {**rf_space, **fs_space}

        sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)
        results = []
        detailed = []
        start = time.time()

        if n_jobs == 1:
            for i, params in enumerate(sampler, 1):
                self.logger.info(f"Evaluating set {i}/{n_iter}")
                eval_res = self._evaluate_params_seq(params, cv_folds)
                folds = eval_res.pop('fold_details', [])
                results.append({**params, **eval_res})
                detailed.append({'iteration': i, 'params': params, 'aggregated_metrics': eval_res, 'fold_details': folds})
                pm = self.config.get('primary_metric', 'qwk')
                self.logger.info(f"  {pm}: {eval_res.get(pm, 0):.4f} (±{eval_res.get(pm + '_std', 0):.4f})")
        else:
            self.logger.info(f"Using {min(n_jobs, mp.cpu_count())} parallel processes")
            param_args = [(params, self.score_col, str(self.base_dir), self.name_col, self.labels.tolist(), cv_folds)
                          for params in sampler]
            with ProcessPoolExecutor(max_workers=min(n_jobs, mp.cpu_count())) as ex:
                fut2params = {ex.submit(evaluate_single_param_set, a): a[0] for a in param_args}
                for i, fut in enumerate(as_completed(fut2params), 1):
                    try:
                        params = fut2params[fut]
                        eval_res = fut.result()
                        folds = eval_res.pop('fold_details', [])
                        results.append({**params, **eval_res})
                        detailed.append({'iteration': i, 'params': params, 'aggregated_metrics': eval_res, 'fold_details': folds})
                        pm = self.config.get('primary_metric', 'qwk')
                        self.logger.info(f"Completed {i}/{n_iter} - {pm}: {eval_res.get(pm, 0):.4f}")
                    except Exception as e:
                        self.logger.warning(f"Parallel eval error: {e}")

        # pick best
        pm = self.config.get('primary_metric', 'qwk')
        maximize = bool(self.config.get('maximize_metric', True))
        valid = [r for r in results if r.get('n_valid_folds', 0) > 0]
        if not valid:
            raise ValueError("No valid results during search")

        best = max(valid, key=lambda x: x.get(pm, -np.inf)) if maximize else min(valid, key=lambda x: x.get(pm, np.inf))
        elapsed = time.time() - start
        self.logger.info(f"Random Search done in {elapsed:.2f}s — Best {pm}: {best.get(pm, 0):.4f}")

        # save
        pd.DataFrame(results).to_csv(self.dirs['raw_results'] / f"{self.score_col}_random_search_results.csv", index=False)
        with open(self.dirs['raw_results'] / f"{self.score_col}_detailed_results.json", 'w') as f:
            json.dump(detailed, f, indent=2)

        best_params = {k: v for k, v in best.items()
                       if k not in ['accuracy', 'rmse', 'mae', 'r2', 'f1_macro', 'precision_macro',
                                    'recall_macro', 'qwk', 'n_valid_folds']
                       and not k.endswith('_std')}

        return {
            'best_params': best_params,
            'best_score': best.get(pm, 0.0),
            'best_metrics': {k: best[k] for k in ['accuracy', 'rmse', 'mae', 'r2', 'f1_macro', 'precision_macro', 'recall_macro', 'qwk'] if k in best},
            'all_results': results,
            'detailed_results': detailed,
            'search_time': elapsed,
            'n_valid_results': len(valid)
        }

    def _evaluate_params_seq(self, params: Dict[str, Any], folds_to_use: List[int]) -> Dict[str, float]:
        args = (params, self.score_col, str(self.base_dir), self.name_col, self.labels.tolist(), folds_to_use)
        return evaluate_single_param_set(args)

    # -------------- Orchestration --------------
    def run(self):
        self.logger.info("Running comprehensive tuning for A_HOL (LASSO FS)")
        res = self.random_search(
            n_iter=int(self.config.get('n_iter', 40)),
            n_jobs=int(self.config.get('n_jobs', 1)),
            cv_folds=self.config.get('cv_folds', None)
        )
        self.best_params[self.score_col] = res['best_params']
        self._save_all_outputs({self.score_col: res})
        return res

    # -------------- Save reports/plots --------------
    def _save_all_outputs(self, results: Dict[str, Dict[str, Any]]):
        self.logger.info("Saving results, visualizations, and reports...")
        try:
            # best params summary
            best_df = pd.DataFrame([{'Score_Column': self.score_col, **results[self.score_col]['best_params']}])
            best_df.to_csv(self.dirs['summary_reports'] / "best_parameters_summary.csv", index=False)

            # performance summary
            perf = results[self.score_col]['best_metrics']
            summary_df = pd.DataFrame([{
                'score_column': self.score_col,
                **{f'best_{k}': v for k, v in perf.items()},
                'search_time': results[self.score_col]['search_time'],
                'n_valid_results': results[self.score_col]['n_valid_results']
            }])
            summary_df.to_csv(self.dirs['summary_reports'] / "performance_summary.csv", index=False)

            # compact JSON
            with open(self.dirs['summary_reports'] / "comprehensive_results.json", 'w') as f:
                json.dump({
                    self.score_col: {
                        'best_params': results[self.score_col]['best_params'],
                        'best_metrics': results[self.score_col]['best_metrics'],
                        'search_time': float(results[self.score_col]['search_time']),
                        'n_valid_results': int(results[self.score_col]['n_valid_results'])
                    }
                }, f, indent=2)

            # Confusion matrix for best model (aggregate across folds)
            self._generate_best_cm(results[self.score_col]['best_params'])

            # Parameter correlation quick-look (where numeric)
            self._parameter_heatmap(results[self.score_col])

            # Final markdown report
            self._final_report(results[self.score_col])

            self.logger.info(f"Saved all artifacts to: {self.session_dir}")
        except Exception as e:
            self.logger.warning(f"Saving outputs failed: {e}")

    def _generate_best_cm(self, best_params: Dict[str, Any]):
        self.logger.info("Generating aggregate confusion matrix for best params...")
        labels = self.labels
        label_indices = np.arange(len(labels))

        all_true_idx, all_pred_idx = [], []
        for fold in range(1, 11):
            try:
                tr_path = self.base_dir / f"fold_{fold}" / f"fold{fold}tr.csv"
                te_path = self.base_dir / f"fold_{fold}" / f"fold{fold}te.csv"
                if not tr_path.exists() or not te_path.exists():
                    continue
                tr = pd.read_csv(tr_path); te = pd.read_csv(te_path)
                tr, te = tr.align(te, axis=1, fill_value=np.nan)
                feats = [c for c in tr.columns if c not in [self.name_col, self.score_col]]
                X_tr, X_te = tr[feats].values, te[feats].values
                y_tr, y_te = tr[self.score_col].values.astype(float), te[self.score_col].values.astype(float)

                X_tr_imp, X_te_imp = preprocess_impute(X_tr, X_te, best_params.get('imputation_strategy', 'median'))
                X_tr_fs, X_te_fs, _, _ = lasso_feature_selection(X_tr_imp, X_te_imp, y_tr, best_params)

                rf_params = {k: v for k, v in best_params.items()
                             if k in ['n_estimators', 'max_depth', 'min_samples_split',
                                      'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']}
                rf, borders, _, classes_sorted = fit_fabof_rf(X_tr_fs, y_tr, rf_params)
                y_pred_rounded, _ = predict_fabof_rf(rf, borders, classes_sorted, X_te_fs)

                all_true_idx.extend(encode_to_index(y_te, labels).tolist())
                all_pred_idx.extend(encode_to_index(y_pred_rounded, labels).tolist())
            except Exception:
                continue

        if not all_true_idx:
            self.logger.warning("No predictions aggregated for CM.")
            return

        cm = confusion_matrix(all_true_idx, all_pred_idx, labels=label_indices)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'}, square=True)
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('True', fontweight='bold')
        plt.title('A_HOL - Best Params Aggregate Confusion Matrix', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.dirs['confusion_matrices'] / f"{self.score_col}_best_cm.png", dpi=300, bbox_inches='tight')
        plt.close()

        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            self.dirs['confusion_matrices'] / f"{self.score_col}_best_cm_data.csv"
        )

    def _parameter_heatmap(self, result_block: Dict[str, Any]):
        try:
            df = pd.DataFrame(result_block['all_results'])
            # Keep only numeric params (RF + lasso knobs) for correlation with qwk
            metric = 'qwk'
            num = df.select_dtypes(include=[np.number])
            if metric in num.columns and num.shape[1] > 2:
                corr = num.corr()[metric].sort_values(ascending=False)
                corr.to_csv(self.dirs['feature_analysis'] / "param_vs_qwk_correlation.csv")
                plt.figure(figsize=(8, max(6, 0.4 * len(corr))))
                corr.drop(metric, errors='ignore').plot(kind='barh')
                plt.gca().invert_yaxis()
                plt.xlabel('Correlation with QWK')
                plt.title('Parameter correlation with QWK')
                plt.tight_layout()
                plt.savefig(self.dirs['visualizations'] / "param_qwk_correlation.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            self.logger.warning(f"Parameter heatmap skipped: {e}")

    def _final_report(self, result_block: Dict[str, Any]):
        try:
            lines = []
            lines.append("# A_HOL Hyperparameter Tuning (LASSO FS)")
            lines.append("")
            lines.append(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Base Directory**: {self.base_dir}")
            lines.append(f"- **Score Column**: {self.score_col}")
            lines.append(f"- **Labels**: {self.labels.tolist()}")
            lines.append(f"- **Primary Metric**: {self.config.get('primary_metric', 'qwk')}")
            lines.append("")
            lines.append("## Best Metrics")
            for k, v in result_block['best_metrics'].items():
                lines.append(f"- **{k}**: {v:.4f}")
            lines.append("")
            lines.append("## Best Parameters")
            for k, v in result_block['best_params'].items():
                lines.append(f"- **{k}**: {v}")
            with open(self.dirs['summary_reports'] / "final_report.md", 'w') as f:
                f.write('\n'.join(lines))
        except Exception as e:
            self.logger.warning(f"Final report generation failed: {e}")


# =========================
# Config helper
# =========================
def create_ahol_tuning_config(search_type='random', n_iter=40, primary_metric='qwk',
                              maximize_metric=True, n_jobs=1, cv_folds=None):
    return {
        'search_type': search_type,
        'n_iter': n_iter,
        'primary_metric': primary_metric,  # 'qwk' recommended for ordinal
        'maximize_metric': maximize_metric,
        'n_jobs': n_jobs,
        'cv_folds': cv_folds
    }


# =========================
# Main
# =========================
if __name__ == "__main__":
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = 'Inputfile'
    score_col = 'A_HOL'
    A_HOL_scores = [
        81.25, 68.75, 75, 56.25, 50, 62.5, 43.75, 87.5, 100, 31.25, 93.75, 37.5, 25
    ]

    tuning_config = create_ahol_tuning_config(
        search_type='random',
        n_iter=40,              # number of random combos
        primary_metric='qwk',   # optimize Quadratic Weighted Kappa
        maximize_metric=True,
        n_jobs=1,               # start with 1 to keep logs tidy; increase if needed
        cv_folds=None           # use all 10 folds
    )

    try:
        print("=" * 80)
        print("A_HOL HYPERPARAMETER TUNING (LASSO FEATURE SELECTION)")
        print("=" * 80)
        tuner = AHOLHyperparameterTuner(base_dir, name_col, score_col, A_HOL_scores, tuning_config)
        results = tuner.run()

        print("\n" + "=" * 80)
        print("A_HOL TUNING COMPLETED!")
        print("=" * 80)
        print("Best parameters:")
        for k, v in tuner.best_params[score_col].items():
            print(f"  {k}: {v}")

        print(f"\nSession saved to: {tuner.session_dir}")
        print("Artifacts:")
        for k, v in tuner.dirs.items():
            print(f"  - {k}: {v}")

    except Exception as e:
        print(f"Tuning failed: {e}")
        import traceback
        traceback.print_exc()
