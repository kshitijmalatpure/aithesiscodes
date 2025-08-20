"""
fabOF_v8_4_0_hyperparameter_tuning_WITH_FS_ALT_OUTPUT.py
========================================================
(Updated) Also writes a copy-pasteable BEST_PARAMS snippet to:
  <base_dir>/comprehensive_fabOF_tuning/aggregated_results/best_params_snippet.txt

CHANGES (per user request):
- KBest-only feature selection (no LASSO anywhere)
- Keep Script 2's tuning structure and output layout
- No visualizations, confusion matrices, or feature analysis
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import logging
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, ParameterSampler

# Future-proof RMSE
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse
    def rmse(y_true, y_pred): return _mse(y_true, y_pred, squared=False)


# ---------------------- Search spaces (KBest-only) ----------------------
class HyperparameterSpace:
    @staticmethod
    def get_random_forest_space(search_type: str = 'grid') -> Dict[str, List]:
        if search_type == 'grid':
            return {
                'n_estimators': [200, 500, 800],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
                'bootstrap': [True],
                'random_state': [42],
            }
        # random
        return {
            'n_estimators': [100, 200, 300, 500, 800, 1000],
            'max_depth': [None, 8, 12, 16, 20, 30],
            'min_samples_split': [2, 3, 5, 7, 10, 15],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
            'bootstrap': [True],
            'random_state': [42],
        }

    @staticmethod
    def get_preprocessing_space(search_type: str = 'grid') -> Dict[str, List]:
        # Keep Script 2’s original behavior: only tune imputation here.
        return {
            'imputation_strategy': ['mean', 'median', 'most_frequent'],
            # Note: optional scaling_method can still be passed explicitly by the caller,
            # but we don't include it in the search space per original Script 2.
        }

    @staticmethod
    def get_feature_selection_space(search_type: str = 'grid') -> Dict[str, List]:
        """
        KBest-only feature selection.
        Use kbest_k == "all" (or None) to skip FS.
        """
        if search_type == 'grid':
            return {
                'feature_selection_method': ['kbest'],
                'kbest_k': [50, 100, 150, 200, "all"],
            }
        # random
        return {
            'feature_selection_method': ['kbest'],
            'kbest_k': [40, 60, 80, 100, 120, 150, 200, 250, "all"],
        }


# ---------------------- Tuner ----------------------
class FabOFHyperparameterTuner:
    def __init__(self, base_dir: str, name_col: str, score_cols: List[str],
                 labels: List[int], tuning_config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.name_col = name_col
        self.score_cols = score_cols
        self.labels = np.array(labels)
        self.config = tuning_config

        # Output layout (fixed)
        self.results_root = self.base_dir / "comprehensive_fabOF_tuning"
        self.logs_dir = self.results_root / "logs"
        self.agg_dir = self.results_root / "aggregated_results"
        for p in (self.results_root, self.logs_dir, self.agg_dir):
            p.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("FabOF Hyperparameter Tuning (KBest-only FS + nested output + TXT snippet)")
        self.logger.info("=" * 60)
        self.logger.info(f"Base dir: {self.base_dir}")
        self.logger.info(f"Results root: {self.results_root}")
        self.logger.info(f"Tuning config: {self.config}")

    def _setup_logging(self) -> logging.Logger:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"hyperparameter_tuning_{ts}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                      logging.StreamHandler()]
        )
        return logging.getLogger(__name__)

    # ---------- Data / preprocessing ----------
    def _load_fold(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tr_path = self.base_dir / f"fold_{fold}" / f"fold{fold}tr.csv"
        te_path = self.base_dir / f"fold_{fold}" / f"fold{fold}te.csv"
        if not tr_path.exists() or not te_path.exists():
            raise FileNotFoundError(f"Missing files for fold {fold}")
        tr = pd.read_csv(tr_path)
        te = pd.read_csv(te_path)
        tr, te = tr.align(te, axis=1, fill_value=np.nan)
        return tr, te

    def _prep(self, X_tr: np.ndarray, X_te: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        imputer = SimpleImputer(strategy=params.get('imputation_strategy', 'median'))
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        # Optional scaling support if provided explicitly (not tuned here by default)
        scaling = params.get('scaling_method', None)
        scaler = None
        if scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'robust':
            scaler = RobustScaler()
        elif scaling == 'minmax':
            scaler = MinMaxScaler()

        if scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        return X_tr, X_te

    def _feature_select(
        self,
        X_tr: np.ndarray,
        X_te: np.ndarray,
        y_tr: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        KBest-only FS. If kbest_k == "all"/None or invalid, skip FS.
        """
        method = str(params.get('feature_selection_method', 'kbest')).lower()
        if method != 'kbest':
            # Enforce KBest-only: ignore any other value silently
            method = 'kbest'

        k_param: Union[int, str, None] = params.get('kbest_k', "all")
        n_features = X_tr.shape[1]

        if isinstance(k_param, str):
            if k_param.lower() == 'all':
                return X_tr, X_te
            # If it's a string that's not 'all', attempt to parse integer
            try:
                k_param = int(k_param)
            except Exception:
                return X_tr, X_te

        if k_param is None:
            return X_tr, X_te

        try:
            k_int = int(k_param)
        except Exception:
            return X_tr, X_te

        if k_int <= 0 or k_int >= n_features:
            return X_tr, X_te

        selector = SelectKBest(score_func=f_regression, k=k_int)
        X_tr_fs = selector.fit_transform(X_tr, y_tr.astype(float))
        X_te_fs = selector.transform(X_te)
        return X_tr_fs, X_te_fs

    # ---------- FabOF core ----------
    def _fit_fabof(self, X_tr: np.ndarray, y_tr: np.ndarray, rf_params: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        clean = {k: v for k, v in rf_params.items()
                 if k in ['n_estimators', 'max_depth', 'min_samples_split',
                          'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']}
        rf = RandomForestRegressor(oob_score=True, **clean)
        rf.fit(X_tr, y_tr.astype(float))
        kmax = int(np.max(y_tr))
        oob = rf.oob_prediction_
        pi = np.array([(y_tr <= c).mean() for c in range(1, kmax)])
        borders_inner = np.quantile(oob, pi) if len(pi) else np.array([])
        borders = np.concatenate([[1.0], borders_inner, [float(kmax)]])
        return rf, borders

    def _predict_fabof(self, rf: Any, borders: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_raw = rf.predict(X_te)
        y_ord = np.searchsorted(borders[1:-1], y_raw) + 1
        return y_ord, y_raw

    # ---------- Evaluation ----------
    def _evaluate_params(self, params: Dict[str, Any], score_col: str,
                         folds: Optional[List[int]] = None) -> Dict[str, float]:
        if folds is None:
            folds = list(range(1, 11))
        fold_metrics: List[Dict[str, float]] = []

        for fold in folds:
            try:
                tr, te = self._load_fold(fold)
                feats = [c for c in tr.columns if c not in [self.name_col] + self.score_cols]
                X_tr, X_te = tr[feats].values, te[feats].values
                y_tr, y_te = tr[score_col].values, te[score_col].values

                X_tr, X_te = self._prep(X_tr, X_te, params)
                X_tr, X_te = self._feature_select(X_tr, X_te, y_tr, params)

                rf, borders = self._fit_fabof(X_tr, y_tr, params)
                y_pred, y_raw = self._predict_fabof(rf, borders, X_te)

                metrics = {
                    'accuracy': accuracy_score(y_te, y_pred),
                    'f1_macro': f1_score(y_te, y_pred, labels=self.labels, average='macro', zero_division=0),
                    'precision_macro': precision_score(y_te, y_pred, labels=self.labels, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_te, y_pred, labels=self.labels, average='macro', zero_division=0),
                    'qwk': cohen_kappa_score(y_te, y_pred, labels=self.labels, weights='quadratic'),
                    'rmse': rmse(y_te, y_raw),
                    'mae': float(np.mean(np.abs(y_te - y_raw))),
                }
                fold_metrics.append(metrics)

            except Exception as e:
                self.logger.warning(f"[{score_col}] Fold {fold} failed: {e}")
                continue

        if not fold_metrics:
            return {'accuracy': 0, 'f1_macro': 0, 'precision_macro': 0, 'recall_macro': 0,
                    'qwk': 0, 'rmse': 999, 'mae': 999}

        agg: Dict[str, float] = {}
        for m in fold_metrics[0].keys():
            vals = [fm[m] for fm in fold_metrics]
            agg[m] = float(np.mean(vals))
            agg[m + "_std"] = float(np.std(vals))
        return agg

    # ---------- Search runners ----------
    def _run_grid(self, score_col: str, param_grid: Dict[str, List],
                  cv_folds: Optional[List[int]]) -> Tuple[List[Dict[str, Any]], float]:
        grid = list(ParameterGrid(param_grid))
        results: List[Dict[str, Any]] = []
        start = time.time()
        for i, p in enumerate(grid, 1):
            self.logger.info(f"[Grid] {score_col} - {i}/{len(grid)}")
            metrics = self._evaluate_params(p, score_col, cv_folds)
            results.append({**p, **metrics})
        return results, time.time() - start

    def _run_random(self, score_col: str, param_distributions: Dict[str, List],
                    n_iter: int, cv_folds: Optional[List[int]]) -> Tuple[List[Dict[str, Any]], float]:
        sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))
        results: List[Dict[str, Any]] = []
        start = time.time()
        for i, p in enumerate(sampler, 1):
            self.logger.info(f"[Random] {score_col} - {i}/{len(sampler)}")
            metrics = self._evaluate_params(p, score_col, cv_folds)
            results.append({**p, **metrics})
        return results, time.time() - start

    # ---------- Public API ----------
    def run_comprehensive_tuning(self) -> Dict[str, Dict[str, Any]]:
        self.logger.info("Starting comprehensive tuning...")
        search_type = self.config.get('search_type', 'random')  # 'grid' | 'random'
        cv_folds = self.config.get('cv_folds', None)
        primary_metric = self.config.get('primary_metric', 'f1_macro')
        maximize = self.config.get('maximize_metric', True)
        n_iter = int(self.config.get('n_iter', 50))

        all_results: Dict[str, Dict[str, Any]] = {}
        aggregated_rows: List[Dict[str, Any]] = []
        best_rows: List[Dict[str, Any]] = []

        best_params_flat: Dict[str, Dict[str, Any]] = {}

        for score_col in self.score_cols:
            self.logger.info(f"{'='*40}\nTuning: {score_col}\n{'='*40}")

            rf_space = HyperparameterSpace.get_random_forest_space(search_type)
            prep_space = HyperparameterSpace.get_preprocessing_space(search_type)
            fs_space = HyperparameterSpace.get_feature_selection_space(search_type)
            combined = {**rf_space, **prep_space, **fs_space}

            if search_type == 'grid':
                results, elapsed = self._run_grid(score_col, combined, cv_folds)
            else:
                results, elapsed = self._run_random(score_col, combined, n_iter, cv_folds)

            if not results:
                self.logger.warning(f"No results for {score_col}")
                continue

            keyfunc = (lambda x: x[primary_metric]) if maximize else (lambda x: -x[primary_metric])
            best = max(results, key=keyfunc) if maximize else min(results, key=keyfunc)

            # Store nested result
            best_params = {
                k: v for k, v in best.items()
                if k not in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                             'qwk', 'rmse', 'mae']
                and not k.endswith('_std')
            }

            all_results[score_col] = {
                'best_params': best_params,
                'best_score': float(best[primary_metric]),
                'search_time': float(elapsed),
            }

            # Build flat dict for the TXT snippet (hard-coded runner)
            out_entry: Dict[str, Any] = {}

            # include keys when present in best_params
            for key in ['random_state','n_estimators','min_samples_split','min_samples_leaf',
                        'max_features','max_depth','imputation_strategy','bootstrap','scaling_method']:
                if key in best_params:
                    out_entry[key] = best_params[key]

            # FS method + KBest knob (KBest-only)
            out_entry['feature_selection_method'] = 'kbest'
            if 'kbest_k' in best_params:
                out_entry['kbest_k'] = best_params['kbest_k']

            best_params_flat[score_col] = out_entry

            # Aggregated rows and CSV row
            for r in results:
                aggregated_rows.append({'score_column': score_col, **r})
            best_rows.append({
                'score_column': score_col,
                'best_metric': primary_metric,
                'best_score': float(best[primary_metric]),
                **best_params
            })

        # ----- Write outputs -----
        if aggregated_rows:
            pd.DataFrame(aggregated_rows).to_csv(self.agg_dir / "aggregated_tuning_results.csv", index=False)
        if best_rows:
            pd.DataFrame(best_rows).to_csv(self.agg_dir / "best_configurations.csv", index=False)
        with open(self.results_root / "best_model_config.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # ----- Write BEST_PARAMS snippet (.txt) -----
        snippet_path = self.agg_dir / "best_params_snippet.txt"
        with open(snippet_path, "w", encoding="utf-8") as f:
            f.write("# Paste this into your runner script, replacing the BEST_PARAMS placeholder.\n")
            f.write("BEST_PARAMS = {\n")
            for trait, params in best_params_flat.items():
                f.write(f"    '{trait}': {{\n")
                # deterministic key order for readability
                ordered_keys = [
                    'random_state','n_estimators','min_samples_split','min_samples_leaf',
                    'max_features','max_depth','imputation_strategy','scaling_method',
                    'feature_selection_method','kbest_k','bootstrap'
                ]
                for k in ordered_keys:
                    if k in params:
                        v = params[k]
                        if isinstance(v, str):
                            f.write(f"        '{k}': '{v}',\n")
                        elif v is None:
                            f.write(f"        '{k}': None,\n")
                        else:
                            f.write(f"        '{k}': {v},\n")
                f.write("    },\n")
            f.write("}\n")
        self.logger.info(f"Wrote BEST_PARAMS snippet to: {snippet_path}")

        self.logger.info("Tuning complete.")
        self.logger.info(f"Saved: {self.agg_dir / 'aggregated_tuning_results.csv'}")
        self.logger.info(f"Saved: {self.agg_dir / 'best_configurations.csv'}")
        self.logger.info(f"Saved: {self.results_root / 'best_model_config.json'}")

        return all_results


# ---------------------- Example usage ----------------------
def create_tuning_config(search_type='random', n_iter=30, primary_metric='f1_macro',
                        maximize_metric=True, cv_folds=None):
    return {
        'search_type': search_type,
        'n_iter': n_iter,
        'primary_metric': primary_metric,
        'maximize_metric': maximize_metric,
        'cv_folds': cv_folds
    }


if __name__ == "__main__":
    base_dir = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"
    name_col = 'Inputfile'
    score_cols = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
    labels = [1, 2, 3, 4]

    config = create_tuning_config(
        search_type='random',
        n_iter=30,
        primary_metric='f1_macro',
        maximize_metric=True,
        cv_folds=None
    )

    print("="*60)
    print("FABOF HYPERPARAMETER TUNING (KBest-only FS + nested output + TXT snippet)")
    print("="*60)

    try:
        tuner = FabOFHyperparameterTuner(base_dir, name_col, score_cols, labels, config)
        tuner.run_comprehensive_tuning()
        print("\nSaved outputs:")
        print(f" - {tuner.agg_dir / 'aggregated_tuning_results.csv'}")
        print(f" - {tuner.agg_dir / 'best_configurations.csv'}")
        print(f" - {tuner.results_root / 'best_model_config.json'}")
        print(f" - {tuner.agg_dir / 'best_params_snippet.txt'}  <-- copy/paste into runner")
    except Exception as e:
        print(f"Tuning failed: {e}")
        import traceback
        traceback.print_exc()
