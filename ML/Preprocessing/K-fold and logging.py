import os
import pandas as pd
import numpy as np
import shutil
import logging
from typing import List, Tuple
# This is the required library for multi-label stratification.
# Ensure it is installed by running: pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# --- 1. SCRIPT CONFIGURATION ---

# --- Paths ---
DATA_DIR = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Missing files"
METADATA_PATH = r"C:\Research\AI Folder\Thesis\Data\Data\allfeats.csv"
OUTPUT_DIR = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final"

# --- Cross-Validation Settings ---
N_SPLITS = 10
RANDOM_STATE = 42
STRATIFY_COLUMNS = ['A_WO', 'A_GR', 'A_CO', 'A_TT']

# --- Variant and Column Definitions ---
FILENAME_COL = 'Inputfile'
VARS_SET_1 = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
VARS_SET_2 = ['A_WOB', 'A_GRB', 'A_COB', 'A_TTB']
VARS_SET_3 = ['A_HOL']

VERSION_DEFINITIONS = [
    {'name': 'Set1_Multiclass', 'vars': VARS_SET_1, 'train_suffix': 'tr.csv', 'test_suffix': 'te.csv'},
    {'name': 'Set2_Binary', 'vars': VARS_SET_2, 'train_suffix': 'trb.csv', 'test_suffix': 'teb.csv'},
    {'name': 'Set3_Holistic', 'vars': VARS_SET_3, 'train_suffix': 'trh.csv', 'test_suffix': 'teh.csv'}
]


# --- END OF CONFIGURATION ---


def setup_logging(log_file: str = None) -> None:
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(level=logging.INFO, format=log_format,
                            handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                                      logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)


def validate_inputs(data_dir: str, metadata_path: str, stratify_columns: List[str]) -> pd.DataFrame:
    """Validate input data and return metadata DataFrame with filenames aligned."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        metadata_df = pd.read_csv(metadata_path)
        logging.info(f"Loaded metadata with {len(metadata_df)} rows and {len(metadata_df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Error reading metadata file: {e}")

    text_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    if len(metadata_df) != len(text_files):
        raise ValueError(f"Mismatch: {len(metadata_df)} metadata rows vs {len(text_files)} text files")

    missing_cols = [col for col in stratify_columns if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Stratification columns not found in metadata: {missing_cols}")

    for col in stratify_columns:
        if metadata_df[col].isnull().any():
            n_missing = metadata_df[col].isnull().sum()
            logging.warning(f"Stratify column '{col}' contains {n_missing} missing values. Filling with mode.")
            mode_val = metadata_df[col].mode().iloc[0] if len(metadata_df[col].mode()) > 0 else 'Unknown'
            metadata_df[col] = metadata_df[col].fillna(mode_val)

    metadata_df[FILENAME_COL] = text_files
    logging.info(f"Successfully aligned {len(text_files)} text files with metadata.")
    return metadata_df


def create_balanced_folds(metadata_df: pd.DataFrame, stratify_columns: List[str],
                          n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create balanced folds using MultilabelStratifiedKFold."""
    y = metadata_df[stratify_columns].values
    dummy_X = np.zeros((len(metadata_df), 1))

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = [(train_idx, test_idx) for train_idx, test_idx in mskf.split(dummy_X, y)]

    logging.info("Balanced folds created with MultilabelStratifiedKFold.")
    return splits


def process_and_save_fold(fold_num: int, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          feature_cols: List[str], fold_dir: str, data_dir: str, filename_col: str):
    """
    Handles file copying, saving originals, robust median imputation, and variant creation for a single fold.
    """
    train_df.to_csv(os.path.join(fold_dir, 'train_metadata_original.csv'), index=False)
    test_df.to_csv(os.path.join(fold_dir, 'test_metadata_original.csv'), index=False)
    logging.info(f"  Fold {fold_num}: Saved original, unprocessed metadata splits.")

    train_txt_dir = os.path.join(fold_dir, 'train')
    test_txt_dir = os.path.join(fold_dir, 'test')
    os.makedirs(train_txt_dir, exist_ok=True)
    os.makedirs(test_txt_dir, exist_ok=True)

    for file in train_df[filename_col]:
        src = os.path.join(data_dir, file)
        dst = os.path.join(train_txt_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            logging.warning(f"Source file not found for copy: {src}")

    for file in test_df[filename_col]:
        src = os.path.join(data_dir, file)
        dst = os.path.join(test_txt_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            logging.warning(f"Source file not found for copy: {src}")

    logging.info(f"  Fold {fold_num}: Copied raw .txt files.")

    # --- NEW STEP 1: Force feature columns to be numeric ---
    # This step will convert any non-numeric strings (e.g., 'NA', '?')
    # in your feature columns into NaN, making the data clean for imputation.
    logging.info(f"  Fold {fold_num}: Coercing feature columns to numeric type...")
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # --- STEP 2: Median Imputation (now guaranteed to work on all feature columns) ---
    logging.info(f"  Fold {fold_num}: Imputing missing values with median...")
    medians_map = {}
    for col in feature_cols:
        # The is_numeric_dtype check is no longer needed because we forced it above.
        if train_df[col].isnull().any():
            median_val = train_df[col].median()
            medians_map[col] = median_val

            # Use direct assignment to fill NaNs safely and avoid FutureWarning
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)

    if medians_map:
        logging.info(f"    Filled NaNs for columns: {list(medians_map.keys())}")
    else:
        logging.info("    No missing values found in numeric feature columns after coercion.")

    # --- STEP 3: Create and Save Variants ---
    for version in VERSION_DEFINITIONS:
        final_cols = [filename_col] + version['vars'] + feature_cols
        train_df[final_cols].to_csv(os.path.join(fold_dir, f"fold{fold_num}{version['train_suffix']}"), index=False)
        test_df[final_cols].to_csv(os.path.join(fold_dir, f"fold{fold_num}{version['test_suffix']}"), index=False)

    logging.info(f"  ✅ Fold {fold_num}: All processed metadata variants created and saved.")


def main():
    """Main execution function to run the entire data preparation pipeline."""
    log_file = os.path.join(os.path.dirname(OUTPUT_DIR) or '.', 'data_prep_pipeline_log.txt')
    setup_logging(log_file)

    try:
        logging.info("--- Starting End-to-End Data Preparation Pipeline for 10-Fold CV ---")
        logging.info(f"Output will be saved to: {OUTPUT_DIR}")

        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
            logging.info(f"Removed existing directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created clean output directory: {OUTPUT_DIR}")

        metadata_df = validate_inputs(DATA_DIR, METADATA_PATH, STRATIFY_COLUMNS)

        all_variable_cols = list(set(VARS_SET_1 + VARS_SET_2 + VARS_SET_3))
        feature_cols = [col for col in metadata_df.columns if col not in all_variable_cols and col != FILENAME_COL]
        logging.info(f"Identified {len(feature_cols)} feature columns.")

        splits = create_balanced_folds(metadata_df, STRATIFY_COLUMNS, N_SPLITS, RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            fold_dir = os.path.join(OUTPUT_DIR, f'fold_{fold}')
            os.makedirs(fold_dir)

            train_df = metadata_df.iloc[train_idx].copy()
            test_df = metadata_df.iloc[test_idx].copy()

            process_and_save_fold(fold, train_df, test_df, feature_cols, fold_dir, DATA_DIR, FILENAME_COL)

        summary_file = os.path.join(OUTPUT_DIR, 'pipeline_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("End-to-End Data Preparation Pipeline Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Source Data: {DATA_DIR}\n")
            f.write(f"Source Metadata: {METADATA_PATH}\n")
            f.write(f"Output Directory: {OUTPUT_DIR}\n\n")
            f.write(f"Total Samples: {len(metadata_df)}\n")
            f.write(f"Number of Folds: {N_SPLITS}\n")
            f.write(f"Stratified by: {', '.join(STRATIFY_COLUMNS)}\n")
            f.write("Stratification Method: MultilabelStratifiedKFold\n\n")
            f.write("Processing Steps Performed for each fold:\n")
            f.write("  1. Saved original, unprocessed train/test metadata.\n")
            f.write("  2. Copied raw .txt files to train/ and test/ subdirectories.\n")
            f.write("  3. Coerced all feature columns to numeric type (handling non-numeric strings).\n")
            f.write("  4. Performed median imputation on numeric features (calculated from train set only).\n")
            f.write("  5. Generated 3 processed data variants with different target variables.\n")

        logging.info(f"✅ All {N_SPLITS} folds processed and saved to: {OUTPUT_DIR}")
        logging.info(f"Pipeline summary saved to: {summary_file}")
        logging.info("--- Pipeline Completed Successfully! ---")

    except Exception as e:
        logging.error(f"An error occurred during the pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()