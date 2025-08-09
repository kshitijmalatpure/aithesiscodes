import os
import pandas as pd
import logging

# --- CONFIGURATION ---
# The main directory containing your 'fold_1', 'fold_2', etc. folders.
BASE_DIR = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-balanced-multicolumn"

# The total number of folds to process.
NUM_FOLDS = 10

# Suffixes for the file variants created by the previous script.
# This helps the script find all the files to modify.
VERSION_DEFINITIONS = [
    {'train_suffix': 'tr.csv', 'test_suffix': 'te.csv'},
    {'train_suffix': 'trb.csv', 'test_suffix': 'teb.csv'},
    {'train_suffix': 'trh.csv', 'test_suffix': 'teh.csv'}
]


# --- END OF CONFIGURATION ---

def setup_logging(base_dir):
    """Sets up logging to a file in the base directory."""
    log_file = os.path.join(base_dir, 'remove_last_column_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )


def remove_last_column_from_all_metadata():
    """
    Iterates through all folds and removes the last column from the original
    metadata files and all their created variants.

    WARNING: This operation is destructive and overwrites the files.
    """
    setup_logging(BASE_DIR)
    logging.warning("--- SCRIPT STARTED: REMOVING LAST COLUMN FROM METADATA ---")
    logging.warning("!!! THIS IS A DESTRUCTIVE OPERATION. FILES WILL BE OVERWRITTEN. !!!")

    for fold_num in range(1, NUM_FOLDS + 1):
        fold_name = f'fold_{fold_num}'
        fold_path = os.path.join(BASE_DIR, fold_name)

        if not os.path.isdir(fold_path):
            logging.warning(f"Skipping: Directory not found at {fold_path}")
            continue

        logging.info(f"--- Processing {fold_name} ---")

        # 1. Start with the original metadata files
        files_to_process = ['train_metadata.csv', 'test_metadata.csv']

        # 2. Add all the variant files to the list
        for version_info in VERSION_DEFINITIONS:
            files_to_process.append(f"fold{fold_num}{version_info['train_suffix']}")
            files_to_process.append(f"fold{fold_num}{version_info['test_suffix']}")

        # 3. Process each file in the list
        for filename in files_to_process:
            file_path = os.path.join(fold_path, filename)

            if not os.path.exists(file_path):
                # This is expected if a previous step failed, so just log as info
                logging.info(f"  Skipping non-existent file: {filename}")
                continue

            try:
                # Read the CSV
                df = pd.read_csv(file_path)

                # Check if there are any columns to remove
                if df.shape[1] < 1:
                    logging.warning(f"  Skipping empty file (no columns): {filename}")
                    continue

                # Get the name of the last column
                last_col_name = df.columns[-1]

                # Drop the last column
                df_modified = df.iloc[:, :-1]

                # Save the modified DataFrame back to the same file, overwriting it
                df_modified.to_csv(file_path, index=False)

                logging.info(f"  -> Removed last column ('{last_col_name}') from {filename}")

            except Exception as e:
                logging.error(f"  ERROR processing {filename}: {e}")

    logging.info("--- Process Completed Successfully! ---")


if __name__ == "__main__":
    # A simple confirmation step to prevent accidental runs.
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: This script will permanently modify your data files. !!!")
    print("!!! It will REMOVE THE LAST COLUMN from all metadata CSVs.        !!!")
    print("!!! Make sure you have a backup before proceeding.                !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # _#
    # _#
    user_input = input("Type 'yes' to continue: ")
    if user_input.lower() == 'yes':
        remove_last_column_from_all_metadata()
    else:
        print("Operation cancelled by user.")