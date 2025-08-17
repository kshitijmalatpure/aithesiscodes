import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import sys

# --- Parameters You Must Configure ---
FILE_PATH = r'C:\Research\AI Folder\Thesis\Data\Data\allfeats.csv'
FILENAME_COLUMN = 'Inputfile'
SOURCE_FILES_DIR = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Missing files'
MAIN_OUTPUT_DIR = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)'

# --- Sampling Parameters ---
TARGET_VARIABLES = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
SAMPLE_SIZE = 100
RANDOM_STATE = 42


# --- NEW: Logging Helper Function ---
def log_message(message, file_handle):
    """Prints a message to the console and writes it to the log file."""
    print(message)
    file_handle.write(message + '\n')


def main():
    """Main function to run the sampling and file copying process."""
    print("--- Starting Sampling and File Copying Process ---")

    # --- Step 1: Input Validation and Directory Setup ---
    if not os.path.exists(FILE_PATH):
        print(f"\nFATAL ERROR: The source CSV file was not found at '{FILE_PATH}'")
        sys.exit(1)
    if not os.path.isdir(SOURCE_FILES_DIR):
        print(f"\nFATAL ERROR: The source files directory was not found at '{SOURCE_FILES_DIR}'")
        sys.exit(1)

    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

    # --- NEW: Setup Log File ---
    log_file_path = os.path.join(MAIN_OUTPUT_DIR, 'data_preparation_log.txt')

    # Use a 'with' statement to ensure the log file is properly closed
    with open(log_file_path, 'w') as log_file:
        log_message(f"--- Data Preparation Log ---\n", log_file)

        # --- Step 2: Load the Dataset ---
        try:
            df_full = pd.read_csv(FILE_PATH)
            log_message(f"Successfully loaded '{os.path.basename(FILE_PATH)}'. Initial total rows: {len(df_full)}.",
                        log_file)
        except Exception as e:
            log_message(f"\nFATAL ERROR: Could not read the CSV file. Details: {e}", log_file)
            sys.exit(1)

        if FILENAME_COLUMN not in df_full.columns:
            log_message(f"\nFATAL ERROR: The filename column '{FILENAME_COLUMN}' was not found in your CSV.", log_file)
            sys.exit(1)

        # --- Step 3: Handle Duplicates and Log Them ---
        log_message("\n--- Checking for duplicate filename entries ---", log_file)

        # NEW: Identify which rows are duplicates before dropping them
        duplicate_rows = df_full[df_full.duplicated(subset=[FILENAME_COLUMN], keep='first')]

        if not duplicate_rows.empty:
            log_message(
                f"Found and removed {len(duplicate_rows)} duplicate row(s). Keeping the first instance of each.",
                log_file)
            log_message("--- List of Removed Duplicate Filenames ---", log_file)
            for filename in duplicate_rows[FILENAME_COLUMN]:
                log_message(f"  - {filename}", log_file)
            log_message("------------------------------------------\n", log_file)
        else:
            log_message("No duplicate filenames found in the CSV.", log_file)

        # Now, drop the duplicates from the main DataFrame
        df_full.drop_duplicates(subset=[FILENAME_COLUMN], keep='first', inplace=True)
        log_message(f"Working with {len(df_full)} unique file entries.", log_file)

        # --- Step 4: Validate File Existence and Log Missing Files ---
        log_message("\n--- Performing pre-flight validation for file existence ---", log_file)
        df_full[FILENAME_COLUMN] = df_full[FILENAME_COLUMN].str.strip()
        df_full['file_exists'] = df_full[FILENAME_COLUMN].apply(
            lambda filename: os.path.exists(os.path.join(SOURCE_FILES_DIR, filename))
        )

        num_files_exist = df_full['file_exists'].sum()
        log_message(
            f"Validation Complete: {num_files_exist} of {len(df_full)} unique files were found in source directory.",
            log_file)

        # NEW: Identify which files are missing and log them
        missing_files_df = df_full[~df_full['file_exists']]
        if not missing_files_df.empty:
            log_message(
                f"\nFound {len(missing_files_df)} file(s) listed in the CSV but not found in the source directory.",
                log_file)
            log_message("--- List of Missing Files ---", log_file)
            for filename in missing_files_df[FILENAME_COLUMN]:
                log_message(f"  - {filename}", log_file)
            log_message("-----------------------------\n", log_file)

        if num_files_exist < SAMPLE_SIZE:
            log_message(
                f"FATAL ERROR: Only {num_files_exist} valid files exist, which is less than SAMPLE_SIZE ({SAMPLE_SIZE}).",
                log_file)
            sys.exit(1)

        df_valid = df_full[df_full['file_exists']].copy()
        df_valid.drop(columns=['file_exists'], inplace=True)
        log_message(f"Proceeding with a pool of {len(df_valid)} validated file entries.", log_file)

        # --- Step 5 & 6: Loop, Sample, and Copy ---
        for target_col in TARGET_VARIABLES:
            print(f"\n===== Processing Target Variable: '{target_col}' =====")

            if target_col not in df_valid.columns:
                print(f"Warning: Target column '{target_col}' not found. Skipping.")
                continue

            try:
                sample_df, _ = train_test_split(
                    df_valid,
                    train_size=SAMPLE_SIZE,
                    stratify=df_valid[target_col],
                    random_state=RANDOM_STATE
                )
            except ValueError as e:
                print(f"ERROR: Could not create stratified sample for '{target_col}'.")
                print(f"Details: {e}. Skipping this variable.")
                continue

            sample_specific_dir = os.path.join(MAIN_OUTPUT_DIR, f"{target_col}")
            files_destination_dir = os.path.join(sample_specific_dir, "copied_files")
            os.makedirs(files_destination_dir, exist_ok=True)

            output_df = sample_df[[FILENAME_COLUMN, target_col]].copy()
            output_df.rename(columns={FILENAME_COLUMN: 'filename', target_col: 'score'}, inplace=True)
            csv_output_path = os.path.join(sample_specific_dir, f"data_sample_{target_col}.csv")
            output_df.to_csv(csv_output_path, index=False)
            print(f"Saved 2-column (filename, score) CSV to: '{csv_output_path}'")

            print(f"Copying {len(output_df)} unique files to '{files_destination_dir}'...")
            files_copied_count = 0

            for filename in output_df['filename']:
                source_path = os.path.join(SOURCE_FILES_DIR, filename)
                destination_path = os.path.join(files_destination_dir, filename)
                shutil.copy2(source_path, destination_path)
                files_copied_count += 1

            print(f"Finished. Successfully copied: {files_copied_count} files.")
            final_file_count = len(os.listdir(files_destination_dir))
            print(f"Verified: {final_file_count} files are present in the destination folder.")

    print(f"\n\n===== SCRIPT COMPLETE =====")
    print(f"A detailed log has been saved to: {log_file_path}")


if __name__ == '__main__':
    main()