import pandas as pd
import os
import sys

# --- Configuration: You Must Edit These Three Lines ---

# 1. The path to the folder containing your files.
FOLDER_PATH = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Missing files'

# 2. The path to your CSV file.
CSV_PATH = r'C:\Research\AI Folder\Thesis\Data\Data\allfeats.csv'

# 3. The name of the column in your CSV that contains the filenames.
FILENAME_COLUMN = 'Inputfile'

# 4. (Optional) Set to False for case-insensitive comparison (e.g., 'File.JPG' matches 'file.jpg')
#    Useful when moving between Windows (insensitive) and Linux/Mac (sensitive).
CASE_SENSITIVE = True


# ---------------------------------------------------------

def validate_data_consistency():
    """
    Validates consistency between files in a folder and a CSV manifest.
    Checks for:
    1. Files in CSV not present in the folder.
    2. Files in the folder not listed in the CSV.
    3. Duplicate entries in the CSV's filename column.
    """
    print("--- Starting Data Consistency Validation ---")
    print(f"Folder to check: {FOLDER_PATH}")
    print(f"CSV manifest: {CSV_PATH}")
    print(f"Filename column: '{FILENAME_COLUMN}'")
    print(f"Case-sensitive check: {CASE_SENSITIVE}")
    print("-" * 40)

    # --- Step 1: Basic Input Validation ---
    if not os.path.isdir(FOLDER_PATH):
        print(f"FATAL ERROR: The specified folder does not exist: {FOLDER_PATH}")
        sys.exit(1)

    if not os.path.exists(CSV_PATH):
        print(f"FATAL ERROR: The specified CSV file does not exist: {CSV_PATH}")
        sys.exit(1)

    # --- Step 2: Get the list of actual files from the folder ---
    try:
        print("Reading files from folder...")
        # os.listdir can include subfolders, so we filter to ensure we only get files.
        # os.path.join is used to create a full path for the isfile check.
        actual_files = [f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))]

        # Apply case-insensitivity if requested
        if not CASE_SENSITIVE:
            folder_files_set = {f.lower() for f in actual_files}
        else:
            folder_files_set = set(actual_files)

        print(f"Found {len(folder_files_set)} unique files in the folder.")
    except Exception as e:
        print(f"FATAL ERROR: Could not read files from the folder. Details: {e}")
        sys.exit(1)

    # --- Step 3: Get the list of filenames from the CSV ---
    try:
        print("\nReading filenames from CSV...")
        df = pd.read_csv(CSV_PATH)

        if FILENAME_COLUMN not in df.columns:
            print(f"FATAL ERROR: Column '{FILENAME_COLUMN}' not found in the CSV file.")
            print(f"Available columns are: {list(df.columns)}")
            sys.exit(1)

        # Clean the data: remove leading/trailing whitespace which causes silent errors
        csv_filenames = df[FILENAME_COLUMN].dropna().astype(str).str.strip()

        # Apply case-insensitivity if requested
        if not CASE_SENSITIVE:
            csv_filenames_set = {f.lower() for f in csv_filenames}
        else:
            csv_filenames_set = set(csv_filenames)

        print(
            f"Found {len(csv_filenames)} total entries and {len(csv_filenames_set)} unique filenames in the CSV column.")
    except Exception as e:
        print(f"FATAL ERROR: Could not read or process the CSV file. Details: {e}")
        sys.exit(1)

    print("-" * 40)

    # --- Step 4: Perform the validation checks and build the report ---
    print("\n--- Validation Report ---")

    inconsistencies_found = False

    # Check 1: Duplicates within the CSV file
    duplicates_in_csv = csv_filenames[csv_filenames.duplicated()].unique()
    if len(duplicates_in_csv) > 0:
        inconsistencies_found = True
        print(f"\n[!] WARNING: Found {len(duplicates_in_csv)} duplicate filename(s) in the CSV:")
        for filename in sorted(duplicates_in_csv):
            print(f"  - {filename}")

    # Check 2: Files in CSV but not in the folder
    missing_from_folder = sorted(list(csv_filenames_set - folder_files_set))
    if missing_from_folder:
        inconsistencies_found = True
        print(f"\n[!] ERROR: {len(missing_from_folder)} file(s) are listed in the CSV but NOT found in the folder:")
        for filename in missing_from_folder:
            print(f"  - {filename}")

    # Check 3: Files in the folder but not in the CSV
    unlisted_in_csv = sorted(list(folder_files_set - csv_filenames_set))
    if unlisted_in_csv:
        inconsistencies_found = True
        print(f"\n[!] WARNING: {len(unlisted_in_csv)} file(s) exist in the folder but are NOT listed in the CSV:")
        for filename in unlisted_in_csv:
            print(f"  - {filename}")

    # --- Step 5: Final Summary ---
    print("-" * 40)
    if not inconsistencies_found:
        print("\n[✓] SUCCESS: No inconsistencies found. The CSV and folder are aligned.")
    else:
        print("\nValidation finished. Please address the issues listed above.")
    print("-" * 40)


if __name__ == '__main__':
    validate_data_consistency()