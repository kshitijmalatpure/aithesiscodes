import pandas as pd

# --- Configuration ---
# Set the filenames for your two CSV files.
# Make sure these files are in the same folder as this script.
base_file = r'C:\Research\AI Folder\Thesis\Data\Data\reduced_allfeats1.csv'
file_to_clean = r'C:\Users\kshit\Downloads\total.document (6).csv'

# Set the names for your output files.
output_txt_file = r'C:\Research\AI Folder\Thesis\Data\Data\extra_headers.txt'
output_csv_file = r'C:\Research\AI Folder\Thesis\Data\Data\total.document_cleaned_and_reordered.csv'
output_txt_file_missing = r'C:\Research\AI Folder\Thesis\Data\Data\missing_headers.txt'


print("--- Starting the data cleaning process ---")

try:
    # --- Step 1: Load both CSV files ---
    print(f"Loading base file: {base_file}")
    df_base = pd.read_csv(base_file)
    base_headers = df_base.columns.tolist()

    print(f"Loading file to clean: {file_to_clean}")
    df_to_clean = pd.read_csv(file_to_clean)
    to_clean_headers = df_to_clean.columns.tolist()

    print(f"\nBase file has {len(base_headers)} columns.")
    print(f"File to clean has {len(to_clean_headers)} columns.")

    # --- Step 2: Case-Insensitive Standardization ---
    # This is the new, crucial step.
    # We create a map from a lowercase header to the correctly-cased header from the base file.
    lower_to_base_case_map = {h.lower(): h for h in base_headers}

    # Now, we build a dictionary to rename columns in df_to_clean
    rename_map = {}
    for col in to_clean_headers:
        if col.lower() in lower_to_base_case_map:
            # If this column (case-insensitive) exists in the base file...
            # ...map its current name to the name with the base file's capitalization.
            rename_map[col] = lower_to_base_case_map[col.lower()]

    df_to_clean.rename(columns=rename_map, inplace=True)
    print("\nStandardized column capitalization to match the base file.")

    # --- Step 3: Identify True Extra and Missing Headers (Post-Standardization) ---
    # Now that capitalization is fixed, we can do a proper comparison.
    standardized_clean_headers = df_to_clean.columns.tolist()

    # Extra headers are those in the cleaned file that are NOT in the base file
    extra_headers = sorted(list(set(standardized_clean_headers) - set(base_headers)))

    # Missing headers are those in the base file that are NOT in the cleaned file
    missing_headers = sorted(list(set(base_headers) - set(standardized_clean_headers)))

    # --- Step 4: Report and Save Differences ---
    print(f"\nFound {len(extra_headers)} true extra headers to remove:")
    print(extra_headers)
    with open(output_txt_file, 'w') as f:
        f.write('\n'.join(extra_headers))
    print(f"Saved list of extra headers to '{output_txt_file}'")

    if missing_headers:
        print(f"\nWARNING: Found {len(missing_headers)} headers from base file that are missing in the target file:")
        print(missing_headers)
        with open(output_txt_file_missing, 'w') as f:
            f.write('\n'.join(missing_headers))
        print(f"Saved list of missing headers to '{output_txt_file_missing}'")

    # --- Step 5: Create the Final DataFrame ---
    # We select only the columns that are in the base file's list.
    # This automatically removes the "extra" columns and handles the "missing" ones.
    final_columns = [col for col in base_headers if col in df_to_clean.columns]
    df_final = df_to_clean[final_columns]

    print(f"\nCreated final dataframe with {len(df_final.columns)} columns, matching the base file's format.")

    # --- Step 6: Save the final, cleaned, and reordered dataframe ---
    df_final.to_csv(output_csv_file, index=False)
    print(f"\nProcess complete! The cleaned and reordered data has been saved to '{output_csv_file}'")

except FileNotFoundError as e:
    print(f"\nERROR: File not found. Please make sure the file '{e.filename}' exists and the path is correct.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")