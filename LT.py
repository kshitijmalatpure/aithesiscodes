import os
import pandas as pd
import language_tool_python
import time
import requests  # Import requests to catch its specific exceptions

# --- Configuration ---
# Initialize LanguageTool (using the public API)
# You can also run a local server for better performance and privacy:
# tool = language_tool_python.LanguageTool('nl', remote_server='http://localhost:8081')
tool = language_tool_python.LanguageTool('nl')

# Set the root folder containing text files
root_folder = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Missing files"

# Output filenames
summary_csv = "spellcheck_summary.csv"
detailed_csv = "spellcheck_output_with_counts.csv"
rule_counts_csv = "spellcheck_rule_counts_by_file.csv"
failed_files_csv = "failed_files.csv"

# --- Initialization ---
# Container for all results
results = []
failed_files = []  # Keep a list of files that failed
processed_files_count = 0
error_free_files_count = 0

print("🚀 Starting file processing...")
start_time = time.time()

# Walk through all subfolders and process .txt files
for subdir, _, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".txt"):
            filepath = os.path.join(subdir, file)
            processed_files_count += 1

            # Announce which file is being processed
            print(f"\nProcessing file {processed_files_count}: {filepath}")

            # Try decoding with multiple encodings
            text = None
            encodings = ["utf-8", "windows-1252", "iso-8859-1"]
            for enc in encodings:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                print(f"❌ Could not read file with any tested encoding. Skipping.")
                failed_files.append({"filepath": filepath, "error": "UnicodeDecodeError"})
                continue

            # Wrap the tool check in a try...except block
            try:
                # Run LanguageTool check
                matches = tool.check(text)

                # Using a short delay can help avoid overwhelming a public server
                time.sleep(0.5)

                # <<< ANNOTATION START >>>
                # This is the key change. We now check if the 'matches' list has any items.
                # If it does, we proceed as before. If it's empty, we print a success message.
                if matches:
                    # If errors are found, announce the count for this specific file.
                    print(f"⚠️ Found {len(matches)} potential errors in this file.")

                    # Record each match
                    for match in matches:
                        results.append({
                            "subfolder": os.path.relpath(subdir, root_folder),
                            "filename": file,
                            "Rule ID": match.ruleId,
                            "Message": match.message,
                            "Replacements": ", ".join(match.replacements),
                            "Context": match.context,
                            "Offset": match.offset,
                            "Error Length": match.errorLength,
                            "Sentence": match.sentence
                        })
                else:
                    # If 'matches' is empty, this block runs.
                    # This explicitly "annotates" that the file was checked and found to be clean.
                    print("✅ OK - No errors found.")
                    error_free_files_count += 1
                # <<< ANNOTATION END >>>

            # Catch the specific timeout error and other potential issues
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                print(f"❌ TIMEOUT or CONNECTION ERROR. Skipping. Error: {e}")
                failed_files.append({"filepath": filepath, "error": str(e)})
                # When the server times out, it's good practice to restart the tool object
                print("...Restarting LanguageTool to ensure a fresh connection...")
                tool.close()
                tool = language_tool_python.LanguageTool('nl')
            except Exception as e:
                # Catch any other unexpected errors from LanguageTool
                print(f"❌ An unexpected error occurred. Skipping. Error: {e}")
                failed_files.append({"filepath": filepath, "error": str(e)})

# --- Final Reporting ---
print("\n" + "=" * 50)
print("✅ File processing complete.")
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(f"Files processed: {processed_files_count}")
print(f"Files with no errors: {error_free_files_count}")
print(f"Files with errors: {processed_files_count - error_free_files_count - len(failed_files)}")
print(f"Files that failed to process: {len(failed_files)}")
print("=" * 50 + "\n")

# --- Save Results to CSVs ---
if results:
    print("Generating output CSV files...")
    df_results = pd.DataFrame(results)

    # 1) Build a summary DataFrame with error counts per file
    summary = (
        df_results
        .groupby(["subfolder", "filename"], as_index=False)
        .size()
        .rename(columns={"size": "Error Count"})
    )
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary to '{summary_csv}'")

    # 2) Add the error count to the detailed results file
    df_results = df_results.merge(summary, on=["subfolder", "filename"], how="left")
    df_results.to_csv(detailed_csv, index=False)
    print(f"Saved detailed results to '{detailed_csv}'")

    # 3) Create a wide table with counts for each specific rule
    error_counts = summary[["subfolder", "filename", "Error Count"]]
    rule_counts = (
        df_results
        .groupby(["subfolder", "filename", "Rule ID"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    third_output = (
        error_counts
        .merge(rule_counts, on=["subfolder", "filename"], how="left")
        .sort_values(["subfolder", "filename"])
    )
    third_output.to_csv(rule_counts_csv, index=False)
    print(f"Saved rule counts to '{rule_counts_csv}'")

    print("\n✅ All CSV reports have been written.")

else:
    print("⚠️ No errors were found in any of the processed files.")

# Save the list of failed files for review
if failed_files:
    pd.DataFrame(failed_files).to_csv(failed_files_csv, index=False)
    print(f"ℹ️ A list of files that could not be processed has been saved to '{failed_files_csv}'")