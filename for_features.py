import csv

# --- User-defined file paths ---
txt_file = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final\fabOF_results_v8.4.0_TUNED_PIPELINE_FIXED\aggregated_predictions\WO.txt"   # text file with one string per line
input_csv = r"C:\Research\AI Folder\Thesis\Data\Data\allfeats.csv"           # original csv
output_csv = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final\fabOF_results_v8.4.0_TUNED_PIPELINE_FIXED\aggregated_predictions\WO.csv"       # new csv with filtered rows

# Read the strings from the txt file into a set for fast lookup
with open(txt_file, "r", encoding="utf-8") as f:
    filter_strings = set(line.strip() for line in f if line.strip())

# Open the CSV, read rows, and filter
with open(input_csv, "r", encoding="utf-8", newline='') as csv_in:
    reader = csv.reader(csv_in)
    header = next(reader)  # keep the header
    filtered_rows = [row for row in reader if row and row[0] in filter_strings]

# Save filtered rows to a new CSV
with open(output_csv, "w", encoding="utf-8", newline='') as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(header)       # write header
    writer.writerows(filtered_rows)

print(f"Filtering complete. {len(filtered_rows)} rows saved to '{output_csv}'.")
