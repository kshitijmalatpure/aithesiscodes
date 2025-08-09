import pandas as pd
import os
import glob

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# Folder with your .txt files
file_destination = (r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\pythonProject\Stuff\fold_2\test")

# Folder with the single reference CSV
ref_folder = r"C:\Research\AI Folder\Thesis\Data\Main\Usable_data\STRT"
# ────────────────────────────────────────────────────────────────────────────────

# 1. Locate and read the reference CSV (assumes exactly one .csv in ref_folder)
ref_csvs = glob.glob(os.path.join(ref_folder, "*.csv"))
if not ref_csvs:
    raise FileNotFoundError(f"No CSVs found in reference folder: {ref_folder}")
ref_df = pd.read_csv(ref_csvs[0])

# 2. Verify expected columns in reference
metadata_cols = [
    "kandidaatcode",
    "transcribent",
    "examen",
    "instellingscode",
    "A_WO",
    "A_GR",
    "A_CO",
    "A_TT",
    "A-type",
]
missing = [c for c in metadata_cols if c not in ref_df.columns]
if missing:
    raise KeyError(f"Reference CSV is missing columns: {missing}")

# 3. Prepare for matching
output_cols = metadata_cols + ["filename"]
results = []

# 4. Loop through each .txt file in file_destination
for fname in os.listdir(file_destination):
    if not fname.lower().endswith(".txt"):
        continue

    seen = set()  # to dedupe identical metadata rows per filename

    # Check every partial string (kandidaatcode) against this filename
    for _, row in ref_df.iterrows():
        partial = str(row["kandidaatcode"])
        if partial in fname:
            # Extract the metadata tuple
            meta_tuple = tuple(row[col] for col in metadata_cols)
            # Only add if it differs from previously added metadata for this file
            if meta_tuple not in seen:
                seen.add(meta_tuple)
                results.append(list(meta_tuple) + [fname])

# 5. Build result DataFrame and write out
out_df = pd.DataFrame(results, columns=output_cols)
out_path = os.path.join(file_destination, "data.csv")
out_df.to_csv(out_path, index=False)

print(f"✔ Generated '{out_path}' with {len(out_df)} rows")
