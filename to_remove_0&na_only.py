import pandas as pd
from pathlib import Path

# --- Load the data -----------------------------------------------------------
file_path = Path(r"C:\Research\AI Folder\Thesis\Data\Data\Upcoming2.csv")
df = pd.read_csv(file_path)

# --- Identify sparse columns: only 0 or NA -----------------------------------
to_drop = [
    col for col in df.columns
    if ((df[col] == 0) | (df[col].isna())).all()
]

# --- Identify first column (e.g., filename) to retain ------------------------
id_column = df.columns[0]  # first column name (e.g., "filename")

# --- Create DataFrame of dropped columns + id column -------------------------
df_removed = df[[id_column] + to_drop] if to_drop else pd.DataFrame(columns=[id_column])

# --- Drop only the sparse columns from main set ------------------------------
df_reduced = df.drop(columns=to_drop)

# --- Save both versions ------------------------------------------------------
out_dir = file_path.parent
df_reduced.to_csv(out_dir / "reduced_allfeats1.csv", index=False)
df_removed.to_csv(out_dir / "removed_only_allfeats1.csv", index=False)

print(f"Removed {len(to_drop)} columns containing only 0 or NA.")
print(f"Saved reduced dataset to     → 'reduced_allfeats1HIHI.csv'")
print(f"Saved removed features with ID to → 'removed_only_allfeats1HIHI.csv'")
