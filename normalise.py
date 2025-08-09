import pandas as pd
from pathlib import Path

# --- Load the data -----------------------------------------------------------
file_path = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\pythonProject\reduced_allfeats1.csv")
df = pd.read_csv(file_path)

# --- Identify the first column (e.g., filename) to exclude from scaling -------
id_column = df.columns[0]

# --- Apply min-max scaling column by column ----------------------------------
df_scaled = df.copy()
for col in df.columns:
    if col == id_column:
        continue  # Skip the ID column
    if pd.api.types.is_numeric_dtype(df[col]):
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df_scaled[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_scaled[col] = 0.0  # or np.nan, if you prefer

# --- Save the scaled DataFrame -----------------------------------------------
df_scaled.to_csv(file_path.parent / "minmax_scaled_allfeats.csv", index=False)
print("Min-max scaling applied column by column. Saved to 'minmax_scaled_allfeats.csv'.")