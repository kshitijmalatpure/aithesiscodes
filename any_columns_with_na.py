import pandas as pd

# Load your DataFrame
df = pd.read_csv(r'CC:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\Usable_data\allfeats.csv')

# Get the first column name (filename column)
filename_col = df.columns[0]

# Select columns with at least one NA value
cols_with_na = df.columns[df.isna().any()].tolist()

# Ensure the filename column is included (at the start)
if filename_col not in cols_with_na:
    cols_with_na = [filename_col] + cols_with_na

# Create a new DataFrame with only those columns
df_na = df[cols_with_na]

# Save to a new CSV file
df_na.to_csv('columns_with_na.csv', index=False)

print(f"Saved {len(cols_with_na)} columns (including filename column) with at least one NA to columns_with_na.csv")
