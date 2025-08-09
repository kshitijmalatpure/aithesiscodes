import pandas as pd

# 1. Read the CSV (adjust filename/path as needed)
#    dtype=str ensures everything is loaded as text
df = pd.read_csv(r'C:\Research\AI Folder\Thesis\Data\Main\Usable_data\data_with_splits.csv', dtype=str)

# 2. (Optional) Strip whitespace from headers in case they have stray spaces
df.columns = df.columns.str.strip()

# 3. Build the new column:
#    transcribent + '_' + kandidaatcode + '_' + A-type
df['kandidaatcode_new'] = (
    df['transcribent'].fillna('') + '_'
  + df['kandidaatcode'].fillna('') + '_'
  + df['A-type'].fillna('')
)

# 4. Write out to a new CSV
df.to_csv(r'output.csv', index=False)

print("✔ Created 'kandidaatcode_new' and saved to output.csv")
