import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1. CONFIGURE PATHS  (⇩ change these three lines as needed)
# ------------------------------------------------------------------
csv1_path  = Path(r"C:\Research\AI Folder\Thesis\Data\Data\allfeats1.csv")
txt_path   = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs\prompt_tt.txt")            # filenames, one per line
csv2_path  = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs\selected_rows.csv")

# ------------------------------------------------------------------
# 2. READ INPUTS
# ------------------------------------------------------------------
df   = pd.read_csv(csv1_path)
with open(txt_path, "r", encoding="utf-8") as f:
    wanted = {line.strip() for line in f if line.strip()}   # remove blank lines

# ------------------------------------------------------------------
# 3. SPLIT THE DATAFRAME
# ------------------------------------------------------------------
mask       = df.iloc[:, 0].isin(wanted)    # first column is the filename
matches    = df[mask]
remaining  = df[~mask]

# ------------------------------------------------------------------
# 4. WRITE OUTPUTS
# ------------------------------------------------------------------
matches.to_csv(csv2_path, index=False)     # the "moved" rows

# Optional: update csv1 to contain only the remaining rows
remaining.to_csv(csv1_path, index=False)

print(f"{len(matches)} rows moved to {csv2_path}")
print(f"{len(remaining)} rows remain in {csv1_path}")
