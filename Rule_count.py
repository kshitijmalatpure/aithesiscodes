import pandas as pd

# ── 1. Load and clean the header row ───────────────────────────────────────────
df = pd.read_csv("spellcheck_output.csv", dtype=str)    # dtype=str avoids dtype fuss
df.columns = df.columns.str.strip()                     # remove leading/trailing spaces
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]      # drop old index cols
df = df.loc[:, ~df.columns.duplicated()]                # drop duplicate headers

# ── 2. Make sure the two key columns exist ─────────────────────────────────────
required = {"filename", "Rule ID"}
missing  = required - set(df.columns)
if missing:
    raise KeyError(f"Missing expected column(s): {', '.join(missing)}")

# ── 3. Count total errors per file ─────────────────────────────────────────────
totals = (
    df.groupby("filename", as_index=False)
      .size()
      .rename(columns={"size": "Error Count"})
)

# ── 4. Count each Rule-ID per file (wide format) ───────────────────────────────
rule_counts = pd.crosstab(df["filename"], df["Rule ID"])   # rows=filename, cols=Rule ID

# ── 5. Merge and save ──────────────────────────────────────────────────────────
out = totals.merge(rule_counts, left_on="filename", right_index=True)
out.to_csv("spellcheck_rule_counts.csv", index=False)
print("✅ spellcheck_rule_counts.csv written.")
