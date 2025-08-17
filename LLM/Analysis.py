"""
Model-comparison utility
------------------------
1. Reads a 3-column CSV  (filename | y_true | y_pred)
2. Computes:
      • Quadratic-weighted κ  (QWK)
      • Macro F1
      • Accuracy
3. Writes:
      • metrics.csv   and  metrics.xlsx   to  <out_dir>
      • <file_stem>_confmat.png           to  <out_dir>
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# 0. Configure paths  (⇩ change to match your setup)
# ---------------------------------------------------------------------------
csv_path = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\A_WO\WO.csv")
out_dir  = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\WO_final")
out_dir.mkdir(parents=True, exist_ok=True)

# A nice label for plots/files: use the CSV file's stem
display_name = csv_path.stem
# If you instead want the value from the first column (e.g., a model/run name),
# uncomment the following line:
# display_name = str(pd.read_csv(csv_path, nrows=1).iloc[0, 0])

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(csv_path)

true_col = df.columns[1]    # second column (y_true)
pred_col = df.columns[2]    # third column (y_pred)

print(f"True labels column: {true_col}")
print(f"Predictions column: {pred_col}")

# ---------------------------------------------------------------------------
# 2. Handle missing/invalid values
# ---------------------------------------------------------------------------
def clean_data(df, true_col, pred_col):
    """Clean the data by handling missing and invalid values."""
    print(f"\nOriginal data shape: {df.shape}")
    true_missing = df[true_col].isnull().sum()
    pred_missing = df[pred_col].isnull().sum()
    print(f"Missing values in {true_col}: {true_missing}")
    print(f"Missing values in {pred_col}: {pred_missing}")

    # Remove rows with missing or non-finite predictions
    valid_mask = df[true_col].notna() & df[pred_col].notna() & np.isfinite(df[pred_col])
    df_clean = df[valid_mask].copy()
    removed_rows = len(df) - len(df_clean)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing/invalid values")

    print(f"Clean data shape: {df_clean.shape}")
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after cleaning!")
    return df_clean

df_clean = clean_data(df, true_col, pred_col)

# Convert to integers (round if needed)
try:
    y_true = df_clean[true_col].astype(int)
    y_pred = df_clean[pred_col].astype(int)
    print("Successfully converted to integers")
except Exception:
    y_true = df_clean[true_col].round().astype(int)
    y_pred = df_clean[pred_col].round().astype(int)
    print("Converted after rounding")

labels = sorted(y_true.unique())
print(f"Unique labels found: {labels}")

# ---------------------------------------------------------------------------
# 3. Metric helpers
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    """Return QWK, macro-F1, accuracy as a dict."""
    return {
        "QWK":       cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "F1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Accuracy":  accuracy_score(y_true, y_pred),
    }

# Calculate metrics and confusion matrix
scores = {display_name: evaluate(y_true, y_pred)}
conf_mats = {display_name: confusion_matrix(y_true, y_pred, labels=labels)}

# ---------------------------------------------------------------------------
# 4. Metrics table  ➜  print + save
# ---------------------------------------------------------------------------
score_df = (
    pd.DataFrame(scores)
    .T
    .sort_values("QWK", ascending=False)
    .round(4)
)

print("\n=== Model comparison metrics ===")
print(score_df.to_string())
print("\nSaving metric tables …")

score_df.to_csv(out_dir / "metrics.csv",  index=True)
score_df.to_excel(out_dir / "metrics.xlsx", index=True)

# ---------------------------------------------------------------------------
# 5. Confusion-matrix plots  ➜  save PNGs (seaborn 'Blues' heatmap)
# ---------------------------------------------------------------------------
print("Saving confusion-matrix images …")

sns.set_theme(style="white")
for name, cm in conf_mats.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        cbar_kws={"label": "Count"},
        square=True,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 20}
    )
    ax.set_title(f"Confusion Matrix – {name}", fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()

    safe_name = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in name)
    fig.savefig(out_dir / f"{safe_name}_confmat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"Done. All outputs saved to:\n  {out_dir}")
