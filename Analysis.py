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
      • pred_confmat.png                  to  <out_dir>
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# 0. Configure paths  (⇩ change to match your setup)
# ---------------------------------------------------------------------------
csv_path = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\A_WO\WOoss.csv")
out_dir  = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\Baseline")
out_dir.mkdir(parents=True, exist_ok=True)

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
    
    # Check for missing values
    true_missing = df[true_col].isnull().sum()
    pred_missing = df[pred_col].isnull().sum()
    
    print(f"Missing values in {true_col}: {true_missing}")
    print(f"Missing values in {pred_col}: {pred_missing}")
    
    # Check for infinite values
    pred_inf = np.isinf(df[pred_col].replace([np.inf, -np.inf], np.nan)).sum()
    print(f"Infinite values in {pred_col}: {pred_inf}")
    
    # Remove rows with missing or infinite values in either column
    valid_mask = (
        df[true_col].notna() & 
        df[pred_col].notna() & 
        np.isfinite(df[pred_col])
    )
    
    df_clean = df[valid_mask].copy()
    removed_rows = len(df) - len(df_clean)
    
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing/invalid values")
    
    print(f"Clean data shape: {df_clean.shape}")
    
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after cleaning!")
    
    return df_clean

# Clean the data
df_clean = clean_data(df, true_col, pred_col)

# Convert to integers
try:
    y_true = df_clean[true_col].astype(int)
    y_pred = df_clean[pred_col].astype(int)
    print(f"Successfully converted to integers")
except Exception as e:
    print(f"Error converting to integers: {e}")
    # Try rounding first in case of float values
    try:
        y_true = df_clean[true_col].round().astype(int)
        y_pred = df_clean[pred_col].round().astype(int)
        print(f"Successfully converted after rounding")
    except Exception as e2:
        print(f"Failed even after rounding: {e2}")
        raise

labels = sorted(y_true.unique())  # keep label order consistent
print(f"Unique labels found: {labels}")

# ---------------------------------------------------------------------------
# 3. Metric helpers
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    """Return QWK, macro-F1, accuracy as a dict."""
    return {
        "QWK":       cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "F1_macro":  f1_score(y_true, y_pred, average="macro"),
        "Accuracy":  accuracy_score(y_true, y_pred),
    }

# Calculate metrics
scores = {"predictions": evaluate(y_true, y_pred)}
conf_mats = {"predictions": confusion_matrix(y_true, y_pred, labels=labels)}

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

score_df.to_csv(out_dir / "TTBmetrics.csv",  index=True)
score_df.to_excel(out_dir / "TTBmetrics.xlsx", index=True)

# ---------------------------------------------------------------------------
# 5. Confusion-matrix plots  ➜  save PNGs
# ---------------------------------------------------------------------------
print("Saving confusion-matrix images …")

for name, cm in conf_mats.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix – {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # write cell counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_confmat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)   # close to free memory

print(f"Done. All outputs saved to:\n  {out_dir}")