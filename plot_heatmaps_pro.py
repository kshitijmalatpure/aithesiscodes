
# plot_heatmaps_pro.py
# Generate high-quality heatmaps (matplotlib-only) with large, outlined annotations
# and crisp cell boundaries. Produces PNG (300 dpi), plus PDF and SVG for print.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from pathlib import Path

csv_path = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final\comprehensive_results_regression_v2\aggregated_results_regression.csv")
df = pd.read_csv(csv_path)
agg = df[df["ScoreColumn"].astype(str).str.upper() == "AGGREGATE"].copy()

#row_order = ["DecisionTree", "GaussianNB", "RandomForest", "SVM", "XGBoost"]
row_order = ["RandomForest", "SVR", "XGBoost", "GradientBoosting", "Ridge", "ElasticNet"]
col_order = ["Baseline", "KBest_100", "KBest_150", "KBest_200", "KBest_50", "LASSO"]

def make_matrix(values_col):
    piv = (
        agg.pivot_table(index="Model", columns="FeatureSelection", values=values_col)
        .reindex(index=row_order, columns=col_order)
    )
    return piv

def professional_heatmap(piv, title, outfile_base, fmt="{:.3f}"):
    nrows, ncols = piv.shape
    cell_w, cell_h = 1.00, 1.25  # inches per cell (taller than wide -> "narrower" look)
    figsize = (ncols * cell_w, nrows * cell_h)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    data = piv.values.astype(float)
    X, Y = np.meshgrid(np.arange(ncols+1), np.arange(nrows+1))
    im = ax.pcolormesh(X, Y, data, shading="flat", linewidth=1.2, edgecolors="white")
    
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(piv.columns, fontsize=8)
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, pad=12)
    
    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            if np.isfinite(val):
                txt = ax.text(j + 0.5, i + 0.5, fmt.format(val),
                              ha="center", va="center",
                              fontsize=10)
                #txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black", alpha=0.35)])
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    fig.tight_layout()
    base = Path(outfile_base)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    svg = base.with_suffix(".svg")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, pdf, svg

if __name__ == "__main__":
    piv_qwk = make_matrix("QWK_mean")
    piv_f1  = make_matrix("F1_macro_mean")
    professional_heatmap(piv_qwk, "Mean QWK for AGGREGATE", "heatmap_qwk_pro")
    professional_heatmap(piv_f1,  "Mean F1 for AGGREGATE",  "heatmap_f1_pro")
    print("Saved heatmap_qwk_pro.(png|pdf|svg), heatmap_f1_pro.(png|pdf|svg)")
