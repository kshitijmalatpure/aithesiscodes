import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math
from scipy import stats


def load_shap_data(csv_path):
    """Load SHAP values from CSV file"""
    df = pd.read_csv(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final\fabOF_results_v8.4.0_TUNED_PIPELINE_FIXED\shap_analysis\Tuned_KBest_A_CO_shap_importance.csv")
    return df


def calculate_feature_importance(shap_df):
    """Calculate feature importance as mean absolute SHAP values"""
    numeric_cols = shap_df.select_dtypes(include=[np.number]).columns

    importance_dict = {}
    for col in numeric_cols:
        importance_dict[col] = np.mean(np.abs(shap_df[col]))

    return importance_dict


def filter_important_features(importance_dict, threshold=0.005):
    """Filter features with importance above threshold"""
    important_features = {k: v for k, v in importance_dict.items() if v >= threshold}
    return dict(sorted(important_features.items(), key=lambda x: x[1], reverse=True))


def create_beeswarm_positions(shap_values, max_display_size=2000):
    """Create beeswarm positions with proper density-based spacing"""
    if len(shap_values) > max_display_size:
        indices = np.random.choice(len(shap_values), max_display_size, replace=False)
        shap_values = shap_values[indices]

    # Sort by SHAP values
    order = np.argsort(shap_values)
    shap_sorted = shap_values[order]

    # Create bins for positioning
    bin_size = 0.01  # Adjust this to control density
    bins = np.arange(shap_sorted.min() - bin_size, shap_sorted.max() + bin_size, bin_size)

    positions = np.zeros_like(shap_sorted, dtype=float)

    for i in range(len(bins) - 1):
        mask = (shap_sorted >= bins[i]) & (shap_sorted < bins[i + 1])
        count = np.sum(mask)

        if count > 0:
            # Create symmetric positioning around y=0
            if count == 1:
                positions[mask] = 0
            else:
                spacing = min(0.8 / count, 0.15)  # Max spacing to avoid overlap
                pos_range = np.linspace(-spacing * (count - 1) / 2,
                                        spacing * (count - 1) / 2, count)
                positions[mask] = pos_range

    # Restore original order
    final_positions = np.zeros_like(positions)
    final_positions[order] = positions

    return shap_values, final_positions


def create_shap_beeswarm_plot(shap_df, features, page_num, total_pages, feature_values_df=None):
    """Create SHAP-style beeswarm plot"""
    n_features = len(features)

    # Set up the plot with proper SHAP styling
    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.4)))

    # SHAP color scheme: blue (low) to red (high)
    colors = ['#1f77b4', '#ff7f0e', '#d62728']  # Blue to orange to red
    cmap = LinearSegmentedColormap.from_list('shap', colors, N=256)

    y_positions = []
    feature_labels = []

    for i, feature in enumerate(features):
        y_pos = len(features) - i - 1
        y_positions.append(y_pos)
        feature_labels.append(feature)

        shap_values = shap_df[feature].values

        # Remove any NaN values
        valid_mask = ~np.isnan(shap_values)
        shap_values = shap_values[valid_mask]

        if len(shap_values) == 0:
            continue

        # Get feature values for coloring
        if feature_values_df is not None and feature in feature_values_df.columns:
            feature_vals = feature_values_df[feature].values[valid_mask]
            # Normalize feature values for coloring
            if len(np.unique(feature_vals)) > 1:
                norm_vals = (feature_vals - np.min(feature_vals)) / (np.max(feature_vals) - np.min(feature_vals))
            else:
                norm_vals = np.ones_like(feature_vals) * 0.5
        else:
            # Use SHAP values themselves for coloring (absolute values normalized)
            abs_shap = np.abs(shap_values)
            if len(np.unique(abs_shap)) > 1:
                norm_vals = (abs_shap - np.min(abs_shap)) / (np.max(abs_shap) - np.min(abs_shap))
            else:
                norm_vals = np.ones_like(abs_shap) * 0.5

        # Create beeswarm positions
        display_shap, y_offsets = create_beeswarm_positions(shap_values)

        if feature_values_df is not None and feature in feature_values_df.columns:
            # If we subsampled, also subsample the color values
            if len(display_shap) < len(norm_vals):
                indices = np.random.choice(len(norm_vals), len(display_shap), replace=False)
                display_norm_vals = norm_vals[indices]
            else:
                display_norm_vals = norm_vals
        else:
            abs_display_shap = np.abs(display_shap)
            if len(np.unique(abs_display_shap)) > 1:
                display_norm_vals = (abs_display_shap - np.min(abs_display_shap)) / (
                            np.max(abs_display_shap) - np.min(abs_display_shap))
            else:
                display_norm_vals = np.ones_like(abs_display_shap) * 0.5

        # Plot points
        scatter = ax.scatter(display_shap, y_pos + y_offsets,
                             c=display_norm_vals, cmap=cmap,
                             s=16, alpha=0.7, edgecolors='none')

    # Customize the plot to match SHAP style
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_labels, fontsize=10)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
    ax.set_title(f'SHAP Summary Plot - Page {page_num} of {total_pages}',
                 fontsize=14, fontweight='bold', pad=20)

    # Add vertical line at x=0
    ax.axvline(x=0, color='#999999', linestyle='-', alpha=0.8, linewidth=1)

    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#666666')
    ax.set_facecolor('white')

    # Set y-axis limits with some padding
    ax.set_ylim(-0.5, len(features) - 0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=20)
    cbar.set_label('Feature value', rotation=270, labelpad=20, fontsize=10)
    cbar.ax.set_yticklabels(['Low', '', '', '', 'High'])
    cbar.outline.set_visible(False)

    # Remove grid
    ax.grid(False)

    plt.tight_layout()
    return fig


def generate_shap_beeswarm_plots(csv_path, feature_values_csv=None, importance_threshold=0.005,
                                 features_per_page=15, output_prefix='shap_beeswarm'):
    """
    Generate SHAP-style beeswarm plots

    Parameters:
    csv_path: str - Path to SHAP values CSV
    feature_values_csv: str - Optional path to feature values CSV for coloring
    importance_threshold: float - Minimum feature importance to include
    features_per_page: int - Number of features per plot
    output_prefix: str - Prefix for output files
    """

    print("Loading SHAP data...")
    shap_df = load_shap_data(csv_path)

    feature_values_df = None
    if feature_values_csv:
        print("Loading feature values for coloring...")
        feature_values_df = pd.read_csv(feature_values_csv)

    print("Calculating feature importance...")
    importance_dict = calculate_feature_importance(shap_df)

    important_features = filter_important_features(importance_dict, importance_threshold)

    print(f"Found {len(important_features)} features with importance >= {importance_threshold}")
    print("Top 10 most important features:")
    for i, (feature, importance) in enumerate(list(important_features.items())[:10]):
        print(f"  {i + 1}. {feature}: {importance:.6f}")

    if len(important_features) == 0:
        print("No features meet the importance threshold!")
        return

    feature_list = list(important_features.keys())
    total_pages = math.ceil(len(feature_list) / features_per_page)

    print(f"\nGenerating {total_pages} SHAP beeswarm plot(s)...")

    for page in range(total_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, len(feature_list))
        page_features = feature_list[start_idx:end_idx]

        print(f"Creating plot {page + 1}/{total_pages} with {len(page_features)} features...")

        fig = create_shap_beeswarm_plot(shap_df, page_features, page + 1, total_pages, feature_values_df)

        output_filename = f"{output_prefix}_page_{page + 1}.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_filename}")

        plt.show()
        plt.close()

    print("\nAll SHAP beeswarm plots generated successfully!")

    print(f"\nSummary:")
    print(f"- Total features processed: {len(shap_df.columns)}")
    print(f"- Features above threshold ({importance_threshold}): {len(important_features)}")
    print(f"- Plots generated: {total_pages}")
    print(f"- Features per plot: {features_per_page}")


# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file paths
    shap_csv_path = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final\fabOF_results_v8.4.0_TUNED_PIPELINE_FIXED\shap_analysis\Tuned_KBest_A_CO_shap_values.csv"  # Path to your SHAP values CSV
    #feature_values_csv_path = "feature_values.csv"  # Optional: path to feature values CSV

    # Generate the plots
    generate_shap_beeswarm_plots(
        csv_path=shap_csv_path,
        #feature_values_csv=feature_values_csv_path,
        importance_threshold=0.005,
        features_per_page=15,
        output_prefix='shap_beeswarm'
    )

    # If you only have SHAP values (no separate feature values file):
    # generate_shap_beeswarm_plots(
    #     csv_path="shap_values.csv",
    #     feature_values_csv=None,
    #     importance_threshold=0.005,
    #     features_per_page=15,
    #     output_prefix='shap_beeswarm'
    # )