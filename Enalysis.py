import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# ==============================================================================
# --- CONFIGURATION: PLEASE UPDATE THIS SECTION ---
# ==============================================================================

# 1. Enter the full path to your prediction CSV file.
#    (Use a raw string r"..." for Windows paths to avoid errors with backslashes)
FILE_PATH = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold-final\coral_results\individual_predictions\CORAL_Small_Baseline_fold1_predictions.csv"

# 2. Define the base names of the score columns in your file.
SCORE_COLUMNS = ['A_WO', 'A_GR', 'A_CO', 'A_TT']


# ==============================================================================
# --- ANALYSIS SCRIPT ---
# ==============================================================================

def analyze_prediction_file(file_path, score_cols):
    """
    Loads a prediction CSV and calculates per-variable and aggregate metrics.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"The file was not found at the path you provided.")
        print(f"Please check that this path is correct: {file_path}")
        return

    # Prepare lists to hold all true and predicted values for aggregation
    all_true_labels = []
    all_pred_labels = []

    # Get just the filename for a clean title
    file_name = file_path.split('\\')[-1].split('/')[-1]

    print(f"\n--- Analysis for: {file_name} ---")
    print("\n**Per-Variable Accuracy:**")

    # Loop through each score column to calculate its accuracy
    for col in score_cols:
        true_col_name = f'{col}_true'
        pred_col_name = f'{col}_pred_rounded'

        # Check if the necessary columns exist in the CSV
        if true_col_name not in df.columns or pred_col_name not in df.columns:
            print(f"  - WARNING: Columns for '{col}' not found. Skipping.")
            continue

        # Get the true and predicted values
        y_true = df[true_col_name]
        y_pred = df[pred_col_name]

        # Calculate and print the accuracy for the current variable
        acc = accuracy_score(y_true, y_pred)
        print(f"  - {col:<4}: {acc:.1%}")

        # Add this variable's data to the master lists for aggregation
        all_true_labels.extend(y_true.tolist())
        all_pred_labels.extend(y_pred.tolist())

    # Check if any data was actually collected before calculating aggregate scores
    if not all_true_labels:
        print("\nCould not calculate aggregate metrics. No valid data found in the file.")
        return

    # --- Calculate Aggregate Metrics ---
    # Determine the unique labels present in the true data for scoring
    labels = sorted(list(set(all_true_labels)))

    aggregate_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    aggregate_f1 = f1_score(all_true_labels, all_pred_labels, labels=labels, average='macro', zero_division=0)
    aggregate_qwk = cohen_kappa_score(all_true_labels, all_pred_labels, labels=labels, weights='quadratic')

    # --- Print Aggregate Results ---
    print("\n**Aggregate Metrics (all variables combined):**")
    print(f"  - Aggregate Accuracy: {aggregate_accuracy:.2%}")
    print(f"  - F1-Score (Macro):   {aggregate_f1:.4f}")
    print(f"  - Quadratic Weighted Kappa (QWK): {aggregate_qwk:.4f}")
    print("-" * 50)


# This part makes the script runnable
if __name__ == "__main__":
    analyze_prediction_file(FILE_PATH, SCORE_COLUMNS)