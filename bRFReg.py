import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score, f1_score,
    accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, cohen_kappa_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Set your columns
name_column = 'Inputfile'
score_column = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
labels = [0,1]

base_dir = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold")
summary_rows = []
confusion_matrices = {col: np.zeros((len(labels), len(labels)), dtype=int) for col in score_column}

for fold_num in range(1, 11):
    fold_dir = base_dir / f"fold_{fold_num}"
    train_path = fold_dir / f"fold{fold_num}trb.csv"
    test_path = fold_dir / f"fold{fold_num}teb.csv"
    output_path = fold_dir / f"bRF_Standardised_Results_fold{fold_num}.txt"

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Features and targets
    X_train = train_df.drop(columns=[name_column] + score_column)
    X_test = test_df.drop(columns=[name_column] + score_column)
    y_train = train_df[score_column].values
    y_test = test_df[score_column].values

    # Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train_scaled, y_train)

    # Predict
    y_pred = reg.predict(X_test_scaled)

    # Clip predictions to valid score range [0, 1]
    y_pred_clipped = np.clip(y_pred, 0, 1)

    # Round clipped predictions to nearest integer for classification metrics
    y_pred_int = np.round(y_pred_clipped).astype(int)
    y_test_int = np.round(y_test).astype(int)

    with open(output_path, "w") as f:
        print(f"\n{'=' * 20} Fold {fold_num} {'=' * 20}\n", file=f)
        for i, col in enumerate(score_column):
            y_true = y_test[:, i]
            y_pred_fold = y_pred[:, i]
            y_pred_clipped_fold = y_pred_clipped[:, i]
            y_true_int = y_test_int[:, i]
            y_pred_int_fold = y_pred_int[:, i]

            mse = mean_squared_error(y_true, y_pred_clipped_fold)
            r2 = r2_score(y_true, y_pred_clipped_fold)
            f1 = f1_score(y_true_int, y_pred_int_fold, labels=labels, average='macro')
            acc = accuracy_score(y_true_int, y_pred_int_fold)
            prec = precision_score(y_true_int, y_pred_int_fold, labels=labels, average='macro', zero_division=0)
            rec = recall_score(y_true_int, y_pred_int_fold, labels=labels, average='macro', zero_division=0)
            if len(np.unique(y_true_int)) > 1 and len(np.unique(y_pred_int_fold)) > 1:
                qwk = cohen_kappa_score(y_true_int, y_pred_int_fold, labels=labels, weights='quadratic')
            else:
                qwk = np.nan

            summary_rows.append({
                'Fold': fold_num,
                'Score': col,
                'MSE': mse,
                'R2': r2,
                'Accuracy': acc,
                'F1': f1,
                'Precision (macro)': prec,
                'Recall (macro)': rec,
                'QWK': qwk
            })

            confmat = confusion_matrix(y_true_int, y_pred_int_fold, labels=labels)
            confusion_matrices[col] += confmat

            print(f"\n--- Results for {col} ---", file=f)
            print("MSE:", mse, file=f)
            print("R^2:", r2, file=f)
            print("F1 (macro):", f1, file=f)
            print("Accuracy:", acc, file=f)
            print("Precision (macro):", prec, file=f)
            print("Recall (macro):", rec, file=f)
            print("QWK:", qwk, file=f)
            print(classification_report(y_true_int, y_pred_int_fold, labels=labels, zero_division=0), file=f)
            print("Confusion Matrix:\n", confmat, file=f)
            print("First 20 true values:", y_true[:20], file=f)
            print("First 20 predicted values:", y_pred_fold[:20], file=f)
            print("First 20 clipped predicted values:", y_pred_clipped_fold[:20], file=f)
            print("-" * 40, file=f)
        print(f"{'=' * 50}\n", file=f)

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = base_dir / "bRF_Standardised_Regression_Metrics_Summary.csv"
summary_df.to_csv(summary_csv_path, index=False)

# Aggregate (mean and std) over folds for each score and metric
agg_rows = []
for col in score_column:
    df_col = summary_df[summary_df['Score'] == col]
    agg = {
        'Score': col,
        'MSE_mean': df_col['MSE'].mean(),
        'MSE_std': df_col['MSE'].std(),
        'R2_mean': df_col['R2'].mean(),
        'R2_std': df_col['R2'].std(),
        'Accuracy_mean': df_col['Accuracy'].mean(),
        'Accuracy_std': df_col['Accuracy'].std(),
        'F1_mean': df_col['F1'].mean(),
        'F1_std': df_col['F1'].std(),
        'Precision_mean': df_col['Precision (macro)'].mean(),
        'Precision_std': df_col['Precision (macro)'].std(),
        'Recall_mean': df_col['Recall (macro)'].mean(),
        'Recall_std': df_col['Recall (macro)'].std(),
        'QWK_mean': df_col['QWK'].mean(),
        'QWK_std': df_col['QWK'].std()
    }
    agg_rows.append(agg)

agg_df = pd.DataFrame(agg_rows)
agg_csv_path = base_dir / "bRF_Standardised_Regression_Metrics_Aggregated.csv"
agg_df.to_csv(agg_csv_path, index=False)

# Save all summed confusion matrices to a text file
confmat_path = base_dir / "bRF_Standardised_Regression_AllFolds_SummedConfusionMatrices.txt"
with open(confmat_path, "w") as f:
    for col in score_column:
        print(f"Summed Confusion Matrix for {col}:", file=f)
        print(confusion_matrices[col], file=f)
        print("-" * 40, file=f)

print(f"Per-fold results saved in each fold's directory.")
print(f"Summary metrics saved to {summary_csv_path}")
print(f"Aggregated metrics saved to {agg_csv_path}")
print(f"Summed confusion matrices saved to {confmat_path}")
