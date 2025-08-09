import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score, f1_score,
    accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, cohen_kappa_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Set your columns
name_column = 'Inputfile'
score_column = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
labels = [1, 2, 3, 4]
k_best = 200

class_weights_list = [
    {1: 10, 2: 1, 3: 1, 4: 2},      # For A_WO
    {1: 3, 2: 1, 3: 1, 4: 2},       # For A_GR
    {1: 6, 2: 1, 3: 1, 4: 3},       # For A_CO
    {1: 3, 2: 1, 3: 1, 4: 1}        # For A_TT
]

base_dir = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold")
summary_rows = []
confusion_matrices = {col: np.zeros((len(labels), len(labels)), dtype=int) for col in score_column}

for fold_num in range(1, 11):
    fold_dir = base_dir / f"fold_{fold_num}"
    train_path = fold_dir / f"fold{fold_num}tr.csv"
    test_path = fold_dir / f"fold{fold_num}te.csv"
    output_path = fold_dir / f"w200RF_RegressionResults_fold{fold_num}2.txt"

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

    for i, col in enumerate(score_column):
        class_weights = class_weights_list[i]
        y_train_col = y_train[:, i]
        y_test_col = y_test[:, i]

        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k_best, X_train_scaled.shape[1]))
        X_train_kbest = selector.fit_transform(X_train_scaled, y_train_col)
        X_test_kbest = selector.transform(X_test_scaled)

        # Oversample
        ros = RandomOverSampler(random_state=42)
        X_train_bal, y_train_bal = ros.fit_resample(X_train_kbest, y_train_col)

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42, n_jobs = -1)
        clf.fit(X_train_bal, y_train_bal)

        # Predict
        y_pred = clf.predict(X_test_kbest)

        # Metrics
        y_true = y_test_col
        y_pred_fold = y_pred

        mse = mean_squared_error(y_true, y_pred_fold)
        r2 = r2_score(y_true, y_pred_fold)
        f1 = f1_score(y_true, y_pred_fold, labels=labels, average='macro')
        acc = accuracy_score(y_true, y_pred_fold)
        prec = precision_score(y_true, y_pred_fold, labels=labels, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred_fold, labels=labels, average='macro', zero_division=0)
        qwk = cohen_kappa_score(y_true, y_pred_fold, labels=labels, weights='quadratic')
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
        confmat = confusion_matrix(y_true, y_pred_fold, labels=labels)
        confusion_matrices[col] += confmat

        # Write results for this target
        with open(output_path, "a") as f:  # Use "a" to append for each target
            print(f"\n{'=' * 20} Fold {fold_num} - {col} {'=' * 20}\n", file=f)
            print("MSE:", mse, file=f)
            print("R^2:", r2, file=f)
            print("F1 (macro):", f1, file=f)
            print("Accuracy:", acc, file=f)
            print("Precision (macro):", prec, file=f)
            print("Recall (macro):", rec, file=f)
            print("QWK:", qwk, file=f)
            print(classification_report(y_true, y_pred_fold, labels=labels, zero_division=0), file=f)
            print("Confusion Matrix:\n", confmat, file=f)
            print("First 20 true values:", y_true[:20], file=f)
            print("First 20 predicted values:", y_pred_fold[:20], file=f)
            print("-" * 40, file=f)

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = base_dir / "w200RF_Standardised_Regression_Metrics_Summary2.csv"
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
agg_csv_path = base_dir / "w200RF_Standardised_Regression_Metrics_Aggregated2.csv"
agg_df.to_csv(agg_csv_path, index=False)

# Save all summed confusion matrices to a text file
confmat_path = base_dir / "w200RF_Standardised_AllFolds_SummedConfusionMatrices2.txt"
with open(confmat_path, "w") as f:
    for col in score_column:
        print(f"Summed Confusion Matrix for {col}:", file=f)
        print(confusion_matrices[col], file=f)
        print("-" * 40, file=f)

print(f"Per-fold results saved in each fold's directory.")
print(f"Summary metrics saved to {summary_csv_path}")
print(f"Aggregated metrics saved to {agg_csv_path}")
print(f"Summed confusion matrices saved to {confmat_path}")
