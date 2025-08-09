import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score, f1_score,
    accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, cohen_kappa_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

name_column = 'Inputfile'
score_column = ['A_WO', 'A_GR', 'A_CO', 'A_TT']
labels = [1, 2, 3, 4]
k_best = 150

base_dir = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold")
summary_rows = []
confusion_matrices = {col: np.zeros((len(labels), len(labels)), dtype=int) for col in score_column}

# For feature tracking
all_feature_names = None
feature_usage = {col: [] for col in score_column}

for fold_num in range(1, 11):
    fold_dir = base_dir / f"fold_{fold_num}"
    train_path = fold_dir / f"fold{fold_num}tr.csv"
    test_path = fold_dir / f"fold{fold_num}te.csv"
    output_path = fold_dir / f"fsMLP_Results_fold{fold_num}.txt"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[name_column] + score_column)
    X_test = test_df.drop(columns=[name_column] + score_column)
    y_train = train_df[score_column].values
    y_test = test_df[score_column].values

    if all_feature_names is None:
        all_feature_names = X_train.columns.tolist()

    y_train_int = np.round(y_train).astype(int)
    y_test_int = np.round(y_test).astype(int)

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(output_path, "w") as f:
        print(f"\n{'=' * 20} Fold {fold_num} {'=' * 20}\n", file=f)
        for i, col in enumerate(score_column):
            y_true = y_test_int[:, i]
            y_train_col = y_train_int[:, i]

            # K-best feature selection
            selector = SelectKBest(score_func=f_classif, k=min(k_best, X_train_scaled.shape[1]))
            X_train_kbest = selector.fit_transform(X_train_scaled, y_train_col)
            X_test_kbest = selector.transform(X_test_scaled)
            selected_mask = selector.get_support()
            selected_features = [fname for fname, sel in zip(all_feature_names, selected_mask) if sel]
            not_selected_features = [fname for fname, sel in zip(all_feature_names, selected_mask) if not sel]
            feature_usage[col].append(selected_mask)

            # Save per-fold, per-target feature usage
            pd.DataFrame({'Used_Features': selected_features}).to_csv(
                fold_dir / f"{col}_Fold{fold_num}_Used_Features.csv", index=False)
            pd.DataFrame({'Not_Used_Features': not_selected_features}).to_csv(
                fold_dir / f"{col}_Fold{fold_num}_Not_Used_Features.csv", index=False)



            # Two-layer neural network: one hidden layer (e.g., 64 units), output layer
            clf = MLPClassifier(hidden_layer_sizes=(48,), max_iter=200000, random_state=42)
            clf.fit(X_train_kbest, y_train_col)
            y_pred = clf.predict(X_test_kbest)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, labels=labels, average='macro')
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            qwk = cohen_kappa_score(y_true, y_pred, labels=labels, weights='quadratic')
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
            confmat = confusion_matrix(y_true, y_pred, labels=labels)
            confusion_matrices[col] += confmat
            print(f"\n--- Results for {col} ---", file=f)
            print("MSE:", mse, file=f)
            print("R^2:", r2, file=f)
            print("F1 (macro):", f1, file=f)
            print("Accuracy:", acc, file=f)
            print("Precision (macro):", prec, file=f)
            print("Recall (macro):", rec, file=f)
            print("QWK:", qwk, file=f)
            print(classification_report(y_true, y_pred, labels=labels, zero_division=0), file=f)
            print("Confusion Matrix:\n", confmat, file=f)
            print("First 20 true values:", y_true[:20], file=f)
            print("First 20 predicted values:", y_pred[:20], file=f)
            print("-" * 40, file=f)
        print(f"{'=' * 50}\n", file=f)

summary_df = pd.DataFrame(summary_rows)
summary_csv_path = base_dir / "fsMLP_Metrics_Summary.csv"
summary_df.to_csv(summary_csv_path, index=False)

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
agg_csv_path = base_dir / "fsMLP_Metrics_AggregatedSTD.csv"
agg_df.to_csv(agg_csv_path, index=False)

confmat_path = base_dir / "fsMLP_AllFolds_SummedConfusionMatrices.txt"
with open(confmat_path, "w") as f:
    for col in score_column:
        print(f"Summed Confusion Matrix for {col}:", file=f)
        print(confusion_matrices[col], file=f)
        print("-" * 40, file=f)

# --- Aggregate feature usage across folds ---
for col in score_column:
    usage_matrix = np.array(feature_usage[col])  # shape: (n_folds, n_features)
    always_used = []
    sometimes_used = []
    never_used = []
    for idx, fname in enumerate(all_feature_names):
        used_count = usage_matrix[:, idx].sum()
        if used_count == usage_matrix.shape[0]:
            always_used.append(fname)
        elif used_count == 0:
            never_used.append(fname)
        else:
            sometimes_used.append(fname)
    pd.DataFrame({'Always_Used_Features': always_used}).to_csv(base_dir / f"{col}_Always_Used_Features.csv", index=False)
    pd.DataFrame({'Sometimes_Used_Features': sometimes_used}).to_csv(base_dir / f"{col}_Sometimes_Used_Features.csv", index=False)
    pd.DataFrame({'Never_Used_Features': never_used}).to_csv(base_dir / f"{col}_Never_Used_Features.csv", index=False)

print(f"Per-fold results saved in each fold's directory.")
print(f"Summary metrics saved to {summary_csv_path}")
print(f"Aggregated metrics saved to {agg_csv_path}")
print(f"Summed confusion matrices saved to {confmat_path}")
print(f"Feature selection summary saved to {base_dir}")

plt.plot(clf.loss_curve_)
plt.title('MLP Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()