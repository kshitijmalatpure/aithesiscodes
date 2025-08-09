import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, classification_report, f1_score,cohen_kappa_score, root_mean_squared_error

# Load the CSV file
df = pd.read_csv(r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\Usable_data\STRT\dsGR.csv')

# Assume the columns are 'predicted' and 'actual'
y_pred = df['Content Number']
y_true = df['A_GR']
labels=[1, 2, 3, 4]
# Calculate precision for each class
precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

from collections import Counter
print("True label counts:", Counter(y_true))
print("Predicted label counts:", Counter(y_pred))

# Precision, Recall (macro = unweighted average over all classes)
precision = precision_score(y_true, y_pred, average='macro', labels=[1, 2, 3, 4], zero_division=0)
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro', labels=[1,2,3,4], zero_division=0)
rmse = root_mean_squared_error(y_true, y_pred)
qwk = cohen_kappa_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
# Print precision for each class
for class_label, precision in zip([1, 2, 3, 4], precision_per_class):
    print(f'Precision for Class {class_label}: {precision:.3f}')
print(f"Precision (macro): {precision:.3f}")
print(f"Recall (macro): {recall:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1:{f1:.3f}")
print(f"qwk:{qwk:.3f}")
print(f"RootMSE:{rmse:.3f}")
# Metrics
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)