import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('C:\\Research\\AI Folder\\Thesis\\Data\\Main\\Usable_data\\full_set.csv')

# Separate features and target
X = df.drop(columns='kandidaatcode')  # Replace 'label' with your actual target column name
y = df['kandidaatcode']

# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: 10% test, 10% eval (i.e., split the remaining 20%)
X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Optional: Combine back into DataFrames if needed
train_df = X_train.copy()
train_df['kandidaatcode'] = y_train

test_df = X_test.copy()
test_df['kandidaatcode'] = y_test

eval_df = X_eval.copy()
eval_df['kandidaatco  de'] = y_eval

# Combine all splits into a single DataFrame
final_df = pd.concat([train_df, test_df, eval_df], ignore_index=True)

# Save to a new CSV
final_df.to_csv('C:\\Research\\AI Folder\\Thesis\\Data\\Main\\Usable_data\\data_with_splits.csv', index=False)

# Confirm the splits
print(f"Train: {len(train_df)}, Test: {len(test_df)}, Eval: {len(eval_df)}")

