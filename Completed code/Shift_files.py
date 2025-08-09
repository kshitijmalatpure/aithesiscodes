import os
import pandas as pd

# Path to your input CSV
csv_path = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\Usable_data\reduced_allfeats1.csv'
base_folder = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\10-fold'

# Loop through all 10 folds
for fold_num in range(1, 11):
    for split in ['train', 'test']:
        # Set folder and output file names
        folder_path = os.path.join(base_folder, f'fold_{fold_num}', split)
        output_csv = os.path.join(base_folder, f'fold_{fold_num}', f'fold{fold_num}{"tr" if split=="train" else "te"}.csv')

        # Skip if folder doesn't exist
        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # Load CSV and round
        df = pd.read_csv(csv_path)
        df = df.round(3)

        # Get the list of files in the folder
        files_in_folder = set(os.listdir(folder_path))

        # Filter rows where the filename in column 1 exists in the folder
        filtered_df = df[df.iloc[:, 0].isin(files_in_folder)]

        # Save to new CSV
        filtered_df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
