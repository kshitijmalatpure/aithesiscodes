import os
import pandas as pd
import re

root_dir = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\Usable_data\STRT\Language Tool Errors'
rows = []

# Regex patterns for each field
patterns = {
    'Rule ID': re.compile(r'Rule ID:\s*(.*)'),
    'Message': re.compile(r'Message:\s*(.*)'),
    'Replacements': re.compile(r'Replacements:\s*\[(.*)\]'),
    'Context': re.compile(r'Context:\s*(.*)'),
    'Offset': re.compile(r'Offset:\s*(.*)'),
    'Error Length': re.compile(r'Error Length:\s*(.*)'),
    'Sentence': re.compile(r'Sentence:\s*(.*)')
}

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(subdir, file)
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
            row = {
                'subfolder': os.path.relpath(subdir, root_dir),
                'filename': file
            }
            for key, pat in patterns.items():
                match = pat.search(content)
                row[key] = match.group(1).strip() if match else ''
            rows.append(row)

# Create DataFrame and save to CSV
df = pd.DataFrame(rows)
df.to_csv('spellcheck_results.csv', index=False)
print("CSV saved as spellcheck_results.csv")