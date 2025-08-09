import csv

# Read filenames from 'ftd' (strip whitespace)
with open('ftd.txt', 'r') as f:
    filenames = set(line.strip() for line in f if line.strip())

with open('data_with_splits.csv', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    print(reader.fieldnames)

# Open the original CSV and the output CSV
with open('data_with_splits.csv', newline='', encoding='utf-8') as infile, \
     open('matched_rows.csv', 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row['kandidaatcode1'] in filenames:
            writer.writerow(row)

print("Done! Matching rows have been written to matched_rows.csv")