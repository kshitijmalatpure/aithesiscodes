import os
import json
import csv
import re


SCORE_RE = re.compile(r"score\s*[:\-]?\s*([1-4])\b", re.I)

def extract_score_from_content(value):
    """
    Return an int 1-4 if `value` (a str | int | float) contains a valid score,
    otherwise None.
    """
    # Numeric value already
    if isinstance(value, (int, float)) and 1 <= value <= 4:
        return int(value)

    # Look for the score inside a string
    if isinstance(value, str):
        # explicit 'Score: 2' style
        m = SCORE_RE.search(value)
        if m:
            return int(m.group(1))

        # a string that is *only* the digit (with or without leading newline)
        m = re.match(r"^\s*\n?([1-4])\s*$", value)
        if m:
            return int(m.group(1))

    return None


def find_score(obj):
    """
    Depth-first search through `obj` (dict | list) for a `"content"` key that
    contains a score. Returns the first score found or None.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "content":
                score = extract_score_from_content(v)
                if score is not None:
                    return score
            # keep digging
            nested = find_score(v)
            if nested is not None:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = find_score(item)
            if nested is not None:
                return nested
    return None


def process_json_files(folder_path, output_csv):
    rows = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".json"):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            score = find_score(data)
            if score is not None:
                rows.append([filename, score])
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Write results
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["JSON Filename", "Score"])
        writer.writerows(rows)


if __name__ == "__main__":
    folder = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\A_GR\scoring_results_oss\openai_gpt-oss-120b"
    output = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\A_GR\GRoss.csv"
    process_json_files(folder, output)
    print(f"CSV file '{output}' generated.")
