import os
import requests
import json

# CONFIGURATION
FOLDER_PATH = r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs\LLM_WO"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"  # Keep this secret!


def get_text_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.txt')]

with open(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs\WOPrompt.txt", "r", encoding="utf-8") as f:

    long_prompt = f.read()
def send_to_openrouter(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "google/gemma-3-27b-it:free",  # Change model if needed
        "messages": [
            {"role": "system", "content": long_prompt},
            {"role": "user", "content": f"Essay:\n{text}\n\nONLY OUTPUT THE SCORE (1, 2, 3, or 4). DO NOT EXPLAIN."}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def main():

    files = get_text_files(FOLDER_PATH)
    for filename in files:
        filepath = os.path.join(FOLDER_PATH, filename)
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        print(f"Sending {filename}...")
        try:
            response = send_to_openrouter(content)
            print("Response:", response)
            # Save the response to a file
            with open(filepath + ".response.json", "w", encoding="utf-8") as out:
                json.dump(response, out, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error with {filename}: {e}")

if __name__ == "__main__":
    main()