"""
Batch-score Dutch B2 essays on Vocabulary (CEFR rubric) using multiple models.

Usage:
1.  Save the assessment prompt as a .txt file.

2.  Put the prompt's path, the essay-folder path, and a list of models
    in the CONFIG section below.

3.  Set your OpenRouter API key as an environment variable
    called OPENROUTER_API_KEY (recommended) or paste it directly.

4.  Run the script

5.  Results will be saved in a main results folder, with a subfolder
    for each model's outputs.
"""

import os
import json
import requests
from pathlib import Path
import time

TIMEOUT = 90
# ─── CONFIG ──────────────────────────────────────────────────────────────────
# The folder containing the .txt essay files to be scored.
ESSAY_DIR   = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Missing files")

# The path to your system prompt file.
PROMPT_FILE = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\Prompts\WOPrompt.txt")

# The folder where all results will be saved.
RESULTS_DIR = Path(r"C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\A_WO\scoring_results_oss")

# List of models to iterate over. Add or remove models as needed.
MODELS      = [
    "openai/gpt-oss-120b",
]

# --- API Configuration ---
API_URL     = ""
# To set it, run this in your terminal: export OPENROUTER_API_KEY="sk-or-..."
API_KEY     = os.getenv("")
# ─────────────────────────────────────────────────────────────────────────────

def load_system_prompt(path: Path) -> str:
    """Return the entire robust-assessment prompt as a string."""
    return path.read_text(encoding="utf-8").strip()

def list_essay_files(folder: Path):
    """Yield all .txt files inside *folder (non-recursive)."""
    print(f"Listing essays from: {folder}")
    yield from folder.glob("*.txt")

def score_essay(system_prompt: str, essay_text: str, model_name: str) -> dict:
    """Call the OpenRouter API for a specific model and return the raw JSON response."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name, # Use the model name passed as an argument
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Essay:\n{essay_text}"},
        ],
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def main() -> None:
    if not API_KEY:
        raise EnvironmentError(
            "API key not found. Set OPENROUTER_API_KEY in your environment "
            "(e.g. export OPENROUTER_API_KEY=sk-...)"
        )

    # Load prompt and essay list once
    system_prompt = load_system_prompt(PROMPT_FILE)
    essay_files = list(list_essay_files(ESSAY_DIR))
    print(f"Found {len(essay_files)} essays to score with {len(MODELS)} models.")

    # Create the main results directory if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Main Loop: Iterate over each model ---
    for model_name in MODELS:
        print(f"\n===== Processing Model: {model_name} =====")

        # Create a sanitized, safe directory name from the model name
        # (e.g., "qwen/qwen3-235b-a22b" becomes "qwen_qwen3-235b-a22b")
        sanitized_model_name = model_name.replace("/", "_").replace(":", "-")
        model_output_dir = RESULTS_DIR / sanitized_model_name
        model_output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved in: {model_output_dir}")

        # --- Inner Loop: Iterate over each essay for the current model ---
        for essay_path in essay_files:
            print(f"  → Scoring {essay_path.name} ...")

            # Construct the output path to check if it already exists
            out_file = model_output_dir / essay_path.with_suffix(".response.json").name
            if out_file.exists():
                print(f"    ✓ Already exists, skipping: {out_file.name}")
                continue

            essay = essay_path.read_text(encoding="utf-8", errors="replace")

            try:
                # Pass the current model_name to the scoring function
                result = score_essay(system_prompt, essay, model_name)
            except Exception as err:
                print(f"    ✖ Error: {err}")
                # Optional: you could write an error file to track failures
                error_file = out_file.with_suffix(".error.txt")
                error_file.write_text(str(err), encoding="utf-8")
                continue # Move to the next essay

            # Save the successful response
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    ✓ Saved: {out_file.name}")

            # Optional: Add a small delay to respect API rate limits
            time.sleep(1)

    print("\n===== All models and essays processed. =====")


if __name__ == "__main__":
    main()