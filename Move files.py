import os
import shutil
from collections import defaultdict

# === CONFIGURE THESE ===
txt_file_path = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Main\LLMs (3.0)\A_WO\Done.txt'  # Path to your .txt file
search_root_dir = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Missing files'  # Folder to search in (includes subfolders)
destination_dir = r'C:\Research\AI Folder\Thesis\Data\data_CTO_Kshitij\Missing files\new'  # Folder to move matching files to

# --- Configuration for log files ---
issues_log_filename = 'issues_log.txt'  # For unmatched fragments AND fragments with odd file counts
error_log_filename = 'move_errors.log'  # For exceptions encountered during move operations

# === INITIALIZATION ===
files_moved_count = 0
move_errors = []
# This dictionary will store each fragment and a list of the files it matched
# e.g., {'patient_123': ['patient_123_CT.nii', 'patient_123_RT.nii']}
fragment_to_files_map = defaultdict(list)

print("Script started...")

# Create destination directory once at the beginning
try:
    os.makedirs(destination_dir, exist_ok=True)
except OSError as e:
    print(f"FATAL: Could not create destination directory '{destination_dir}'. Error: {e}")
    exit()

# === STEP 1: Read lines from the .txt file ===
try:
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        name_fragments = [line.strip() for line in f if line.strip()]
    print(f"Successfully read {len(name_fragments)} name fragments from '{txt_file_path}'.")
except FileNotFoundError:
    print(f"FATAL: The source .txt file was not found at '{txt_file_path}'.")
    exit()

# === STEP 2: Walk through all files, find matches, and move them ===
print(f"Searching for files in '{search_root_dir}' and its subfolders...")
for root, dirs, files in os.walk(search_root_dir):
    # CRITICAL: Prevent os.walk from descending into the destination directory.
    # We do this by modifying the 'dirs' list in-place.
    if os.path.basename(destination_dir) in dirs:
        dirs.remove(os.path.basename(destination_dir))

    for file in files:
        for fragment in name_fragments:
            if fragment in file:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(destination_dir, file)

                # Track the match before attempting to move
                fragment_to_files_map[fragment].append(file)

                # Attempt to move the file, with error handling
                try:
                    # To prevent re-moving if script is re-run on a partial result
                    if not os.path.exists(dst_path):
                        # *** THE KEY CHANGE IS HERE: shutil.move instead of shutil.copy2 ***
                        shutil.move(src_path, dst_path)
                        print(f"Moved: {file}")
                        files_moved_count += 1
                    else:
                        # This case is less likely with 'move' unless the script was interrupted
                        print(f"Skipped (already exists in destination): {file}")

                except Exception as e:
                    error_message = f"FAILED to move '{src_path}' to '{dst_path}'. Reason: {e}"
                    print(f"ERROR: {error_message}")
                    move_errors.append(error_message)

                # Once a file is matched, move to the next file.
                # This prevents multiple fragments from matching the same file.
                break

# === STEP 3: Analyze results and generate logs ===
print("\n--- Operation Summary ---")
print(f"Total files moved successfully: {files_moved_count}")

# Analyze the fragment matches for issues
truly_unmatched_fragments = []
fragments_with_odd_counts = {}

all_found_fragments = fragment_to_files_map.keys()

for frag in name_fragments:
    if frag not in all_found_fragments:
        truly_unmatched_fragments.append(frag)
    else:
        matched_files = fragment_to_files_map[frag]
        if len(matched_files) % 2 != 0:
            fragments_with_odd_counts[frag] = matched_files

# Write a consolidated issues log
if truly_unmatched_fragments or fragments_with_odd_counts:
    issues_log_path = os.path.join(destination_dir, issues_log_filename)
    with open(issues_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("This log contains details of name fragments with issues.\n")
        log_file.write("========================================================\n\n")

        if truly_unmatched_fragments:
            log_file.write("[FRAGMENTS WITH NO MATCHING FILES]\n")
            log_file.write("------------------------------------\n")
            for frag in truly_unmatched_fragments:
                log_file.write(f"{frag}\n")
            log_file.write("\n")

        if fragments_with_odd_counts:
            log_file.write("[FRAGMENTS WITH AN ODD NUMBER OF FILES (Expected 2, 4, etc.)]\n")
            log_file.write("--------------------------------------------------------------\n")
            for frag, files in fragments_with_odd_counts.items():
                log_file.write(f"Fragment: {frag}  (Found {len(files)} file(s))\n")
                for f in files:
                    log_file.write(f"  - {f}\n")
            log_file.write("\n")

    print(
        f"Detected {len(truly_unmatched_fragments) + len(fragments_with_odd_counts)} fragments with issues. See log: {issues_log_path}")
else:
    print("Success! All name fragments were matched to an even number of files.")

# Write move errors to a separate log file
if move_errors:
    error_log_path = os.path.join(destination_dir, error_log_filename)
    with open(error_log_path, 'w', encoding='utf-8') as error_file:
        error_file.write("# The following files were found but could not be moved due to an error:\n")
        for error in move_errors:
            error_file.write(f"{error}\n")
    print(f"Encountered {len(move_errors)} move errors. Details saved to: {error_log_path}")

print("\nDone.")