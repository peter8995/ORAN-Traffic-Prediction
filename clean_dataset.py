import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Keep only *_metrics.csv files, move them up one level, and remove empty directories.")
    parser.add_argument("--dir", type=str, default="Dataset/colosseum-oran-coloran-dataset", help="Target directory to process.")
    parser.add_argument("--execute", action="store_true", help="Actually modify files. If not set, runs in dry-run mode.")
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    print(f"Processing directory: {target_dir}")
    if not args.execute:
        print("--- DRY RUN MODE: No files will be actually modified ---")
        print("Run with --execute to perform operations.\n")

    deleted_non_metrics = 0
    kept_metrics = 0
    moved_metrics = 0
    conflict_count = 0
    empty_removed = 0

    # Step 1: Delete non-metrics files
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('_metrics.csv'):
                kept_metrics += 1
            else:
                if args.execute:
                    try:
                        os.remove(file_path)
                        deleted_non_metrics += 1
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                else:
                    deleted_non_metrics += 1

    # Step 2: Move metrics files up one level and clean empty directories
    # Use topdown=False so we can process children before their parent directories
    for root, dirs, files in os.walk(target_dir, topdown=False):
        # Move files
        for file in files:
            if file.lower().endswith('_metrics.csv'):
                current_path = os.path.join(root, file)
                
                # Prevent moving files out of the main target directory
                if os.path.abspath(root) == os.path.abspath(target_dir):
                    continue

                parent_dir = os.path.dirname(root)
                new_path = os.path.join(parent_dir, file)
                
                if os.path.exists(new_path) and current_path != new_path:
                     conflict_count += 1
                     continue

                if args.execute:
                    try:
                        shutil.move(current_path, new_path)
                        moved_metrics += 1
                    except Exception as e:
                        print(f"Failed to move {current_path}: {e}")
                else:
                    moved_metrics += 1
        
        # Remove empty directories
        if args.execute and os.path.abspath(root) != os.path.abspath(target_dir):
            try:
                if not os.listdir(root):  # Directory is empty
                    os.rmdir(root)
                    empty_removed += 1
            except Exception as e:
                pass

    print("--- Summary ---")
    print(f"Files kept          (*_metrics.csv): {kept_metrics}")
    print(f"Files deleted       (non-metrics)  : {deleted_non_metrics}")
    print(f"Files moved up one level           : {moved_metrics}")
    print(f"Conflicts (not moved)              : {conflict_count}")
    if args.execute:
        print(f"Empty directories removed          : {empty_removed}")

if __name__ == "__main__":
    main()

# python3 clean_dataset.py --dir Dataset/colosseum-oran-coloran-dataset
# python3 clean_dataset.py --dir Dataset/colosseum-oran-coloran-dataset --execute