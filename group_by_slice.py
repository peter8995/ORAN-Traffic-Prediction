import os
import shutil
import argparse
import re

# UEs belong to different traffic classes:
# - eMBB UEs (slice 0): 3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48
# - MTC UEs (slice 1): 4, 7, 11, 14, 18, 21, 25, 28, 32, 35, 39, 42, 46, 49
# - URLLC UEs (slice 2): 2, 5, 9, 12, 16, 19, 23, 26, 30, 33, 37, 40, 44, 47

# Node mappings to UEs (To figure out the UE from the IMSI)
# IMSI format: 1 0 1 0 1 2 3 4 5 6 0 {UE_ID}
# The last 2 digits of the IMSI correspond to the UE ID.

UE_MAPPING = {
    'embb': [3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48],
    'mtc': [4, 7, 11, 14, 18, 21, 25, 28, 32, 35, 39, 42, 46, 49],
    'urllc': [2, 5, 9, 12, 16, 19, 23, 26, 30, 33, 37, 40, 44, 47]
}

def get_slice_type(ue_id):
    for slice_name, ues in UE_MAPPING.items():
        if ue_id in ues:
            return slice_name
    return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Group dataset by slices (eMBB, MTC, URLLC).")
    parser.add_argument("--dir", type=str, default="Dataset/colosseum-oran-coloran-dataset", help="Target directory to process.")
    parser.add_argument("--execute", action="store_true", help="Actually move files. If not set, runs in dry-run mode.")
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    print(f"Processing directory: {target_dir}")
    if not args.execute:
        print("--- DRY RUN MODE: No files will be actually moved ---")
        print("Run with --execute to perform operations.\n")

    stats = {
        'embb': 0,
        'mtc': 0,
        'urllc': 0,
        'unknown': 0
    }

    # Regex to extract the UE ID from the filename like "1010123456021_metrics.csv"
    # Assuming IMSI format, the last 2 digits before "_metrics.csv" is the UE ID
    filename_pattern = re.compile(r'10101234560(\d{2})_metrics\.csv')

    for root, dirs, files in os.walk(target_dir):
        # We don't want to process files if they are already inside the slice folders
        if any(sc in root for sc in ['/embb', '/mtc', '/urllc']):
             continue

        for file in files:
            if file.lower().endswith('_metrics.csv'):
                current_path = os.path.join(root, file)

                # extract UE ID
                match = filename_pattern.search(file)
                if match:
                    ue_id = int(match.group(1))
                    slice_type = get_slice_type(ue_id)
                else:
                    # if filename doesn't match standard IMSI, let's just mark unknown
                    slice_type = 'unknown'

                stats[slice_type] += 1

                if slice_type != 'unknown' and args.execute:
                    # Create the slice directory if it doesn't exist inside the current experiment folder
                    # Basically moving from: tr1/exp1/bs3/1010123456021_metrics.csv
                    # to: tr1/exp1/bs3/mtc/1010123456021_metrics.csv
                    slice_dir = os.path.join(root, slice_type)
                    if not os.path.exists(slice_dir):
                        os.makedirs(slice_dir)
                    
                    new_path = os.path.join(slice_dir, file)
                    try:
                        shutil.move(current_path, new_path)
                    except Exception as e:
                        print(f"Failed to move {current_path}: {e}")

    print("--- Summary ---")
    for s_type, count in stats.items():
        print(f"Files for {s_type.ljust(7)}: {count}")
    
    if args.execute:
        print("\nSuccessfully moved files into slice-specific subdirectories.")

if __name__ == "__main__":
    main()
