# =========================================================
# RUN THIS SCRIPT ONCE to convert raw files into a single binary dataset.
# =========================================================
import os
import re
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DESCRIPTOR_ROOT = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/hydrogen_atom/sum_predictor_model/descriptor_output"
TARGET_SUBFOLDERS = ["CoGe_100_GBs", "CoGe_110_GBs", "CoGe_111_GBs"]
#Change name of output file as needed. You can have many unique ones if you just change the name each time 
#you run this script with different data processing steps.
OUTPUT_FILENAME = "New_coge_dataset_processed.npz"

def parse_information_file(filepath):
    sym_map = {}
    if not os.path.exists(filepath):
        print(f"  [!] Missing Information file: {filepath}")
        return sym_map
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r'^(\d+):\s*(\d+)\s*symmetry operations', line)
            if match:
                sym_map[int(match.group(1))] = int(match.group(2))
    print(f"  [*] Parsed {len(sym_map)} labels from {os.path.basename(os.path.dirname(filepath))}/Information.txt")
    return sym_map

def load_raw_dataset():
    features, targets, labels, paths = [], [], [], []
    print(f"Scanning {DESCRIPTOR_ROOT}...")
    if not os.path.exists(DESCRIPTOR_ROOT): return None

    for sub in TARGET_SUBFOLDERS:
        full_path = os.path.join(DESCRIPTOR_ROOT, sub)
        if not os.path.exists(full_path): continue
        
        print(f"\nProcessing {sub}...")
        sym_map = parse_information_file(os.path.join(full_path, "Information.txt"))
        
        file_paths = [os.path.join(dp, "soap_descriptor.npy") for dp, _, fn in os.walk(full_path) if "soap_descriptor.npy" in fn]

        if not file_paths: continue

        print(f"  [*] Found {len(file_paths)} descriptors. Loading & Processing...")
        for path in tqdm(file_paths, leave=False):
            try:
                parts = path.split(os.sep)
                sym_folder = [p for p in parts if p.startswith('sym')][0]
                interface_idx = int(sym_folder.replace('sym', ''))
                
                # --- Load Descriptor & Compute Target ---
                local_desc = np.load(path)
                global_desc = np.mean(local_desc, axis=0)
                target_val = np.sum(global_desc)
                # --------------------------
                
                features.append(global_desc)
                targets.append(target_val)
                labels.append(sym_map.get(interface_idx, -1))
                paths.append(os.path.dirname(path))
            except Exception: pass

    return np.array(features), np.array(targets), np.array(labels), np.array(paths)

if __name__ == "__main__":
    print("--- STAGE 1: RAW DATA PROCESSING ---")
    X, y, labels, paths = load_raw_dataset()
    
    if X is None or len(X) == 0:
        print("Failed to load data.")
        exit()

    print(f"\nFinal Dataset Shape: {X.shape}")
    print(f"Saving processed data to {OUTPUT_FILENAME}...")
    
    # Save all arrays into one compressed binary file
    np.savez_compressed(
        OUTPUT_FILENAME,
        features=X,
        targets=y,
        sym_labels=labels,
        paths=paths
    )
    print("Done! You can now run the analysis scripts instantly.")