import os
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# INPUT: The master dictionary from Phase 3
SOAP_DATA_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl"

# INPUT: The base directory to find the Information.txt files
RAW_BASE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries"

# OUTPUT: A text file listing the 200 selected structures
OUTPUT_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/results/selected_200_structures.txt"

TARGET_TOTAL_STRUCTURES = 200

import re
import os

def get_symmetry_mapping():
    """
    Crawls the Information.txt files to map each index to its symXXXXX folder and symmetry count.
    Returns: Dict[str, int] -> e.g., {"CoGe_100_GBs/sym00000": 12}
    """
    print("Parsing Information.txt files for symmetry labels...")
    sym_mapping = {}
    
    gb_types = ["CoGe_100_GBs", "CoGe_110_GBs", "CoGe_111_GBs"]
    
    # NEW BULLETPROOF REGEX
    # Matches exactly: "Interface 0 : Num of Syms = 12"
    pattern = re.compile(r"^Interface\s+(\d+)\s+:\s+Num of Syms\s+=\s+(\d+)")
    
    for gb in gb_types:
        info_path = os.path.join(RAW_BASE, gb, "Interface", "Information.txt")
        if not os.path.exists(info_path):
            print(f"[!] Warning: Could not find {info_path}")
            continue
            
        with open(info_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    idx_str = match.group(1)
                    sym_count = int(match.group(2))
                    
                    # Pad the integer with leading zeros to match your folder structure (e.g., 0 -> sym00000)
                    sym_folder = f"sym{int(idx_str):05d}"
                    
                    # Create the master key: e.g., "CoGe_100_GBs/sym00000"
                    key = f"{gb}/{sym_folder}"
                    sym_mapping[key] = sym_count
                    
    return sym_mapping

def farthest_point_sampling(features, n_samples):
    """
    Standard FPS algorithm using Euclidean distance.
    """
    N = features.shape[0]
    if n_samples >= N:
        return list(range(N))
        
    selected_indices = []
    
    # 1. Start with a random index
    current_idx = np.random.randint(0, N)
    selected_indices.append(current_idx)
    
    # 2. Keep track of the minimum distance from each point to the selected set
    distances = pairwise_distances(features, features[current_idx].reshape(1, -1)).flatten()
    
    # 3. Iteratively select the point that is furthest away
    for _ in tqdm(range(1, n_samples), desc=f"FPS (Target: {n_samples})", leave=False):
        current_idx = np.argmax(distances)
        selected_indices.append(current_idx)
        
        # Update distances: if the new point is closer to an unselected point, update the min distance
        new_distances = pairwise_distances(features, features[current_idx].reshape(1, -1)).flatten()
        distances = np.minimum(distances, new_distances)
        
    return selected_indices

def main():
    print("Starting Stratified Furthest Point Sampling . . .")
    
    # 1. Load the SOAP Data
    print(f"Loading SOAP vectors from {SOAP_DATA_FILE} . . .")
    with open(SOAP_DATA_FILE, 'rb') as f:
        soap_data = pickle.load(f)
    print(f"Loaded {len(soap_data)} structures.")
    
    # 2. Get Symmetry Mappings
    sym_mapping = get_symmetry_mapping()
    
    # 3. Group Structures into "Buckets" based on their Symmetry Count
    # Example bucket format: {1: ["key1", "key2"], 6: ["key3"]}
    buckets = {}
    
    for key in soap_data.keys():
        # The key looks like "CoGe_100_GBs/sym00003/4x"
        # We need to extract "CoGe_100_GBs/sym00003" to match the Information.txt mapping
        parts = key.split("/")
        if len(parts) >= 2:
            base_sym_key = f"{parts[0]}/{parts[1]}"
            sym_count = sym_mapping.get(base_sym_key, -1) # Default to -1 if not found
            
            if sym_count not in buckets:
                buckets[sym_count] = []
            buckets[sym_count].append(key)
            
    # Remove the "Not Found" bucket if you only want explicitly labeled structures
    if -1 in buckets:
        print(f"[!] Warning: Found {len(buckets[-1])} structures not listed in Information.txt. Ignoring them.")
        del buckets[-1]
        
    # 4. Calculate Proportional Targets for each Bucket
    total_valid_structures = sum([len(keys) for keys in buckets.values()])
    print(f"\nTotal structures ready for FPS: {total_valid_structures}")
    
    final_selected_keys = []
    
    print("\nRunning FPS by symmetry group . . .")
    for sym_count, keys in sorted(buckets.items()):
        group_size = len(keys)
        
        # Proportional math: (Group Size / Total Size) * Target 200
        # The max(1) ensures even tiny groups get at least 1 structure selected
        target_for_group = max(1, int(round((group_size / total_valid_structures) * TARGET_TOTAL_STRUCTURES)))
        
        print(f"\nSymmetry Ops [{sym_count}]: Total Structures = {group_size} -> Target Selection = {target_for_group}")
        
        if target_for_group == 0:
            continue
            
        # Extract the actual SOAP arrays for just this bucket
        group_features = np.array([soap_data[k] for k in keys])
        
        # Run FPS
        selected_indices = farthest_point_sampling(group_features, target_for_group)
        
        # Map the selected numerical indices back to their string filepaths
        for idx in selected_indices:
            final_selected_keys.append(keys[idx])
            
    # 5. Save the Final List
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for key in final_selected_keys:
            f.write(f"{key}\n")
            
    print(f"\nProcess Complete! Happy Machine Learning! :)")
    print(f"Total structures selected: {len(final_selected_keys)}")
    print(f"Final list written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()