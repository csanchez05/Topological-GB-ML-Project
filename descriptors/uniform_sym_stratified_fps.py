import os
import pickle
import numpy as np
import re
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SOAP_DATA_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl"
RAW_BASE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries"
OUTPUT_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/results/uniform_selected_200_structures.txt"

TARGET_TOTAL_STRUCTURES = 200

def get_symmetry_mapping():
    """
    Crawls the Information.txt files using exact regex for the PI's format.
    """
    print("Parsing Information.txt files for symmetry labels...")
    sym_mapping = {}
    
    gb_types = ["CoGe_100_GBs", "CoGe_110_GBs", "CoGe_111_GBs"]
    
    # Bulletproof Regex for "Interface 0 : Num of Syms = 12"
    pattern = re.compile(r"^Interface\s+(\d+)\s+:\s+Num of Syms\s+=\s+(\d+)")
    
    for gb in gb_types:
        info_path = os.path.join(RAW_BASE, gb, "Interface", "Information.txt")
        if not os.path.exists(info_path):
            continue
            
        with open(info_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    idx_str = match.group(1)
                    sym_count = int(match.group(2))
                    
                    sym_folder = f"sym{int(idx_str):05d}"
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
    
    current_idx = np.random.randint(0, N)
    selected_indices.append(current_idx)
    
    distances = pairwise_distances(features, features[current_idx].reshape(1, -1)).flatten()
    
    for _ in tqdm(range(1, n_samples), desc=f"FPS (Target: {n_samples})", leave=False):
        current_idx = np.argmax(distances)
        selected_indices.append(current_idx)
        
        new_distances = pairwise_distances(features, features[current_idx].reshape(1, -1)).flatten()
        distances = np.minimum(distances, new_distances)
        
    return selected_indices

def main():
    print("Starting Stratified Furthest Point Sampling . . .")
    
    print(f"Loading SOAP vectors from {SOAP_DATA_FILE}...")
    with open(SOAP_DATA_FILE, 'rb') as f:
        soap_data = pickle.load(f)
    print(f"Loaded {len(soap_data)} structures.")
    
    sym_mapping = get_symmetry_mapping()
    
    buckets = {}
    for key in soap_data.keys():
        parts = key.split("/")
        if len(parts) >= 2:
            base_sym_key = f"{parts[0]}/{parts[1]}"
            sym_count = sym_mapping.get(base_sym_key, -1)
            
            if sym_count not in buckets:
                buckets[sym_count] = []
            buckets[sym_count].append(key)
            
    if -1 in buckets:
        print(f"[!] Warning: Found {len(buckets[-1])} structures not listed in Information.txt. Ignoring.")
        del buckets[-1]
        
    total_valid = sum([len(keys) for keys in buckets.values()])
    print(f"\nTotal strictly mapped structures: {total_valid}")
    
    # ---------------------------------------------------------
    # THE FIX: UNIFORM TARGET ALLOCATION
    # ---------------------------------------------------------
    total_buckets = len(buckets)
    base_target = TARGET_TOTAL_STRUCTURES // total_buckets
    remainder = TARGET_TOTAL_STRUCTURES % total_buckets
    
    # Sort buckets by size descending to give the remainder to the largest buckets
    sorted_bucket_keys = sorted(buckets.keys(), key=lambda k: len(buckets[k]), reverse=True)
    
    targets = {}
    for i, sym_count in enumerate(sorted_bucket_keys):
        targets[sym_count] = base_target + (1 if i < remainder else 0)
        # Safety check: if a bucket has fewer structures than the target, cap it.
        targets[sym_count] = min(targets[sym_count], len(buckets[sym_count]))

    final_selected_keys = []
    
    print("\nRunning FPS by symmetry group . . .")
    for sym_count in sorted(buckets.keys()):
        keys = buckets[sym_count]
        target = targets[sym_count]
        
        print(f"\nSymmetry Ops [{sym_count}]: Total Pool = {len(keys)} -> Selected = {target}")
        
        if target == 0:
            continue
            
        group_features = np.array([soap_data[k] for k in keys])
        selected_indices = farthest_point_sampling(group_features, target)
        
        for idx in selected_indices:
            final_selected_keys.append(keys[idx])
            
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for key in final_selected_keys:
            f.write(f"{key}\n")
            
    print(f"\nProcess Complete! Happy Machine Learning! :)")
    print(f"Total structures uniformly selected: {len(final_selected_keys)}")
    print(f"Final list written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()