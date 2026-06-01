import os
import pickle
import numpy as np
import re
from collections import Counter
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# INPUT: The master dictionary from descriptor generation phase
SOAP_DATA_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl"

# INPUT: The base directory to find the Information.txt files
RAW_BASE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries"

# OUTPUT: A text file listing the 200 selected structures
OUTPUT_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/results/global_selected_200_structures.txt"

TARGET_TOTAL_STRUCTURES = 200

def get_symmetry_mapping():
    """
    Crawls the Information.txt files to map each index to its symXXXXX folder and symmetry count.
    Returns: Dict[str, int] -> e.g., {"CoGe_100_GBs/sym00000": 12}
    """
    sym_mapping = {}
    gb_types = ["CoGe_100_GBs", "CoGe_110_GBs", "CoGe_111_GBs"]
    
    # Bulletproof Regex for the PI's format
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
    print("Starting Global Furthest Point Sampling . . .")
    
    # 1. Load the SOAP Data
    print(f"Loading SOAP vectors from {SOAP_DATA_FILE} . . .")
    with open(SOAP_DATA_FILE, 'rb') as f:
        soap_data = pickle.load(f)
        
    all_keys = list(soap_data.keys())
    total_structures = len(all_keys)
    print(f"Loaded {total_structures} structures.")
    
    # 2. Stack vectors into a single 2D NumPy array
    print("Compiling feature matrix...")
    all_features = np.array([soap_data[k] for k in all_keys])
    
    # 3. Run Global FPS (No Tampering, Pure Math)
    print("\nExecuting Global FPS on the entire dataset...")
    selected_indices = farthest_point_sampling(all_features, TARGET_TOTAL_STRUCTURES)
    
    # Map indices back to file paths
    final_selected_keys = [all_keys[idx] for idx in selected_indices]
    
    # 4. OBSERVE THE RESULTS (Cross-reference with Information.txt)
    print("\nAnalyzing the symmetry distribution of the selected structures...")
    sym_mapping = get_symmetry_mapping()
    
    selected_sym_counts = Counter()
    for key in final_selected_keys:
        parts = key.split("/")
        if len(parts) >= 2:
            base_sym_key = f"{parts[0]}/{parts[1]}"
            sym = sym_mapping.get(base_sym_key, -1) # Default to -1 if somehow missing
            selected_sym_counts[sym] += 1
            
    print("\nStructure Distribution by Symmetry Count")
    for sym_count in sorted(selected_sym_counts.keys()):
        label = sym_count if sym_count != -1 else "Unknown/Unlisted"
        print(f"Symmetry Ops [{label}]: {selected_sym_counts[sym_count]} structures selected")
            
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