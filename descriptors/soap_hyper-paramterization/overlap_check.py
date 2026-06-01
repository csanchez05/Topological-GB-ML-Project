import os
import glob
import random
import numpy as np
from ase.io import read
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT_FOLDERS = [
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_110_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_111_GBs/Interface"
]

SAMPLE_SIZE = 500
CRITICAL_DISTANCE = 1.25  # Angstroms. Anything below this is a guaranteed VASP crash.

def get_random_poscars(folders, n_samples):
    all_poscars = []
    for folder in folders:
        search_pattern = os.path.join(folder, "sym*", "4x", "POSCAR")
        all_poscars.extend(glob.glob(search_pattern))
    return random.sample(all_poscars, min(n_samples, len(all_poscars)))

def main():
    print("--- STARTING READ-ONLY GEOMETRIC DIAGNOSTIC ---")
    print(f"Checking for physically impossible atomic clashes (< {CRITICAL_DISTANCE} Å)...\n")
    
    test_poscars = get_random_poscars(ROOT_FOLDERS, SAMPLE_SIZE)
    if not test_poscars:
        print("No POSCARs found.")
        return

    clash_count = 0
    total_clashing_pairs = 0

    for path in tqdm(test_poscars, desc="Scanning Structures"):
        try:
            atoms = read(path)
            # Calculate distance matrix taking into account periodic boundary conditions
            distances = atoms.get_all_distances(mic=True)
            
            # We only care about the upper triangle to avoid double counting, 
            # and we ignore the diagonal (distance from an atom to itself, which is 0)
            np.fill_diagonal(distances, np.inf)
            
            # Find all pairs closer than the critical distance
            clashes = np.where(distances < CRITICAL_DISTANCE)
            
            # Each clash is counted twice in a symmetric matrix (i,j and j,i), so divide by 2
            num_clashing_pairs = len(clashes[0]) // 2
            
            if num_clashing_pairs > 0:
                clash_count += 1
                total_clashing_pairs += num_clashing_pairs
                
        except Exception as e:
            pass # Ignore read errors for the diagnostic

    print("\n--- DIAGNOSTIC RESULTS ---")
    print(f"Structures Scanned: {len(test_poscars)}")
    print(f"Structures with FATAL Clashes: {clash_count}")
    print(f"Total Impossible Atomic Overlaps Detected: {total_clashing_pairs}")
    
    percentage = (clash_count / len(test_poscars)) * 100
    print(f"\nFailure Rate: {percentage:.1f}%")
    
    if clash_count == 0:
        print("\nVERDICT: Your dataset is physically sane. We can run Farthest Point Sampling immediately.")
    else:
        print("\nVERDICT: Your dataset contains mathematical artifacts. Farthest Point Sampling will fail.")
        print("We must purge the overlapping atoms before proceeding.")

if __name__ == "__main__":
    main()