import os
import sys
import numpy as np
import pickle
from ase.io import read
from dscribe.descriptors import SOAP
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- CONFIGURATION ---
ROOT_FOLDERS = [
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_110_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_111_GBs/Interface"
]
OUTPUT_FILE = "/home/calvi/Research_Group/ML_Interface_Project/data/processed/viable_soap_vectors.pkl"
CRASH_THRESHOLD = 2.0

def collect_jobs():
    jobs = []
    print("Scanning directories for POSCARs...")
    for root_folder in ROOT_FOLDERS:
        if not os.path.exists(root_folder):
            continue
        parent_dir = os.path.dirname(root_folder) 
        gb_type_name = os.path.basename(parent_dir)

        for dirpath, dirnames, filenames in os.walk(root_folder):
            dirnames[:] = [d for d in dirnames if 'relax' not in d.lower()]
            target_files = [f for f in filenames if f == "POSCAR" or f == "CONTCAR"]
            
            for filename in target_files:
                poscar_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(dirpath, root_folder)
                unique_key = os.path.join(gb_type_name, rel_path).replace("\\", "/")
                jobs.append((poscar_path, unique_key))
    return jobs

def process_structure(job):
    poscar_path, unique_key = job
    try:
        if os.path.getsize(poscar_path) == 0:
            return (unique_key, "Skipped (Empty File)")

        structure = read(poscar_path)
        
        # Skip absurdly large structures to prevent worker Out-of-Memory (OOM) kills
        if len(structure) > 15000:
            return (unique_key, f"Skipped (Too large: {len(structure)} atoms)")
        
        # 1. THE VIABILITY FILTER (Steric Crashes)
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list('ijd', structure, CRASH_THRESHOLD)
        
        # Count pairs closer than threshold, ignoring self-interactions (d ~ 0)
        num_crashes = np.sum(d > 0.1) / 2
        
        if num_crashes > 1.0:
            return (unique_key, f"Skipped (Unphysical: {num_crashes} crashes)")

        # 2. THE FEATURE EXTRACTION (SOAP) - Reverted to your known-working masking method
        Lx = structure.cell.cellpar()[0]
        soap = SOAP(
            species=["Co", "Ge"],
            r_cut=6.0, n_max=12, l_max=12, sigma=0.20,
            weighting={"function": "pow", "r0": 5.0, "m": 6, "c": 1, "d": 1},
            periodic=True, sparse=False
        )
        
        # Compute the full descriptor array safely
        raw_descriptor = soap.create(structure, n_jobs=1)
        abs_x = structure.get_positions()[:, 0]
        
        mid_mask = (abs_x > (Lx/2 - 3.0)) & (abs_x < (Lx/2 + 3.0))
        edge_mask = (abs_x < 3.0) | (abs_x > (Lx - 3.0))
        
        max_mid = np.max(raw_descriptor[mid_mask], axis=0) if any(mid_mask) else np.zeros(soap.get_number_of_features())
        max_edge = np.max(raw_descriptor[edge_mask], axis=0) if any(edge_mask) else np.zeros(soap.get_number_of_features())
        
        return [
            (f"{unique_key}_MID", max_mid),
            (f"{unique_key}_EDGE", max_edge)
        ]

    except Exception as e:
        return (unique_key, f"Error: {str(e)}")

if __name__ == "__main__":
    from collections import Counter
    
    jobs = collect_jobs()
    if not jobs:
        sys.exit("No structures found.")
        
    print(f"Evaluating {len(jobs)} total structures for thermodynamic viability...")
    n_cores = max(1, cpu_count() - 2)
    
    results_dict = {}
    skip_reasons = []  # We will track exactly why things are failing
    
    with Pool(n_cores) as pool:
        for result in tqdm(pool.imap_unordered(process_structure, jobs), total=len(jobs)):
            if isinstance(result, list):
                for k, v in result:
                    results_dict[k] = v
            else:
                # result is a tuple: (unique_key, reason_string)
                skip_reasons.append(result[1])

    print("\n" + "="*50)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results_dict, f)
        
    print(f"Successfully encoded {len(results_dict)//2} viable boundaries.")
    
    print("\n--- DIAGNOSTIC BREAKDOWN OF DISCARDED STRUCTURES ---")
    reason_counts = Counter(skip_reasons)
    for reason, count in reason_counts.most_common(10):
        print(f"{count:5d} structures -> {reason}")
    print("="*50)