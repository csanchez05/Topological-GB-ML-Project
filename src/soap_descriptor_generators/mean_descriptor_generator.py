import os
import sys
import numpy as np
import pickle
from ase.io import read
from dscribe.descriptors import SOAP
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT_FOLDERS = [
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_110_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_111_GBs/Interface"
]

# Changed filename to preserve your previous max-pooled data
OUTPUT_FILE = "/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_mean_vectors_compiled.pkl"

def collect_jobs():
    jobs = []
    print("Scanning directories for structures...")
    for root_folder in ROOT_FOLDERS:
        if not os.path.exists(root_folder):
            continue
        parent_dir = os.path.dirname(root_folder) 
        gb_type_name = os.path.basename(parent_dir)

        for dirpath, dirnames, filenames in os.walk(root_folder):
            dirnames[:] = [d for d in dirnames if 'relax' not in d.lower()]
            target_files = [f for f in filenames if "POSCAR" in f or f.endswith(".vasp")]
            
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
        
        soap = SOAP(
            species=["Co", "Ge"],
            r_cut=6.0, 
            n_max=12, 
            l_max=12, 
            sigma=0.20,
            weighting={"function": "pow", "r0": 5.0, "m": 6, "c": 1, "d": 1},
            periodic=True,
            sparse=False
        )
        
        raw_descriptor = soap.create(structure, n_jobs=1)
        
        # GLOBAL AVERAGE POOLING: Compress all atoms into one mean structural fingerprint
        # Downcast to float32 immediately to save RAM
        mean_descriptor = np.mean(raw_descriptor, axis=0).astype(np.float32)
        
        # Return a single tuple containing the key and the mean vector
        return (unique_key, mean_descriptor)

    except Exception as e:
        return (unique_key, f"Error: {str(e)}")

if __name__ == "__main__":
    jobs = collect_jobs()
    total_jobs = len(jobs)
    
    if total_jobs == 0:
        print("No structures found! Check your paths.")
        sys.exit()
        
    print(f"Found {total_jobs} structures. Extracting global mean descriptors...")
    n_cores = max(1, cpu_count() - 2)
    
    results_dict = {}
    errors = []
    
    with Pool(n_cores) as pool:
        for result in tqdm(pool.imap_unordered(process_structure, jobs), total=total_jobs):
            unique_key, data = result
            
            if isinstance(data, str):  # It caught an error message string
                errors.append(f"[{unique_key}] {data}")
            else:
                results_dict[unique_key] = data

    print("\n" + "="*40)
    print("Clearing parallel memory and saving data...")
    gc.collect()
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results_dict, f)
        
    print(f"Successfully processed {len(results_dict)} distinct boundaries.")
    if errors:
        print(f"Errors encountered: {len(errors)}")