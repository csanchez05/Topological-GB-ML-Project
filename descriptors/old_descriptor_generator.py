import os
import sys
import numpy as np
import pickle
from ase.io import read
from dscribe.descriptors import SOAP
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# INPUT: Your raw structures (Read Only)
ROOT_FOLDERS = [
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_110_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_111_GBs/Interface"
]

# OUTPUT: A SINGLE compiled dictionary file containing all 13,842 vectors
OUTPUT_FILE = "/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_mean_vectors_compiled.pkl"

def collect_jobs():
    jobs = []
    print("Scanning directories for structures...")
    
    for root_folder in ROOT_FOLDERS:
        if not os.path.exists(root_folder):
            print(f"[!] Warning: Directory not found: {root_folder}")
            continue
            
        parent_dir = os.path.dirname(root_folder) 
        gb_type_name = os.path.basename(parent_dir)

        for dirpath, dirnames, filenames in os.walk(root_folder):
            # Ignore any pre-existing relaxation folders
            dirnames[:] = [d for d in dirnames if 'relax' not in d.lower()]
            
            target_files = [f for f in filenames if "POSCAR" in f or f.endswith(".vasp")]
            
            for filename in target_files:
                poscar_path = os.path.join(dirpath, filename)
                
                # Construct relative path (e.g. sym00001/4x)
                rel_path = os.path.relpath(dirpath, root_folder)
                
                # Construct a unique string key for the dictionary: "CoGe_100_GBs/sym00001/4x"
                unique_key = os.path.join(gb_type_name, rel_path).replace("\\", "/")
                
                jobs.append((poscar_path, unique_key))
                
    return jobs

def process_structure(job):
    poscar_path, unique_key = job
    
    try:
        if os.path.getsize(poscar_path) == 0:
            return (unique_key, f"Skipped (Empty File): {poscar_path}")

        try:
            structure = read(poscar_path)
        except Exception as e:
            return (unique_key, f"Read Error: {e}")

        # Setup SOAP inside the worker to prevent multiprocessing memory leaks
        species = ["Co", "Ge"]
        soap = SOAP(
            species=species,
            r_cut=6.0, 
            n_max=12, 
            l_max=12, 
            sigma=0.3,
            weighting={"function": "pow", "r0": 5.0, "m": 6, "c": 1, "d": 1},
            periodic=True,
            sparse=False
        )
        
        # Generate raw 2D array (Shape: N_atoms x N_features)
        raw_descriptor = soap.create(structure, n_jobs=1)
        
        # Extract fractional coordinates to find the interface split
        scaled_positions = structure.get_scaled_positions()
        x_coords = scaled_positions[:, 0]
        
        # Separate atomic indices based on the mathematical center (x=0.5)
        left_indices = np.where(x_coords < 0.5)[0]
        right_indices = np.where(x_coords >= 0.5)[0]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return (unique_key, "Error: Missing atoms on one side of the interface.")
            
        # Slice the raw SOAP descriptors into Left and Right grains
        left_descriptors = raw_descriptor[left_indices]
        right_descriptors = raw_descriptor[right_indices]
        
        # Max-pool locally within each grain to preserve local symmetry
        left_pooled = np.max(left_descriptors, axis=0)
        right_pooled = np.max(right_descriptors, axis=0)
        
        # Calculate the Asymmetry Vector (Delta SOAP)
        delta_descriptor = left_pooled - right_pooled
        
        # Return the key and the asymmetry array
        return (unique_key, delta_descriptor)

    except Exception as e:
        return (unique_key, f"Error: {e}")

if __name__ == "__main__":
    jobs = collect_jobs()
    total_jobs = len(jobs)
    
    if total_jobs == 0:
        print("No structures found! Check your paths.")
        sys.exit()
        
    print(f"Found {total_jobs} structures. Data will be compiled to: {OUTPUT_FILE}")

    # Leave 2 cores free so your laptop doesn't completely lock up
    n_cores = max(1, cpu_count() - 2)
    print(f"Utilizing {n_cores} CPU cores for DScribe calculations . . .\n")
    
    results_dict = {}
    errors = []
    
    with Pool(n_cores) as pool:
        # Process and collect results dynamically
        for result in tqdm(pool.imap_unordered(process_structure, jobs), total=total_jobs):
            unique_key, data = result
            
            if isinstance(data, str):  # It's an error message
                errors.append(f"[{unique_key}] {data}")
            else:
                results_dict[unique_key] = data

    print("\n" + "="*40)
    print("Saving data to master file . . .")
    print("="*40)
    
    # Save the aggregated dictionary to a single file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results_dict, f)
        
    print(f"Successfully processed: {len(results_dict)} / {total_jobs}")
    print(f"Master file saved: {OUTPUT_FILE}")
    
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for err in errors[:5]:
            print(err)
            
    print("="*40)