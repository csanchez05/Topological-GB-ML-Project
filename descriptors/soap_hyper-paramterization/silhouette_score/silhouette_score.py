import os
import re
import glob
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT_FOLDERS = [
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_110_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_111_GBs/Interface"
]

OUTPUT_BASE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/descriptors/soap_hyper-paramterization/silhouette_score"

# We need a slightly larger sample size to ensure every symmetry class has enough members to form a cluster
SAMPLE_SIZE = 200
FIXED_R_CUT = 5.54 

# The grid of parameters to test
N_MAX_OPTIONS = [2, 4, 6, 8, 10, 12]
L_MAX_OPTIONS = [2, 4, 6, 8, 10, 12]
SIGMA_OPTIONS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

def parse_master_info(interface_dir):
    """Reads the master Information.txt one level up from the Interface directory"""
    info_path = os.path.join(interface_dir, "Information.txt")
    sym_map = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            for line in f:
                match = re.match(r'^(\d+):\s*(\d+)\s*symmetry operations', line)
                if match:
                    sym_map[int(match.group(1))] = int(match.group(2))
    return sym_map

def get_labeled_poscars(folders, n_samples):
    """Grabs POSCARs and their known physical symmetry operations"""
    all_data = []
    
    for folder in folders:
        sym_map = parse_master_info(folder)
        search_pattern = os.path.join(folder, "sym*", "4x", "POSCAR")
        poscars = glob.glob(search_pattern)
        
        for p in poscars:
            try:
                # Extract the sym ID to get the label
                parts = p.split(os.sep)
                sym_folder = [part for part in parts if part.startswith('sym')][0]
                interface_idx = int(sym_folder.replace('sym', ''))
                
                label = sym_map.get(interface_idx, -1)
                if label != -1:  # Only keep data we actually have labels for
                    all_data.append((p, label))
            except Exception:
                pass

    # Sample randomly, but guarantee we don't exceed what's available
    if len(all_data) < n_samples:
        return all_data
    return random.sample(all_data, n_samples)

def main():
    print("--- STARTING TOPOLOGICAL SILHOUETTE SCORE SWEEP ---")
    
    if not os.path.exists(OUTPUT_BASE):
        os.makedirs(OUTPUT_BASE)

    print(f"\nSampling {SAMPLE_SIZE} labeled POSCARs...")
    test_data = get_labeled_poscars(ROOT_FOLDERS, SAMPLE_SIZE)
    if not test_data: 
        print("No valid POSCARs with labels found. Exiting.")
        return

    # Load geometries into memory
    atoms_list = []
    labels_list = []
    for p, label in tqdm(test_data, desc="Loading Geometries"):
        try:
            atoms_list.append(read(p))
            labels_list.append(label)
        except Exception: pass

    labels_array = np.array(labels_list)

    grid = list(itertools.product(zip(N_MAX_OPTIONS, L_MAX_OPTIONS), SIGMA_OPTIONS))
    results = []

    print(f"\nEvaluating {len(grid)} parameter combinations...")
    
    for (n, l), sig in grid:
        print(f"  -> Testing: n={n}, l={l}, sigma={sig}")
        
        try:
            soap = SOAP(
                species=["Co", "Ge"],
                r_cut=FIXED_R_CUT, 
                n_max=n, 
                l_max=l, 
                sigma=sig,
                weighting={"function": "pow", "r0": 5.0, "m": 6, "c": 1, "d": 1},
                periodic=True,
                sparse=False
            )
            
            features = []
            for atoms in atoms_list:
                local_desc = soap.create(atoms, n_jobs=1)
                # THE CRITICAL FIX: Max Pooling isolates the interface defect features
                #features.append(np.max(local_desc, axis=0))
                # THE CONTROL: Mean Pooling (Dilutes interface with bulk)
                features.append(np.mean(local_desc, axis=0))
                
            X = np.array(features)
            feature_dim = X.shape[1]
            
            # Calculate Silhouette Score using the physical creation symmetry labels
            score = silhouette_score(X, labels_array)
            
            results.append({
                'n_max': n, 'l_max': l, 'sigma': sig,
                'dimensions': feature_dim,
                'silhouette_score': score
            })
            
        except Exception as e:
            print(f"     [!] Failed: {e}")

    # --- Analysis & Plotting ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_BASE, "topological_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        df['dimensions'], 
        df['silhouette_score'], 
        c=df['sigma'], 
        s=df['n_max'] * 20, 
        cmap='plasma',
        alpha=0.8,
        edgecolors='k'
    )
    
    for i, row in df.iterrows():
        plt.annotate(f"{int(row['n_max'])}/{int(row['l_max'])}, sig={row['sigma']}", 
                     (row['dimensions'], row['silhouette_score']),
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

    plt.colorbar(scatter, label='Sigma Value')
    plt.title("Topological Descriptor Sweep (Max-Pooling)\nHigher Silhouette Score = Better Physical Separation")
    plt.xlabel("Descriptor Dimensionality (Compute Cost)")
    plt.ylabel("Silhouette Score (Clustering Quality)")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_BASE, "silhouette_mean_test.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"\n[+] Sweep complete. Data saved to {csv_path}")
    print(f"[+] Pareto plot saved to {plot_path}")
    print("\nHOW TO DECIDE: Look at the plot. Pick the point furthest to the TOP-LEFT.")

if __name__ == "__main__":
    main()