import os
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
from sklearn.decomposition import PCA
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT_FOLDERS = [
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_110_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_111_GBs/Interface"
]

OUTPUT_BASE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/descriptors/soap_pca_variance_sweep"

SAMPLE_SIZE = 100
FIXED_R_CUT = 5.54 
TARGET_COMPONENTS = 15

# The grid of parameters to test
N_MAX_OPTIONS = [4, 6, 8, 10, 12]
L_MAX_OPTIONS = [4, 6, 8, 10, 12]
SIGMA_OPTIONS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

def get_random_poscars(folders, n_samples):
    all_poscars = []
    for folder in folders:
        search_pattern = os.path.join(folder, "sym*", "4x", "POSCAR")
        all_poscars.extend(glob.glob(search_pattern))
    return random.sample(all_poscars, min(n_samples, len(all_poscars)))

def main():
    print("--- STARTING 2D HYPERPARAMETER PARETO SWEEP ---")
    
    if not os.path.exists(OUTPUT_BASE):
        os.makedirs(OUTPUT_BASE)

    test_poscars = get_random_poscars(ROOT_FOLDERS, SAMPLE_SIZE)
    if not test_poscars: return

    atoms_list = []
    for p in tqdm(test_poscars, desc="Loading Geometries"):
        try:
            atoms_list.append(read(p))
        except Exception: pass

    # Link n_max and l_max to avoid redundant mismatched arrays
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
                features.append(np.mean(local_desc, axis=0))
                
            X = np.array(features)
            feature_dim = X.shape[1]
            
            n_comps = min(TARGET_COMPONENTS, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_comps)
            pca.fit(X)
            var_captured = np.sum(pca.explained_variance_ratio_)
            
            results.append({
                'n_max': n, 'l_max': l, 'sigma': sig,
                'dimensions': feature_dim,
                'variance_at_15': var_captured
            })
            
        except Exception as e:
            print(f"     [!] Failed: {e}")

    # --- Analysis & Plotting ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_BASE, "hyperparameter_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        df['dimensions'], 
        df['variance_at_15'], 
        c=df['sigma'], 
        s=df['n_max'] * 20, 
        cmap='viridis',
        alpha=0.8,
        edgecolors='k'
    )
    
    # --- UPDATED ANNOTATION ---
    for i, row in df.iterrows():
        plt.annotate(f"{int(row['n_max'])}/{int(row['l_max'])}, sig={row['sigma']}", 
                     (row['dimensions'], row['variance_at_15']),
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

    plt.colorbar(scatter, label='Sigma Value')
    plt.title(f"Pareto Frontier: SOAP Hyperparameters\n(Variance captured in first {TARGET_COMPONENTS} dimensions)")
    plt.xlabel("Descriptor Dimensionality (Compute Cost)")
    plt.ylabel(f"Explained Variance (Information Density)")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_BASE, "pareto_hyperparameter_sweep.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"\n[+] Sweep complete. Data saved to {csv_path}")
    print(f"[+] Pareto plot saved to {plot_path}")

if __name__ == "__main__":
    main()