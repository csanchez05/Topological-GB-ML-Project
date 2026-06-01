import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
from tqdm import tqdm

def extract_chirality(info_filepath):
    """Parses Information.txt to extract the determinant/chirality for each interface."""
    data = []
    with open(info_filepath, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Interface") and "Num of Syms" in line:
            parts = line.split(":")
            struct_id = int(parts[0].replace("Interface", "").strip())
            
            # Fast-forward to the FIRST matrix for this interface
            while i < len(lines) and not lines[i].strip().startswith("[["):
                i += 1
                
            if i < len(lines):
                # Clean brackets and parse the 3x3 rotation matrix
                row1 = [float(x) for x in lines[i].replace('[', ' ').replace(']', ' ').split()]
                row2 = [float(x) for x in lines[i+1].replace('[', ' ').replace(']', ' ').split()]
                row3 = [float(x) for x in lines[i+2].replace('[', ' ').replace(']', ' ').split()]
                
                A = np.array([row1, row2, row3])[:, :3] 
                det_A = np.round(np.linalg.det(A))
                
                data.append({
                    "ID": struct_id,
                    "Chirality_Preserved": (det_A == 1.0)
                })
        i += 1
        
    return pd.DataFrame(data)

def count_steric_crashes(base_dir, df_merged, threshold=2.0):
    """Loads POSCARs and counts severely compressed bonds (< threshold Å)."""
    crash_data = []

    print(f"Scanning {len(df_merged)} POSCARs for steric crashes (< {threshold} \u00c5)...")
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged)):
        struct_id = int(row["ID"])
        energy = row["Gamma_J_per_m2"]
        chirality = row["Chirality_Preserved"]
        
        folder_name = f"sym{struct_id:05d}"
        poscar_path = os.path.join(base_dir, folder_name, "4x", "POSCAR")
        
        if not os.path.exists(poscar_path):
            poscar_path = os.path.join(base_dir, folder_name, "4x", "CONTCAR")

        if os.path.exists(poscar_path):
            try:
                atoms = read(poscar_path)
                # Minimum Image Convention is crucial for periodic boundaries
                dist_mat = atoms.get_all_distances(mic=True)
                
                # Ignore self-distance (0.0) and count atoms closer than threshold
                crash_mask = (dist_mat > 0.1) & (dist_mat <= threshold)
                num_crashes = np.sum(crash_mask) / 2 # Divide by 2 because matrix is symmetric
                
                crash_data.append({
                    "ID": struct_id,
                    "Gamma_J_per_m2": energy,
                    "Num_Crashes": num_crashes,
                    "Chirality_Preserved": chirality
                })
            except Exception as e:
                print(f"Error reading {folder_name}: {e}")

    return pd.DataFrame(crash_data)

if __name__ == "__main__":
    # --- PATH SETUP ---
    info_file = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface/Information.txt" 
    base_dir = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface"
    energy_csv = "/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv"
    output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/Topological_Baseline_Analysis.png"

    print("1. Extracting Determinants (Chirality)...")
    df_chirality = extract_chirality(info_file)

    print("2. Loading DFT Energies...")
    df_energies = pd.read_csv(energy_csv)

    print("3. Merging Core Data...")
    df_merged = pd.merge(df_chirality, df_energies, on="ID", how="inner")

    print("4. Calculating Microscopic Collisions...")
    df_final = count_steric_crashes(base_dir, df_merged, threshold=2.0)

    # --- VISUALIZATION ---
    if not df_final.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel 1: Full Steric Crash Correlation
        for chirality_state, color, label in [(True, 'blue', 'Det = +1 (Classical)'), (False, 'red', 'Det = -1 (Enantiomorphic)')]:
            subset = df_final[df_final["Chirality_Preserved"] == chirality_state]
            axes[0].scatter(subset["Num_Crashes"], subset["Gamma_J_per_m2"], 
                            c=color, alpha=0.7, s=100, label=label)
            
        axes[0].set_title("Formation Energy vs. Number of Bonds < 2.0 \u00c5")
        axes[0].set_xlabel("Number of Steric Crashes (Bond Length < 2.0 \u00c5)")
        axes[0].set_ylabel("Interface Formation Energy \u03b3 (J/m\u00b2)")
        axes[0].legend()

        # Panel 2: The Topological Baseline (Zoomed in on 0 and 1 crashes)
        df_low_crash = df_final[df_final["Num_Crashes"] <= 1]
        
        for chirality_state, color, label in [(True, 'blue', 'Det = +1 (Classical)'), (False, 'red', 'Det = -1 (Enantiomorphic)')]:
            subset = df_low_crash[df_low_crash["Chirality_Preserved"] == chirality_state]
            
            axes[1].scatter(subset["Num_Crashes"], subset["Gamma_J_per_m2"], c=color, alpha=0.8, s=150,label=label)

        axes[1].set_title("Formation Energy vs. Number of Bonds < 2.0 \u00c5 (0 and 1 Crashes Only)")
        axes[1].set_xlabel("Number of Steric Crashes")
        axes[1].set_ylabel("Interface Formation Energy \u03b3 (J/m\u00b2)")
        axes[1].set_xticks([0, 1])
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        print(f"\nSaved analysis to: {output_png}")
    else:
        print("No data available to plot.")