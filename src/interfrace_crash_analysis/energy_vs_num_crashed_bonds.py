import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
from tqdm import tqdm

def count_steric_crashes(base_dir, df_energies, threshold=2.0):
    """
    Loads POSCARs and counts the number of severely compressed bonds (< threshold Å).
    """
    crash_data = []

    print(f"Scanning {len(df_energies)} POSCARs for steric crashes (< {threshold} Å)...")
    for idx, row in tqdm(df_energies.iterrows(), total=len(df_energies)):
        struct_id = int(row["ID"])
        chirality = row.get("Chirality_Preserved", True)
        energy = row["Gamma_J_per_m2"]
        
        # Format ID to 5 digits based on your naming convention (e.g., sym00078)
        folder_name = f"sym{struct_id:05d}"
        poscar_path = os.path.join(base_dir, folder_name, "4x", "POSCAR")
        
        if not os.path.exists(poscar_path):
            # Fallback for alternative naming
            poscar_path = os.path.join(base_dir, folder_name, "4x", "CONTCAR")

        if os.path.exists(poscar_path):
            try:
                atoms = read(poscar_path)
                
                # Calculate all pairwise distances using Minimum Image Convention (Periodic Boundaries)
                dist_mat = atoms.get_all_distances(mic=True)
                
                # Count pairs where distance is between 0.1 and threshold
                # (We use > 0.1 to ignore the diagonal where an atom's distance to itself is 0)
                crash_mask = (dist_mat > 0.1) & (dist_mat <= threshold)
                
                # Divide by 2 because the distance matrix is symmetric (atom A to B, and atom B to A)
                num_crashes = np.sum(crash_mask) / 2
                
                crash_data.append({
                    "ID": struct_id,
                    "Gamma_J_per_m2": energy,
                    "Num_Crashes": num_crashes,
                    "Chirality_Preserved": chirality
                })
            except Exception as e:
                print(f"Error reading {folder_name}: {e}")
        else:
            print(f"File not found: {poscar_path}")

    return pd.DataFrame(crash_data)

if __name__ == "__main__":
    # --- PATH SETUP ---
    # WSL Path to your raw POSCARs
    base_dir = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface"
    
    # WSL Path to the previously calculated energies
    energy_csv = "/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv"
    
    # Output Image
    output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/Steric_Crash_Correlation.png"

    # 1. Load the energies (Ensure it has 'ID', 'Gamma_J_per_m2', and 'Chirality_Preserved')
    # If Chirality_Preserved isn't in this CSV, we will just plot them all the same color.
    df_energies = pd.read_csv(energy_csv)

    # 2. Extract the crash counts
    df_crashes = count_steric_crashes(base_dir, df_energies, threshold=2.0)

    # 3. Plotting the physics
    if not df_crashes.empty:
        plt.figure(figsize=(10, 6))
        
        # Check if we successfully merged Chirality data earlier
        if "Chirality_Preserved" in df_crashes.columns:
            colors = df_crashes["Chirality_Preserved"].map({True: 'blue', False: 'red'})
            labels = df_crashes["Chirality_Preserved"].map({True: 'Det = +1 (Proper)', False: 'Det = -1 (Improper)'})
            
            for chirality_state, color in zip([True, False], ['blue', 'red']):
                subset = df_crashes[df_crashes["Chirality_Preserved"] == chirality_state]
                label_name = 'Det = +1 (Classical)' if chirality_state else 'Det = -1 (Enantiomorphic)'
                plt.scatter(subset["Num_Crashes"], subset["Gamma_J_per_m2"], 
                            c=color, alpha=0.7, s=100, label=label_name)
        else:
            plt.scatter(df_crashes["Num_Crashes"], df_crashes["Gamma_J_per_m2"], 
                        alpha=0.7, s=100)

        plt.title("Macroscopic Energy vs. Microscopic Atomic Collisions")
        plt.xlabel("Number of Steric Crashes (Bond Length < 2.0 \u00c5)")
        plt.ylabel("Interface Formation Energy \u03b3 (J/m\u00b2)")
        plt.legend()

        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"\nSaved collision analysis to: {output_png}")
    else:
        print("No valid data to plot.")