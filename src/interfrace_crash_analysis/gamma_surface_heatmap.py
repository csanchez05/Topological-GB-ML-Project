import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_2d_surface_data(info_filepath):
    """Extracts the X and Y grid shifts and the matrix determinant correctly."""
    data = []
    
    with open(info_filepath, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Target the detailed block: "Interface 0 : Num of Syms = 12"
        if line.startswith("Interface") and "Num of Syms" in line:
            parts = line.split(":")
            struct_id = int(parts[0].replace("Interface", "").strip())
            
            # The next line is the X distance
            i += 1
            x_dist = float(lines[i].split("=")[1].strip())
            
            # The next line is the Total distance
            i += 1
            total_dist = float(lines[i].split("=")[1].strip())
            
            # Calculate Y distance using Pythagorean theorem
            y_dist = np.sqrt(max(0, total_dist**2 - x_dist**2))
            
            # Fast-forward a few lines to the FIRST matrix
            while i < len(lines) and not lines[i].strip().startswith("[["):
                i += 1
                
            if i < len(lines):
                # Clean the brackets and parse the 3x3 rotation matrix
                row1 = [float(x) for x in lines[i].replace('[', ' ').replace(']', ' ').split()]
                row2 = [float(x) for x in lines[i+1].replace('[', ' ').replace(']', ' ').split()]
                row3 = [float(x) for x in lines[i+2].replace('[', ' ').replace(']', ' ').split()]
                
                A = np.array([row1, row2, row3])[:, :3] 
                det_A = np.round(np.linalg.det(A))
                
                data.append({
                    "ID": struct_id,
                    "X_Shift": x_dist,
                    "Y_Shift": y_dist,
                    "Total_Distance": total_dist,
                    "Chirality_Preserved": (det_A == 1.0)
                })
        i += 1
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    # --- PATH SETUP ---
    info_file = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface/Information.txt" 
    energy_csv = "/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv"
    output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/Gamma_Surface_2D_Map.png"
    
    print("Extracting 2D grid shifts and chirality...")
    df_grid = extract_2d_surface_data(info_file)
    
    print("Loading normalized DFT energies...")
    df_energies = pd.read_csv(energy_csv)
    
    print("Merging datasets...")
    df_final = pd.merge(df_grid, df_energies, on="ID", how="inner")
    
    # Separate the topological regimes
    df_proper = df_final[df_final["Chirality_Preserved"] == True]
    df_improper = df_final[df_final["Chirality_Preserved"] == False]
    
    print(f"Plotting {len(df_proper)} Proper and {len(df_improper)} Improper interfaces...")
    
    # --- VISUALIZATION ---
    # Determine global min/max energy so both colorbars are on the same scale
    vmin = df_final["Gamma_J_per_m2"].min()
    vmax = df_final["Gamma_J_per_m2"].max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Proper Rotations (det = +1)
    # Using scatter with large markers to simulate a heatmap for sparse data
    sc1 = axes[0].scatter(df_proper["X_Shift"], df_proper["Y_Shift"], 
                          c=df_proper["Gamma_J_per_m2"], cmap='viridis', 
                          s=200, vmin=vmin, vmax=vmax, edgecolors='black')
    axes[0].set_title("Classical \u03b3-Surface (det = +1)\nChirality Preserved")
    axes[0].set_xlabel("X Translation (\u00c5)")
    axes[0].set_ylabel("Y Translation (\u00c5)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Improper Rotations (det = -1)
    sc2 = axes[1].scatter(df_improper["X_Shift"], df_improper["Y_Shift"], 
                          c=df_improper["Gamma_J_per_m2"], cmap='viridis', 
                          s=200, vmin=vmin, vmax=vmax, edgecolors='black')
    axes[1].set_title("Enantiomorphic \u03b3-Surface (det = -1)\nChirality Inverted")
    axes[1].set_xlabel("X Translation (\u00c5)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Add a single shared colorbar
    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label("Interface Formation Energy \u03b3 (J/m\u00b2)")
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Saved 2D maps to {output_png}")

    plt.figure(figsize=(10, 6))

# Plot Energy vs Minimum Atomic Separation (Total Distance)
plt.scatter(df_final["Total_Distance"], df_final["Gamma_J_per_m2"], 
            c=df_final["Chirality_Preserved"], cmap='coolwarm', alpha=0.8, s=100)
plt.title("Steric Crash: Formation Energy vs. Minimum Interfacial Bond Length")
plt.xlabel("Minimum Atomic Separation Across Interface (\u00c5)")
plt.ylabel("Interface Formation Energy \u03b3 (J/m\u00b2)")
# Add a vertical line where a normal Co-Ge bond should be
plt.axvline(x=2.35, color='red', linestyle='--', label='Ideal Co-Ge Bond Length (~2.35 \u00c5)')
plt.legend()

plt.savefig("Steric_Crash_Plot.png", dpi=300)