import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def extract_linear_algebra(info_filepath):
    data = []
    
    with open(info_filepath, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Find the Interface block
        if line.startswith("Interface") and ":" in line:
            struct_id = int(line.split(":")[0].replace("Interface", "").strip())
            
            # Fast-forward until we hit a line that actually starts with the matrix brackets
            while i < len(lines) and not lines[i].strip().startswith("[["):
                i += 1
                
            if i < len(lines):
                # Replace all brackets with spaces, then split by whitespace to isolate the numbers
                row1 = [float(x) for x in lines[i].replace('[', ' ').replace(']', ' ').split()]
                row2 = [float(x) for x in lines[i+1].replace('[', ' ').replace(']', ' ').split()]
                row3 = [float(x) for x in lines[i+2].replace('[', ' ').replace(']', ' ').split()]
                
                M = np.array([row1, row2, row3]) # 3x4 matrix
                
                # --- APPLY THE LINEAR ALGEBRA ---
                A = M[:, :3] # 3x3 Linear Map
                t = M[:, 3]  # 3x1 Translation Vector
                
                # 1. Trace
                trace_A = np.trace(A)
                
                # 2. Translation Norm (Shear magnitude)
                norm_t = np.linalg.norm(t)
                
                # 3. Frobenius Distance from Identity
                I = np.eye(3)
                frob_dist = np.linalg.norm(A - I, 'fro')
                
                # Check Determinant for Chirality flip
                det_A = np.round(np.linalg.det(A))
                
                data.append({
                    "ID": struct_id,
                    "Trace": trace_A,
                    "Translation_Norm": norm_t,
                    "Frobenius_Dist": frob_dist,
                    "Chirality_Preserved": (det_A == 1.0)
                })
        i += 1
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    # --- PATH SETUP ---
    # Since you are running this from the 'src' folder, we step up one level ('..') 
    # to access the other folders in your ML_Interface_Project directory.
    
    # Verify this is where Information.txt lives in your data folder:
    info_file = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface/Information.txt" 
    
    # Point to the CSV you just downloaded into the results folder:
    energy_csv = "/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv"
    
    # Set the output image to also drop into the results folder:
    output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/simple_matrix_analysis/Linear_Algebra_Transformations_new.png"
    
    print("Extracting linear algebra transformations...")
    # 1. Run the Linear Algebra extraction
    df_la = extract_linear_algebra(info_file)
    
    print("Loading DFT energies...")
    # 2. Load your previously extracted energies
    df_energies = pd.read_csv(energy_csv)
    
    print("Merging datasets...")
    # 3. Merge them
    df_final = pd.merge(df_la, df_energies, on="ID", how="inner")
    
    print(f"Data merged. Generating plots for {len(df_final)} interfaces...")
    
    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Spectral Trace vs Energy
    axes[0].scatter(df_final["Trace"], df_final["Gamma_J_per_m2"], c=df_final["Chirality_Preserved"], cmap='coolwarm', alpha=0.7)
    axes[0].set_title("Trace vs. Energy")
    axes[0].set_xlabel("Trace(A)")
    axes[0].set_ylabel("Interface Formation Energy \u03b3 (J/m\u00b2)")
    
    # Plot 2: Translation Norm vs Energy
    axes[1].scatter(df_final["Translation_Norm"], df_final["Gamma_J_per_m2"], c=df_final["Chirality_Preserved"], cmap='coolwarm', alpha=0.7)
    axes[1].set_title("Shear Translation vs. Energy")
    axes[1].set_xlabel("|| t || (Magnitude)")
    
    # Plot 3: Frobenius Distance vs Energy
    axes[2].scatter(df_final["Frobenius_Dist"], df_final["Gamma_J_per_m2"], c=df_final["Chirality_Preserved"], cmap='coolwarm', alpha=0.7)
    axes[2].set_title("Frobenius Distance vs. Energy")
    axes[2].set_xlabel("|| A - I ||_F")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=600)
    print(f"Saved mathematical mappings to {output_png}")