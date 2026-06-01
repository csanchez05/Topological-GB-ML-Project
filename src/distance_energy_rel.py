import pandas as pd
import matplotlib.pyplot as plt

def extract_distances(info_filepath):
    data = []
    with open(info_filepath, 'r') as f:
        for line in f:
            # Target line: "0: 12 symmetry operations, x distance = ..., total distance = ..."
            if "symmetry operations, x distance =" in line:
                try:
                    parts = line.split(":")
                    struct_id = int(parts[0].strip())
                    
                    metrics = parts[1].split(",")
                    num_syms = int(metrics[0].strip().split()[0])
                    x_dist = float(metrics[1].split("=")[1].strip())
                    total_dist = float(metrics[2].split("=")[1].strip())
                    
                    data.append({
                        "ID": struct_id,
                        "Num_Syms": num_syms,
                        "X_Distance": x_dist,
                        "Total_Distance": total_dist
                    })
                except Exception:
                    continue # Skip malformed lines
                    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # --- PATH SETUP ---
    # Verify this is where Information.txt lives in your data folder:
    info_file = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface/Information.txt" 
    
    # Point to the CSV you just downloaded into the results folder:
    energy_csv = "/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv"
    
    # Set the output image to also drop into the results folder:
    output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/Distance_vs_Energy.png"
    
    print("Parsing pre-calculated geometric distances...")
    df_dist = extract_distances(info_file)
    
    print("Loading DFT energies...")
    df_energies = pd.read_csv(energy_csv)
    
    print("Merging datasets...")
    df_final = pd.merge(df_dist, df_energies, on="ID", how="inner")
    
    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Total Distance vs Energy
    scatter1 = axes[0].scatter(df_final["Total_Distance"], df_final["Energy_Per_Atom_eV"], 
                               c=df_final["Num_Syms"], cmap='viridis', alpha=0.8)
    axes[0].set_title("Geometric Shift (Total Distance) vs. Energy")
    axes[0].set_xlabel("Total Distance (Å)")
    axes[0].set_ylabel("Energy per Atom (eV)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    fig.colorbar(scatter1, ax=axes[0], label="Num Symmetry Ops")
    
    # Plot 2: X Distance vs Energy
    scatter2 = axes[1].scatter(df_final["X_Distance"], df_final["Energy_Per_Atom_eV"], 
                               c=df_final["Num_Syms"], cmap='viridis', alpha=0.8)
    axes[1].set_title("Perpendicular Separation (X Distance) vs. Energy")
    axes[1].set_xlabel("X Distance (Å)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.colorbar(scatter2, ax=axes[1], label="Num Symmetry Ops")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Saved correlation plot to {output_png}")