import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your clean isolated data
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/Isolated_Manifold_Data.csv")

# 2. Load the dual-max vectors
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

data = []
for index, row in df_energies.iterrows():
    struct_id = int(row['ID'])
    base_key = f"CoGe_100_GBs/sym{struct_id:05d}/4x"
    
    mid_key = f"{base_key}_MID"
    edge_key = f"{base_key}_EDGE"
    
    if mid_key in soap_dict and edge_key in soap_dict:
        vec_mid = soap_dict[mid_key]
        vec_edge = soap_dict[edge_key]
        
        # Total frustration is the distortion at the middle PLUS the distortion at the edge
        total_distortion = np.linalg.norm(vec_mid) + np.linalg.norm(vec_edge)
        
        data.append({
            "ID": struct_id,
            "Energy": row['Energy_Per_Atom_eV'],
            "Total_Distortion": total_distortion,
            "Chirality": row['Chirality_Preserved']
        })

df_plot = pd.DataFrame(data)

if df_plot.empty:
    print("Error: No matching keys found.")
    exit()

# 3. Plot the physics
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_plot["Total_Distortion"], df_plot["Energy"], 
                      c=df_plot["Chirality"], cmap='coolwarm', 
                      alpha=0.8, s=80, edgecolor='k')

plt.title("Total Box Structural Frustration vs. Unrelaxed Energy\n(Locked at Termination 0.035 Å)", fontsize=14)
plt.xlabel("Total Magnitude of Delta SOAP (||Mid_Max|| + ||Edge_Max||)", fontsize=12)
plt.ylabel("Energy per Atom (eV)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label('Chirality Preserved (Red=True, Blue=False)')

plt.tight_layout()
output_path = "/home/calvi/Research_Group/ML_Interface_Project/results/dual_soap_energy_analysis.png"
plt.savefig(output_path, dpi=300)


print(f"Successfully mapped {len(df_plot)} structures.")
print(f"Analysis saved to {output_path}")