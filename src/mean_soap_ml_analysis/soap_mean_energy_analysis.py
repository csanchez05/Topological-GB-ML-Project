import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your clean isolated data
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/Isolated_Manifold_Data.csv")

# 2. Load the GLOBAL MEAN vectors
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_mean_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

data = []
for index, row in df_energies.iterrows():
    struct_id = int(row['ID'])
    
    # The mean generator script saved the data under the exact base key, no _MID or _EDGE suffixes.
    key = f"CoGe_100_GBs/sym{struct_id:05d}/4x"
    
    if key in soap_dict:
        vec_mean = soap_dict[key]
        
        # Calculate the magnitude of the mean structural fingerprint
        mean_distortion = np.linalg.norm(vec_mean)
        
        data.append({
            "ID": struct_id,
            "Energy": row['Energy_Per_Atom_eV'],
            "Mean_Distortion": mean_distortion,
            "Chirality": row['Chirality_Preserved']
        })

df_plot = pd.DataFrame(data)

if df_plot.empty:
    print("Error: No matching keys found. Check your dictionary generation.")
    exit()

# 3. Plot the physics (or lack thereof)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_plot["Mean_Distortion"], df_plot["Energy"], 
                      c=df_plot["Chirality"], cmap='coolwarm', 
                      alpha=0.8, s=80, edgecolor='k')

plt.title("Global Mean Structural Distortion vs. Unrelaxed Energy\n(Testing the Bulk Dilution Hypothesis)", fontsize=14)
plt.xlabel("Magnitude of Mean SOAP Descriptor (||Mean||)", fontsize=12)
plt.ylabel("Energy per Atom (eV)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label('Chirality Preserved (Red=True, Blue=False)')

plt.tight_layout()
output_path = "/home/calvi/Research_Group/ML_Interface_Project/results/mean_soap_energy_analysis.png"
plt.savefig(output_path, dpi=300)

print(f"Successfully mapped {len(df_plot)} structures.")
print(f"Analysis saved to {output_path}")