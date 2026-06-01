import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your extracted energies
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv")

# 2. Load your new SOAP asymmetry vectors
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

# 3. Match them up
data = []
for index, row in df_energies.iterrows():
    struct_id = int(row['ID'])
    # Construct the key format your script used: "CoGe_100_GBs/symXXXXX/4x"
    # Assuming we are only looking at the 100 GBs right now based on our earlier isolation
    key = f"CoGe_100_GBs/sym{struct_id:05d}/4x"
    
    if key in soap_dict:
        delta_vector = soap_dict[key]
        
        # Calculate the mathematical magnitude of the asymmetry
        asymmetry_norm = np.linalg.norm(delta_vector)
        
        data.append({
            "ID": struct_id,
            "Energy": row['Energy_Per_Atom_eV'],
            "Asymmetry_Norm": asymmetry_norm
        })

df_plot = pd.DataFrame(data)

# 4. Plot the physics
plt.figure(figsize=(8, 6))
plt.scatter(df_plot["Asymmetry_Norm"], df_plot["Energy"], alpha=0.7, edgecolor='k')
plt.title("Structural Asymmetry vs. Unrelaxed Energy")
plt.xlabel("Magnitude of Delta SOAP (||Left_Max - Right_Max||)")
plt.ylabel("Energy per Atom (eV)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("/home/calvi/Research_Group/ML_Interface_Project/results/soap_energy_analysis.png", dpi=300)
print("Analysis saved to Asymmetry_vs_Energy.png")