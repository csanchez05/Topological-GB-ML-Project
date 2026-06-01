import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Loading 13,870 Dual SOAP Vectors...")
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

# 1. Build the "Universe" of all structures
X_all = []
keys_all = []

# Group the MID and EDGE vectors for every structure in the dictionary
structure_bases = set([k.replace("_MID", "").replace("_EDGE", "") for k in soap_dict.keys()])

for base_key in structure_bases:
    mid_key = f"{base_key}_MID"
    edge_key = f"{base_key}_EDGE"
    
    if mid_key in soap_dict and edge_key in soap_dict:
        combined_vector = np.concatenate([soap_dict[mid_key], soap_dict[edge_key]])
        X_all.append(combined_vector)
        keys_all.append(base_key)

X_all = np.array(X_all)
print(f"Constructed Universe Matrix: {X_all.shape[0]} structures, {X_all.shape[1]} dimensions.")

# 2. Compress the Universe to 2D
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

pca = PCA(n_components=2)
print("Performing PCA on the entire structural phase space...")
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the Universe
df_universe = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_universe['Base_Key'] = keys_all

# 3. Load the 100 "Beacons" (Known VASP Energies)
print("Loading DFT Energy Beacons...")
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/Isolated_Manifold_Data.csv")

# Match the beacons to their coordinates in the Universe
beacon_data = []
for index, row in df_energies.iterrows():
    struct_id = int(row['ID'])
    base_key = f"CoGe_100_GBs/sym{struct_id:05d}/4x"
    
    # Find where this structure sits in the PCA dataframe
    match = df_universe[df_universe['Base_Key'] == base_key]
    if not match.empty:
        beacon_data.append({
            'PC1': match['PC1'].values[0],
            'PC2': match['PC2'].values[0],
            'Energy': row['Energy_Per_Atom_eV']
        })

df_beacons = pd.DataFrame(beacon_data)
print(f"Matched {len(df_beacons)} calculated energies to the map.")

# 4. Plot the Map and the Beacons
plt.figure(figsize=(10, 8))

# Plot the background Universe (Uncalculated structures)
plt.scatter(df_universe['PC1'], df_universe['PC2'], 
            c='lightgray', alpha=0.3, s=15, label='Uncalculated Structures')

# Plot the Beacons (Calculated energies)
scatter = plt.scatter(df_beacons['PC1'], df_beacons['PC2'], 
                      c=df_beacons['Energy'], cmap='magma', 
                      s=50, linewidth=1.5, label='DFT Calculated')

plt.colorbar(scatter, label='DFT Unrelaxed Energy (eV/atom)')
plt.title("Structural Latent Space", fontsize=14)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.legend(loc='best')

output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/Latent_Space_Overlay.png"
plt.tight_layout()
plt.savefig(output_png, dpi=300)
print(f"Analysis saved to {output_png}")