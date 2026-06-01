import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import gc 

print("Loading the FULL 13,000+ structural universe...")
# Make sure this points to the full, unfiltered pickle file you generated yesterday
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

# 1. Build the Universe Matrix
X_all = []
keys_all = []
structure_bases = list(set([k.replace("_MID", "").replace("_EDGE", "") for k in soap_dict.keys()]))

print(f"Assembling high-dimensional matrix for {len(structure_bases)} structures...")
for base_key in structure_bases:
    mid_key = f"{base_key}_MID"
    edge_key = f"{base_key}_EDGE"
    if mid_key in soap_dict and edge_key in soap_dict:
        combined_vector = np.concatenate([soap_dict[mid_key], soap_dict[edge_key]])
        X_all.append(combined_vector)
        keys_all.append(base_key)

# DOWNCAST to save WSL memory
X_all = np.array(X_all, dtype=np.float32)
keys_all = np.array(keys_all)

print("Clearing dictionary from RAM...")
del soap_dict
gc.collect()

# Compress to 2D
print("Scaling and performing PCA on the FULL noisy dataset...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

del X_all
gc.collect()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

del X_scaled
gc.collect()

# Create a master DataFrame
df_universe = pd.DataFrame({
    'Base_Key': keys_all,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})

# 2. Isolate the Known VASP Runs (USING THE FULL UNFILTERED DATASET)
print("Loading existing DFT runs (Anchor Pool)...")
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv")
known_keys = [f"CoGe_100_GBs/sym{int(row['ID']):05d}/4x" for _, row in df_energies.iterrows()]
df_known = df_universe[df_universe['Base_Key'].isin(known_keys)]

# 3. Isolate the NEW FPS Targets
print("Loading new viable FPS targets...")
with open("/home/calvi/Research_Group/ML_Interface_Project/results/VASP_Targets_FPS.txt", "r") as f:
    fps_keys = [line.strip() for line in f.readlines()]
df_fps = df_universe[df_universe['Base_Key'].isin(fps_keys)]

# 4. Plot the Strategic Overlay
print("Generating plot...")
plt.figure(figsize=(11, 8))

# Layer 1: The Universe (Uncalculated, Mostly Crashed)
plt.scatter(df_universe['PC1'], df_universe['PC2'],
            c='lightgray', alpha=0.3, s=20, label=f'Full Universe ({len(df_universe)})')

# Layer 2: The Old Data (Existing VASP Runs)
plt.scatter(df_known['PC1'], df_known['PC2'],
            c='midnightblue', alpha=0.6, s=50, linewidth=0.5, edgecolor='black',
            label=f'Existing Runs ({len(df_known)})')

# Layer 3: The New Targets (FPS Selections)
plt.scatter(df_fps['PC1'], df_fps['PC2'],
            c='crimson', alpha=0.9, s=200, linewidth=1.5, edgecolor='black', marker='*',
            label=f'New Viable Targets ({len(df_fps)})')

plt.title("Phase Space Mapping: Viable Targets vs. Full Noisy Universe", fontsize=15)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=14)
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/FPS_Targets_Overlay_Macro.png"
plt.tight_layout()
plt.savefig(output_png, dpi=300)
print(f"SUCCESS: Plot saved to {output_png}")