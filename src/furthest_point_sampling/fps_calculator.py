import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import gc  # Garbage Collector

print("Loading structural universe...")
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

# 1. Build the Data Matrices
X_all = []
keys_all = []
structure_bases = list(set([k.replace("_MID", "").replace("_EDGE", "") for k in soap_dict.keys()]))

print("Assembling high-dimensional matrix...")
for base_key in structure_bases:
    mid_key = f"{base_key}_MID"
    edge_key = f"{base_key}_EDGE"
    if mid_key in soap_dict and edge_key in soap_dict:
        X_all.append(np.concatenate([soap_dict[mid_key], soap_dict[edge_key]]))
        keys_all.append(base_key)

# DOWNCAST to float32 to instantly cut RAM usage by 50%
X_all = np.array(X_all, dtype=np.float32)
keys_all = np.array(keys_all)

# THE CRITICAL STEP: Destroy the massive dictionary to free up gigabytes of RAM
print("Clearing dictionary from RAM...")
del soap_dict
gc.collect()

# Standardize
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Destroy the unscaled array to free more RAM
del X_all
gc.collect()

# 2. Identify Known vs. Unknown Structures (USING ALL UNFILTERED RUNS)
print("Isolating calculated vs uncalculated boundaries...")
# Changed this to your raw dataset with all ~650 structures
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv")
known_keys = [f"CoGe_100_GBs/sym{int(row['ID']):05d}/4x" for _, row in df_energies.iterrows()]

known_mask = np.isin(keys_all, known_keys)
unknown_mask = ~known_mask

X_known = X_scaled[known_mask]
X_unknown = X_scaled[unknown_mask]
keys_unknown = keys_all[unknown_mask]

print(f"Known VASP runs (The Anchor): {len(X_known)}")
print(f"Unknown structures available: {len(X_unknown)}")

# 3. Farthest Point Sampling (FPS)
N_TARGETS = 25
selected_keys = []

current_known_pool = X_known.copy()
remaining_unknown_X = X_unknown.copy()
remaining_unknown_keys = list(keys_unknown)

print(f"Hunting for the {N_TARGETS} most diverse structures...")

for i in range(N_TARGETS):
    # cdist computes the distance matrix. By freeing RAM earlier, we have space for this.
    distances = cdist(remaining_unknown_X, current_known_pool, metric='euclidean')
    min_dist_to_known = np.min(distances, axis=1)
    farthest_idx = np.argmax(min_dist_to_known)
    
    target_key = remaining_unknown_keys[farthest_idx]
    selected_keys.append(target_key)
    
    current_known_pool = np.vstack([current_known_pool, remaining_unknown_X[farthest_idx]])
    remaining_unknown_X = np.delete(remaining_unknown_X, farthest_idx, axis=0)
    remaining_unknown_keys.pop(farthest_idx)

# 4. Output the Action Plan
print("\n" + "="*50)
print("ACTION REQUIRED: RUN THESE IN VASP TO FIX SAMPLING BIAS")
print("="*50)
for idx, key in enumerate(selected_keys):
    print(f"{idx+1:02d}. {key}")

with open("/home/calvi/Research_Group/ML_Interface_Project/results/VASP_Targets_FPS.txt", "w") as f:
    for key in selected_keys:
        f.write(f"{key}\n")
        
print("Successfully generated target file.")