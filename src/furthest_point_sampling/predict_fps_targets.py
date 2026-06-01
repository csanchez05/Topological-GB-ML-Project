import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Loading SOAP high-dimensional manifold...")
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

# --- 1. Rebuild the Mathematical Scaler ---
# We MUST fit the scaler on the original training data, or the PySR equation is useless.
print("Rebuilding standard scaler from the 908-structure phase space...")
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv")

target_indices = [813, 1630, 5090] 
X_train = []
supercell_depths = ['4x', '5x', '6x', '7x', '8x', '9x', '10x', '11x', '12x']

for index, row in df_energies.iterrows():
    struct_id = int(row['ID'])
    for depth in supercell_depths:
        base_key = f"CoGe_100_GBs/sym{struct_id:05d}/{depth}"
        mid_key = f"{base_key}_MID"
        edge_key = f"{base_key}_EDGE"
        
        if mid_key in soap_dict and edge_key in soap_dict:
            combined_vector = np.concatenate([soap_dict[mid_key], soap_dict[edge_key]])
            X_train.append(combined_vector[target_indices])

X_train = np.array(X_train)
scaler = StandardScaler()
scaler.fit(X_train)  # The scaler is now mathematically locked to the training distribution

# --- 2. Execute Validated Predictions ---
print("Loading the 25 VASP targets currently in the supercomputer queue...")
with open("/home/calvi/Research_Group/ML_Interface_Project/results/VASP_Targets_FPS.txt", "r") as f:
    fps_keys = [line.strip() for line in f.readlines()]

predictions = []

for base_key in fps_keys:
    mid_key = f"{base_key}_MID"
    edge_key = f"{base_key}_EDGE"
    
    if mid_key in soap_dict and edge_key in soap_dict:
        combined_vector = np.concatenate([soap_dict[mid_key], soap_dict[edge_key]])
        raw_features = combined_vector[target_indices].reshape(1, -1)
        
        # CRITICAL STEP: Scale the raw features before feeding them to the equation
        scaled_features = scaler.transform(raw_features)[0]
        
        co_co_mid = scaled_features[0]
        co_ge_mid = scaled_features[1]
        co_ge_edge = scaled_features[2]
        
        # Execute the newly evolved analytical physics equation for \u03b3
        gamma_pred = (co_co_mid + co_ge_mid + co_ge_edge) * 0.08564798 + 0.30675054
        
        predictions.append({
            "Target_Boundary": base_key,
            "Predicted_Gamma_eV_A2": round(gamma_pred, 4)
        })

df_preds = pd.DataFrame(predictions)
output_file = "/home/calvi/Research_Group/ML_Interface_Project/results/FPS_Blind_Predictions_Corrected.csv"
df_preds.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"BLIND PREDICTIONS LOCKED IN FOR {len(df_preds)} TARGETS.")
print("="*60)
print(df_preds.head())