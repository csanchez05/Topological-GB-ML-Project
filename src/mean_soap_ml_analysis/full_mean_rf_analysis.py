import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load ALL Raw Data (Bypassing the Manifold Filter)
print("Loading unfiltered dataset...")
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv")

# Load your descriptors (using the mean vectors you just generated)
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_mean_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

X_data = []
y_data = []
kept_ids = []

# 2. Extract Data
for index, row in df_energies.iterrows():
    struct_id = int(row['ID'])
    key = f"CoGe_100_GBs/sym{struct_id:05d}/4x"
    
    if key in soap_dict:
        X_data.append(soap_dict[key])
        y_data.append(row['Energy_Per_Atom_eV'])
        kept_ids.append(struct_id)

X = np.array(X_data)
y = np.array(y_data)

print(f"Brute Force Data Shape: {X.shape[0]} structures, {X.shape[1]} structural dimensions.")

# 3. Random Forest Pipeline (No scaling or PCA required for Trees)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Outlier-Resistant Random Forest...")
# Using 500 trees to handle the complex non-linear collision landscape
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 4. Predict and Evaluate
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Test MAE: {mae_test:.4f} eV/atom")
print(f"Test R^2: {r2_test:.4f}")

# 5. The Parity Plot
plt.figure(figsize=(9, 8))

min_val = min(np.min(y), np.min(y_pred_test))
max_val = max(np.max(y), np.max(y_pred_test))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

# Plot the physical regime vs the collision regime to expose the dataset reality
plt.scatter(y_train, y_pred_train, c='blue', alpha=0.4, label='Training Data', edgecolor='none')
plt.scatter(y_test, y_pred_test, c='red', alpha=0.9, s=60, label=f'Test Data (R^2={r2_test:.2f})', edgecolor='k')

plt.title("Random Forest: Predicting Unfiltered Unrelaxed Energies", fontsize=14)
plt.xlabel("True DFT Unrelaxed Energy (eV/atom)", fontsize=12)
plt.ylabel("ML Predicted Energy (eV/atom)", fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/Unfiltered_RF_Parity_Plot.png"
plt.tight_layout()
plt.savefig(output_png, dpi=300)
print(f"Saved Parity Plot to {output_png}")