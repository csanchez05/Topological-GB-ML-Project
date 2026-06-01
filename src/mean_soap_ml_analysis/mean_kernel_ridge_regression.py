import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
df_energies = pd.read_csv("/home/calvi/Research_Group/ML_Interface_Project/results/Isolated_Manifold_Data.csv")

# Point to the new MEAN vectors
with open("/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_mean_vectors_compiled.pkl", "rb") as f:
    soap_dict = pickle.load(f)

X_data = []
y_data = []

# 2. Extract Mean Vectors
for index, row in df_energies.iterrows():
    struct_id = int(row['ID'])
    # The mean generator used the exact base key with no suffixes
    key = f"CoGe_100_GBs/sym{struct_id:05d}/4x"
    
    if key in soap_dict:
        X_data.append(soap_dict[key])
        y_data.append(row['Energy_Per_Atom_eV'])

X = np.array(X_data)
y = np.array(y_data)

print(f"Raw Data Shape: {X.shape[0]} structures, {X.shape[1]} structural dimensions.")

# 3. Build the Machine Learning Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)), 
    ('krr', KernelRidge(kernel='rbf'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Hyperparameter Grid Search
param_grid = {
    'krr__alpha': np.logspace(-3, 2, 6),
    'krr__gamma': np.logspace(-4, 1, 6)
}

print("Executing Grid Search with Cross-Validation...")
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Best hyperparameters found: {grid.best_params_}")

# 5. Predict and Evaluate
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Test MAE: {mae_test:.4f} eV/atom")
print(f"Test R^2: {r2_test:.4f}")

# 6. The Parity Plot
plt.figure(figsize=(8, 8))

min_val = min(np.min(y), np.min(y_pred_test), np.min(y_pred_train))
max_val = max(np.max(y), np.max(y_pred_test), np.max(y_pred_train))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

plt.scatter(y_train, y_pred_train, c='blue', alpha=0.5, label='Training Data', edgecolor='k')
plt.scatter(y_test, y_pred_test, c='red', alpha=0.9, s=80, label=f'Test Data (R^2={r2_test:.2f})', edgecolor='k')

plt.title("PCA + KRR (Global Mean Descriptor): Predicted vs. DFT Energy", fontsize=14)
plt.xlabel("True DFT Unrelaxed Energy (eV/atom)", fontsize=12)
plt.ylabel("ML Predicted Energy (eV/atom)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# Save to a distinct file so we can compare it to the max-pooled run
output_png = "/home/calvi/Research_Group/ML_Interface_Project/results/KRR_PCA_Mean_Parity_Plot.png"
plt.tight_layout()
plt.savefig(output_png, dpi=300)
print(f"Saved Parity Plot to {output_png}")