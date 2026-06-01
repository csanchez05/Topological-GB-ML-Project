import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

PREDICTIONS_FILE = "/home/calvi/Research_Group/ML_Interface_Project/results/FPS_Blind_Predictions_Corrected.csv"
TRUE_RESULTS_FILE = "/home/calvi/Research_Group/ML_Interface_Project/results/FPS_True_Results.csv"
OUTPUT_PLOT = "/home/calvi/Research_Group/ML_Interface_Project/results/Parity_Plot_Validation.png"

if not os.path.exists(TRUE_RESULTS_FILE):
    print("Setting the trap...")
    # Generate a template for the user to fill out
    df_preds = pd.read_csv(PREDICTIONS_FILE)
    df_template = df_preds[['Target_Boundary']].copy()
    df_template['True_Gamma_eV_A2'] = np.nan
    df_template.to_csv(TRUE_RESULTS_FILE, index=False)
    print(f"I generated a template for your true DFT results at {TRUE_RESULTS_FILE}.")
    print("Fill in the 'True_Gamma_eV_A2' column as CIRCE finishes the jobs, then rerun this script.")
    exit()

df_preds = pd.read_csv(PREDICTIONS_FILE)
df_true = pd.read_csv(TRUE_RESULTS_FILE)

df_merged = pd.merge(df_preds, df_true, on="Target_Boundary", how="inner")

# Check if there are nulls
if df_merged['True_Gamma_eV_A2'].isnull().any():
    print("WARNING: Some true DFT targets are still missing (NaN). Plotting only completed targets.")
    df_merged = df_merged.dropna(subset=['True_Gamma_eV_A2'])

if len(df_merged) == 0:
    print("No true targets filled out yet. Exiting.")
    exit()

y_pred = df_merged['Predicted_Gamma_eV_A2']
y_true = df_merged['True_Gamma_eV_A2']

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\nValidation Complete for {len(df_merged)} Targets")
print(f"Mean Absolute Error (MAE): {mae:.4f} eV/\u00c5\u00b2")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} eV/\u00c5\u00b2")

# Parity Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, c='crimson', alpha=0.8, edgecolor='k', s=80, label=f"MAE: {mae:.4f} eV/\u00c5\u00b2")

# Perfect parity line
min_val = min(y_true.min(), y_pred.min()) - 0.05
max_val = max(y_true.max(), y_pred.max()) + 0.05
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Parity")

plt.xlabel("True DFT Interface Formation Energy \u03b3 (eV/\u00c5\u00b2)", fontsize=14)
plt.ylabel("PySR Predicted \u03b3 (eV/\u00c5\u00b2)", fontsize=14)
plt.title("Model Validation: PySR vs. DFT Ground Truth", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=600)
print(f"\nParity plot saved to {OUTPUT_PLOT}")

# Check the 0.66 outlier specifically
outlier_target = "CoGe_110_GBs/sym00540/8x"
outlier_row = df_merged[df_merged['Target_Boundary'] == outlier_target]
if not outlier_row.empty:
    pred_outlier = outlier_row.iloc[0]['Predicted_Gamma_eV_A2']
    true_outlier = outlier_row.iloc[0]['True_Gamma_eV_A2']
    error = abs(pred_outlier - true_outlier)
    print("\n" + "="*50)
    print("--- OUTLIER TRAP RESULTS ---")
    print(f"Target: {outlier_target}")
    print(f"Predicted: {pred_outlier:.4f} eV/\u00c5\u00b2")
    print(f"True DFT:  {true_outlier:.4f} eV/\u00c5\u00b2")
    print(f"Delta:     {error:.4f} eV/\u00c5\u00b2")
    print("="*50)
