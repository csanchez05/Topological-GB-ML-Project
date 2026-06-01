import os
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SOAP_DATA_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl"
RAW_BASE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries"
SELECTED_FILE = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/results/proportional_selected_200_structures.txt"
PLOT_OUTPUT_DIR = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/results/figures/fps_visualizations"

def get_symmetry_mapping():
    sym_mapping = {}
    gb_types = ["CoGe_100_GBs", "CoGe_110_GBs", "CoGe_111_GBs"]
    pattern = re.compile(r"^Interface\s+(\d+)\s+:\s+Num of Syms\s+=\s+(\d+)")
    
    for gb in gb_types:
        info_path = os.path.join(RAW_BASE, gb, "Interface", "Information.txt")
        if not os.path.exists(info_path): continue
            
        with open(info_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    idx_str = match.group(1)
                    sym_count = int(match.group(2))
                    key = f"{gb}/sym{int(idx_str):05d}"
                    sym_mapping[key] = sym_count
    return sym_mapping

def main():
    print("--- LOADING DATA ---")
    with open(SOAP_DATA_FILE, 'rb') as f:
        soap_data = pickle.load(f)
        
    all_keys = list(soap_data.keys())
    features = np.array([soap_data[k] for k in all_keys])
    
    sym_mapping = get_symmetry_mapping()
    
    # Extract symmetry labels for coloring (Default to 1 if not found to prevent crashes)
    labels = []
    for key in all_keys:
        parts = key.split("/")
        if len(parts) >= 2:
            base_sym_key = f"{parts[0]}/{parts[1]}"
            labels.append(sym_mapping.get(base_sym_key, 1))
        else:
            labels.append(1)
    labels = np.array(labels)
    
    # Load the 200 selected structures
    selected_keys = set()
    if os.path.exists(SELECTED_FILE):
        with open(SELECTED_FILE, 'r') as f:
            selected_keys = set([line.strip() for line in f.readlines()])
    
    is_selected = np.array([1 if k in selected_keys else 0 for k in all_keys])

    print("--- RUNNING DIMENSIONALITY REDUCTION ---")
    print("Calculating PCA (Global Variance)...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    var_explained = pca.explained_variance_ratio_
    
    print("Calculating t-SNE (Local Clusters) - This will take a minute...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_tsne = tsne.fit_transform(features)

    print("--- GENERATING PLOTS ---")
    sns.set_theme(style="whitegrid")
    
    # Convert labels to strings for discrete categorical plotting
    str_labels = [f"{L} Ops" for L in labels]
    
    # PLOT 1: PCA by Symmetry
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=str_labels, 
                    palette="viridis", s=15, alpha=0.7, edgecolor=None)
    plt.title(f"PCA Projection of CoGe Grain Boundaries\nVariance Explained: PC1={var_explained[0]:.2f}, PC2={var_explained[1]:.2f}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Symmetry", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "proportional_pca_symmetry.png"), dpi=300)
    plt.close()

    # PLOT 2: t-SNE by Symmetry
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=str_labels, 
                    palette="plasma", s=15, alpha=0.7, edgecolor=None)
    plt.title("t-SNE Projection (Topological Clustering)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Symmetry", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "proportional_tsne_symmetry.png"), dpi=300)
    plt.close()

    # PLOT 3: The FPS Proof (PCA highlighting selections)
    plt.figure(figsize=(10, 8))
    # Plot background (unselected)
    plt.scatter(features_pca[is_selected==0, 0], features_pca[is_selected==0, 1], 
                c='lightgrey', s=10, alpha=0.5, label='Unselected Pool')
    # Plot foreground (selected)
    plt.scatter(features_pca[is_selected==1, 0], features_pca[is_selected==1, 1], 
                c='red', s=40, edgecolors='black', marker='X', label='FPS Selected (200)')
    
    plt.title("Global Farthest Point Sampling Verification")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "proportional_fps_verification_pca.png"), dpi=300)
    plt.close()

    print(f"--- SUCCESS: Plots saved to {PLOT_OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()