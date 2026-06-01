import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import os

def parse_information_full(info_filepath):
    data = []
    with open(info_filepath, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Interface") and ":" in line:
            struct_id = int(line.split(":")[0].replace("Interface", "").strip())
            
            i += 1
            x_dist = float(lines[i].split("=")[1].strip())
            
            while i < len(lines) and not lines[i].strip().startswith("[["):
                i += 1
                
            if i < len(lines):
                row1 = [float(x) for x in lines[i].replace('[', ' ').replace(']', ' ').split()]
                row2 = [float(x) for x in lines[i+1].replace('[', ' ').replace(']', ' ').split()]
                row3 = [float(x) for x in lines[i+2].replace('[', ' ').replace(']', ' ').split()]
                
                M = np.array([row1, row2, row3])
                A = M[:, :3] 
                
                data.append({
                    "ID": struct_id,
                    "X_Distance_Rounded": str(round(x_dist, 3)), # String for filtering
                    "Chirality_Preserved": str(np.round(np.linalg.det(A)) == 1.0)
                })
        i += 1
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 1. Parse all geometric data
    info_file = "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface/Information.txt"
    df_info = parse_information_full(info_file)
    
    # 2. Load all unrelaxed DFT energies
    energy_csv = "/home/calvi/Research_Group/ML_Interface_Project/results/energy_symmetry_data.csv"
    df_energies = pd.read_csv(energy_csv)
    
    df_full = pd.merge(df_info, df_energies, on="ID", how="inner")
    
    # 3. Load the Dual SOAP Vectors
    pkl_path = "/home/calvi/Research_Group/ML_Interface_Project/data/processed/soap_max_vectors_compiled.pkl"
    with open(pkl_path, "rb") as f:
        soap_dict = pickle.load(f)

    # 4. Match and extract X, Y, Z coordinates
    plot_data = []
    for index, row in df_full.iterrows():
        struct_id = int(row['ID'])
        base_key = f"CoGe_100_GBs/sym{struct_id:05d}/4x"
        
        mid_key = f"{base_key}_MID"
        edge_key = f"{base_key}_EDGE"
        
        if mid_key in soap_dict and edge_key in soap_dict:
            vec_mid = soap_dict[mid_key]
            vec_edge = soap_dict[edge_key]
            
            norm_mid = np.linalg.norm(vec_mid)
            norm_edge = np.linalg.norm(vec_edge)
            
            # Filter out catastrophic overlap explosions
            if row['Energy_Per_Atom_eV'] < 0.0: 
                plot_data.append({
                    "ID": struct_id,
                    "Norm_MID (X)": norm_mid,
                    "Norm_EDGE (Y)": norm_edge,
                    "Energy (Z)": row['Energy_Per_Atom_eV'],
                    "Termination Plane": row['X_Distance_Rounded'],
                    "Chirality": row['Chirality_Preserved']
                })

    df_plot = pd.DataFrame(plot_data)
    print(f"Successfully mapped {len(df_plot)} total interfaces.")

    # 5. Build isolated plots for target termination planes
    target_planes = ['0.035', '0.029']
    
    for plane in target_planes:
        df_filtered = df_plot[df_plot['Termination Plane'] == plane]
        
        if df_filtered.empty:
            print(f"Warning: No data found for termination plane {plane} Å.")
            continue
            
        fig = px.scatter_3d(
            df_filtered, 
            x='Norm_MID (X)', 
            y='Norm_EDGE (Y)', 
            z='Energy (Z)',
            color='Chirality',             # Now coloring strictly by Topology
            symbol='Chirality',
            hover_data=['ID'],
            title=f"3D Energy Manifold: Mid vs Edge Frustration (Termination {plane} Å)",
            opacity=0.8,
            color_discrete_map={'True': '#1f77b4', 'False': '#d62728'} # Blue=Preserved, Red=Flipped
        )

        fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
        
        output_html = f"/home/calvi/Research_Group/ML_Interface_Project/results/interactive_3d_manifold_{plane}.html"
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        fig.write_html(output_html)
        
        print(f"DONE. Saved isolated plot for {plane} Å to: {output_html}")