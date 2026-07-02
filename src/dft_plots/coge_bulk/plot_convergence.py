from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("/home/calvi/Research_Group/ML_Interface_Project/data/dft_calculations/CoGe/bulk")
CONV = DATA / "convergence"

for fname, xcol, xlabel, title in [
    ("kpoint_convergence.csv", "kmesh",    "k-mesh (N×N×N)", "k-point convergence — CoGe SCF"),
    ("encut_convergence.csv",  "encut_eV", "ENCUT (eV)",     "ENCUT convergence — CoGe SCF"),
]:
    df = pd.read_csv(CONV / fname)

    # energy relative to the densest/highest setting (the reference), in meV/atom
    df["dE"] = df["energy_per_atom_meV"] - df["energy_per_atom_meV"].iloc[-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[xcol], df["dE"], "o-")
    ax.axhline(0,  color="k",    lw=0.5)
    ax.axhline(1,  ls="--", color="gray")
    ax.axhline(-1, ls="--", color="gray", label="±1 meV/atom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy/atom − converged (meV)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out = fname.replace(".csv", ".png")
    fig.savefig(out, dpi=300)
    print("wrote", out)