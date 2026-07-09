from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("/work_bgfs/c/calvinsanchez/USF/ML_Interface_Project/calculations/CoGe_Bulk_Pristine")
CONV = DATA / "SCF" / "convergence"
output_dir = Path("/work_bgfs/c/calvinsanchez/USF/ML_Interface_Project/plots/dft_plots/CoGe/bulk/regular/convergence")
output_dir.mkdir(parents=True, exist_ok=True)

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

    out = output_dir / fname.replace(".csv", ".png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)