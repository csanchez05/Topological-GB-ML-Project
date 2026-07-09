from pathlib import Path
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter

DATA = Path("/work_bgfs/c/calvinsanchez/USF/ML_Interface_Project/calculations/CoGe_Bulk_Pristine")
output_dir = Path("/work_bgfs/c/calvinsanchez/USF/ML_Interface_Project/plots/dft_plots/CoGe/bulk/regular/non_soc")
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fermi levels: bands use the SCF run's E_F (line-mode E_F is unreliable),
# DOS uses its own run's E_F (dense uniform mesh -> reliable on its own).
# Same split logic as the SOC overlay script.
# ---------------------------------------------------------------------------
def scf_efermi(scf_dir):
    return Vasprun(str(scf_dir / "vasprun.xml"), parse_dos=True,
                   parse_eigen=False, parse_potcar_file=False).efermi

ef_bands = scf_efermi(DATA / "SCF" / "production")
print("bands E-fermi (from SCF) =", ef_bands)

bs = BSVasprun(str(DATA / "bandstructure" / "vasprun.xml"),
               parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(DATA / "bandstructure" / "KPOINTS"),
    line_mode=True, efermi=ef_bands)

dosv = Vasprun(str(DATA / "dos" / "vasprun.xml"),
               parse_dos=True, parse_eigen=False, parse_potcar_file=False)
cdos = dosv.complete_dos
print("DOS E-fermi (own run)   =", dosv.efermi)

# ---------------------------------------------------------------------------
# Combined plot: bands (left) + DOS (right), sharing the energy axis.
# bs_projection=None  -> plain bands, no orbital coloring needed here.
# dos_projection='elements' -> Co/Ge breakdown in the DOS panel (uses the
#   same complete_dos you already plotted in plot_dos.py, no extra cost).
#   Set to None instead if you just want a single black total-DOS curve.
# vb_energy_range / cb_energy_range: window below/above E_F, in eV.
#   Tightened to 3 eV each since CoGe is a metal -- no "valence/conduction"
#   split in the traditional sense, just a symmetric window around E_F.
# ---------------------------------------------------------------------------
plotter = BSDOSPlotter(
    bs_projection=None,
    dos_projection="elements",
    vb_energy_range=3,
    cb_energy_range=3,
    egrid_interval=1,
)
plotter.get_plot(bs, dos=cdos)

plt.savefig(output_dir / "bands_dos_combined.png", dpi=800, bbox_inches="tight")
print("wrote", output_dir / "bands_dos_combined.png")