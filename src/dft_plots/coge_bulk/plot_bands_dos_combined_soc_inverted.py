from pathlib import Path
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter

DATA = Path("/work_bgfs/c/calvinsanchez/USF/ML_Interface_Project/calculations/CoGe_Bulk_Pristine/inverted")
SOC = DATA / "SOC"
output_dir = Path("/work_bgfs/c/calvinsanchez/USF/ML_Interface_Project/plots/dft_plots/CoGe/bulk/inverted/soc")
output_dir.mkdir(parents=True, exist_ok=True)

# SOC run uses its own SCF's E_F, NOT the non-SOC one -- they differ slightly
# (same reasoning as plot_soc_spin_texture.py).
def scf_efermi(scf_dir):
    return Vasprun(str(scf_dir / "vasprun.xml"), parse_dos=True,
                   parse_eigen=False, parse_potcar_file=False).efermi

ef = scf_efermi(SOC / "scf")
print("Inverted SOC bands/DOS E-fermi (from SOC/scf) =", ef)

bs = BSVasprun(str(SOC / "bandstructure" / "vasprun.xml"),
               parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(SOC / "bandstructure" / "KPOINTS"),
    line_mode=True, efermi=ef)

dosv = Vasprun(str(SOC / "dos" / "vasprun.xml"),
               parse_dos=True, parse_eigen=False, parse_potcar_file=False)
cdos = dosv.complete_dos
print("Inverted SOC DOS E-fermi (own run)   =", dosv.efermi)

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
