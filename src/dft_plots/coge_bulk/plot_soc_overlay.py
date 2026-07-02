from pathlib import Path
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter

DATA = Path("/home/calvi/Research_Group/ML_Interface_Project/data/dft_calculations/CoGe/bulk")
output_dir = Path("/home/calvi/Research_Group/ML_Interface_Project/plots/dft_plots/CoGe/bulk")
output_dir.mkdir(parents=True, exist_ok=True)

def scf_efermi(scf_dir):
    # efermi lives inside the <dos> block, so parse_dos must be True or
    # pymatgen returns None.
    return Vasprun(str(scf_dir / "vasprun.xml"), parse_dos=True,
                   parse_eigen=False, parse_potcar_file=False).efermi

ef_nosoc = scf_efermi(DATA / "scf")
ef_soc   = scf_efermi(DATA / "SOC" / "scf")
print("E-fermi  non-SOC =", ef_nosoc, " SOC =", ef_soc)

bs_nosoc = BSVasprun(str(DATA / "bandstructure" / "vasprun.xml"),
                     parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(DATA / "bandstructure" / "KPOINTS"),
    line_mode=True, efermi=ef_nosoc)

bs_soc = BSVasprun(str(DATA / "SOC" / "bandstructure" / "vasprun.xml"),
                   parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(DATA / "SOC" / "bandstructure" / "KPOINTS"),
    line_mode=True, efermi=ef_soc)

plotter = BSPlotter(bs_nosoc)   # non-SOC first
plotter.add_bs(bs_soc)          # SOC overlaid in a second color
plotter.get_plot(ylim=(-2, 2))
plt.savefig(output_dir / "bands_soc_vs_nosoc.png", dpi=300)
print("wrote", output_dir / "bands_soc_vs_nosoc.png")