from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

plotter = BSPlotter(bs_nosoc)   # non-SOC first  -> color C0
plotter.add_bs(bs_soc)          # SOC overlaid   -> color C1
ax = plotter.get_plot(ylim=(-2, 2))

# BSPlotter labels each line by its internal band/spin index ("Band 0 up",
# "Band 0 down", "Band 1 up"), which says nothing about SOC. Replace the legend
# with one entry per calculation. BSPlotter colors by the order band structures
# were added (0 -> non-SOC, 1 -> SOC) using matplotlib's default color cycle.
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ax.legend(handles=[
    Line2D([], [], color=colors[0], lw=2, label="non-SOC"),
    Line2D([], [], color=colors[1], lw=2, label="SOC"),
], fontsize=16, loc="upper right")

plt.savefig(output_dir / "bands_soc_vs_nosoc.png", dpi=800)
print("wrote", output_dir / "bands_soc_vs_nosoc.png")