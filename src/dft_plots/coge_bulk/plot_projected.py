from pathlib import Path
import pyprocar
from pymatgen.io.vasp.outputs import Vasprun

DATA = Path("/home/calvi/Research_Group/ML_Interface_Project/data/dft_calculations/CoGe/bulk")
BS = str(DATA / "bandstructure")

efermi = Vasprun(str(DATA / "scf" / "vasprun.xml"),
                 parse_dos=False, parse_eigen=False,
                 parse_potcar_file=False).efermi

projections = [
    ("Co_s", [0, 1, 2, 3], [0]),
    ("Co_p", [0, 1, 2, 3], [1, 2, 3]),
    ("Co_d", [0, 1, 2, 3], [4, 5, 6, 7, 8]),
    ("Ge_s", [4, 5, 6, 7], [0]),
    ("Ge_p", [4, 5, 6, 7], [1, 2, 3]),
    ("Ge_d", [4, 5, 6, 7], [4, 5, 6, 7, 8]),
]

for label, atoms, orbitals in projections:
    pyprocar.bandsplot(
        code="vasp", dirname=BS, mode="parametric",
        fermi=efermi, atoms=atoms, orbitals=orbitals,
        cmap="jet", elimit=[-3, 3],
        title=label.replace("_", " "),
        savefig=f"proj_{label}.png", show=False,
    )
    print("wrote", f"proj_{label}.png")