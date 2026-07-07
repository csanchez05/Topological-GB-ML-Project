from pathlib import Path
import pyvista.core.utilities as _pv_util
from pyvista.core.utilities.helpers import _NORMALS as _pv_normals
if not hasattr(_pv_util, "NORMALS"):
    _pv_util.NORMALS = _pv_normals

import pyprocar
from pymatgen.io.vasp.outputs import Vasprun

DATA = Path("/home/calvi/Research_Group/ML_Interface_Project/data/dft_calculations/CoGe/bulk/inverted")
BS = str(DATA / "bandstructure")
output_dir = Path("/home/calvi/Research_Group/ML_Interface_Project/plots/dft_plots/CoGe/bulk/inverted")
output_dir.mkdir(parents=True, exist_ok=True)

# efermi lives inside the <dos> block of vasprun.xml, so parse_dos must be True
# or pymatgen returns efermi=None and the bands are not shifted to E_F.
efermi = Vasprun(str(DATA / "scf" / "vasprun.xml"),
                 parse_dos=True, parse_eigen=False,
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
        savefig=str(output_dir / f"proj_{label}.png"), show=False,
    )
    print("wrote", output_dir / f"proj_{label}.png")