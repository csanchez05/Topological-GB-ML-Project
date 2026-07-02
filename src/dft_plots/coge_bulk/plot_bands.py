from pathlib import Path

# Compatibility shim: pyvista >= 0.45 renamed the public NORMALS constant to the
# private _NORMALS, which pyprocar 6.x still imports at package init (for its 3D
# Fermi plotting, unused here). Restore the alias before importing pyprocar.
import pyvista.core.utilities as _pv_util
from pyvista.core.utilities.helpers import _NORMALS as _pv_normals
if not hasattr(_pv_util, "NORMALS"):
    _pv_util.NORMALS = _pv_normals

import pyprocar
from pymatgen.io.vasp.outputs import Vasprun

DATA = Path("/home/calvi/Research_Group/ML_Interface_Project/data/dft_calculations/CoGe/bulk")
output_dir = Path("/home/calvi/Research_Group/ML_Interface_Project/plots/dft_plots/CoGe/bulk")
output_dir.mkdir(parents=True, exist_ok=True)

# Reliable Fermi level from the SCF run (NOT the band-structure run).
# If you only copied scf/OUTCAR, run `grep E-fermi scf/OUTCAR` and replace
# the next line with:  efermi = <that number>
# efermi lives inside the <dos> block of vasprun.xml, so parse_dos must be True
# or pymatgen returns efermi=None and the bands are not shifted to E_F.
efermi = Vasprun(str(DATA / "scf" / "vasprun.xml"),
                 parse_dos=True, parse_eigen=False,
                 parse_potcar_file=False).efermi
print("SCF E-fermi =", efermi)

pyprocar.bandsplot(
    code="vasp",
    dirname=str(DATA / "bandstructure"),
    mode="plain",
    fermi=efermi,
    elimit=[-3, 3],
    colorbar_title_size=16,
    colorbar_tick_labelsize=12,
    savefig=str(output_dir / "bands_nosoc.png"),
)
print("wrote", output_dir / "bands_nosoc.png")