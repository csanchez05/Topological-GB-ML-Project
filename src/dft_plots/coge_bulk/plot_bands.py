from pathlib import Path
import pyprocar
from pymatgen.io.vasp.outputs import Vasprun

DATA = Path("/home/calvi/Research_Group/ML_Interface_Project/data/dft_calculations/CoGe/bulk")

# Reliable Fermi level from the SCF run (NOT the band-structure run).
# If you only copied scf/OUTCAR, run `grep E-fermi scf/OUTCAR` and replace
# the next line with:  efermi = <that number>
efermi = Vasprun(str(DATA / "scf" / "vasprun.xml"),
                 parse_dos=False, parse_eigen=False,
                 parse_potcar_file=False).efermi
print("SCF E-fermi =", efermi)

pyprocar.bandsplot(
    code="vasp",
    dirname=str(DATA / "bandstructure"),
    mode="plain",
    fermi=efermi,
    elimit=[-3, 3],
    savefig="bands_nosoc.png",
)