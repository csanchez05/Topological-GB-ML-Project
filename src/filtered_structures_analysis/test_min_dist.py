from ase.io import read
from ase.neighborlist import neighbor_list
import numpy as np
import os
from viable_soap_extractor import collect_jobs

jobs = collect_jobs()
print("\nChecking minimum distances for the first 10 structures:")
for poscar_path, key in jobs[:10]:
    structure = read(poscar_path)
    i, j, d = neighbor_list('ijd', structure, 3.0)  # use 3.0 to see what's actually there
    if len(d) > 0:
        d_valid = d[d > 0.1]
        if len(d_valid) > 0:
            min_d = np.min(d_valid)
            num_under_2 = np.sum(d_valid < 2.0) / 2
            print(f"{key}: min_dist = {min_d:.3f} A, pairs < 2.0A = {num_under_2}")
        else:
            print(f"{key}: no pairs under 3.0A (excluding self)")
    else:
         print(f"{key}: empty distance list")
