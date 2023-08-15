# This script is used to find atoms connected with specified atomic index.
from pymatgen.core.structure import Structure
import numpy as np
import sys
from collections import Counter


def find_adsorption_site():
    assert len(sys.argv) > 3, 'Usage: command + file name + atomic index (start from zero) + distance cutoff'
    fname = sys.argv[1]
    atom_num = sys.argv[2]
    dist_cutoff = sys.argv[3]



    crystal = Structure.from_file(fname)
    element_list = [crystal.sites[x].specie.name for x in range(crystal.num_sites)]
    dist_mat = crystal.distance_matrix[int(atom_num)]
    OneNN_index = np.where(dist_mat < float(dist_cutoff))

    # sort_index = sorted(range(len(dist_mat)), key=lambda k: dist_mat[k])
    ele = np.array(element_list)[OneNN_index]
    for i in ['O', 'H']:
        location = np.where(ele == i)
        ele = np.delete(ele, location)
    for i in ele:
        print(i, end='  ')
    print()
