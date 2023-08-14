# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:16:03 2022

@author: 18326

Prepare the POSCAR files for additional dataset (NiCoFePdPt) for transfer learning.
Nonequi aotmic compositions.

formulars:
Ni.10000000000000000000Co.10000000000000000000Fe.10000000000000000000Pd0.1Pt.6
Ni.10000000000000000000Co.10000000000000000000Fe.10000000000000000000Pd0.2Pt.5
Ni.10000000000000000000Co.10000000000000000000Fe.10000000000000000000Pd0.3Pt.4
Ni.10000000000000000000Co.10000000000000000000Fe.10000000000000000000Pd0.4Pt.3
Ni.10000000000000000000Co.10000000000000000000Fe.10000000000000000000Pd0.5Pt.2
Ni.10000000000000000000Co.10000000000000000000Fe.10000000000000000000Pd0.6Pt.1
Ni.10000000000000000000Co.10000000000000000000Fe.10000000000000000000Pd0.7Pt0
Ni.13333333333333333333Co.13333333333333333333Fe.13333333333333333333Pd0.1Pt.5
Ni.13333333333333333333Co.13333333333333333333Fe.13333333333333333333Pd0.2Pt.4
Ni.13333333333333333333Co.13333333333333333333Fe.13333333333333333333Pd0.3Pt.3
Ni.13333333333333333333Co.13333333333333333333Fe.13333333333333333333Pd0.4Pt.2
Ni.13333333333333333333Co.13333333333333333333Fe.13333333333333333333Pd0.5Pt.1
Ni.13333333333333333333Co.13333333333333333333Fe.13333333333333333333Pd0.6Pt0
Ni.16666666666666666666Co.16666666666666666666Fe.16666666666666666666Pd0.1Pt.4
Ni.16666666666666666666Co.16666666666666666666Fe.16666666666666666666Pd0.2Pt.3
Ni.16666666666666666666Co.16666666666666666666Fe.16666666666666666666Pd0.3Pt.2
Ni.16666666666666666666Co.16666666666666666666Fe.16666666666666666666Pd0.4Pt.1
Ni.16666666666666666666Co.16666666666666666666Fe.16666666666666666666Pd0.5Pt0
Ni.20000000000000000000Co.20000000000000000000Fe.20000000000000000000Pd0.1Pt.3
Ni.20000000000000000000Co.20000000000000000000Fe.20000000000000000000Pd0.2Pt.2
Ni.20000000000000000000Co.20000000000000000000Fe.20000000000000000000Pd0.3Pt.1
Ni.20000000000000000000Co.20000000000000000000Fe.20000000000000000000Pd0.4Pt0
Ni.23333333333333333333Co.23333333333333333333Fe.23333333333333333333Pd0.1Pt.2
Ni.23333333333333333333Co.23333333333333333333Fe.23333333333333333333Pd0.2Pt.1
Ni.23333333333333333333Co.23333333333333333333Fe.23333333333333333333Pd0.3Pt0
Ni.26666666666666666666Co.26666666666666666666Fe.26666666666666666666Pd0.1Pt.1
Ni.26666666666666666666Co.26666666666666666666Fe.26666666666666666666Pd0.2Pt0
Ni.30000000000000000000Co.30000000000000000000Fe.30000000000000000000Pd0.1Pt0
"""

from pymatgen.core.ion import Ion
from ase.io import read, write
from ase.visualize import view
from ase.build import sort
import numpy as np

def get_frac_from_fromula(chemical_formula):
    comp             = Ion.from_formula(chemical_formula)
    elements         = comp.elements
    # elements = [x.name for x in elements]
    atomic_fracions  = [comp.get_atomic_fraction(x) for x in elements]
    return dict(zip(elements, atomic_fracions))

atoms = read('POSCAR_example')
total_num = len(atoms)

with open('formulars.txt', 'r') as f:
    formulas = f.readlines()
formulas = [x.strip() for x in formulas]

for f in formulas:
    new_atoms = atoms.copy()
    atomic_frac_dict   = get_frac_from_fromula(f)
    atomic_fracions    = list(atomic_frac_dict.values())
    num_atom_list      = np.array(atomic_fracions) * total_num
    num_atom_list      = np.around(num_atom_list, decimals=0)
    total_tmp          = np.sum(num_atom_list)
    deviation          = total_num - total_tmp
    num_atom_list[-1] += deviation

    elements = list(atomic_frac_dict.keys())
    elements = [e.name for e in elements]
    new_symbols    = []
    for i_index, i in enumerate(num_atom_list):
        for j in range(int(i)):
            new_symbols.append(elements[i_index])
    np.random.shuffle(new_symbols)
    new_atoms.set_chemical_symbols(new_symbols)
    new_atoms = sort(new_atoms)
    write(f'POSCAR_{f}', new_atoms)
    print(num_atom_list)


# for f in formulas:
#     con = get_frac_from_fromula(f)
#     for i in ['Ni', 'Co', 'Fe', 'Pd', 'Pt']:
#         if not con.__contains__(i):
#             con[i]=0.0
#     print(con['Ni']+con['Co']+con['Fe'], con['Pd'], con['Pt'])
