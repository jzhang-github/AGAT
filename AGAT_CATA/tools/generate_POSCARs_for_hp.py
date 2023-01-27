# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:47:49 2022

@author: 18326
"""

import numpy as np
from ase.io import read, write
from pymatgen.core.ion import Ion

fname = 'POSCAR_example'

atoms = read(fname)
tot_num = len(atoms)

formulas = np.loadtxt('formulas.txt', dtype=str)
for chemical_formula in formulas:
    comp             = Ion.from_formula(chemical_formula)
    elements         = [x.name for x in comp.elements]
    atomic_fracions  = [comp.get_atomic_fraction(x) for x in elements]

    atom_num = [int(x * tot_num) for x in atomic_fracions]
    atom_num[-1] += tot_num - np.sum(atom_num)
    symbols = []
    for i,n in enumerate(atom_num):
        symbols += [elements[i] for y in range(n)]
    atoms.set_chemical_symbols(symbols)
    write(f'POSCAR_{chemical_formula}', atoms)
