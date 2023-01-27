# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:59:09 2022

@author: ZHANG Jun
"""


import numpy as np
from ase.io import read, write

def perturb_poscar(fname='POSCAR', amplitude=0.1):
    atoms = read(fname)
    posistions = atoms.arrays['positions']
    num_atoms = len(atoms)
    increment = np.random.uniform(-amplitude, amplitude, (num_atoms,3))
    posistions += increment
    atoms.set_positions(posistions)
    write(f'{fname}_perturb', atoms)

perturb_poscar()
