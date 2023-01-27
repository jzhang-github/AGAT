# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:40:38 2022

@author: 18326
"""

import sys
assert len(sys.argv) > 2, 'Usage: command + file name + atom layer in this file'

from ase.io import read, write
from ase.constraints import FixAtoms
import numpy as np

fname = sys.argv[1]
atom_layer = int(sys.argv[2])

out_name = fname+'_with_vacuum'
atoms = read(fname)
atoms.wrap()
cart   = atoms.get_positions()
cell   = atoms.cell.array
layer_space = 1 / atom_layer * cell[2][2]
pop_list = np.concatenate((np.where(cart[:,2] < 0.5 * layer_space)[0],
                      np.where(cart[:,2] > cell[2][2] - 0.5 * layer_space)[0]), axis=0)

cell[2][2] += 10
atoms.set_cell(cell)
z_frac_cutoff = 2.5 * layer_space
fix_location = np.where(cart[:,2] < z_frac_cutoff)
c = FixAtoms(indices=fix_location[0])

atoms.set_constraint(c)
del atoms[pop_list]
write(out_name, atoms)
