# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:30:37 2022

@author: 18326
"""

import sys
assert len(sys.argv) > 1, 'Usage: command + file name'

from ase.io import read, write
# from ase.constraints import FixAtoms
import numpy as np

fname = sys.argv[1]

atoms = read(fname)
pos = atoms.positions

np.random.shuffle(pos)
atoms.set_positions(pos)
write(f'{fname}_random', atoms)
