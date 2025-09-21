# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:36:19 2023

@author: ZHANG Jun
"""

import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
import time
from ase.io import read
from ase import Atoms

model_save_dir = os.path.join('acatal', 'test', 'agat_model')
graph_build_scheme_dir = os.path.join('acatal', 'test', 'agat_model')
fname = os.path.join('acatal', 'test', 'POSCAR_NiCoFePdPt')
# fname = os.path.join('acatal', 'test', 'POSCAR_2')
# fname = os.path.join('acatal', 'test', 'POSCAR_big')
# fname = os.path.join('acatal', 'test', 'CONTCAR_new_3')

###############################################################################
# ase.optimize.MDMin optimizer, PyTorch-based graph construction
from ase.optimize import MDMin
from acatal.ase_torch.calculators import AgatCalculatorFastGraphNumpy

acfgn = AgatCalculatorFastGraphNumpy(model_save_dir,
                                     graph_build_scheme_dir, device='cpu')

start_time = time.time()
for _ in range(1):
    acfgn.ag.reset() # important !!!
    atoms = read(fname)
    atoms = Atoms(atoms, calculator=acfgn)
    dyn = MDMin(atoms, trajectory=os.path.join('acatal', 'test', 'test_new_ttt.traj'))
    dyn.run(fmax=0.01, steps=3)

print('ase.optimize.MDMin, PyTorch Graph:', time.time()-start_time, 's')

###############################################################################
# MDMinTorch optimizer, PyTorch-based graph construction
from acatal.ase_torch.ase_dyn import MDMinTorch
from acatal.ase_torch.calculators import AgatCalculatorFastGraph

acfg = AgatCalculatorFastGraph(model_save_dir,
                               graph_build_scheme_dir, device='cpu')

start_time = time.time()
for _ in range(1):
    acfg.ag.reset() # important !!!
    atoms = read(fname)
    atoms = Atoms(atoms, calculator=acfg)
    dyn = MDMinTorch(atoms, trajectory=os.path.join('acatal', 'test', 'test_new_ttt.traj'),
                     device='cpu')
    dyn.run(fmax=0.01, steps=30)

print('MDMinTorch, PyTorch Graph:', time.time()-start_time, 's')

###############################################################################
# ase.optimize.BFGS optimizer, PyTorch-based graph construction
from ase.optimize import BFGS
from acatal.ase_torch.calculators import AgatCalculatorFastGraphNumpy

acfgn = AgatCalculatorFastGraphNumpy(model_save_dir,
                                      graph_build_scheme_dir, device='cpu')

start_time = time.time()
for _ in range(1):
    acfgn.ag.reset() # important !!!
    atoms = read(fname)
    atoms = Atoms(atoms, calculator=acfgn)
    dyn = BFGS(atoms, trajectory=os.path.join('acatal', 'test', 'test_new_ttt.traj'))
    dyn.run(fmax=0.01, steps=3)

print('ase.optimize.BFGS, PyTorch Graph:', time.time()-start_time, 's')

###############################################################################
# previous optimizer, previous graph construction
from ase.optimize import BFGS
from acatal.ase_torch.calculators import AgatCalculator

calculator=AgatCalculator(model_save_dir,
                          graph_build_scheme_dir, device='cpu')

start_time = time.time()
for _ in range(1):
    atoms = read(fname)
    atoms = Atoms(atoms, calculator=calculator)
    dyn = BFGS(atoms, trajectory=os.path.join('acatal', 'test', 'test.traj'))
    dyn.run(steps=3, fmax=0.01)
print('ase.optimize.BFGS, ASE graph:', time.time()-start_time, 's')

##############################################################################
# PyTorch-based BFGS, PyTorch-based graph construction

from acatal.ase_torch.calculators import AgatCalculatorFastGraph
from acatal.ase_torch.bfgs_torch import BFGSTorch

acfg = AgatCalculatorFastGraph(model_save_dir,
                               graph_build_scheme_dir, device='cpu')

start_time = time.time()
for _ in range(1):
    acfg.ag.reset() # important !!!
    atoms = read(fname)
    atoms = Atoms(atoms, calculator=acfg)
    dyn = BFGSTorch(atoms, trajectory=os.path.join('acatal', 'test', 'test_new.traj'), device='cpu')
    dyn.run(fmax=0.01, steps=3)

print('BFGSTorch, PyTorch Graph:', time.time()-start_time, 's')

# # compare
# force0 = calculator.force_log[0]
# force1 = calculator.force_log[5]
# force2 = calculator.force_log[11]
# es = calculator.energy_log
# xdatcar = read(os.path.join('acatal', 'test', 'test.traj'), index=':')
# xdatcar[0].write(os.path.join('acatal', 'test', 'CONTCAR_0'))
# xdatcar[1].write(os.path.join('acatal', 'test', 'CONTCAR_1'))

# force_new0 = acfg.force_log[0]
# force_new1 = acfg.force_log[5]
# force_new2 = acfg.force_log[11]
# es_new = acfg.energy_log
# xdatcar_new = read(os.path.join('acatal', 'test', 'test_new.traj'), index=':')
# xdatcar_new[0].write(os.path.join('acatal', 'test', 'CONTCAR_new_0'))
# xdatcar_new[1].write(os.path.join('acatal', 'test', 'CONTCAR_new_1'))
# xdatcar_new[3].write(os.path.join('acatal', 'test', 'CONTCAR_new_3'))

# force0 - force_new0
# force1 - force_new1
# force2 - force_new2

###############################################################################
# import torch
# class GeoAdam(object):
#     def __init__(self, lr=0.01):
#         self.lr = lr
#         optimizer = torch.optim.Adam(pos, lr=self.lr)

#     def run(atoms, steps=200, fmax=0.05):

#         for step in range(steps):
#             self.optimizer.zero_grad()
#             if f > fmax:
#                 break
