# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 10:52:44 2025

@author: ZHANG Jun
"""

###############################################################################
# Detect GPU card
###############################################################################
import torch
if torch.cuda.is_available():
    device='cuda'
    print("CUDA is available.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    device='cpu'
    print("CUDA is NOT available.")

import dgl
u, v = torch.tensor([0, 0, 0, 1], device=device), torch.tensor([1, 2, 3, 3], 
                                                               device=device)
g = dgl.graph((u, v), device=device)
print(f'DGL graph device: {g.device}')

###############################################################################
# AGAT calculators with torch
###############################################################################
from agat.app.calculators import AgatCalculator, AgatCalculatorAseGraphTorch
from agat.app.calculators import AgatCalculatorAseGraphTorchNumpy
from agat.app.calculators import AgatEnsembleCalculator, OnTheFlyCalculator
import os
from ase.io import read, write

# --- 1 --- AgatCalculator
model_save_dir = os.path.join('potential_models', 'agat_model_1')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatCalculator(model_save_dir, graph_build_scheme_dir, 
                          device=device)
calculator.calculate(atoms)
print(calculator.results)

# --- 2 --- AgatCalculatorAseGraphTorch
model_save_dir = os.path.join('potential_models', 'agat_model_1')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatCalculatorAseGraphTorch(model_save_dir, graph_build_scheme_dir,
                                       device=device)
calculator.calculate(atoms)
print(calculator.results)

# --- 3 --- AgatCalculatorAseGraphTorchNumpy
model_save_dir = os.path.join('potential_models', 'agat_model_1')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatCalculatorAseGraphTorchNumpy(model_save_dir, 
                                            graph_build_scheme_dir, 
                                            device=device)
calculator.calculate(atoms)
print(calculator.results)

# --- 4 --- AgatEnsembleCalculator
model_ensemble_dir = os.path.join('potential_models')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatEnsembleCalculator(model_ensemble_dir, graph_build_scheme_dir, 
                                  device=device)
calculator.calculate(atoms)
print(calculator.results)

# --- 5 --- OnTheFlyCalculator
# model_ensemble_dir = os.path.join('potential_models')
# graph_build_scheme_dir = os.path.join('potential_models')
# atoms = read(os.path.join('potential_models', 'POSCAR'))
# calculator=OnTheFlyCalculator(model_ensemble_dir, graph_build_scheme_dir, 
#                                   device=device)
# calculator.calculate(atoms)
# print(calculator.results)

###############################################################################
# Optimizers
###############################################################################
import os
from ase.io import read, write
from ase import Atoms
from agat.app.optimizers import BFGSTorch, MDMinTorch
from agat.app.calculators import AgatCalculator
# --- 1 --- BFGSTorch
model_save_dir = os.path.join('potential_models', 'agat_model_1')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatCalculator(model_save_dir,
                          graph_build_scheme_dir,
                          device=device)
atoms = Atoms(atoms, calculator=calculator)
dyn = BFGSTorch(atoms, trajectory='test.traj', device=device)
dyn.run(fmax=1.0)
traj = read('test.traj', index=':')
write("XDATCAR.gat", traj)

# --- 2 --- MDMinTorch
model_save_dir = os.path.join('potential_models', 'agat_model_1')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatCalculator(model_save_dir,
                          graph_build_scheme_dir,
                          device=device)
atoms = Atoms(atoms, calculator=calculator)
dyn = MDMinTorch(atoms, trajectory='test.traj', device=device)
dyn.run(fmax=1.0)

# --- 3 --- ase.optimize.BFGS
from ase.optimize import BFGS
model_save_dir = os.path.join('potential_models', 'agat_model_1')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatCalculator(model_save_dir,
                          graph_build_scheme_dir,
                          device=device)
atoms = Atoms(atoms, calculator=calculator)
dyn = BFGS(atoms, trajectory='test.traj')
dyn.run(fmax=1.0)

###############################################################################
# ASE ensembles for AGAT
###############################################################################
import os
from ase.io import read
from ase import units
from ase.md.npt import NPT
from ase.md import MDLogger
from agat.app.ensembles import ModifiedNPT
from agat.app.calculators import AgatEnsembleCalculator
model_ensemble_dir = os.path.join('potential_models')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatEnsembleCalculator(model_ensemble_dir, graph_build_scheme_dir, 
                                  device=device)
atoms.set_calculator(calculator)

dyn = ModifiedNPT(atoms,
          timestep=1.0 * units.fs,
          temperature_K=300,
          ttime = 25 * units.fs,
          pfactor = 75 * units.fs,
          externalstress = [0.0] * 6,
          mask=[[1,0,0],
                [0,1,0],
                [0,0,1]],
          trajectory=os.path.join('md_NPT.traj'))

dyn.attach(MDLogger(dyn, atoms, os.path.join('md_NPT.log'),
                    header=True,
                    stress=True,
                    peratom=False,
                    mode="a"),
           interval=1)

dyn.run(200)

###############################################################################
# dataset / database
###############################################################################
from agat.data import Dataset, concat_dataset
from agat.data import select_graphs_from_dataset_random
dataset = Dataset('graphs.bin')
da = dataset[11]
db = dataset[1:4]
dc = concat_dataset(da, db)
dd = select_graphs_from_dataset_random(dataset, 10, save_file=False,
                                       fname=None)
print(dataset, da, db, dc, dd)

from agat.data import select_graphs_from_dataset_random
de = select_graphs_from_dataset_random(dd, 3, save_file=False, fname=None)
print(de)

from agat.data import save_dataset
save_dataset(dd, fname='new_dataset.bin')