# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 10:52:44 2025

@author: JXZ
"""

# GPU card
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
u, v = torch.tensor([0, 0, 0, 1], device=device), torch.tensor([1, 2, 3, 3], device=device)
g = dgl.graph((u, v), device='cuda')
print(f'DGL graph device: {g.device}')

###############################################################################
# AGAT calculators with torch
###############################################################################
from agat.app.calculators import AgatCalculator, AgatCalculatorAseGraphTorch, AgatCalculatorAseGraphTorchNumpy, AgatEnsembleCalculator, OnTheFlyCalculator
import os
from ase.io import read

# --- 1 --- AgatCalculator
model_save_dir = os.path.join('potential_models', 'agat_model_1')
graph_build_scheme_dir = os.path.join('potential_models')
atoms = read(os.path.join('potential_models', 'POSCAR'))
calculator=AgatCalculator(model_save_dir, graph_build_scheme_dir, device='cuda')
calculator.calculate(atoms)
print(calculator.results)

# --- 2 --- AgatCalculatorAseGraphTorch

# --- 3 --- AgatCalculatorAseGraphTorchNumpy

# --- 4 --- AgatEnsembleCalculator

# --- 5 --- OnTheFlyCalculator




###############################################################################
# Optimizers
###############################################################################


###############################################################################
# ASE ensembles for AGAT
###############################################################################


###############################################################################
# dataset / database
###############################################################################

