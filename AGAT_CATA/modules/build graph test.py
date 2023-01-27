# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 20:47:16 2022

@author: ZHANG Jun
"""

import numpy as np
import os
from pymatgen.core.structure import Structure
from modules.PymatgenStructureAnalyzer import VoronoiConnectivity
from modules.get_atomic_features import get_atomic_feature_onehot, get_atomic_features
import dgl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dgl.data.utils import save_graphs, load_graphs
import multiprocessing
from tqdm import tqdm
import json
from ase.io import read
import ase
from ase.neighborlist import natural_cutoffs
import itertools


ase_atoms = read('POSCAR.txt')
num_sites = len(ase_atoms)
ase_cutoffs = natural_cutoffs(ase_atoms, mult=1.25, H=3.0, O=3.0)
i, j, d, D = ase.neighborlist.neighbor_list('ijdD',
                                                    ase_atoms,
                                                    cutoff=ase_cutoffs,
                                                    self_interaction=False)

bonds = np.vstack((i,j)).T
bonds = np.sort(bonds, axis=1)
bonds = np.unique(bonds, axis=0)
num_bonds = np.shape(bonds)[0]

bond_around_atom = [[]for x in range(num_sites)]
for bond in bonds:
    bond_around_atom[bond[0]].append(bond[1])
    bond_around_atom[bond[1]].append(bond[0])

sender, receiver, edge = [], [], []
for atom_i in range(num_sites):
    pass












result = []
for atom_i in range(num_sites):
    Ns = bond[np.where(bond==atom_i)]
    for N in itertools.combinations(Ns, 2):
        result.append(list(N)+[atom_i])









import time
import itertools
start = time.time()
result = []
num_atom = len(ase_atoms)





# import time
# start = time.time()
# nodes = np.vstack((i,j)).T
# bond_num = len(i)
# result = []
# for bond_i in range(bond_num):
#     for bond_j in range(bond_i+1, bond_num):
#         result.append(np.vstack((nodes[i], nodes[j])))
# print(time.time()-start)
### Too expensive!!!

# start = time.time()
# bond_num = len(i)
# nodes = list(np.vstack((i,j)).T)
# ii, jj = np.meshgrid(nodes, nodes, sparse=False)
# dstack = np.dstack((ii,jj))
# np.vstack(np.dstack((ii,jj)))


# kk = ii + jj



# i = [0,0,0,1,1,2]
# j = [1,2,3,2,3,3]
# nodes = np.vstack((i,j)).T

# ii, jj, ij, ji = np.meshgrid(i,j,i,j, sparse=False)
# np.dstack([ii, jj, ij, ji])

# aaa = np.array()

# nodes = list(np.vstack((i,j)).T)
# ii, jj = np.meshgrid(nodes, nodes, sparse=True)
# edges =
# print(time.time()-start)




# angle = ase_atoms.get_angle(1,0,2)
# angles = ase_atoms.get_angles([[1,0,2],[0,1,2],[1,2,0],[2,0,1]])

# # distance_matrix = ase_atoms.get_all_distances()


# x = [1,2,3]
# y = [1,2,3]
# xx, yy = np.meshgrid(x, y)
# zz = xx**2 + yy**2

