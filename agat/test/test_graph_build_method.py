# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:18:57 2023

@author: ZHANG Jun
"""

''' Using old way '''
from agat.data import CrystalGraph
from agat.default_parameters import default_data_config
import time

default_data_config['mode_of_NN'] = 'ase_dist'
default_data_config['build_properties']['energy'] = False
default_data_config['build_properties']['forces'] = False
default_data_config['build_properties']['cell'] = False
default_data_config['build_properties']['stress'] = False

cg = CrystalGraph(**default_data_config)

if __name__ == '__main__':
    start = time.time()
    for i in range(50):
        g = cg.get_graph('POSCAR')

    print('Dur:', time.time() - start)


''' New method '''
from agat.lib.model_lib import config_parser
from agat.data.atomic_feature import get_atomic_feature_onehot
from ase.io import read
import torch
import numpy as np

class AseGraph(object):
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}
        self.all_atom_feat = get_atomic_feature_onehot(self.data_config['species'])
        self.cutoff = self.data_config['cutoff']
        self.skin = 1.0 # angstrom
        self.new_graph = True
        self.step = 0
        self.new_graph_steps = 20 # buid new graph every * steps.
        self.inner_senders = []
        self.inner_receivers = []
        self.skin_senders = []
        self.skin_receivers = []
        self.graph = None
        self.device = 'cuda'

    def reset(self):
        # reset parameters
        self.new_graph = True
        self.step = 0
        self.g_senders = []
        self.g_receivers = []
        self.skin_senders = []
        self.skin_receivers = []
        self.graph = None

    def get_scaled_positions(self, cell, positions): # cell = np.mat(cell), positions = positions
        scaled_positions = torch.matmul(C_coord_tmp, cell.I)
        scaled_positions = np.dot(C_coord_tmp, cell.I)
        pass

    def get_connections(self, ase_atoms):
        # get senders and receivers, including inner and skin connections.
        positions = ase_atoms.positions
        cell = ase_atoms.cell.array
        pass

    def build(self, fname):
        # build graph from file
        # Include all possible properties.
        ndata = self.get_ndata(ase_atoms)
        pass

    def update(self):
        # calculate d and D, reassign i and j
        pass

    def get_graph(self, fname):
        pass

    def get_graph_from_connections(self): # ase_atoms or file name

        i, j, d, D = ase.neighborlist.neighbor_list('ijdD', # i: sender; j: receiver; d: distance; D: direction
                                                    ase_atoms,
                                                    cutoff=ase_cutoffs,
                                                    self_interaction=True)

        ndata                   = self.get_ndata(ase_atoms)
        bg                      = dgl.graph((i, j))
        bg.ndata['h']           = torch.tensor(ndata, dtype=self.dtype)
        if self.data_config['build_properties']['distance']:
            bg.edata['dist']        = torch.tensor(d, dtype=self.dtype)
        if self.data_config['build_properties']['direction']:
            bg.edata['direction']   = torch.tensor(D, dtype=self.dtype)
        if self.data_config['build_properties']['constraints']:
            constraints             = [[1, 1, 1]] * num_sites
            for c in ase_atoms.constraints:
                if isinstance(c, ase.constraints.FixScaled):
                    constraints[c.a] = c.mask
                elif isinstance(c, ase.constraints.FixAtoms):
                    for i in c.index:
                        constraints[i] = [0, 0, 0]
                elif isinstance(c, ase.constraints.FixBondLengths):
                    pass
                else:
                    raise TypeError(f'Wraning!!! Undefined constraint type: {type(c)}')
            bg.ndata['constraints'] = torch.tensor(constraints, dtype=self.dtype)
        if self.data_config['build_properties']['forces']:
            forces_true             = torch.tensor(np.load(fname+'_force.npy'), dtype=self.dtype)
            bg.ndata['forces_true'] = forces_true
        if self.data_config['build_properties']['cart_coords']:
            bg.ndata['cart_coords'] = torch.tensor(ase_atoms.positions, dtype=self.dtype)
        if self.data_config['build_properties']['frac_coords']:
            bg.ndata['frac_coords'] = torch.tensor(ase_atoms.get_scaled_positions(), dtype=self.dtype)
        if self.data_config['has_adsorbate']:
            element_list            = ase_atoms.get_chemical_symbols()
            bg.ndata['adsorbate']   = self.get_adsorbate_bool(element_list)

        graph_info = {}
        if self.data_config['build_properties']['energy']:
            energy_true = torch.tensor(np.load(fname+'_energy.npy'), dtype=self.dtype)
            graph_info['energy_true'] = energy_true
        if self.data_config['build_properties']['stress']:
            stress_true = torch.tensor(np.load(fname+'_stress.npy'), dtype=self.dtype)
            graph_info['stress_true'] = stress_true
        if self.data_config['build_properties']['cell']:
            cell_true = torch.tensor(ase_atoms.cell.array, dtype=self.dtype)
            graph_info['cell_true'] = cell_true
        if self.data_config['build_properties']['path']:
            graph_info['path'] = fname

        return bg, graph_info
