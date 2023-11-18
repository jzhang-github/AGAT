# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:18:57 2023

@author: ZHANG Jun
"""

''' Using old way '''
# from agat.data import CrystalGraph
# from agat.default_parameters import default_data_config
# import time

# default_data_config['mode_of_NN'] = 'ase_dist'
# default_data_config['build_properties']['energy'] = False
# default_data_config['build_properties']['forces'] = False
# default_data_config['build_properties']['cell'] = False
# default_data_config['build_properties']['stress'] = False

# cg = CrystalGraph(**default_data_config)

# from ase.io import read
# ase_atoms = read('POSCAR')
# ase_atoms.write('POSCAR_50')
# ase_atoms.repeat((1,1,2)).write('POSCAR_100')
# ase_atoms.repeat((1,2,2)).write('POSCAR_200')
# ase_atoms.repeat((2,2,2)).write('POSCAR_400')
# ase_atoms.repeat((2,2,4)).write('POSCAR_800')
# ase_atoms.repeat((2,4,4)).write('POSCAR_1600')
# ase_atoms.repeat((4,4,4)).write('POSCAR_3200')

# if __name__ == '__main__':
#     for n in [50, 100, 200, 400, 800, 1600, 3200]:
#         start = time.time()
#         for i in range(30):
#             g = cg.get_graph(f'POSCAR_{n}')

#         print(time.time() - start)

''' New method '''
from agat.lib.model_lib import config_parser
from agat.data.atomic_feature import get_atomic_feature_onehot
from agat.default_parameters import default_data_config
import ase
from ase.io import read
import torch
import numpy as np
import dgl

class AseGraph(object):
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}
        self.all_atom_feat = get_atomic_feature_onehot(self.data_config['species'])
        self.cutoff = self.data_config['cutoff']
        self.skin = 1.0 # angstrom
        # self.adsorbate_skin = 2.0 # in many cases, the adsorbates experience larger geometry relaxation than surface atoms.

        self.step = 0
        self.new_graph_steps = 20 # buid new graph every * steps.
        self.inner_senders = None
        self.inner_receivers = None
        self.skin_senders = None
        self.skin_receivers = None
        self.graph = None
        self.num = None
        self.device = 'cuda'

    def reset(self):
        # reset parameters for new optimization.
        # self.new_graph = True
        self.step = 0
        self.inner_senders = None
        self.inner_receivers = None
        self.skin_senders = None
        self.skin_receivers = None
        self.graph = None
        self.num = None

    def get_ndata(self, ase_atoms):
        # print('get_ndata')
        ndata = []
        for i in ase_atoms.get_chemical_symbols():
            ndata.append(self.all_atom_feat[i])
        return torch.tensor(ndata, dtype=torch.float32)

    def get_adsorbate_bool(self, element_list):
        # print('get_adsorbate_bool')
        """
       .. py:method:: get_adsorbate_bool(self)

          Identify adsorbates based on elements： H and O.

          :return: a list of bool values.
          :rtype: tf.constant

        """
        element_list = np.array(element_list)
        return torch.tensor(np.where((element_list == 'H') | (element_list == 'O'),
                                     1, 0))

    def get_scaled_positions(self, cell_I_tensor, positions): # cell = np.mat(cell), positions = positions
        # print('get_scaled_positions')
        # cell_I: Returns the (multiplicative) inverse of invertible
        scaled_positions = torch.matmul(positions, cell_I_tensor)
        scaled_positions = torch.where(scaled_positions < 0.0, scaled_positions+1, scaled_positions)
        scaled_positions = torch.where(scaled_positions > 1.0, scaled_positions-1, scaled_positions)
        return scaled_positions

    def fractional2cartesian(self, cell_tensor, scaled_positions):
        # print('fractional2cartesian')
        positions = torch.matmul(scaled_positions, cell_tensor)
        return positions

    def get_pair_distances(self, a, b, ase_atoms):
        # print('get_pair_distances')
        # print(len(a))
        cell = ase_atoms.cell.array
        cell_tensor = torch.tensor(cell, dtype=torch.float32)
        cell_I_tensor = torch.tensor(np.array(np.mat(cell).I), dtype=torch.float32)
        positions = torch.tensor(ase_atoms.positions, dtype=torch.float32)
        scaled_positions = self.get_scaled_positions(cell_I_tensor, positions)
        a_positions = positions[a,:]
        a_scaled_positions = scaled_positions[a,:]
        b_scaled_positions = scaled_positions[b,:]
        diff_scaled_positions = b_scaled_positions - a_scaled_positions
        b_scaled_positions = torch.where(diff_scaled_positions > 0.5,
                                         b_scaled_positions-1.0,
                                         b_scaled_positions)
        b_scaled_positions = torch.where(diff_scaled_positions < -0.5,
                                         b_scaled_positions+1.0,
                                         b_scaled_positions)
        b_positions_new = self.fractional2cartesian(cell_tensor, b_scaled_positions)
        D = b_positions_new - a_positions
        d = torch.norm(D, dim=1)
        return d, D

    def get_all_distances(self, ase_atoms):
        # print('get_all_distances')
        # get senders and receivers, including inner and skin connections.
        # torch.from_numpy is memory effcient than torch.tensor, especially for large tensors.
        # No self loop and reverse direction.

        a, b = [], []
        for i in range(self.num):
            for j in range(i+1, self.num):
                a.append(i)
                b.append(j)
        a, b = torch.tensor(a), torch.tensor(b)
        d, D = self.get_pair_distances(a, b, ase_atoms)
        return a, b, d, D

    def get_init_connections(self, ase_atoms):
        # print('get_init_connections')
        i, j, d, D = self.get_all_distances(ase_atoms)
        inner_connections = torch.where(d < self.cutoff)
        skin_connections = torch.where((d > self.cutoff) & (d < self.cutoff+1))
        i_i, j_i, d_i, D_i = i[inner_connections], j[inner_connections], d[inner_connections], D[inner_connections]
        i_s, j_s, d_s, D_s = i[skin_connections], j[skin_connections], d[skin_connections], D[skin_connections]
        return i_i, j_i, d_i, D_i, i_s, j_s, d_s, D_s

    def update_connections(self, i_i, j_i, i_s, j_s, ase_atoms):
        # print('update_connections')
        i, j = torch.cat((i_i, i_s)), torch.cat((j_i, j_s))
        d, D = self.get_pair_distances(i, j, ase_atoms)
        inner_connections = torch.where(d < self.cutoff)
        skin_connections = torch.where((d > self.cutoff) & (d < self.cutoff+1))
        i_i, j_i, d_i, D_i = i[inner_connections], j[inner_connections], d[inner_connections], D[inner_connections]
        i_s, j_s, d_s, D_s = i[skin_connections], j[skin_connections], d[skin_connections], D[skin_connections]
        return i_i, j_i, d_i, D_i, i_s, j_s, d_s, D_s

    def build(self, ase_atoms):
        # print('build')
        # build graph
        # Include all possible properties.
        self.num = len(ase_atoms)

        ndata = self.get_ndata(ase_atoms)
        i_i, j_i, d_i, D_i, i_s, j_s, d_s, D_s = self.get_init_connections(ase_atoms)
        self.inner_senders = i_i
        self.inner_receivers = j_i
        self.skin_senders = i_s
        self.skin_receivers = j_s

        # add reverse direction and self loop connections.
        # Some properties will not change when update: h, constraints, adsorbate,
        bg = dgl.graph((torch.cat((self.inner_senders,
                                   self.inner_receivers,
                                   torch.arange(self.num)),
                                  dim=0),
                       torch.cat((self.inner_receivers,
                                  self.inner_senders,
                                  torch.arange(self.num)),
                                 dim=0)))
        bg.ndata['h'] = ndata
        bg.edata['dist'] = torch.cat((d_i, d_i, torch.zeros(self.num)), dim=0)
        bg.edata['direction']   = torch.cat((D_i, D_i, torch.zeros(self.num,3)), dim=0)
        constraints = [[1, 1, 1]] * self.num
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
        bg.ndata['constraints'] = torch.tensor(constraints, dtype=torch.float32)
        # bg.ndata['cart_coords'] = torch.tensor(ase_atoms.positions, dtype=torch.float32)
        # bg.ndata['frac_coords'] = torch.tensor(ase_atoms.get_scaled_positions(), dtype=torch.float32)
        element_list = ase_atoms.get_chemical_symbols()
        bg.ndata['adsorbate'] = self.get_adsorbate_bool(element_list)
        self.graph = bg.clone()
        return bg

    def update(self, ase_atoms):
        # print('update')
        # calculate d and D, reassign i and j
        i_i, j_i, d_i, D_i, i_s, j_s, d_s, D_s = self.update_connections(self.inner_senders,
                self.inner_receivers, self.skin_senders, self.skin_receivers, ase_atoms)
        self.g_senders = i_i
        self.g_receivers = j_i
        self.skin_senders = i_s
        self.skin_receivers = j_s
        bg = dgl.graph((torch.cat((self.g_senders,
                                  self.g_receivers,
                                  torch.arange(self.num)),
                                 dim=0),
                       torch.cat((self.g_receivers,
                                  self.g_senders,
                                  torch.arange(self.num)),
                                 dim=0)))
        bg.ndata['h'] = self.graph.ndata['h']
        bg.edata['dist'] = torch.cat((d_i, d_i, torch.zeros(self.num)),
                                     dim=0)
        bg.edata['direction']   = torch.cat((D_i, D_i, torch.zeros(self.num,3)),
                                            dim=0)
        bg.ndata['constraints'] = self.graph.ndata['constraints']
        bg.ndata['adsorbate'] = self.graph.ndata['adsorbate']
        self.graph = bg.clone()
        return bg

    def get_graph(self, ase_atoms): # this is the high-level API.
        # print('get_graph')
        # if isinstance(structure, str):
        #     ase_atoms = read(structure)
        #     self.reset()
        # elif isinstance(structure, ase.atoms.Atoms):
        #     ase_atoms = structure
        # else:
        #     raise TypeError("Incorrect input structure type.")

        if self.step % self.new_graph_steps == 0:
            bg = self.build(ase_atoms)
        else:
            bg = self.update(ase_atoms)
        self.step += 1
        return bg

if __name__ == '__main__':
    # ase_atoms = read('XDATCAR')
    ag = AseGraph()
    # bg = ag.get_graph(ase_atoms)
    # print('========================')
    # bg_update = ag.get_graph(ase_atoms)
    # self = ag

    import time
    # for n in [50, 100, 200, 400, 800, 1600, 3200]:
    #     start = time.time()
    #     ase_atoms = read(f'POSCAR_{n}')
    #     for i in range(30):
    #         bg = ag.build(ase_atoms)

    #     print(time.time() - start)

    for n in [50, 100, 200, 400, 800, 1600, 3200]:
        start = time.time()
        ase_atoms = read(f'POSCAR_{n}')
        bg = ag.build(ase_atoms)
        for i in range(29):
            bg = ag.update(ase_atoms)

        print(time.time() - start)
