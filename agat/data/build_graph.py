# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:18:57 2023

@author: ZHANG Jun
"""


import os
import json
import numpy as np

import torch
import dgl
import ase
from ase.neighborlist import natural_cutoffs
from ase.io import read

from ..default_parameters import default_data_config
from .atomic_feature import get_atomic_feature_onehot
from ..lib.model_lib import config_parser

class CrystalGraph(object):
    """
    .. py:class:: CrystalGraph(object)

        :param **data_config: Configuration file for building database.
        :type **data_config: str/dict
        :return: A ``DGL.graph``.
        :rtype: ``DGL.graph``.

    .. Hint::
        Although we recommend representing atoms with one hot code, you can
        use the another way with: ``self.all_atom_feat = get_atomic_features()``

    .. Hint::
        In order to build a reasonable graph, a samll cell should be repeated.
        One can modify "self._cell_length_cutoff" for special needs.

    .. Hint::
        We encourage you to use ``ase`` module to build crystal graphs.
        The ``pymatgen`` module needs some dependencies that conflict with
        other modules.

    """
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}

        # check inputs
        if self.data_config['topology_only']:
            self.data_config['build_properties']['energy'] = False
            self.data_config['build_properties']['forces'] = False
            self.data_config['build_properties']['cell'] = False
            self.data_config['build_properties']['stress'] = False
            self.data_config['build_properties']['frac_coords'] = False

        self.all_atom_feat = get_atomic_feature_onehot(self.data_config['species'])

        """ Although we recommend representing atoms with one hot code, you can
        use the another way with the next line: """
        # self.all_atom_feat = get_atomic_features()

        """
        In order to build a reasonable graph, a samll cell should be repeated.
        One can modify "self._cell_length_cutoff" for special needs.
        """
        self._cell_length_cutoff = np.array([11.0, 5.5, 0.0]) # Unit: angstrom. # increase cutoff to repeat more.

        if self.data_config['mode_of_NN'] == 'voronoi' or self.data_config['mode_of_NN'] == 'pymatgen_dist':
            ''' We encourage you to use ``ase`` module to build crystal graphs.
            The ``pymatgen`` module may need dependencies that conflict with
            other modules.'''
            try:
                from pymatgen.core.structure import Structure
                from agat.data.PymatgenStructureAnalyzer import VoronoiConnectivity
            except ModuleNotFoundError:
                raise ModuleNotFoundError('We encourage you to use ``ase`` module \
                to build crystal graphs. The ``pymatgen`` module needs some \
                dependencies that conflict with other modules. Install\
                ``pymatgen`` with ``pip install pymatgen``.')

            self.Structure = Structure
            self.VoronoiConnectivity = VoronoiConnectivity

        # self.device = self.data_config['device']

        if self.data_config['build_properties']['path']:
            print('You choose to store the path of each graph. \
Be noted that this will expose your file structure when you publish your dataset.')
        self.dtype = torch.float

    def get_adsorbate_bool(self, element_list):
        """
       .. py:method:: get_adsorbate_bool(self)

          Identify adsorbates based on elements： H and O.

          :return: a list of bool values.
          :rtype: tf.constant

        """
        # element_list = np.array(self.data_config['species'])
        element_list = np.array(element_list)
        return torch.tensor(np.where((element_list == 'H') | (element_list == 'O'),
                                     1, 0), dtype=self.dtype)

    def get_crystal(self, crystal_fpath):
        """
        .. py:method:: get_crystal(self, crystal_fpath)

           Read structural file and return a pymatgen crystal object.

           :param str crystal_fpath: the path to the crystal structural.
           :return: a pymatgen structure object.
           :rtype: ``pymatgen.core.structure``.

        .. Hint:: If there is only one site in the cell, this site has no other
        neighbors. Thus, the graph cannot be built without repetition.

        """
        assert os.path.exists(crystal_fpath), str(crystal_fpath) + " not found."
        structure = self.Structure.from_file(crystal_fpath)
        """If there is only one site in the cell, this site has no other
        neighbors. Thus, the graph cannot be built without repetition."""
        cell_span = structure.lattice.matrix.diagonal()
        if cell_span.min() < max(self._cell_length_cutoff) and self.data_config['super_cell']:
            repeat = np.digitize(cell_span, self._cell_length_cutoff) + 1
            structure.make_supercell(repeat)
        return structure # pymatgen format

    def get_1NN_pairs_voronoi(self, crystal):
        """
        .. py:method:: get_1NN_pairs_voronoi(self, crystal)

           The ``get_connections_new()`` of ``VoronoiConnectivity`` object is modified.

           :param pymatgen.core.structure crystal: a pymatgen structure object.
           :Returns:
              - index of senders
              - index of receivers
              - a list of distance between senders and receiver

        """

        crystal_connect = self.VoronoiConnectivity(crystal, self.data_config['cutoff'])
        return crystal_connect.get_connections_new() # the outputs are bi-directional and self looped.

    def get_1NN_pairs_distance(self, crystal):
        """

        .. py:method:: get_1NN_pairs_distance(self, crystal)

           Find the index of senders, receivers, and distance between them based on the ``distance_matrix`` of pymargen crystal object.

           :param pymargen.core.structure crystal: pymargen crystal object
           :Returns:
              - index of senders
              - index of receivers
              - a list of distance between senders and receivers

        """

        distance_matrix = crystal.distance_matrix
        sender, receiver = np.where(distance_matrix < self.data_config['cutoff'])
        dist = distance_matrix[(sender, receiver)]
        return sender, receiver, dist # the outputs are bi-directional and self looped.

    def get_1NN_pairs_ase_distance(self, ase_atoms):
        """

        .. py:method:: get_1NN_pairs_ase_distance(self, ase_atoms)

           :param ase.atoms ase_atoms: ``ase.atoms`` object.
           :Returns:
              - index of senders
              - index of receivers
              - a list of distance between senders and receivers

        """

        distance_matrix = ase_atoms.get_all_distances(mic=True)
        sender, receiver = np.where(distance_matrix < self.data_config['cutoff'])
        dist = distance_matrix[(sender, receiver)]
        return sender, receiver, dist # the outputs are bi-directional and self looped.

    def get_ndata(self, crystal):
        """

        .. py:method:: get_ndata(self, crystal)

           :param pymargen.core.structure crystal: a pymatgen crystal object.
           :return: ndata: the atomic representations of a crystal graph.
           :rtype: numpy.ndarray

        """

        ndata = []
        if isinstance(crystal, ase.Atoms):
            # num_sites = len(crystal)
            for i in crystal.get_chemical_symbols():
                ndata.append(self.all_atom_feat[i])
        else: # isinstance(crystal, self.Structure)
            num_sites = crystal.num_sites
            for i in range(num_sites):
                ndata.append(self.all_atom_feat[crystal[i].species_string])

        # else:
        #     print(f'Wraning!!! Unrecognized structure type: {crystal}')

        return np.array(ndata)

    def get_graph_from_ase(self, fname): # ase_atoms or file name
        '''
        .. py:method:: get_graph_from_ase(self, fname)

           Build graphs with ``ase``.

           :param str/``ase.Atoms`` fname: File name or ``ase.Atoms`` object.
           :return: A bidirectional graph with self-loop connection.
           :return: A dict of information of graph-level features.
        '''

        assert not isinstance(fname, ase.Atoms) or not self.data_config['build_properties']['forces'], 'The input cannot be a ase.Atoms object when include_forces is ``True``, try with an ``OUTCAR``'

        if not isinstance(fname, ase.Atoms):
            ase_atoms = read(fname)
        else:
            ase_atoms = fname
        num_sites = len(ase_atoms)
        if self.data_config['mode_of_NN'] == 'ase_dist':
            assert self.data_config['cutoff'], 'The `cutoff` cannot be `None` \
in this case. Provide a float here.'
            ase_cutoffs = self.data_config['cutoff']
        elif self.data_config['mode_of_NN'] == 'ase_natural_cutoffs':
            ase_cutoffs = np.array(natural_cutoffs(ase_atoms, mult=1.25, H=3.0, O=3.0)) # the specified values are radius, not diameters.
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
                    for i in c.index:
                        constraints[i] = [int(c.mask[0]), int(c.mask[1]), int(c.mask[2])]
                elif isinstance(c, ase.constraints.FixAtoms):
                    for i in c.index:
                        constraints[i] = [0, 0, 0]
                elif isinstance(c, ase.constraints.FixedLine):
                    for i in c.index:
                        constraints[i] = c.dir.tolist()
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

    def get_graph_from_pymatgen(self, crystal_fname):
        """
        .. py:method:: get_graph_from_pymatgen(self, crystal_fname)

           Build graphs with pymatgen.

           :param str crystal_fname: File name.
           :return: A bidirectional graph with self-loop connection.
           :return: A dict of information of graph-level features.
        """
        assert bool(not bool(self.data_config['super_cell'] and self.data_config['build_properties']['forces'])), 'include_forces cannot be True when super_cell is True.'
        # https://docs.dgl.ai/guide_cn/graph-feature.html 通过张量分配创建特征时，DGL会将特征赋给图中的每个节点和每条边。该张量的第一维必须与图中节点或边的数量一致。 不能将特征赋给图中节点或边的子集。
        mycrystal               = self.get_crystal(crystal_fname)
        # self.cart_coords       = mycrystal.cart_coords           # reserved for other functions

        if self.data_config['mode_of_NN'] == 'voronoi':
            sender, receiver, dist  = self.get_1NN_pairs_voronoi(mycrystal)   # the dist array is bidirectional and self-looped. # dist需要手动排除cutoff外的邻居
            location                = np.where(np.array(dist) > self.data_config['cutoff'])
            sender                  = np.delete(np.array(sender), location)
            receiver                = np.delete(np.array(receiver), location)
            dist                    = np.delete(np.array(dist), location)
        elif self.data_config['mode_of_NN'] == 'pymatgen_dist':
            sender, receiver, dist  = self.get_1NN_pairs_distance(mycrystal)

        ndata                   = self.get_ndata(mycrystal)
        bg                      = dgl.graph((sender, receiver))
        bg.ndata['h']           = torch.tensor(ndata, dtype=self.dtype)

        if self.data_config['build_properties']['distance']:
            bg.edata['dist']        = torch.tensor(dist, dtype=self.dtype)
        if self.data_config['build_properties']['cart_coords']:
            bg.ndata['cart_coords'] = torch.tensor(mycrystal.cart_coords, dtype=self.dtype)
        if self.data_config['build_properties']['frac_coords']:
            bg.ndata['frac_coords'] = torch.tensor(mycrystal.frac_coords, dtype=self.dtype)
        if self.data_config['build_properties']['direction']:
            frac_coords             = mycrystal.frac_coords
            sender_frac_coords      = frac_coords[sender]
            receiver_frac_coords    = frac_coords[receiver]
            delta_frac_coords       = receiver_frac_coords - sender_frac_coords
            image_top               = np.where(delta_frac_coords < -0.5, -1.0, 0.0)
            image_bottom            = np.where(delta_frac_coords >  0.5,  1.0, 0.0)
            image                   = image_top + image_bottom
            sender_frac_coords     += image
            sender_cart_coords      = mycrystal.lattice.get_cartesian_coords(sender_frac_coords)
            receiver_cart_coords    = mycrystal.cart_coords[receiver]
            real_direction          = receiver_cart_coords - sender_cart_coords
            direction_norm          = np.linalg.norm(real_direction, axis=1, keepdims=True)
            direction_normalized    = real_direction / direction_norm
            direction_normalized[np.where(sender == receiver)] = 0.0
            bg.edata['direction']   = torch.tensor(direction_normalized, dtype=self.dtype)
        if self.data_config['build_properties']['constraints']:
            try:
                bg.ndata['constraints'] = torch.tensor([x.selective_dynamics for x in mycrystal], dtype=self.dtype)
            except AttributeError:
                bg.ndata['constraints'] = torch.tensor([[True, True, True] for x in mycrystal], dtype=self.dtype)
        if self.data_config['build_properties']['forces']:
            forces_true             = torch.tensor(np.load(crystal_fname+'_force.npy'), dtype=self.dtype)
            bg.ndata['forces_true'] = torch.tensor((forces_true), dtype=self.dtype)
        if self.data_config['has_adsorbate']:
            element_list            = [x.specie.name for x in mycrystal.sites]
            bg.ndata['adsorbate']   = self.get_adsorbate_bool(element_list)

        graph_info = {}
        if self.data_config['build_properties']['energy']:
            energy_true = torch.tensor(np.load(crystal_fname+'_energy.npy'), dtype=self.dtype)
            graph_info['energy_true'] = torch.tensor((energy_true), dtype=self.dtype)
        if self.data_config['build_properties']['stress']:
            stress_true = torch.tensor(np.load(crystal_fname+'_stress.npy'), dtype=self.dtype)
            graph_info['stress_true'] = torch.tensor((stress_true), dtype=self.dtype)
        if self.data_config['build_properties']['cell']:
            cell_true = torch.tensor(mycrystal.lattice.matrix, dtype=self.dtype)
            graph_info['cell_true'] = torch.tensor((cell_true), dtype=self.dtype)
        if self.data_config['build_properties']['path']:
            graph_info['path'] = crystal_fname
        return bg, graph_info

    def get_graph(self, crystal_fname):
        """
        .. py:method:: get_graph(self, crystal_fname)

           This method can choose which graph-construction method is used, according to the ``mode_of_NN`` attribute.

           .. Hint:: You can call this method to build one graph.

           :param str crystal_fname: File name.
           :return: A bidirectional graph with self-loop connection.
        """
        if self.data_config['mode_of_NN'] == 'voronoi' or self.data_config['mode_of_NN'] == 'pymatgen_dist':
            return self.get_graph_from_pymatgen(crystal_fname)
        elif self.data_config['mode_of_NN'] == 'ase_natural_cutoffs' or self.data_config['mode_of_NN'] == 'ase_dist':
            return self.get_graph_from_ase(crystal_fname)

class AseGraphTorch(object):
    def __init__(self, **data_config):
        self.data_config = {**default_data_config,
                            **config_parser(data_config)}
        self.device = torch.device(self.data_config['device'])
        self.all_atom_feat = get_atomic_feature_onehot(
            self.data_config['species'])
        self.cutoff = torch.tensor(self.data_config['cutoff'],
                                   device = self.device)
        self.skin = torch.tensor(1.0, device = self.device)  # angstrom
        # self.adsorbate_skin = 2.0 # in many cases, the adsorbates experience larger geometry relaxation than surface atoms.

        self.step = 0
        self.new_graph_steps = 40  # buid new graph every * steps.
        self.inner_senders = None
        self.inner_receivers = None
        self.inner_receivers_image = None
        self.skin_senders = None
        self.skin_receivers = None
        self.skin_receivers_image = None
        self.graph = None

    def reset(self):
        # reset parameters for new optimization.
        # self.new_graph = True
        self.step = 0
        self.inner_senders = None
        self.inner_receivers = None
        self.inner_receivers_image = None
        self.skin_senders = None
        self.skin_receivers = None
        self.skin_receivers_image = None
        self.graph = None

    def get_ndata(self, ase_atoms):
        # print('get_ndata')
        ndata = []
        for i in ase_atoms.get_chemical_symbols():
            ndata.append(self.all_atom_feat[i])
        return torch.tensor(np.array(ndata), dtype=torch.float32,
                            device=self.device)

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
                                     1, 0), device=self.device)

    def get_scaled_positions_wrap(self, cell_I_tensor, positions):
        # print('get_scaled_positions')
        # cell_I: Returns the (multiplicative) inverse of invertible
        scaled_positions = torch.matmul(positions, cell_I_tensor)
        scaled_positions = torch.where(
            scaled_positions < 0.0, scaled_positions+1, scaled_positions)
        scaled_positions = torch.where(
            scaled_positions > 1.0, scaled_positions-1, scaled_positions)
        return scaled_positions

    def get_scaled_positions(self, cell_I_tensor, positions):
        # print('get_scaled_positions')
        # cell_I: Returns the (multiplicative) inverse of invertible
        return torch.matmul(positions, cell_I_tensor)

    def fractional2cartesian(self, cell_tensor, scaled_positions):
        # print('fractional2cartesian')
        positions = torch.matmul(scaled_positions, cell_tensor)
        return positions

    def safe_to_use(self, ase_atoms, critical=0.01):
        cell = ase_atoms.cell.array
        abs_cell = np.abs(cell)
        sum_cell = np.sum(abs_cell,axis=1)
        is_cubic = np.max((sum_cell - np.diagonal(abs_cell)) / sum_cell) < critical
        cutoff = self.cutoff + self.skin
        large_enough = cutoff.item() < np.linalg.norm(cell, axis=0)
        return is_cubic & large_enough.all()

    def get_pair_distances(self, a, b, ase_atoms):
        if not self.safe_to_use(ase_atoms):
            raise ValueError('Input structure is not a cubic system or not large enough, cannot use \
this function to calculate distances. Alternatively, you need to use \
``ase.Atoms.get_distances``: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.get_distances')
        cell = ase_atoms.cell.array
        cell_tensor = torch.tensor(cell, dtype=torch.float32,
                                   device=self.device)
        cell_I_tensor = torch.tensor(
            np.array(np.mat(cell).I), dtype=torch.float32, device=self.device)
        positions = torch.tensor(ase_atoms.positions, dtype=torch.float32,
                                 device=self.device)
        scaled_positions = self.get_scaled_positions(cell_I_tensor, positions)
        a_positions = positions[a, :]
        a_scaled_positions = scaled_positions[a, :]
        b_scaled_positions = scaled_positions[b, :]
        diff_scaled_positions = b_scaled_positions - a_scaled_positions
        b_scaled_positions = torch.where(diff_scaled_positions > 0.5,
                                         b_scaled_positions-1.0,
                                         b_scaled_positions)
        b_scaled_positions = torch.where(diff_scaled_positions < -0.5,
                                         b_scaled_positions+1.0,
                                         b_scaled_positions)
        b_positions_new = self.fractional2cartesian(
            cell_tensor, b_scaled_positions)
        D = b_positions_new - a_positions
        d = torch.norm(D, dim=1)
        return d, D

    def update_pair_distances(self, a, b, b_image, ase_atoms):
#         if not self.safe_to_use(ase_atoms):
#             raise ValueError('Input structure is not a cubic system or not large enough, cannot use \
# this function to calculate distances. Alternatively, you need to use \
# ``ase.Atoms.get_distances``: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.get_distances')
        cell = ase_atoms.cell.array
        cell_tensor = torch.tensor(cell, dtype=torch.float32,
                                    device=self.device)
        cell_I_tensor = torch.tensor(
            np.array(np.mat(cell).I), dtype=torch.float32, device=self.device)
        positions = torch.tensor(ase_atoms.positions, dtype=torch.float32,
                                  device=self.device)
        scaled_positions = self.get_scaled_positions(cell_I_tensor, positions)

        a_positions = positions[a, :]
        # a_scaled_positions = scaled_positions[a, :]
        b_scaled_positions = scaled_positions[b, :] + b_image
        b_positions_new = self.fractional2cartesian(
            cell_tensor, b_scaled_positions)
        D = b_positions_new - a_positions
        d = torch.norm(D, dim=1)
        return d, D

    def get_all_possible_distances(self, ase_atoms):
        # get senders and receivers, including inner and skin connections.
        # torch.from_numpy is memory effcient than torch.tensor, especially for large tensors.
        # No self loop and reverse direction.
        if not self.safe_to_use(ase_atoms):
            raise ValueError('Input structure is not a cubic system or not large enough, cannot use \
this function to calculate distances. Alternatively, you need to use \
``ase.Atoms.get_distances``: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.get_distances')

        # prepare variables
        cell = ase_atoms.cell.array
        cell_tensor = torch.tensor(cell, dtype=torch.float32, device=self.device)
        cell_I_tensor = torch.tensor(
            np.array(np.mat(cell).I), dtype=torch.float32, device=self.device)
        positions = torch.tensor(ase_atoms.positions, dtype=torch.float32,
                                 device=self.device)
        scaled_positions = self.get_scaled_positions(cell_I_tensor, positions)

        a, b = np.triu_indices(len(ase_atoms), k=1)
        a, b = torch.tensor(a, device=self.device), torch.tensor(b, device=self.device)

        # ************************
        # a = torch.tensor([0,1,48,54,2,3])
        # b = torch.tensor([24,25,72,78,26,27])
        # dist_test_1 = ase_atoms.get_distance(0,24,mic=True)
        # dist_test_2 = ase_atoms.get_distance(0,24,mic=False)
        # ************************

        a_scaled_positions = scaled_positions[a, :]
        b_scaled_positions = scaled_positions[b, :]
        diff_scaled_positions = b_scaled_positions - a_scaled_positions
        abs_diff_scaled_positions = torch.absolute(diff_scaled_positions)
        cutoff = self.cutoff + self.skin
        scaled_cutoff = cutoff/torch.norm(cell_tensor, dim=1)

        # remove distant atomic pairs. Test results show that these lines have little effect on boosting efficiency.
        # if torch.all(scaled_cutoff < 0.5):
        #     # print('before:', len(a))
        #     mask1 = abs_diff_scaled_positions > scaled_cutoff
        #     mask2 = abs_diff_scaled_positions < 1-scaled_cutoff
        #     mask = mask1 & mask2
        #     mask = ~mask.all(dim=1)
        #     a = a[mask]
        #     b = b[mask]
        #     a_scaled_positions = scaled_positions[a, :]
        #     b_scaled_positions = scaled_positions[b, :]
        #     diff_scaled_positions = b_scaled_positions - a_scaled_positions
        #     abs_diff_scaled_positions = torch.absolute(diff_scaled_positions)
        #     # print('before:', len(a))

        # case one: connection at current image
        case_one_mask = torch.any(abs_diff_scaled_positions < scaled_cutoff, dim=1)
        case_one_i = torch.where(case_one_mask)[0]
        case_one_image = torch.zeros(len(case_one_i), 3,device=self.device)

        # case two: connection at other images
        case_two_mask = abs_diff_scaled_positions > 1 - scaled_cutoff
        x_mask = case_two_mask[:,0]
        y_mask = case_two_mask[:,1]
        z_mask = case_two_mask[:,2]
        xy_mask = x_mask & y_mask
        xz_mask = x_mask & z_mask
        yz_mask = y_mask & z_mask
        xyz_mask = x_mask & y_mask & z_mask

        xi = torch.where(x_mask)[0]
        yi = torch.where(y_mask)[0]
        zi = torch.where(z_mask)[0]
        xyi = torch.where(xy_mask)[0]
        xzi = torch.where(xz_mask)[0]
        yzi = torch.where(yz_mask)[0]
        xyzi = torch.where(xyz_mask)[0]

        image_x = torch.where(diff_scaled_positions[x_mask,0] > 0.0, -1.0, 1.0)
        image_y = torch.where(diff_scaled_positions[y_mask,1] > 0.0, -1.0, 1.0)
        image_z = torch.where(diff_scaled_positions[z_mask,2] > 0.0, -1.0, 1.0)
        image_xy = torch.where(diff_scaled_positions[xy_mask,0:2] > 0.0, -1.0, 1.0)
        image_xz = torch.where(diff_scaled_positions[xz_mask,:] > 0.0, -1.0, 1.0)
        image_xz_x = image_xz[:,0]
        image_xz_z = image_xz[:,2]
        image_yz = torch.where(diff_scaled_positions[yz_mask,1:3] > 0.0, -1.0, 1.0)
        image_xyz = torch.where(diff_scaled_positions[xyz_mask,:] > 0.0, -1.0, 1.0)

        image_x = torch.stack((image_x,
                               torch.zeros_like(image_x, device=self.device),
                               torch.zeros_like(image_x, device=self.device)))
        image_x = torch.transpose(image_x, 0, 1)
        image_y = torch.stack((torch.zeros_like(image_y, device=self.device),
                               image_y,
                               torch.zeros_like(image_y, device=self.device)))
        image_y = torch.transpose(image_y, 0, 1)
        image_z = torch.stack((torch.zeros_like(image_z, device=self.device),
                               torch.zeros_like(image_z, device=self.device),
                               image_z))
        image_z = torch.transpose(image_z, 0, 1)
        image_xy = torch.cat((image_xy,
                              torch.zeros(len(image_xy), 1,
                                          device=self.device)),
                             dim=1)
        image_xz = torch.stack((image_xz_x,
                                torch.zeros_like(image_xz_x, device=self.device),
                                image_xz_z))
        image_xz = torch.transpose(image_xz, 0, 1)
        image_yz = torch.cat((torch.zeros(len(image_yz), 1, device=self.device),
                              image_yz),
                             dim=1)
        image_xyz = image_xyz

        all_i = torch.cat((case_one_i,
                           xi, yi, zi,
                           xyi, xzi, yzi,
                           xyzi))

        all_image = torch.cat((case_one_image,
                               image_x, image_y, image_z,
                               image_xy, image_xz, image_yz,
                               image_xyz), dim=0)

        b_scaled_positions_all = b_scaled_positions[all_i] + all_image
        b_positions_all = self.fractional2cartesian(
            cell_tensor, b_scaled_positions_all)
        a_positions = positions[a, :]
        a_positions_all = a_positions[all_i]

        a_all, b_all = a[all_i], b[all_i]

        # calculate distance
        D = b_positions_all - a_positions_all
        d = torch.norm(D, dim=1)

        # position_0 = positions[0]
        # position_24 = positions[24]

        return a_all, b_all, d, D, all_image # what about return a dict? or image location

    def get_init_connections(self, ase_atoms):
        # No self loop and reverse direction.
        i, j, d, D, j_image = self.get_all_possible_distances(ase_atoms)
        inner_connections = torch.where(d < self.cutoff)
        skin_connections = torch.where((d > self.cutoff) & (d < self.cutoff+1))
        i_i, j_i, d_i, D_i, j_image_i = i[inner_connections], j[inner_connections], d[inner_connections], D[inner_connections], j_image[inner_connections]
        i_s, j_s, d_s, D_s, j_image_s = i[skin_connections], j[skin_connections], d[skin_connections], D[skin_connections], j_image[skin_connections]
        return i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s

    def update_connections(self, i_i, j_i, j_image_i, i_s, j_s, j_image_s, ase_atoms):
        # print('update instead of build')
        i, j, j_image = torch.cat((i_i, i_s)), torch.cat((j_i, j_s)), torch.cat((j_image_i, j_image_s))
        d, D = self.update_pair_distances(i, j, j_image, ase_atoms)
        inner_connections = torch.where(d < self.cutoff)
        skin_connections = torch.where(
            (d > self.cutoff) & (d < self.cutoff+self.skin))
        i_i, j_i, d_i, D_i = i[inner_connections], j[inner_connections], d[inner_connections], D[inner_connections]
        i_s, j_s, d_s, D_s = i[skin_connections], j[skin_connections], d[skin_connections], D[skin_connections]
        j_image_i = j_image[inner_connections]
        j_image_s = j_image[skin_connections]
        return i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s

    def build(self, ase_atoms):
        print('build new graph.')
        # build graph
        # Include all possible properties.

        ndata = self.get_ndata(ase_atoms)
        # No self loop and reverse direction.
        i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s = self.get_init_connections(
            ase_atoms)
        self.inner_senders = i_i
        self.inner_receivers = j_i
        self.inner_receivers_image = j_image_i
        self.skin_senders = i_s
        self.skin_receivers = j_s
        self.skin_receivers_image = j_image_s

        # add reverse direction and self loop connections.
        # Some properties will not change when update: h, constraints, adsorbate,
        bg = dgl.graph((torch.cat((self.inner_senders,
                                   self.inner_receivers,
                                   torch.arange(len(ase_atoms),
                                                device=self.device)),
                                  dim=0),
                       torch.cat((self.inner_receivers,
                                  self.inner_senders,
                                  torch.arange(len(ase_atoms),
                                               device=self.device)),
                                 dim=0)))
        bg.ndata['h'] = ndata
        bg.edata['dist'] = torch.cat((d_i, d_i, torch.zeros(len(ase_atoms),
                                                            device=self.device)),
                                     dim=0)
        bg.edata['direction'] = torch.cat(
            (D_i, -D_i, torch.zeros(len(ase_atoms), 3,
                                   device=self.device)), dim=0)
        constraints = [[1, 1, 1]] * len(ase_atoms)
        for c in ase_atoms.constraints:
            if isinstance(c, ase.constraints.FixScaled):
                for i in c.index:
                    constraints[i] = [int(c.mask[0]), int(c.mask[1]), int(c.mask[2])]
            elif isinstance(c, ase.constraints.FixAtoms):
                for i in c.index:
                    constraints[i] = [0, 0, 0]
            elif isinstance(c, ase.constraints.FixedLine):
                for i in c.index:
                    constraints[i] = c.dir.tolist()
            elif isinstance(c, ase.constraints.FixBondLengths):
                pass
            else:
                raise TypeError(
                    f'Wraning!!! Undefined constraint type: {type(c)}')
        bg.ndata['constraints'] = torch.tensor(
            constraints, dtype=torch.float32, device=self.device)
        element_list = ase_atoms.get_chemical_symbols()
        bg.ndata['adsorbate'] = self.get_adsorbate_bool(element_list)
        self.graph = bg
        return bg

    def update(self, ase_atoms):
        # print('update')
        # calculate d and D, reassign i and j
        i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s = self.update_connections(
            self.inner_senders, self.inner_receivers, self.inner_receivers_image,
            self.skin_senders, self.skin_receivers, self.skin_receivers_image,
            ase_atoms)
        self.inner_senders = i_i
        self.inner_receivers = j_i
        self.inner_receivers_image = j_image_i
        self.skin_senders = i_s
        self.skin_receivers = j_s
        self.skin_receivers_image = j_image_s
        bg = dgl.graph((torch.cat((self.inner_senders,
                                  self.inner_receivers,
                                  torch.arange(len(ase_atoms),
                                               device=self.device)),
                                  dim=0),
                       torch.cat((self.inner_receivers,
                                  self.inner_senders,
                                  torch.arange(len(ase_atoms),
                                               device=self.device)),
                                 dim=0)))
        bg.ndata['h'] = self.graph.ndata['h']
        bg.edata['dist'] = torch.cat((d_i, d_i, torch.zeros(len(ase_atoms),
                                                            device=self.device)),
                                     dim=0)
        bg.edata['direction'] = torch.cat((D_i, -D_i, torch.zeros(len(ase_atoms),
                                                                 3,
                                                                 device=self.device)),
                                          dim=0)
        bg.ndata['constraints'] = self.graph.ndata['constraints']
        bg.ndata['adsorbate'] = self.graph.ndata['adsorbate']
        self.graph = bg
        return bg

    def get_graph(self, ase_atoms):  # this is the high-level API.
        # if isinstance(ase_atoms, str):
        #     ase_atoms = read(ase_atoms)
        #     self.reset()
        # elif isinstance(ase_atoms, ase.atoms.Atoms):
        #     ase_atoms = ase_atoms
        # else:
        #     raise TypeError("Incorrect input structure type.")
        # ase_atoms_tmp = ase_atoms.copy()
        if self.step % self.new_graph_steps == 0:
            bg = self.build(ase_atoms)
        else:
            bg = self.update(ase_atoms)
        self.step += 1
        return bg

if __name__ == '__main__':
    # ase_atoms = read('XDATCAR')
    import os
    # os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    import json
    graph_build_scheme_path = os.path.join('acatal', 'test', 'agat_model', 'graph_build_scheme.json')
    with open(graph_build_scheme_path, 'r') as f:
        graph_build_scheme = json.load(f)
    graph_build_scheme['device'] = 'cpu'
    graph_build_scheme['topology_only'] = True
    fname = os.path.join('acatal', 'test', 'CONTCAR_new_3')


    ###########################################################################
    # test
    from agat.lib.model_lib import load_model
    model_save_dir = os.path.join('acatal', 'test', 'agat_model')

    ase_atoms = read(fname)
    ag = AseGraphTorch(**graph_build_scheme)

    g1 = ag.get_graph(ase_atoms)
    g2 = ag.get_graph(ase_atoms)
    agat_model = load_model(model_save_dir, 'cpu')
    with torch.no_grad():
        energy_pred1, force_pred1, stress_pred1 = agat_model.forward(g1)
        energy_pred2, force_pred2, stress_pred2 = agat_model.forward(g2)

    ###########################################################################
    # from agat.data import CrystalGraph
    # ase_atoms = read(fname)
    # cg = CrystalGraph(**graph_build_scheme)
    # g, _ = cg.get_graph(ase_atoms)
    # dist = g.edata['dist']
    # edges = g.edges()
    # len(dist)

    # ase_atoms = read(fname)
    # ag = AseGraph(**graph_build_scheme)
    # g_new = ag.get_graph(ase_atoms)
    # dist_new = g_new.edata['dist']
    # edges_new = g_new.edges()


    ###########################################################################
    # compare
    # num_edges = len(dist)
    # pair = []
    # for i in range(num_edges):
    #     pair.append((edges[0][i].item(),
    #                   edges[1][i].item()))

    # num_edges_new = len(dist_new)
    # pair_new = []
    # for i in range(num_edges_new):
    #     pair_new.append((edges_new[0][i].item(),
    #                       edges_new[1][i].item()))

    # for p in pair:
    #     repeat = pair.count(p)
    #     repeat_new = pair_new.count(p)
    #     if repeat > 1:
    #         print(p, repeat, repeat_new)


    # print('======================== PyTorch from scratch')
    # # self = ag
    # ase_atoms = read(fname)
    # ag = AseGraph(**graph_build_scheme)
    # import time
    # for n in [1,2,3,4]:
    #     start = time.time()
    #     ase_atoms_tmp = ase_atoms.copy().repeat(n)
    #     for i in range(30):
    #         bg = ag.build(ase_atoms_tmp)
    #     print(len(ase_atoms_tmp), time.time() - start)

    # print('======================== PyTorch with update')
    # ase_atoms = read(fname)
    # ag = AseGraph(**graph_build_scheme)
    # for n in [1,2,3,4]:
    #     start = time.time()
    #     ase_atoms_tmp = ase_atoms.copy().repeat(n)
    #     bg = ag.build(ase_atoms_tmp)
    #     for i in range(29):
    #         bg = ag.update(ase_atoms_tmp)
    #     print(len(ase_atoms_tmp), time.time() - start)

    # print('======================== ASE from scratch')
    # from agat.data import CrystalGraph
    # ase_atoms = read(fname)
    # cg = CrystalGraph(**graph_build_scheme)
    # for n in [1,2,3,4]:
    #     start = time.time()
    #     ase_atoms_tmp = ase_atoms.copy().repeat(n)
    #     for i in range(30):
    #         g = cg.get_graph(ase_atoms_tmp)
    #     print(len(ase_atoms_tmp), time.time() - start)

    # # start = time.time()
    # # for i in range(1000):
    # #     ase_atoms.get_positions()
    # # print(len(ase_atoms_tmp), time.time() - start)
    # # start = time.time()
    # # for i in range(1000):
    # #     ase_atoms.get_scaled_positions()
    # # print(len(ase_atoms_tmp), time.time() - start)
    # # for i in range(1000):
    # #     positions = torch.tensor(ase_atoms.get_positions(), dtype=torch.float32)
    # #     self.get_scaled_positions(cell_I_tensor, positions)
    # # print(len(ase_atoms_tmp), time.time() - start)
