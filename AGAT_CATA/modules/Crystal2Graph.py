# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:48:04 2021

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

np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# 对于高熵体系，voronoi划分近邻，可能不合理
# =============================================================================

class CrystalGraph(object):
    """
    Description:
    ----------
        Read cif or POSCAR data and return a graph.
    Parameters
    ----------
    cutoff : float
        Cutoff distance when calculate connections of a crystal.
    mode_of_NN : str
        Mode of how to get the neighbors, which can be: \n
        * 'distance': All atoms within cutoff will be neighbors. \n
        * 'voronoi': Consider Voronoi neighbors only.\n
        * 'ase': Build graph from ase which has a dynamic cutoff scheme.\n
            In this case, the `cutoff` is deprecated because `ase` will use the
            dynamic cutoffs in `ase.neighborlist.natural_cutoffs()`.\n
    Speed of building graphs: distance > ase >> voronoi.
    """
    def __init__(self, cutoff=6.0, mode_of_NN='distance', adsorbate=True):
        self.cutoff        = cutoff
        self.all_atom_feat = get_atomic_feature_onehot()

        """ Although we recommend representing atoms with one hot code, you can
        use the another way with the next line: """
        # self.all_atom_feat = get_atomic_features()

        """
        In order to build a reasonable graph, a samll cell should be repeated.
        One can modify "self._cell_length_cutoff" for special needs.
        """
        self._cell_length_cutoff = np.array([11.0, 5.5, 0.0]) # Unit: angstrom. # increase cutoff to repeat more.
        self.mode_of_NN = mode_of_NN

        self.adsorbate  = adsorbate
        # self.adsorbate_index = {'H': 1, 'O': 2}

    def get_adsorbate_bool(self, element_list):
        # num_eles = len(element_list)
        element_list = np.array(element_list)
        return tf.constant(np.where((element_list == 'H') | (element_list == 'O'), True, False), dtype='bool')

    def get_crystal(self, crystal_fpath, super_cell=True):
        assert os.path.exists(crystal_fpath), str(crystal_fpath) + " not found."
        structure = Structure.from_file(crystal_fpath)
        """If there is only one site in the cell, this site has no other
        neighbors. Thus, the graph cannot be built without repetition."""
        cell_span = structure.lattice.matrix.diagonal() # np.array(structure.lattice.abc)
        if cell_span.min() < max(self._cell_length_cutoff) and super_cell:
            repeat = np.digitize(cell_span, self._cell_length_cutoff) + 1
            # np.where(cell_length < self.cell_length_cutoff, 3, 1)
            structure.make_supercell(repeat)
        return structure # pymatgen format

    def get_1NN_pairs_voronoi(self, crystal):
        crystal_connect = VoronoiConnectivity(crystal, self.cutoff)
        return crystal_connect.get_connections_new() # the outputs are bi-directional and self looped.

    def get_1NN_pairs_distance(self, crystal):
        distance_matrix = crystal.distance_matrix
        sender, receiver = np.where(distance_matrix < self.cutoff)
        dist = distance_matrix[(sender, receiver)]
        return sender, receiver, dist # the outputs are bi-directional and self looped.

    def get_1NN_pairs_ase_distance(self, ase_atoms):
        distance_matrix = ase_atoms.get_all_distances(mic=True)
        sender, receiver = np.where(distance_matrix < self.cutoff)
        dist = distance_matrix[(sender, receiver)]
        return sender, receiver, dist # the outputs are bi-directional and self looped.

    def get_ndata(self, crystal):
        ndata = []
        if isinstance(crystal, Structure):
            num_sites = crystal.num_sites
            for i in range(num_sites):
                ndata.append(self.all_atom_feat[crystal[i].species_string])
        elif isinstance(crystal, ase.Atoms):
            # num_sites = len(crystal)
            for i in crystal.get_chemical_symbols():
                ndata.append(self.all_atom_feat[i])
        else:
            print(f'Wraning!!! Unrecognized structure type: {crystal}')

        return np.array(ndata)

    def get_graph_from_ase(self, fname, include_forces=False): # ase_atoms or file name
        '''
        Description:
        ----------
            Build graphs with ase.
        Parameters
        ----------
        fname : str or ase.Atoms object
            File name or `ase.Atoms` object.
        include_forces: bool
            Include forces into graphs or not.
        Return
        ----------
            A bidirectional  graph with self-loop connection.
        '''
        assert not isinstance(fname, ase.Atoms) or not include_forces, 'The input cannot be a ase.Atoms object when include_forces is True'

        if not isinstance(fname, ase.Atoms):
            ase_atoms = read(fname)
        else:
            ase_atoms = fname
        num_sites = len(ase_atoms)
        if self.mode_of_NN == 'ase_dist':
            ase_cutoffs = self.cutoff
        elif self.mode_of_NN == 'ase_natural_cutoffs':
            ase_cutoffs = np.array(natural_cutoffs(ase_atoms, mult=1.25, H=3.0, O=3.0)) # the specified values are radius, not diameters.
        i, j, d, D = ase.neighborlist.neighbor_list('ijdD',
                                                    ase_atoms,
                                                    cutoff=ase_cutoffs,
                                                    self_interaction=True)

        ndata                   = self.get_ndata(ase_atoms)
        bg                      = dgl.graph((i, j))
        bg.ndata['h']           = tf.constant(ndata, dtype='float32')
        bg.edata['dist']        = tf.constant(d, dtype='float32')
        bg.edata['direction']   = tf.constant(D, dtype='float32')

        constraints             = [[True, True, True]] * num_sites
        # try:
        #     for i in ase_atoms.constraints[-1].index:
        #         constraints[i]      = [False, False, False]
        # except IndexError:
        #     pass
        for c in ase_atoms.constraints:
            if isinstance(c, ase.constraints.FixScaled):
                constraints[c.a] = c.mask
            elif isinstance(c, ase.constraints.FixAtoms):
                for i in c.index:
                    constraints[i] = [False, False, False]
            elif isinstance(c, ase.constraints.FixBondLengths):
                pass
            else:
                raise TypeError(f'Wraning!!! Undefined constraint type: {type(c)}')
        bg.ndata['constraints'] = tf.constant(constraints, dtype='bool')

        if include_forces:
            forces_true             = tf.constant(np.load(fname+'_force.npy'), dtype='float32')
            bg.ndata['forces_true'] = tf.constant((forces_true), dtype='float32')

        if self.adsorbate:
            element_list            = ase_atoms.get_chemical_symbols()
            bg.ndata['adsorbate']   = self.get_adsorbate_bool(element_list)
        return bg

    def get_graph_from_pymatgen(self, crystal_fname, super_cell=True, include_forces=False):
        """
        Parameters
        ----------
        crystal_fname : str
            File name of a crystal with cif or VASP formate.
        Returns
        -------
        bg : DGLGraph
            A bidirectional graph.
        """
        assert bool(not bool(super_cell and include_forces)), 'include_forces cannot be True when super_cell is True.'
        # https://docs.dgl.ai/guide_cn/graph-feature.html 通过张量分配创建特征时，DGL会将特征赋给图中的每个节点和每条边。该张量的第一维必须与图中节点或边的数量一致。 不能将特征赋给图中节点或边的子集。
        mycrystal               = self.get_crystal(crystal_fname, super_cell=super_cell)
        # self.cart_coords       = mycrystal.cart_coords           # reserved for other functions

        if self.mode_of_NN == 'voronoi':
            sender, receiver, dist  = self.get_1NN_pairs_voronoi(mycrystal)   # the dist array is bidirectional and self-looped. # dist需要手动排除cutoff外的邻居
            location                = np.where(np.array(dist) > self.cutoff)
            sender                  = np.delete(np.array(sender), location)
            receiver                = np.delete(np.array(receiver), location)
            dist                    = np.delete(np.array(dist), location)
        elif self.mode_of_NN == 'pymatgen_dist':
            sender, receiver, dist  = self.get_1NN_pairs_distance(mycrystal)

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
        # direction_normalized    = np.vstack((direction_normalized, np.zeros(shape=(mycrystal.num_sites, 3))))

        ndata                   = self.get_ndata(mycrystal)
        bg                      = dgl.graph((sender, receiver))
        bg.ndata['h']           = tf.constant(ndata, dtype='float32')
        bg.ndata['cart_coords'] = tf.constant(mycrystal.cart_coords, dtype='float32')
        bg.ndata['frac_coords'] = tf.constant(frac_coords, dtype='float32')
        try:
            bg.ndata['constraints'] = tf.constant([x.selective_dynamics for x in mycrystal], dtype='bool')
        except AttributeError:
            bg.ndata['constraints'] = tf.constant([[True, True, True] for x in mycrystal], dtype='bool')
        # bg                      = dgl.add_self_loop(bg)
        bg.edata['dist']        = tf.constant(dist, dtype='float32')
        bg.edata['direction']   = tf.constant(direction_normalized, dtype='float32') # unit vector

        # print(include_forces)
        if include_forces:
            forces_true             = tf.constant(np.load(crystal_fname+'_force.npy'), dtype='float32')
            bg.ndata['forces_true'] = tf.constant((forces_true), dtype='float32')

        if self.adsorbate:
            element_list            = [x.specie.name for x in mycrystal.sites]
            bg.ndata['adsorbate']   = self.get_adsorbate_bool(element_list)
        return bg # graph 應當同時包含cart_coords, h, forces

    def get_new_graph(self, crystal_fname, super_cell=False, include_forces=True):

        pass

    def get_graph(self, crystal_fname, super_cell=False, include_forces=True):
        """
        Parameters
        ----------
        crystal_fname : str
            File name of a crystal with cif or VASP formate or `ase.Atoms` object.
        super_cell: bool
            Repeat the loaded cell or not. Will be deprecated when call `self.get_graph_from_ase()`
        Return
        -------
        bg : DGLGraph
            A bidirectional graph.
        """
        if self.mode_of_NN == 'voronoi' or self.mode_of_NN == 'pymatgen_dist':
            return self.get_graph_from_pymatgen(crystal_fname, super_cell=super_cell, include_forces=include_forces)
        elif self.mode_of_NN == 'ase_natural_cutoffs' or self.mode_of_NN == 'ase_dist':
            return self.get_graph_from_ase(crystal_fname, include_forces=include_forces)

class ReadGraphs(CrystalGraph):
    """
    Description:
    ----------
        Read graphs from a specified csv file.
    Parameters
    ----------
    csv_file: str
        File name of a csv file that contains the filenames of crystals
        with cif or VASP formate.
    dataset_path: str
        Path to the dataset. The folder contains DGL binary graph files or
        files with cif/VASP formate.
    cutoff: float
        Cutoff distance to calculate site connections.
    from_binary: boolean
        Whether the graphs are read from scratch or DGL binary graph.
    num_of_cores: int
        Number of cores when reading graphs from *cif or VASP file.
        Read graphs from scratch takes much more time than that from binary
        file. So I recommend reading *cif or VASP file with multiple
        CPU cores.
    Note:
    ----------
        Property (or label) of each graph are loaded from *.csv file when build
        new graphs from scratch. On the other hadn, they will be loaded from
        the DGL binary file if all the graphs are loaded from the binary file.
    """
    def __init__(self,
                 csv_file,
                 dataset_path,
                 cutoff = 6.0,
                 mode_of_NN='distance',
                 from_binary=False,
                 num_of_cores=1,
                 super_cell=True,
                 include_forces=True,
                 adsorbate=True):
        super(ReadGraphs, self).__init__(cutoff,adsorbate)
        self.csv_file         = csv_file
        self.dataset_path     = dataset_path
        self.mode_of_NN       = mode_of_NN
        self.cutoff           = cutoff
        self.include_forces   = include_forces
        self.adsorbate        = adsorbate
        assert os.path.exists(self.csv_file), str(self.csv_file) + " not found."
        assert os.path.exists(self.dataset_path), str(self.dataset_path) + " not found."

        self.from_binary      = from_binary
        self.num_of_cores     = num_of_cores
        fname_prop_data       = np.loadtxt(self.csv_file, dtype=str, delimiter=',')
        self.name_list        = fname_prop_data[:,0]
        self.prop_list        = list(map(float, fname_prop_data[:,1]))   # if other codes called this property of this class, it means this function used labels from *.csv file.
        self.number_of_graphs = len(fname_prop_data)
        self.read_super_cell  = super_cell

    def read_batch_graphs(self, batch_index_list, batch_num):
        batch_fname  = [self.name_list[x] for x in batch_index_list]
        batch_g      = [self.get_graph(os.path.join(self.dataset_path ,fname), super_cell=self.read_super_cell, include_forces=self.include_forces) for fname in tqdm(batch_fname, desc='Reading ' + str(batch_num) + ' batch graphs', delay=batch_num)]
        batch_prop   = tf.constant([self.prop_list[x] for x in batch_index_list], dtype=tf.float32)
        batch_labels = {"prop": batch_prop}
        save_graphs(os.path.join(self.dataset_path, 'all_graphs_' + str(batch_num) + '.bin'), batch_g, batch_labels)

    def read_all_graphs(self, scale_prop=False, ckpt_path='.'): # prop_per_node=False Deprecated!
        """
        Description:
        ----------
            Read all graphs.
        Parameters
        ----------
        scale_prop: bool
            if `True`, graph labels will be scaled into the range of `[0, 1]`.

            if `False`, nothing happen.
        prop_per_node: bool (deprecated!)
            if `True`, graph labels will be divided by the number of nodes.

            if `False`, nothing happen.
        ckpt_path: path (str)
            Checkpoint path for save the information about how to process the
            label data.
        Returns
        -------
        graph_list: DGLGraph
            A `list` of graphs.
        graph_labels: dict
            A `dict` contains a `list` of graph labels.
        """
        if self.from_binary:
            try:
                graph_path = os.readlink(os.path.join(self.dataset_path, 'all_graphs.bin'))
            except:
                graph_path = 'all_graphs.bin'
            cwd = os.getcwd()
            os.chdir(self.dataset_path)
            graph_list, graph_labels = load_graphs(graph_path)
            os.chdir(cwd)
        else:
            num_graph_per_core = self.number_of_graphs // self.num_of_cores + 1
            graph_index        = [x for x in range(self.number_of_graphs)]
            batch_index        = [graph_index[x: x + num_graph_per_core] for x in range(0, self.number_of_graphs, num_graph_per_core)]
            p                  = multiprocessing.Pool(self.num_of_cores)

            for batch_num, batch_index_list in enumerate(batch_index):
                p_out = p.apply_async(self.read_batch_graphs, args=(batch_index_list, batch_num))
            p_out.get()
            print('Waiting for all subprocesses...')
            p.close()
            p.join()
            print('All subprocesses done.')
            graph_list = []
            graph_labels = {"prop": []}
            for x in range(self.num_of_cores):
                batch_g, batch_labels = load_graphs(os.path.join(self.dataset_path, 'all_graphs_' + str(x) + '.bin'))
                graph_list.extend(batch_g)
                graph_labels["prop"] = tf.concat([graph_labels["prop"], batch_labels["prop"]], 0)
                os.remove(os.path.join(self.dataset_path, 'all_graphs_' + str(x) + '.bin'))
            save_graphs(os.path.join(self.dataset_path, 'all_graphs.bin'), graph_list, graph_labels)


        prop_max = tf.reduce_max(graph_labels['prop'])
        prop_min = tf.reduce_min(graph_labels['prop'])

        graph_build_scheme = {}
        graph_build_scheme['scale_prop']      = scale_prop
        graph_build_scheme['prop_max']        = float(prop_max.numpy())
        graph_build_scheme['prop_min']        = float(prop_min.numpy())
        graph_build_scheme['cutoff']          = self.cutoff
        graph_build_scheme['mode_of_NN']      = self.mode_of_NN
        graph_build_scheme['read_super_cell'] = self.read_super_cell
        graph_build_scheme['include_forces '] = self.include_forces
        graph_build_scheme['adsorbate ']      = self.adsorbate

        with open(os.path.join(ckpt_path, 'graph_build_scheme.json'), 'w') as fjson:
            json.dump(graph_build_scheme, fjson)

        # Always save the property per node as the label. For example: energy per atom (eV/atom)
        # if prop_per_node:
        #     node_num = tf.constant([x.num_nodes() for x in graph_list], dtype='float32')
        #     graph_labels['prop'] = graph_labels['prop'] / node_num

        # do not scale the label if you don't know what are you doing.
        if scale_prop:
            prop_span = prop_max - prop_min
            graph_labels['prop'] = (graph_labels['prop'] - prop_min) / prop_span

        return graph_list, graph_labels

class TrainValTestSplit(object):
    """
    Description:
    ----------
        Split the dataset.
    Parameters
    ----------
    validation_size: int or float
        int: number of samples of the validation set.
        float: portion of samples of the validation set
    test_size: int or float
        int: number of samples of the validation set.
        float: portion of samples of the validation set
    csv_file: str
        File name of a csv file that contains the filenames of crystals
        with cif or VASP formate.
    new_split: boolean
        Split the dataset by `sklearn.model_selection.train_test_split` or
        loaded from previously saved txt files.
    Returns of `__call__` method
    ----------------------------
    train_index : list
        A list of integers of training dataset.
    validation_index : list
        A list of integers of validation dataset.
    test_index : list
        A list of integers of test dataset.
    """
    def __init__(self,
                 validation_size,
                 test_size,
                 csv_file,
                 new_split=True):
        self.validation_size  = validation_size
        self.test_size        = test_size
        self.new_split        = new_split
        self.csv_file         = csv_file
        self.user_file        = os.path.split(self.csv_file)[0]
        fname_prop_data       = np.loadtxt(self.csv_file, dtype=str, delimiter=',')
        self.number_of_graphs = np.shape(fname_prop_data)[0]
        # print(self.number_of_graphs)

    def __call__(self):
        if self.new_split:
            train_index,      validation_and_test_index = train_test_split([x for x in range(self.number_of_graphs)], test_size=self.test_size+self.validation_size,  shuffle=True)
            validation_index, test_index                = train_test_split(validation_and_test_index, test_size=self.test_size/(self.test_size+self.validation_size), shuffle=True)
            np.savetxt(os.path.join(self.user_file, 'train.txt'),      train_index,      fmt='%.0f')
            np.savetxt(os.path.join(self.user_file, 'validation.txt'), validation_index, fmt='%.0f')
            np.savetxt(os.path.join(self.user_file, 'test.txt'),       test_index,       fmt='%.0f')
        else:
            # assert os.path.exists(os.path.join(self.user_file, 'train.txt')),      str(os.path.join(self.user_file, 'train.txt'))      + " not found."
            # assert os.path.exists(os.path.join(self.user_file, 'validation.txt')), str(os.path.join(self.user_file, 'validation.txt')) + " not found."
            # assert os.path.exists(os.path.join(self.user_file, 'test.txt')),       str(os.path.join(self.user_file, 'test.txt'))       + " not found."
            try:
                train_index      = np.loadtxt(os.path.join(self.user_file, 'train.txt'),      dtype=int)
                validation_index = np.loadtxt(os.path.join(self.user_file, 'validation.txt'), dtype=int)
                test_index       = np.loadtxt(os.path.join(self.user_file, 'test.txt'),       dtype=int)
            except OSError:
                print('User: Index file not found, generate new files...')
                train_index,      validation_and_test_index = train_test_split([x for x in range(self.number_of_graphs)], test_size=self.test_size+self.validation_size,  shuffle=True)
                validation_index, test_index                = train_test_split(validation_and_test_index, test_size=self.test_size/(self.test_size+self.validation_size), shuffle=True)
                np.savetxt(os.path.join(self.user_file, 'train.txt'),      train_index,      fmt='%.0f')
                np.savetxt(os.path.join(self.user_file, 'validation.txt'), validation_index, fmt='%.0f')
                np.savetxt(os.path.join(self.user_file, 'test.txt'),       test_index,       fmt='%.0f')
        return train_index, validation_index, test_index

# debug
if __name__ == '__main__':
    import time
    start = time.time()
    graph_builder = CrystalGraph(mode_of_NN='ase', cutoff=4.0)
    g = graph_builder.get_graph('POSCAR.txt', super_cell=False)
    print(f'Dur: {time.time() - start} s')



    # ase debug
    from ase.visualize import view
    poscar = read('POSCAR.txt')
    ase_atoms = poscar
    # num_atoms = len(poscar)
    ase_cutoffs = ase.neighborlist.natural_cutoffs(ase_atoms, mult=0.85, H=3.0, O=3.0)
    i, j, d, D = ase.neighborlist.neighbor_list('ijdD', ase_atoms, cutoff=2.7, self_interaction=False)
    # nl = NeighborList([2.5] * num_atoms, self_interaction=True, bothways=True)
    # nl.update(poscar)
    # nl.get_distance_indices()

    i, j, d, D = ase.neighborlist.neighbor_list('ijdD',
                                                poscar,
                                                cutoff=ase_cutoffs,
                                                self_interaction=True)

    ase_atoms = read('CONTCAR')
    view(poscar)
