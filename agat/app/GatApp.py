# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:16:24 2021

@author: ZHANG Jun
"""

import json
import os
import sys
import tensorflow as tf
from agat.data.data import CrystalGraph
from agat.model.GatEnergyModel import EnergyGat as GATE
from agat.model.GatForceModel import ForceGat as GATF
from dgl.data.utils import load_graphs
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator
from ase.io import read, write

class GatApp(object):
    """ Basic application of well-trained AGAT model.

    :param energy_model_save_path: Directory for the saved energy model.
    :type energy_model_save_path: str
    :param force_model_save_path: Directory for the saved force model
    :type force_model_save_path: str
    :param force_model_save_path: Directory for the json file of how to build graphs. This file is normally saved when you build your dataset.
    :type force_model_save_path: str
    :param load_model: Load well-trained models or not, defaults to True
    :type load_model: bool, optional
    :param gpu: Predict with GPU or CPU, negative value for GPU, defaults to -1
    :type gpu: int, optional
    :return: Energy and forces
    :rtype: tensorflow EagerTensor.

    Example::

        energy_model_save_dir = os.path.join('out_file', 'energy_ckpt')
        force_model_save_dir = os.path.join('out_file', 'force_ckpt')
        graph_build_scheme_dir = 'dataset'
        app = GatApp(energy_model_save_dir, force_model_save_dir, graph_build_scheme_dir)

        graph, info = app.get_graph('POSCAR')
        energy = app.get_energy(graph)
        forces = app.get_forces(graph)

    """
    def __init__(self, energy_model_save_path, force_model_save_path, graph_build_scheme_path, load_model=True, gpu=-1):

        self.energy_model_save_path = energy_model_save_path
        self.force_model_save_path  = force_model_save_path
        self.graph_build_scheme_path = graph_build_scheme_path
        self.gpu                    = gpu
        if self.gpu < 0:
            self.device             = "/cpu:0"
        else:
            self.device             = "/gpu:{}".format(self.gpu)

        if load_model:
            self.energy_model       = self.load_energy_model(energy_model_save_path)
            self.force_model        = self.load_force_model(force_model_save_path)

        self.graph_build_scheme     = self.load_graph_build_scheme(self.graph_build_scheme_path) # you can also load it from `force_model_save_path`

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False}
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'], **build_properties}
        self.cg                     = CrystalGraph(**self.graph_build_scheme, gpu=self.gpu)

    def load_graph_build_scheme(self, path):
        """ Load graph building scheme. This file is normally saved when you build your dataset.

        :param path: Directory for storing ``graph_build_scheme.json`` file.
        :type path: str
        :return: A dict denotes how to build the graph.
        :rtype: dict

        """

        json_file  = os.path.join(path, 'graph_build_scheme.json')
        assert os.path.exists(json_file), f"{json_file} file dose not exist."
        with open(json_file, 'r') as jsonf:
            graph_build_scheme = json.load(jsonf)
        return graph_build_scheme

    def load_energy_model(self, energy_model_save_path):
        """ Load the energy model.

        :param energy_model_save_path: Directory for the saved energy model.
        :type energy_model_save_path: str
        :return: An AGAT model
        :rtype: agat.model.GatEnergyModel.EnergyGat

        """

        json_file  = os.path.join(energy_model_save_path, 'gat_model.json')
        graph_file = os.path.join(energy_model_save_path, 'graph_tmp.bin')
        ckpt_file  = os.path.join(energy_model_save_path, 'gat.ckpt')

        for f in [json_file, graph_file, ckpt_file + '.index']:
            assert os.path.exists(f), f"{f} file dose not exist."

        # load json file
        with open(json_file, 'r') as jsonf:
            model_config = json.load(jsonf)

        # build a model
        model =  GATE(model_config['num_gat_out_list'],
                      num_readout_out_list = model_config['num_readout_out_list'],
                      head_list_en         = model_config['head_list_en'],
                      embed_activation     = model_config['embed_activation'],
                      readout_activation   = model_config['readout_activation'],
                      bias                 = model_config['bias'],
                      negative_slope       = model_config['negative_slope'])

        # load weights
        graph_tmp, label_tmp = load_graphs(graph_file)
        graph_tmp = graph_tmp[0].to(self.device)
        with tf.device(self.device):
            model(graph_tmp)
        load_status          = model.load_weights(ckpt_file)
        load_status.assert_consumed()
        print(f'Load energy model weights from {ckpt_file} successfully.')
        return model

    def load_force_model(self, force_model_save_path):
        """ Load the force model.

        :param force_model_save_path: Directory for the saved force model.
        :type force_model_save_path: str
        :return: An AGAT model
        :rtype: agat.model.GatForceModel.ForceGat

        """
        json_file  = os.path.join(force_model_save_path, 'gat_model.json')
        graph_file = os.path.join(force_model_save_path, 'graph_tmp.bin')
        ckpt_file  = os.path.join(force_model_save_path, 'gat.ckpt')

        for f in [json_file, graph_file, ckpt_file + '.index']:
            assert os.path.exists(f), f"{f} file dose not exist."

        # load json file
        with open(json_file, 'r') as jsonf:
            model_config = json.load(jsonf)

        # build a model
        model =  GATF(model_config['num_gat_out_list'],
                      model_config['num_readout_out_list'],
                      model_config['head_list_force'],
                      model_config['embed_activation'],
                      model_config['readout_activation'],
                      model_config['bias'],
                      model_config['negative_slope'],
                      model_config['batch_normalization'],
                      model_config['tail_readout_no_act'])

        # load weights
        graph_tmp, label_tmp = load_graphs(graph_file)
        graph_tmp = graph_tmp[0].to(self.device)
        with tf.device(self.device):
            model(graph_tmp)
        load_status          = model.load_weights(ckpt_file)
        load_status.assert_consumed()
        print(f'Load force model weights from {ckpt_file} successfully.')
        return model

    def get_graph(self, fname):
        """ Build agat graph from VASP file.

        :param fname: The file name.
        :type fname: str
        :return: dgl graph.
        :rtype: dgl.heterograph.DGLGraph

        """

        return self.cg.get_graph(fname) # No need to read True forces for application

    def get_energy(self, graph):
        """ Forward the energy model.

        :param graph: dgl graph.
        :type graph: dgl.heterograph.DGLGraph
        :return: Total energy.
        :rtype: tf EagerTensor.

        """

        with tf.device(self.device):
            # graph.to(self.device)
            energy = tf.reduce_sum(self.energy_model(graph)).numpy()
        return energy

    def get_energies(self, graph):
        """ Forward the energy model.

        :param graph: dgl graph.
        :type graph: dgl.heterograph.DGLGraph
        :return: Energies of all atoms.
        :rtype: numpy.ndarray.

        """

        with tf.device(self.device):
            # graph.to(self.device)
            return self.energy_model(graph).numpy()

    def get_forces(self, graph):
        """ Forward the force model.

        :param graph: dgl graph.
        :type graph: dgl.heterograph.DGLGraph
        :return: Atomic forces.
        :rtype: numpy.ndarray.

        """
        with tf.device(self.device):
            # graph.to(self.device)
            return self.force_model(graph).numpy()

    def get_stress(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return None

class GatAseCalculator(Calculator):
    """ Calculator with ASE module.

    :param energy_model_save_path: Directory for the saved energy model.
    :type energy_model_save_path: str
    :param force_model_save_path: Directory for the saved force model.
    :type force_model_save_path: str
    :param graph_build_scheme_path: Directory for the json file of how to build graphs. This file is normally saved when you build your dataset.
    :type graph_build_scheme_path: str
    :param load_model: Load well-trained models or not, defaults to True
    :type load_model: bool, optional
    :param gpu: Predict with GPU or CPU, negative value for GPU, defaults to -1
    :type gpu: int, optional
    :param **kwargs: kwargs
    :type **kwargs: dict/kwargs


    Example::

        energy_model_save_dir = os.path.join('out_file', 'energy_ckpt')
        force_model_save_dir = os.path.join('out_file', 'force_ckpt')
        graph_build_scheme_dir = 'dataset'
        poscar = read('POSCAR')
        calculator=GatAseCalculator(energy_model_save_dir,
                                    force_model_save_dir,
                                    graph_build_scheme_dir)
        poscar = Atoms(poscar, calculator=calculator)
        dyn = BFGS(poscar, trajectory='test.traj')
        dyn.run(fmax=0.005)

        traj = read('test.traj', index=':')
        write("XDATCAR.gat", traj)

    """
    implemented_properties = ['energy', 'energies', 'free_energy', 'forces', 'stress', 'stresses']
    default_parameters     = { }
    def __init__(self,
                 energy_model_save_path,
                 force_model_save_path,
                 graph_build_scheme_path,
                 load_model=True,
                 gpu       = -1,
                 **kwargs):

        Calculator.__init__(self, **kwargs)
        self.app = GatApp(energy_model_save_path,
                          force_model_save_path,
                          graph_build_scheme_path,
                          load_model=True,
                          gpu=gpu)

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph, info = self.app.get_graph(atoms)
        # print(graph.device)
        graph = graph.to(self.app.device)
        # print(graph.device)

        # get results
        energy = self.app.get_energy(graph)
        forces = self.app.get_forces(graph)

        self.results = {'energy': energy,
                        'forces': forces} # ,                        'energies': energies

class GatCalculator(Calculator):
    """ Calculator with ASE module. Pymatgen is also needed.

    :param energy_model_save_path: Directory for the saved energy model.
    :type energy_model_save_path: str
    :param force_model_save_path: Directory for the saved force model.
    :type force_model_save_path: str
    :param graph_build_scheme_path: Directory for the json file of how to build graphs. This file is normally saved when you build your dataset.
    :type graph_build_scheme_path: str
    :param load_model: Load well-trained models or not, defaults to True
    :type load_model: bool, optional
    :param **kwargs: kwargs
    :type **kwargs: kwargs

    Example::

        energy_model_save_dir = os.path.join('out_file', 'energy_ckpt')
        force_model_save_dir = os.path.join('out_file', 'force_ckpt')
        graph_build_scheme_dir = 'dataset'
        poscar = read('POSCAR')
        calculator=GatCalculator(energy_model_save_dir,
                                    force_model_save_dir,
                                    graph_build_scheme_dir)
        poscar = Atoms(poscar, calculator=calculator)
        dyn = BFGS(poscar, trajectory='test.traj')
        dyn.run(fmax=0.005)

        traj = read('test.traj', index=':')
        write("XDATCAR.gat", traj)

    """
    implemented_properties = ['energy', 'energies', 'free_energy', 'forces', 'stress', 'stresses']
    default_parameters     = { }
    def __init__(self,
                 energy_model_save_path,
                 force_model_save_path,
                 graph_build_scheme_path,
                 load_model=True, **kwargs):


        Calculator.__init__(self, **kwargs)
        self.app = GatApp(energy_model_save_path,
                          force_model_save_path,
                          graph_build_scheme_path,
                          load_model=True)


    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        if properties is None:
            properties = self.implemented_properties

        # save `atom` object to a file or to the memory
        write('CONTCAR.gat', atoms)

        # read file with MP
        graph, info = self.app.get_graph('CONTCAR.gat')
        # graph = graph.to(self.app.device)
        # print(graph.device)

        # get results
        energy = self.app.get_energy(graph)
        forces = self.app.get_forces(graph)

        self.results = {'energy': energy,
                        'forces': forces} # ,                        'energies': energies

# debug
if __name__ == '__main__':
    energy_model_save_dir = os.path.join('out_file', 'energy_ckpt')
    force_model_save_dir = os.path.join('out_file', 'force_ckpt')
    graph_build_scheme_dir = 'dataset'
    app = GatApp(energy_model_save_dir, force_model_save_dir, graph_build_scheme_dir)

    import time
    start = time.time()
    for i in range(10):
        graph, info = app.get_graph('POSCAR')
        print(time.time() - start)

    import time
    start = time.time()
    for i in range(10):
        energy = app.get_energy(graph)
        forces = app.get_forces(graph)
        print(time.time() - start)

    #BFGS
    poscar = read('POSCAR')
    calculator=GatAseCalculator(energy_model_save_dir,
                                force_model_save_dir,
                                graph_build_scheme_dir)
    poscar = Atoms(poscar, calculator=calculator)
    dyn = BFGS(poscar, trajectory='test.traj')
    dyn.run(fmax=0.005)

    traj = read('test.traj', index=':')
    write("XDATCAR.gat", traj)

    # with pymatgen
    poscar = read('POSCAR')
    calculator=GatCalculator(energy_model_save_dir,
                                force_model_save_dir,
                                graph_build_scheme_dir)
    poscar = Atoms(poscar, calculator=calculator)
    dyn = BFGS(poscar, trajectory='test.traj')
    dyn.run(fmax=0.005)

    traj = read('test.traj', index=':')
    write("XDATCAR.gat", traj)

    # from ase.optimize import BFGS, LBFGS, LBFGSLineSearch, GPMin, FIRE
    # import time
    # from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

    # for optimizer in (BFGS, LBFGS, LBFGSLineSearch, GPMin, FIRE):
    #     start = time.time()
    #     poscar = read('POSCAR')
    #     calculator=GatAseCalculator(energy_model_save_dir,
    #                                 force_model_save_dir,
    #                                 graph_build_scheme_dir)
    #     poscar = Atoms(poscar, calculator=calculator)
    #     # dyn = optimizer(poscar, trajectory='test.traj', maxstep=0.05)
    #     dyn = optimizer(poscar, trajectory='test.traj', callback_always=True)
    #     dyn.run(fmax=0.05, steps =5)
    #     print(f'Time used for {optimizer} is {time.time()-start}')


    #BFGS
    poscar_init = read('POSCAR')
    f = open('test.log', 'w', buffering=1)
    for i in range(10):
        for j in range(10):
            poscar = poscar_init.repeat(i+1)
            calculator=GatAseCalculator(energy_model_save_dir,
                                        force_model_save_dir,
                                        graph_build_scheme_dir, gpu=-1)
            poscar = Atoms(poscar, calculator=calculator)
            dyn = BFGS(poscar, trajectory='test.traj')
            start = time.time()
            dyn.run(fmax=0.0000005, steps=20)
            print('Time:', time.time() - start, 'Number of atoms:', len(poscar), file=f)

            traj = read('test.traj', index=':')
            write("XDATCAR.gat", traj)
    f.close()
