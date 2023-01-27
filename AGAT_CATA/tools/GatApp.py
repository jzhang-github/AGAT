# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:16:24 2021

@author: ZHANG Jun
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from modules.Crystal2Graph import CrystalGraph
from modules.GatEnergyModel import GAT as GATE
from modules.GatForceModel import GAT as GATF
from dgl.data.utils import load_graphs
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator
from ase.io import read, write

class GatApp(object):
    def __init__(self, energy_model_save_path, force_model_save_path, load_model=True, gpu=-1):
        self.energy_model_save_path = energy_model_save_path
        self.force_model_save_path  = force_model_save_path
        self.gpu                    = gpu
        if gpu < 0:
            self.device             = "/cpu:0"
        else:
            self.device             = "/gpu:{}".format(gpu)
        
        if load_model:
            self.energy_model       = self.load_energy_model(energy_model_save_path)
            self.force_model        = self.load_force_model(force_model_save_path)

        self.graph_build_scheme     = self.load_graph_build_scheme(energy_model_save_path) # you can also load it from `force_model_save_path`
        self.cg                     = CrystalGraph(cutoff     = self.graph_build_scheme['cutoff'],
                                                   mode_of_NN = self.graph_build_scheme['mode_of_NN'])

    def load_graph_build_scheme(self, model_save_path):
        json_file  = os.path.join(model_save_path, 'graph_build_scheme.json')
        assert os.path.exists(json_file), f"{json_file} file dose not exist."
        with open(json_file, 'r') as jsonf:
            graph_build_scheme = json.load(jsonf)
        return graph_build_scheme

    def load_energy_model(self, energy_model_save_path):
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
        return self.cg.get_graph(fname,
                                 super_cell=self.graph_build_scheme['read_super_cell'],
                                 include_forces=False) # No need to read True forces for application

    def get_energy(self, graph):
        with tf.device(self.device):
            # graph.to(self.device)
            energy = tf.reduce_sum(self.energy_model(graph)).numpy()
        return energy

    def get_energies(self, graph):
        with tf.device(self.device):
            # graph.to(self.device)
            return self.energy_model(graph).numpy()

    def get_forces(self, graph):
        with tf.device(self.device):
            # graph.to(self.device)
            return self.force_model(graph).numpy()

    def get_stress(self,):
        return None

class GatAseCalculator(Calculator):
    implemented_properties = ['energy', 'energies', 'free_energy', 'forces', 'stress', 'stresses']
    default_parameters     = { }
    def __init__(self,
                 energy_model_save_path,
                 force_model_save_path,
                 load_model=True,
                 gpu       = -1,
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.app = GatApp(energy_model_save_path,
                          force_model_save_path,
                          load_model=True,
                          gpu=gpu)

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph = self.app.get_graph(atoms)
        # print(graph.device)
        graph = graph.to(self.app.device)
        # print(graph.device)

        # get results
        energy = self.app.get_energy(graph)
        forces = self.app.get_forces(graph)

        self.results = {'energy': energy,
                        'forces': forces} # ,                        'energies': energies

class GatCalculator(Calculator):
    implemented_properties = ['energy', 'energies', 'free_energy', 'forces', 'stress', 'stresses']
    default_parameters     = { }
    def __init__(self,
                 energy_model_save_path,
                 force_model_save_path,
                 load_model=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.app = GatApp(energy_model_save_path,
                          force_model_save_path,
                          load_model=True)

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        if properties is None:
            properties = self.implemented_properties

        # save `atom` object to a file or to the memory
        write('CONTCAR.gat', atoms)

        # read file with MP
        graph = self.app.get_graph('CONTCAR.gat')
        # graph = graph.to(self.app.device)
        # print(graph.device)

        # get results
        energy = self.app.get_energy(graph)
        forces = self.app.get_forces(graph)

        self.results = {'energy': energy,
                        'forces': forces} # ,                        'energies': energies

# debug
if __name__ == '__main__':
    energy_model_save_dir = os.path.join('..', 'GAT_5.1_potential', 'energy_ckpt')
    force_model_save_dir  = os.path.join('..', 'GAT_5.1_potential', 'force_ckpt')
    app = GatApp(energy_model_save_dir, force_model_save_dir)

    graph= app.get_graph('POSCAR_surface')
    energy = app.get_energy(graph)
    forces = app.get_forces(graph)

    #BFGS
    poscar = read('POSCAR')
    calculator=GatCalculator(energy_model_save_dir,
                             force_model_save_dir)
    poscar = Atoms(poscar, calculator=calculator)
    dyn = BFGS(poscar, trajectory='test.traj')
    dyn.run(fmax=0.05)

    traj = read('test.traj', index=':')
    write("XDATCAR.gat", traj)
