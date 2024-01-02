# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:16:24 2021

@author: ZHANG Jun
"""

import json
import os
import sys

import torch
from ase.calculators.calculator import Calculator

from ..data.build_dataset import CrystalGraph
from ..lib.model_lib import load_model

class AgatCalculator(Calculator):
    """Calculator with ASE module.

    :param model_save_dir: Directory storing the well-trained model.
    :type model_save_dir: str
    :param graph_build_scheme_dir: Direcotry storing the ``graph_build_scheme.json`` file.
    :type graph_build_scheme_dir: str
    :param device: model device, defaults to 'cuda'
    :type device: str, optional
    :param **kwargs: other input arguments
    :type **kwargs: dict
    :return: Calculated properties.
    :rtype: dict

    Example::

        model_save_dir = 'agat_model'
        graph_build_scheme_dir = 'dataset'
        atoms = read('CONTCAR')
        calculator=AgatCalculator(model_save_dir,
                                  graph_build_scheme_dir)
        atoms = Atoms(atoms, calculator=calculator)
        dyn = BFGS(atoms, trajectory='test.traj')
        dyn.run(fmax=0.005)

        traj = read('test.traj', index=':')
        write("XDATCAR.gat", traj)

    """
    implemented_properties = ['energy', 'forces']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda',
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        # self.atoms = None  # copy of atoms object from last calculation
        # self.results = {}  # calculated properties (energy, forces, ...)
        # self.parameters = None  # calculational parameters
        # self._directory = None  # Initialize

        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.device = device

        self.model = load_model(self.model_save_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}
        self.cg = CrystalGraph(**self.graph_build_scheme)

    # def set(self):
    #     pass

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

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """

        :param atoms: ase.atoms object, defaults to None
        :type atoms: ase.atoms, optional
        :param properties: calculated properties, defaults to None
        :type properties: none, optional
        :param system_changes: DESCRIPTION, defaults to ['positions', 'numbers', 'cell', 'pbc']
        :type system_changes: TYPE, optional
        :return: calculated results
        :rtype: dict

        """

        if properties is None:
            properties = self.implemented_properties

        if atoms is not None:
            self.atoms = atoms.copy()

        # read graph
        graph, info = self.cg.get_graph(atoms)
        graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)

        self.results = {'energy': energy_pred[0].item() * len(atoms),
                        'forces': force_pred.cpu().numpy()}

# class GatCalculator(Calculator):
#     """ Calculator with ASE module. Pymatgen is also needed.

#     :param energy_model_save_path: Directory for the saved energy model.
#     :type energy_model_save_path: str
#     :param force_model_save_path: Directory for the saved force model.
#     :type force_model_save_path: str
#     :param graph_build_scheme_path: Directory for the json file of how to build graphs. This file is normally saved when you build your dataset.
#     :type graph_build_scheme_path: str
#     :param load_model: Load well-trained models or not, defaults to True
#     :type load_model: bool, optional
#     :param **kwargs: kwargs
#     :type **kwargs: kwargs

#     Example::

#         energy_model_save_dir = os.path.join('out_file', 'energy_ckpt')
#         force_model_save_dir = os.path.join('out_file', 'force_ckpt')
#         graph_build_scheme_dir = 'dataset'
#         poscar = read('POSCAR')
#         calculator=GatCalculator(energy_model_save_dir,
#                                     force_model_save_dir,
#                                     graph_build_scheme_dir)
#         poscar = Atoms(poscar, calculator=calculator)
#         dyn = BFGS(poscar, trajectory='test.traj')
#         dyn.run(fmax=0.005)

#         traj = read('test.traj', index=':')
#         write("XDATCAR.gat", traj)

#     """
#     implemented_properties = ['energy', 'energies', 'free_energy', 'forces', 'stress', 'stresses']
#     default_parameters     = { }
#     def __init__(self,
#                  energy_model_save_path,
#                  force_model_save_path,
#                  graph_build_scheme_path,
#                  load_model=True, **kwargs):


#         Calculator.__init__(self, **kwargs)
#         self.app = GatApp(energy_model_save_path,
#                           force_model_save_path,
#                           graph_build_scheme_path,
#                           load_model=True)


#     def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
#         if properties is None:
#             properties = self.implemented_properties

#         # save `atom` object to a file or to the memory
#         write('CONTCAR.gat', atoms)

#         # read file with MP
#         graph, info = self.app.get_graph('CONTCAR.gat')
#         # graph = graph.to(self.app.device)
#         # print(graph.device)

#         # get results
#         energy = self.app.get_energy(graph)
#         forces = self.app.get_forces(graph)

#         self.results = {'energy': energy,
#                         'forces': forces} # ,                        'energies': energies

# debug
if __name__ == '__main__':
    pass

