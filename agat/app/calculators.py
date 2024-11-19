# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:56:28 2023

@author: ZHANG Jun
"""

import os
import json
from shutil import move

import torch

from ase.calculators.calculator import Calculator
from ase.io import read

from ..data.build_graph import CrystalGraph, AseGraphTorch
from ..lib.file_lib import get_INCAR, get_KPOINTS_gamma, get_KPOINTS, get_POTCAR, modify_INCAR, run_vasp, file_force_action
from ..lib.model_lib import load_model, load_model_ensemble

class AgatCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
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

        # self.force_log = []
        # self.energy_log = []

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

        # print('call calculate')
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph, info = self.cg.get_graph(atoms)
        graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)

        self.results = {'energy': energy_pred[0].item() * len(atoms),
                        'forces': force_pred.cpu().numpy(),
                        'stress': stress_pred[0].cpu().numpy()}

        # self.force_log.append(force_pred.cpu().numpy())
        # self.energy_log.append(energy_pred[0].item() * len(atoms))

class AgatCalculatorAseGraphTorch(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda',
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.device = torch.device(device)

        self.model = load_model(self.model_save_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}
        self.graph_build_scheme['device'] = self.device
        self.ag = AseGraphTorch(**self.graph_build_scheme)

        # self.force_log = []
        # self.energy_log = []

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

        # print('AGAT forward in the AgatCalculatorFastGraph.')

        # atoms_tmp = atoms.copy()
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph = self.ag.get_graph(atoms)
        # graph = self.ag.build(atoms_tmp)
        # graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)

        self.results = {'energy': energy_pred * len(atoms),
                        'forces': force_pred,
                        'stress': stress_pred[0]}

        # self.force_log.append(force_pred.cpu().numpy())
        # self.energy_log.append(energy_pred[0].item() * len(atoms_tmp))

class AgatCalculatorAseGraphTorchNumpy(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda',
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.device = torch.device(device)

        self.model = load_model(self.model_save_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}
        self.graph_build_scheme['device'] = self.device
        self.ag = AseGraphTorch(**self.graph_build_scheme)

        # self.force_log = []
        # self.energy_log = []

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

        # print('AGAT forward in the AgatCalculatorFastGraph.')

        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph = self.ag.get_graph(atoms)
        # graph = self.ag.build(atoms_tmp)
        # graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)

        self.results = {'energy': energy_pred[0].item() * len(atoms),
                        'forces': force_pred.cpu().numpy(),
                        'stress': stress_pred[0].cpu().numpy()}

        # self.force_log.append(force_pred.cpu().numpy())
        # self.energy_log.append(energy_pred[0].item() * len(atoms_tmp))

class AgatEnsembleCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_ensemble_dir,
                 graph_build_scheme_dir,
                 start_step=0,
                 device = 'cuda',
                 # collect_mode = True, # collect snapshots, instead of using DFT
                 io = None,
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model_ensemble_dir = model_ensemble_dir
        self.step = start_step
        self.device = torch.device(device)
        self.model_list = load_model_ensemble(self.model_ensemble_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.

        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}

        self.graph_build_scheme['device'] = self.device
        self.g = CrystalGraph(**self.graph_build_scheme)

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

        # print(self.step)
        if atoms is not None:
            self.atoms = atoms.copy()
        if properties is None:
            properties = self.implemented_properties

        # read graph
        # self.atg.reset()
        # graph = self.atg.get_graph(atoms)
        graph, _ = self.g.get_graph(atoms)
        graph = graph.to(self.device)

        energies, forces, stresses = [], [], []
        with torch.no_grad():
            for model in self.model_list:
                energy_pred, force_pred, stress_pred = model.forward(graph)
                energies.append(energy_pred)
                forces.append(force_pred)
                stresses.append(stress_pred)

        energies = torch.stack(energies)
        forces = torch.stack(forces)
        stresses = torch.stack(stresses)[0]

        energy = torch.mean(energies).item() * len(atoms)
        force = torch.mean(forces, dim=0).cpu().numpy()
        stress = torch.mean(stresses, dim=0).cpu().numpy()

        self.results = {'energy': energy,
                        'forces': force,
                        'stress': stress}
        self.step += 1

class OnTheFlyCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_ensemble_dir,
                 graph_build_scheme_dir,
                 use_vasp=False,
                 start_step=0,
                 vasp_work_dir='.',
                 vasp_inputs_dir='.',
                 gamma_only=False,
                 vasp_potential_generator='getpotential.sh',
                 device = 'cuda',
                 energy_threshold = 0.5,
                 force_threshold = 0.5,
                 stress_threshold = 0.5,
                 # collect_mode = True, # collect snapshots, instead of using DFT
                 io = None,
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model_ensemble_dir = model_ensemble_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.use_vasp = use_vasp
        self.collected_snapshot_num = 0
        self.step = start_step
        self.vasp_work_dir = vasp_work_dir
        self.vasp_inputs_dir = vasp_inputs_dir
        self.gamma_only = gamma_only
        self.vasp_potential_generator = vasp_potential_generator
        self.device = torch.device(device)

        self.root_dir = os.getcwd()

        self.model_list = load_model_ensemble(self.model_ensemble_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.

        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}

        self.graph_build_scheme['device'] = self.device
        self.g = CrystalGraph(**self.graph_build_scheme)

        self.energy_std = None
        self.force_std = None
        self.stress_std = None

        self.energy_threshold = energy_threshold
        self.force_threshold = force_threshold
        self.stress_threshold = stress_threshold

        self.io = io
        # if not isinstance(self.io, _io.TextIOWrapper):
        #     self.io = open(self.io, 'a+')

        if not os.path.exists(self.vasp_work_dir):
            os.makedirs(self.vasp_work_dir)

    def run_vasp(self, work_dir):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        r = run_vasp()
        os.chdir(self.root_dir)
        return r

    def prepare_vasp_calculation(self, atoms, dst_dir):
        atoms.write(os.path.join(dst_dir, 'POSCAR'), format='vasp')
        get_INCAR(os.path.join(self.vasp_inputs_dir, 'INCAR'),
                  os.path.join(dst_dir, 'INCAR'))
        if self.gamma_only:
            get_KPOINTS_gamma(dst_dir)
        else:
            get_KPOINTS(os.path.join(self.vasp_inputs_dir, 'KPOINTS'),
                      os.path.join(dst_dir, 'KPOINTS'))

        get_POTCAR(cmd=self.vasp_potential_generator,
                   line=1, working_dir=dst_dir)
        modify_INCAR(dst_dir, 'NSW', '0')
        modify_INCAR(dst_dir, 'ISTART', '1')
        modify_INCAR(dst_dir, 'ICHARG', '1')
        modify_INCAR(dst_dir, 'LWAVE', '.TRUE.')
        modify_INCAR(dst_dir, 'LCHARG', '.TRUE.')

        # Search WAVECAR and CHGCAR files from latest 3 steps.
        for s in range(self.step-1, self.step-4, -1):
            wavecar = os.path.join(self.vasp_work_dir, str(s), 'WAVECAR')
            chgcar = os.path.join(self.vasp_work_dir, str(s), 'CHGCAR')
            if os.path.exists(wavecar) and os.path.exists(chgcar):
                print(f'WAVECAR and CHGCAR files detected in {wavecar} and {chgcar}',
                      file=self.io)
                file_force_action(move,
                                  wavecar,
                                  os.path.join(dst_dir,
                                               'WAVECAR'))
                file_force_action(move,
                                  chgcar,
                                  os.path.join(dst_dir,
                                               'CHGCAR'))
            break

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

        # print(self.step)
        if atoms is not None:
            self.atoms = atoms.copy()
        if properties is None:
            properties = self.implemented_properties

        # read graph
        # self.atg.reset()
        # graph = self.atg.get_graph(atoms)
        graph, _ = self.g.get_graph(atoms)
        graph = graph.to(self.device)

        energies, forces, stresses = [], [], []
        with torch.no_grad():
            for model in self.model_list:
                energy_pred, force_pred, stress_pred = model.forward(graph)
                energies.append(energy_pred)
                forces.append(force_pred)
                stresses.append(stress_pred)

        energies = torch.stack(energies)
        forces = torch.stack(forces)
        stresses = torch.stack(stresses)[0]

        energy = torch.mean(energies).item() * len(atoms)
        force = torch.mean(forces, dim=0).cpu().numpy()
        stress = torch.mean(stresses, dim=0).cpu().numpy()

        self.energy_std = torch.std(energies)
        self.force_std = torch.mean(torch.std(forces, dim=0))
        self.stress_std = torch.mean(torch.std(stresses, dim=0))

        over_threshold = self.energy_std > self.energy_threshold or\
                         self.force_std > self.force_threshold or\
                         self.stress_std > self.stress_threshold

        if over_threshold:
            self.collected_snapshot_num += 1
            work_dir = os.path.join(self.vasp_work_dir, str(self.step))
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)

            self.prepare_vasp_calculation(self.atoms, work_dir)

            if self.use_vasp:
                self.run_vasp(work_dir) # run vasp

                # read dft results
                atoms = read(os.path.join(work_dir, 'OUTCAR'))

                energy = atoms.get_total_energy() # torch.tensor(, dtype=torch.float32)
                force = atoms.get_forces(apply_constraint=False)
                stress = atoms.get_stress(apply_constraint=False)

        print(f'Step: {self.step} over_threshold: {over_threshold} use_vasp: {self.use_vasp} ',
              file=self.io)

        self.results = {'energy': energy,
                        'forces': force,
                        'stress': stress}

        self.step += 1

        # print(self.energy_std, self.force_std, self.stress_std)

