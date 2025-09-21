# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:26:30 2023

@author: ZHANG Jun
"""

import os
import sys

import numpy as np
from ase.io import read, write

from agat.lib.file_lib import modify_INCAR
from agat.default_parameters import default_hp_dft_config

class DataGenerator(object):
    def __init__(self, bulk_fname):
        self.bulk_fname = bulk_fname
        self.template_atoms = read(bulk_fname)
        self.root_dir = os.getcwd()

    def set_structure(self, fname):
        self.bulk_fname = fname
        self.template_atoms = read(fname)

    def get_INCAR(self, dst='.'):
        path = os.path.join(dst, 'INCAR')
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(default_hp_dft_config['INCAR_static'])

    def get_KPOINTS(self, dst='.'):
        path = os.path.join(dst, 'KPOINTS')
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(default_hp_dft_config['KPOINTS'])

    def get_POTCAR(self, line=1, working_dir='.'):
        if sys.platform != 'linux':
            print('The POTCAR file can only be generated on a Linux OS.')
            return None
        path = os.path.join(working_dir, 'POTCAR')
        if not os.path.exists(path):
            os.chdir(working_dir)
            os.system(f"getpotential.sh {str(line)}")
            os.chdir(self.root_dir)

    def apply_strain(self, ase_atoms, strain: float):
        atoms = ase_atoms.copy()
        cell = atoms.get_cell() * (1 + strain)
        atoms.set_cell(cell, scale_atoms=True)
        return atoms

    def apply_perturbation(self, ase_atoms, amplitude: float = 0.2):
        atoms = ase_atoms.copy()
        posistions = atoms.get_positions()
        num_atoms = len(atoms)
        increment = np.clip(np.random.normal(0.0, amplitude / 3,  size=(num_atoms,3)), -amplitude, amplitude) # np.random.uniform(-amplitude, amplitude, (num_atoms,3))
        constraints = atoms.constraints
        if len(constraints) > 0:
            increment[constraints[0].index] = 0.0
        posistions += increment
        atoms.set_positions(posistions)
        return atoms

    def create_vacancy(self, ase_atoms, index='random'):
        atoms = ase_atoms.copy()
        if index == 'random':
            del atoms[np.random.randint(0, len(atoms))]
        elif isinstance(index, [list, int, np.ndarray]):
            del atoms[index]
        else:
            raise RuntimeError('Wrong input index. Options: "random", int, list of int')
        return atoms

    def create_species_vacancy(self, ase_atoms, species='Li', num=1):
        atoms = ase_atoms.copy()
        symbols = np.array(atoms.get_chemical_symbols())
        species_i = np.where(symbols==species)[0]
        index = np.random.choice(species_i, size=num, replace=False)
        del atoms[index]
        return atoms

    def relocate_atoms(self, ase_atoms, displacement=[0.0, 0.0, 0.1], species='Li'):
        raise RuntimeError('This function is not prepared yet.')
        atoms = ase_atoms.copy()
        symbols = np.array(atoms.get_chemical_symbols())
        species_i = np.where(symbols==species)[0]
        index = np.random.choice(species_i, replace=False)
        ...
        return atoms

    def static(self, dst='.', strain=[-0.02, -0.01, 0.0, 0.01, 0.02], perturbation_num=10):
        path = os.path.join(dst, 'static')
        if not os.path.exists(path):
            os.mkdir(path)

        for s in strain:
            for p in range(perturbation_num):
                path_tmp = os.path.join(path, f'strain{s}_perturbation{p}')
                if not os.path.exists(path_tmp):
                    os.mkdir(path_tmp)
                atoms = self.apply_strain(
                    self.apply_perturbation(self.template_atoms), s)
                write(os.path.join(path_tmp, 'POSCAR'), atoms, format='vasp')
                self.get_INCAR(path_tmp)
                self.get_KPOINTS_gamma(path_tmp)
                self.get_POTCAR(line=6, working_dir=path_tmp)

    def aimd(self, dst='.',
             strain=[-0.02, -0.01, 0.0, 0.01, 0.02],
             start_T=[100, 300, 500],
             end_T=[400, 600, 800]):
        assert len(start_T) == len(end_T), f'Expect same length of start_T and end_T. Got length {len(start_T)} and {len(end_T)}, respectively.'
        path = os.path.join(dst, 'aimd')
        if not os.path.exists(path):
            os.mkdir(path)

        # NVT ensemble with Nose-Hoover thermostat
        for s in strain:
            for st, et in zip(start_T, end_T):
                path_tmp = os.path.join(path, f'strain{s}_t{st}-{et}')
                if not os.path.exists(path_tmp):
                    os.mkdir(path_tmp)
                atoms = self.apply_strain(
                    self.apply_perturbation(self.template_atoms), s)
                write(os.path.join(path_tmp, 'POSCAR'), atoms, format='vasp')
                self.get_INCAR(path_tmp)
                self.get_KPOINTS_gamma(path_tmp)
                self.get_POTCAR(line=6, working_dir=path_tmp)

                modify_INCAR(working_dir=path_tmp, key='ISIF', value='2', s='NVT')
                modify_INCAR(working_dir=path_tmp, key='IBRION', value='0')
                modify_INCAR(working_dir=path_tmp, key='MDALGO', value='2')
                modify_INCAR(working_dir=path_tmp, key='TEBEG', value=st)
                modify_INCAR(working_dir=path_tmp, key='TEEND', value=et)
                modify_INCAR(working_dir=path_tmp, key='NSW', value='500')
                modify_INCAR(working_dir=path_tmp, key='POTIM', value=str((-1/400 * st + 2.25) / 2)) # 2 is used to adjust time step.
                modify_INCAR(working_dir=path_tmp, key='SMASS', value='1.0')
                modify_INCAR(working_dir=path_tmp, key='EDIFFG', value='#')

        # NPT ensemble with Nose-Hoover thermostat
        for st, et in zip(start_T, end_T):
            path_tmp = os.path.join(path, f'NPT_t{st}-{et}')
            if not os.path.exists(path_tmp):
                os.mkdir(path_tmp)
            atoms = self.apply_perturbation(self.template_atoms)
            write(os.path.join(path_tmp, 'POSCAR'), atoms, format='vasp')
            self.get_INCAR(path_tmp)
            self.get_KPOINTS_gamma(path_tmp)
            self.get_POTCAR(line=6, working_dir=path_tmp)

            modify_INCAR(working_dir=path_tmp, key='ISIF', value='3', s='NPT')
            modify_INCAR(working_dir=path_tmp, key='IBRION', value='0')
            modify_INCAR(working_dir=path_tmp, key='MDALGO', value='2')
            modify_INCAR(working_dir=path_tmp, key='TEBEG', value=st)
            modify_INCAR(working_dir=path_tmp, key='TEEND', value=et)
            modify_INCAR(working_dir=path_tmp, key='NSW', value='1000')
            modify_INCAR(working_dir=path_tmp, key='POTIM', value=str((-1/400 * st + 2.25) / 2)) # 2 is used to adjust time step.
            modify_INCAR(working_dir=path_tmp, key='SMASS', value='1.0')
            modify_INCAR(working_dir=path_tmp, key='EDIFFG', value='#')

if __name__ == '__main__':
    dg = DataGenerator(bulk_fname=os.path.join('ion_diffusion', 'structures', 'CONTCAR_LGPS_gamma'))
    # dg.static(dst=os.path.join('ion_diffusion', 'test'))
    dg.aimd(dst=os.path.join('ion_diffusion', 'test'))
