# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:17:36 2023

@author: ZHANG Jun
"""

import platform
import numpy as np
import os
import stat
import shutil

from ase.formula import Formula
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic
from ase.data import covalent_radii, atomic_numbers

def perturb_positions(atoms, amplitude=0.1):
    calculator  = atoms.get_calculator()
    atoms       = atoms.copy()
    atoms.set_calculator(calculator)
    posistions  = atoms.arrays['positions']
    num_atoms   = len(atoms)
    increment   = np.clip(np.random.normal(0.0, amplitude / 3,  size=(num_atoms,3)), -amplitude, amplitude) # np.random.uniform(-amplitude, amplitude, (num_atoms,3))
    constraints = atoms.constraints
    if len(constraints) > 0:
        increment[constraints[0].index] = 0.0
    posistions += increment
    atoms.set_positions(posistions)
    return atoms

def scale_atoms(atoms, scale_factor=1.0):
    calculator = atoms.get_calculator()
    new_atoms  = atoms.copy()
    new_atoms.set_calculator(calculator)
    cell        = new_atoms.get_cell()
    frac_coords = new_atoms.get_scaled_positions()
    new_cell    = cell * scale_factor
    new_atoms.set_cell(new_cell)
    new_atoms.set_scaled_positions(frac_coords)
    return new_atoms

def get_concentration_from_ase_formula(formula):
    f_dict = Formula(formula).count()
    tot = np.sum(list(f_dict.values()))
    c_dict = {k: v/tot for k, v in f_dict.items()}
    c_dict = {k:c_dict[k] for k in c_dict if c_dict[k] > 0}
    return c_dict

def get_v_per_atom(chemical_formula):
    frac_dict = get_concentration_from_ase_formula(chemical_formula)
    frac_dict = {k:(0.0 if k not in frac_dict else frac_dict[k]) for k in ['Ni', 'Co', 'Fe', 'Pd', 'Pt']}
    return -282.7957531391954 * (frac_dict['Ni'] + frac_dict['Co'] + frac_dict['Fe'])\
        - 278.79605077419797 * frac_dict['Pd'] - 278.6228860885035 * frac_dict['Pt']\
            + 293.66128761358624

def get_ase_atom_from_formula(chemical_formula, v_per_atom=None):
    # interpret formula
    atomic_fracions = get_concentration_from_ase_formula(chemical_formula)
    elements = [x for x in atomic_fracions]
    element_number = [atomic_numbers[x] for x in elements]
    mean_radii       = np.sum([covalent_radii[n] * atomic_fracions[e] for n, e in zip(element_number, elements)]) #

    Pt_radii           = covalent_radii[78]
    latticeconstant    = mean_radii / Pt_radii * 3.92
    atoms              = FaceCenteredCubic('Pt', directions=[[1,-1,0], [0,1,-1], [1,1,1]], size=(4, 4, 2), latticeconstant=latticeconstant, pbc=True)
    total_atom         = len(atoms)
    num_atom_list      = np.array(list(atomic_fracions.values())) * total_atom
    num_atom_list      = np.around(num_atom_list, decimals=0)
    total_tmp          = np.sum(num_atom_list)
    deviation          = total_atom - total_tmp
    num_atom_list[np.random.randint(len(elements))] += deviation

    # shuffle atoms
    ase_number    = []
    for i_index, i in enumerate(num_atom_list):
        for j in range(int(i)):
            ase_number.append(element_number[i_index])
    np.random.shuffle(ase_number)
    atoms.set_atomic_numbers(ase_number)

    # scale atoms
    if isinstance(v_per_atom, (float, int)):
        volume = atoms.cell.volume
        volume_per_atom = volume / len(atoms)
        volume_ratio = v_per_atom / volume_per_atom
        scale_factor = pow(volume_ratio, 1/3)
        atoms = scale_atoms(atoms, scale_factor)
    return atoms

def get_ase_atom_from_formula_template( chemical_formula, v_per_atom=None,
                                       template_file='POSCAR_temp'):
    # interpret formula
    # the template file should be a bulk structure
    atomic_fracions    = get_concentration_from_ase_formula(chemical_formula)
    elements           = [x for x in atomic_fracions]
    element_number     = [atomic_numbers[x] for x in elements]
    atoms              = read(template_file)
    total_atom         = len(atoms)
    num_atom_list      = np.array(list(atomic_fracions.values())) * total_atom
    num_atom_list      = np.around(num_atom_list, decimals=0)
    total_tmp          = np.sum(num_atom_list)
    deviation          = total_atom - total_tmp
    num_atom_list[np.random.randint(len(elements))] += deviation

    # shuffle atoms
    ase_number    = []
    for i_index, i in enumerate(num_atom_list):
        for j in range(int(i)):
            ase_number.append(element_number[i_index])
    np.random.shuffle(ase_number)
    atoms.set_atomic_numbers(ase_number)

    # scale atoms
    if isinstance(v_per_atom, (float, int)):
        volume = atoms.cell.volume
        volume_per_atom = volume / len(atoms)
        volume_ratio = v_per_atom / volume_per_atom
        scale_factor = pow(volume_ratio, 1/3)
        atoms = scale_atoms(atoms, scale_factor)
    return atoms

def run_vasp(vasp_bash_path):
    """

    :raises ValueError: VASP can only run on a Linux platform


    .. warning:: Setup your own VAPS package and Intel libraries before using this function.

    """

    # os_type = platform.system()
    # if not os_type == 'Linux':
    #     raise ValueError(f'VASP can only be executed on Linux OS, instead of {os_type}.')
    # shell_script = '''#!/bin/bash
# . /home/jzhang/software/intel/oneapi/setvars.sh
# mpirun /home/jzhang/software/vasp/vasp_std
    # '''

    # with open('vasp_run.sh', 'w') as f:
    #     f.write(shell_script)
    # 
    # os.chmod('vasp_run.sh', stat.S_IRWXU)

    shutil.copyfile(vasp_bash_path, os.path.join('./vasp_run.sh'))
    os.chmod('vasp_run.sh', stat.S_IRWXU)
    os.system('./vasp_run.sh')

# remove imported objects
# del np, Formula, read, FaceCenteredCubic, covalent_radii, atomic_numbers, platform
