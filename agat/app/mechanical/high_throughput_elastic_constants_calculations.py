# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:35:53 2022

@author: ZHANG Jun
"""

import numpy as np
import sys
import os
from tools.GatApp import GatApp, GatAseCalculator
from ase.optimize import BFGS
from ase.io import read, write
from ase.lattice.cubic import FaceCenteredCubic
from ase.visualize import view
from ase.data import covalent_radii, atomic_numbers # index of `ase.data` starts from ONE.
from ase import Atoms
from ase.formula import Formula
from ase.build import sort
from scipy.optimize import curve_fit

def geo_opt(atoms_with_calculator, **kwargs):
    calculator = atoms_with_calculator.get_calculator()
    atoms  = atoms_with_calculator.copy()
    atoms.set_calculator(calculator)

    config = {'fmax'             : kwargs['fmax']              if kwargs.__contains__('fmax')              else 0.05,
              'steps'            : kwargs['steps']             if kwargs.__contains__('steps')             else 200,
              'maxstep'          : kwargs['maxstep']           if kwargs.__contains__('maxstep')           else 0.1,
              'restart'          : kwargs['restart']           if kwargs.__contains__('restart')           else None,
              'restart_steps'    : kwargs['restart_steps']     if kwargs.__contains__('restart_steps')     else 5,
              'perturb_steps'    : kwargs['perturb_steps']     if kwargs.__contains__('perturb_steps')     else 0,
              'perturb_amplitude': kwargs['perturb_amplitude'] if kwargs.__contains__('perturb_amplitude') else 0.05,
              'out'              : kwargs['out']               if kwargs.__contains__('out')               else None}

    if isinstance(config["out"], type(None)):
        logfile    = '-'
        trajectory = None
    else:
        logfile    = f'{config["out"]}.log'
        trajectory = f'{config["out"]}.traj'
    force_opt, energy_opt, atoms_list = [], [], []
    for i in range(config["perturb_steps"]+1):
        dyn = BFGS(atoms,
                   logfile=logfile,
                   trajectory=trajectory,
                   restart=config["restart"],
                   maxstep=config["maxstep"])

        return_code  = dyn.run(fmax=config["fmax"], steps=config["steps"])
        restart_step = 0
        while not return_code and restart_step < config["restart_steps"]:
            restart_step += 1
            config["maxstep"]      /= 2.0
            dyn = BFGS(atoms,
                       logfile=logfile,
                       trajectory=trajectory,
                       restart=config["restart"],
                       maxstep=config["maxstep"])
            return_code  = dyn.run(fmax=config["fmax"], steps=config["steps"])
        force_opt.append(atoms.get_forces())
        energy_opt.append(atoms.get_potential_energy(apply_constraint=False))
        atoms_list.append(atoms.copy())
        if config["perturb_steps"] > 0:
            atoms = perturb_positions(atoms, amplitude=config["perturb_amplitude"])
    argmin = np.argmin(energy_opt)
    energy, force, atoms = energy_opt[argmin], force_opt[argmin], atoms_list[argmin]
    force_max = np.linalg.norm(force, axis=1).max()
    return energy, force, atoms, force_max

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

def deform_atoms(atoms, new_cell_matrix):
    calculator = atoms.get_calculator()
    new_atoms  = atoms.copy()
    new_atoms.set_calculator(calculator)
    frac_coords = new_atoms.get_scaled_positions()
    new_atoms.set_cell(new_cell_matrix)
    new_atoms.set_scaled_positions(frac_coords)
    return new_atoms

def get_bulk_structure(chemical_formula, size=[2,2,2]):
    metallic_elements     = list(Formula(chemical_formula).count().keys())
    element_number        = [atomic_numbers[x] for x in metallic_elements]
    reference_radii       = 1.615
    num_cations           = 4 * size[0] * size[1]* size[2]
    metal_index     = np.array_split(np.arange(num_cations), len(metallic_elements))
    element_numbers = list(np.concatenate([[element_number[x]    for y in _] for x, _ in enumerate(metal_index)]).flat)
    element_symbols = list(np.concatenate([[metallic_elements[x] for y in _] for x, _ in enumerate(metal_index)]).flat)
    mean_radii      = np.mean(covalent_radii[element_numbers])
    radii_ratio     = mean_radii / reference_radii
    # generate structure (ase `atoms` object)
    a     = 4.43 * radii_ratio
    atoms = Atoms('Ti4C4',
                  positions=[[0.0,0.0,0.0],
                             [a/2,a/2,0.0],
                             [0.0,a/2,a/2],
                             [a/2,0.0,a/2],
                             [0.0,a/2,0.0],
                             [a/2,0.0,0.0],
                             [0.0,0.0,a/2],
                             [a/2,a/2,a/2]],
                  cell=[[a,0.0,0.0],[0.0,a,0.0],[0.0,0.0,a]],
                  pbc=(True,True,True))
    atoms = atoms.repeat(size)
    symbols   = atoms.get_chemical_symbols()
    np.random.shuffle(element_symbols)
    cations_i = np.where(np.array(symbols) == 'Ti')[0]
    for i, x in enumerate(cations_i):
        symbols[x] = element_symbols[i]
    atoms.set_chemical_symbols(symbols)
    return atoms

def bulk_opt(ase_atoms, calculator,
             deform_list=np.linspace(0.80, 1.20, num=41, endpoint=True),
             **kwargs):
    # ase calculator should be set for ase_atoms
    energy_bulk, force_bulk, atoms_bulk = geo_opt(ase_atoms, **kwargs)
                # fmax=0.05, steps=200, maxstep=0.05,
                # restart=None,
                # restart_steps=3,
                # perturb_steps=2,
                # perturb_amplitude=0.1
    energy_bulk_old = energy_bulk

    scale_num   = np.linspace(0.80, 1.20, num=41, endpoint=True)
    energy_list = [None for x in range(41)]
    atoms_list  = [None for x in range(41)]
    direction   = 1
    init_index  = 20
    energy_list[init_index] = energy_bulk
    atoms_list[init_index]  = atoms_bulk
    step        = 1

    index_new = init_index + 1
    scale_factor = scale_num[index_new]
    while step < 10:
        if isinstance(energy_list[index_new], float):
            energy_bulk_new = energy_list[index_new]
        else:
            atoms_bulk_new = scale_atoms(energy_bulk, scale_factor)
            energy_bulk_new, force_bulk_new, atoms_bulk_new = geo_opt(atoms_bulk_new,
                                                                  fmax=0.05, steps=200, maxstep=0.05,
                                                                  restart=None,
                                                                  restart_steps=3,
                                                                  perturb_steps=0,
                                                                  perturb_amplitude=0)
            energy_list[index_new] = energy_bulk_new
            atoms_list[index_new]  = atoms_bulk_new
        if energy_bulk_new < energy_bulk_old:
            direction *= 1
        else:
            direction *= -1
        energy_bulk_old = energy_bulk_new

        index_new += direction
        scale_factor = scale_num[index_new]

        step += 1
    return # atoms_opt

def quadrafunc(x, a, b, c):
    y = a*x**2 + b*x + c
    return y

if __name__ == '__main__':
    fname                 = os.path.join('CONTCAR')
    energy_model_save_dir = os.path.join('HECC_potential_natural_cutoffs', 'energy_ckpt')
    force_model_save_dir  = os.path.join('HECC_potential_natural_cutoffs', 'force_ckpt')
    calculator            = GatAseCalculator(energy_model_save_dir, force_model_save_dir)
    opt_config            = {'fmax'             : 0.1,
                             'steps'            : 200,
                             'maxstep'          : 0.05,
                             'restart'          : None,
                             'restart_steps'    : 0,
                             'perturb_steps'    : 0,
                             'perturb_amplitude': 0.05,
                             'out'              : None}

    atoms_bulk = read(fname)

    # Generate structure for elastic constants calculations
    set1_atoms, set2_atoms, set3_atoms = [], [], []
    deform_ratios = np.linspace(-0.015, 0.015, num=7, endpoint=True)

    for ratio in deform_ratios:
        set1_deform_mat = np.diag([1.0+ratio, 1.0+ratio, 1.0])
        set2_deform_mat = np.diag([1.0+ratio, 1.0+ratio, 1.0+ratio])
        set3_deform_mat = np.array([[1.0, 0.5 * ratio, 0.5 * ratio],
                                    [0.5 * ratio, 1.0, 0.5 * ratio],
                                    [0.5 * ratio, 0.5 * ratio, 1]])

        atoms_tmp = deform_atoms(atoms_bulk, np.matmul(atoms_bulk.cell.array, set1_deform_mat))
        atoms_tmp.set_calculator(calculator)
        set1_atoms.append(atoms_tmp)
        atoms_tmp = deform_atoms(atoms_bulk, np.matmul(atoms_bulk.cell.array, set2_deform_mat))
        atoms_tmp.set_calculator(calculator)
        set2_atoms.append(atoms_tmp)
        atoms_tmp = deform_atoms(atoms_bulk, np.matmul(atoms_bulk.cell.array, set3_deform_mat))
        atoms_tmp.set_calculator(calculator)
        set3_atoms.append(atoms_tmp)

    # geometrical optimization
    set1_energies, set2_energies, set3_energies = [], [], []
    set1_fmax, set2_fmax, set3_fmax = [], [], []
    for i in range(7):
        out_tmp = geo_opt(set1_atoms[i], **opt_config)
        set1_energies.append(out_tmp[0])
        set1_fmax.append(out_tmp[3])

        out_tmp = geo_opt(set2_atoms[i], **opt_config)
        set2_energies.append(out_tmp[0])
        set2_fmax.append(out_tmp[3])

        out_tmp = geo_opt(set3_atoms[i], **opt_config)
        set3_energies.append(out_tmp[0])
        set3_fmax.append(out_tmp[3])

    # fitting # ev/A^3=1.6*1e11 Pa=1.6*1e2 GPa # 1 eV/Angstrom3 = 160.21766208 GPa
    set_energies = [set1_energies, set2_energies, set3_energies]
    set_fmaxs    = [set1_fmax, set2_fmax, set3_fmax]
    set_results  = np.vstack([set_energies, set_fmaxs]).T
    np.savetxt(f'{fname}_set_results.txt', set_results, fmt='%f')
    E0s          = [x[3] for x in set_energies]
    V0s          = [x[3].get_volume() for x in [set1_atoms, set2_atoms, set3_atoms]]
    quadr_coeffs = []
    for i in range(3):
        E0, V0 = E0s[i], V0s[i]
        E      = (set_energies[i] - E0) / V0 * 160.21766208

        [a, b, c], covariance = curve_fit(quadrafunc, deform_ratios, E)
        std = np.sqrt(np.diag(covariance))
        quadr_coeffs.append(a)

    # compute results
    C11 = 2*quadr_coeffs[0]     - 2*quadr_coeffs[1]/3.0
    C12 = 2*quadr_coeffs[1]/3.0 -   quadr_coeffs[0]
    C44 = 2*quadr_coeffs[2]/3.0
    B   = 1/3 * (C11 + 2*C12) # BR=BV=B
    Gv  = 1/5 * (3*C44 + C11 - C12)
    GR  = 5 * (C11 - C12) * C44 / (4*C44 + 3*(C11 - C12))
    G   = (Gv + GR) / 2
    E   = 9*G*B/(3*B + G)
    v   = (3*B - 2*G)/(6*B + 2*G)
    Hv  = 0.92*(G/B)**1.137*G**0.708

    # print(f'C11: {C11}')
    # print(f'C12: {C12}')
    # print(f'C44: {C44}')

    # print(f'{chemical_formula}: C11: {C11}  C12: {C12}  C44: {C44}')
    print(f'{fname}: C11: {C11}  C12: {C12}  C44: {C44}')
