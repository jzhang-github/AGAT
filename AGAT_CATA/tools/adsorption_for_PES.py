# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:59:30 2021

@author: ZHANG Jun
"""

from GatApp import GatAseCalculator
from ase.optimize import BFGS
from ase.io import read, write
from ase.lattice.cubic import FaceCenteredCubic
from ase.visualize import view
from ase.data import covalent_radii # index of `ase.data` starts from ONE.
from pymatgen.core.ion import Ion
from ase import Atom
from ase.constraints import FixAtoms
import numpy as np
import os
from ase.build import add_vacuum, sort
from ase.constraints import FixAtoms
import sys

def generate_file_name(fname):
    while os.path.exists(fname):
        fname = fname + '_new'
    return fname

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

if __name__ == '__main__':
    fname = 'CONTCAR_surf_NiCoFePdPt'
    grid_mesh = [15,15]
    
    # model save path
    energy_model_save_dir = os.path.join('..', 'RuRhPdIrPt_energy_ckpt')
    force_model_save_dir  = os.path.join('..', 'RuRhPdIrPt_force_ckpt')

    # instantiate a calculator
    calculator=GatAseCalculator(energy_model_save_dir, force_model_save_dir, gpu=-1)

    # most settings are effective for the adsorption relaxation, except 'fmax',
    # which applies to every relaxation process.
    opt_config            = {'fmax'             : 0.5,
                             'steps'            : 200,
                             'maxstep'          : 0.05,
                             'restart'          : None,
                             'restart_steps'    : 3,
                             'perturb_steps'    : 0,
                             'perturb_amplitude': 0.05,
                             'out'              : None}
    
    atoms_surf = read(fname)
    cell = atoms_surf.get_cell_lengths_and_angles()
    x_length = cell[0]
    y_length = cell[1]
    for x_scale in np.linspace(0, 1, grid_mesh[0], endpoint=False):
        for y_scale in np.linspace(0, 1, grid_mesh[1], endpoint=False):
            atoms_ads = atoms_surf.copy()
            atom_O = Atom('O', position=(x_scale * x_length, y_scale * y_length, 13.14))
            atom_H = Atom('H', position=(x_scale * x_length, y_scale * y_length, 14.14))
            atoms_ads.append(atom_O)
            atoms_ads.append(atom_H)
            write(os.path.join('poscar_for_PES_NiCoFePdPt', f'POSCAR_{x_scale}_{y_scale}'),  atoms_ads)

    # 先手动固定部分自由度
    


