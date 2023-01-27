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
import numpy as np
import os
from ase.build import add_vacuum, sort
from ase.constraints import FixAtoms
from generate_adsorption_sites import AddAtoms
import sys
from dgl.data.utils import load_graphs
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as mae
# import multiprocessing
# import pandas as pd

def generate_file_name(fname):
    while os.path.exists(fname):
        fname = fname + '_new'
    return fname

def perturb_positions(atoms, amplitude=0.1):
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
    cell        = atoms.get_cell()
    frac_coords = atoms.get_scaled_positions()
    new_cell    = cell * scale_factor
    atoms.set_cell(new_cell)
    atoms.set_scaled_positions(frac_coords)
    return atoms

def get_ase_atom_from_formula(chemical_formula, v_per_atom=None):
    # interpret formula
    comp             = Ion.from_formula(chemical_formula)
    elements         = comp.elements
    element_number   = [x.number for x in elements]
    atomic_fracions  = [comp.get_atomic_fraction(x) for x in elements]
    mean_radii       = np.sum([covalent_radii[x] * atomic_fracions[i] for i, x in enumerate(element_number)]) #

    # create and scale a (111) terminated cell
    Pt_radii           = covalent_radii[78]
    latticeconstant    = mean_radii / Pt_radii * 3.92
    atoms              = FaceCenteredCubic('Pt', directions=[[1,-1,0], [1,1,-2], [1,1,1]], size=(4, 3, 2), latticeconstant=latticeconstant, pbc=True)
    total_atom         = len(atoms)
    num_atom_list      = np.array(atomic_fracions) * total_atom
    num_atom_list      = np.around(num_atom_list, decimals=0)
    total_tmp          = np.sum(num_atom_list)
    deviation          = total_atom - total_tmp
    num_atom_list[-1] += deviation

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

def geo_opt(atoms_with_calculator,
            fmax=0.05, steps=200, maxstep=0.2,
            restart=None,
            restart_steps=5,
            perturb_steps=0,
            perturb_amplitude=0.05):

    force_opt, energy_opt, atoms_list = [], [], []
    for i in range(perturb_steps+1):
        dyn = BFGS(atoms_with_calculator,
                   logfile='-',
                   trajectory=None,
                   restart=restart,
                   maxstep=maxstep)

        return_code  = dyn.run(fmax=fmax, steps=steps)
        restart_step = 0
        while not return_code and restart_step < restart_steps:
            restart_step += 1
            maxstep      /= 2.0
            dyn = BFGS(atoms_with_calculator,
                       logfile='-',
                       trajectory=None,
                       restart=restart,
                       maxstep=maxstep)
            return_code  = dyn.run(fmax=fmax, steps=steps)
        force_opt.append(atoms_with_calculator.get_forces())
        energy_opt.append(atoms_with_calculator.get_potential_energy(apply_constraint=False))
        atoms_list.append(atoms_with_calculator.copy())
        if perturb_steps > 0:
            atoms_with_calculator = perturb_positions(atoms_with_calculator, amplitude=perturb_amplitude)
    argmin = np.argmin(energy_opt)
    energy, force, atoms = energy_opt[argmin], force_opt[argmin], atoms_list[argmin]
    return energy, force, atoms

def read_output(dir_path):
    CURDIR = os.getcwd()
    os.chdir(dir_path)
    fnames = os.listdir('.')
    energies, forces = [], []
    for fname in fnames:
        short_name, suffix = os.path.splitext(fname)
        if suffix == '.log':
            with open(fname, 'r') as f:
                lines = f.readlines()
            try:
                energy, force = lines[-1].split()[3:5]
                energy, force = float(energy), float(force)
            except:
                energy, force = None, None
            energies.append(energy)
            forces.append(force)
    os.chdir(CURDIR)
    return energies, forces # a list of energies

def ads_calc(formula, calculator, v_per_atom=14.045510416666668,
             calculation_index=None):
    # generate bulk structure
    chemical_formula = formula
    atoms            = get_ase_atom_from_formula(chemical_formula, v_per_atom)
    atoms.set_calculator(calculator)

    energy_bulk, force_bulk, atoms_bulk = geo_opt(atoms, fmax=0.05, steps=200, maxstep=0.05,
                                                  restart=None, restart_steps=5,
                                                  perturb_steps=5, perturb_amplitude=0.05)

    # energy = []
    # for i in np.linspace(0.80, 1.05, 21):
    #     print(f'Optimizing the bulk structure with scaled factor: {i}')
    #     scaled_atoms = scale_atoms(atoms_bulk.copy(), i)
    #     scaled_atoms.set_calculator(calculator)
    #     energy_opt, force_opt = geo_opt(scaled_atoms, fmax=0.05, steps=200, maxstep=0.1,
    #                             restart=None, restart_steps=5,
    #                             perturb_steps=0, perturb_amplitude=0.05)
    #     energy.append(scaled_atoms.get_potential_energy(apply_constraint=False))

    # add vacuum space and fix bottom atoms
    len_z = atoms_bulk.cell.array[2][2]
    c     = FixAtoms(indices=np.where(atoms_bulk.positions[:,2] < len_z / 2 - 1.0)[0])
    atoms_bulk.set_constraint(c)
    add_vacuum(atoms_bulk, 10.0)
    atoms_bulk.positions += 1.0 # avoid PBC error

    # surface optimization
    atoms_bulk.set_calculator(calculator)
    energy_surf, force_surf, atoms_surf = geo_opt(atoms_bulk, fmax=0.05, steps=200, maxstep=0.05,
                                                  restart=None, restart_steps=5,
                                                  perturb_steps=5, perturb_amplitude=0.05)
    write(f'POSCAR_surf_opt_{calculation_index}.gat', sort(atoms_surf))

    # generate adsorption configurations
    adder     = AddAtoms(f'POSCAR_surf_opt_{calculation_index}.gat', species='OH', sites='bridge', dist_from_surf=2.0, num_atomic_layer_along_Z=6)
    all_sites = adder.write_file_with_adsorption_sites(file_format='Cartesian', calculation_index=calculation_index)

    # adsorption optimization
    ase_atoms = [read(f'POSCAR_{calculation_index}_{x}') for x in range(all_sites)]
    [os.remove(f'POSCAR_{calculation_index}_{x}') for x in range(all_sites)]

    energy_ads_list = []
    for ads_atoms in ase_atoms:
        ads_atoms.set_calculator(calculator)
        energy_ads, force_ads, atoms_ads = geo_opt(ads_atoms, fmax=0.05, steps=200, maxstep=0.05,
                                                    restart=None, restart_steps=5,
                                                    perturb_steps=0, perturb_amplitude=0.05)
        energy_ads_list.append(energy_ads)

    energy_surf = np.array([energy_surf] * len(energy_ads_list))
    energy_ads_list = np.array(energy_ads_list)

    out = np.vstack([energy_ads_list, energy_surf]).T
    np.savetxt(f'ads_surf_energy_{calculation_index}.txt', out, fmt='%f')
    return energy_surf, energy_ads_list

if __name__ == '__main__':
    # v_per_atom_RuRhPdIrPt = 14.045510416666668

    # model save path
    energy_model_save_dir = os.path.join('..', 'energy_ckpt')
    force_model_save_dir  = os.path.join('..', 'force_ckpt')

    # instantiate a calculator
    calculator=GatAseCalculator(energy_model_save_dir, force_model_save_dir, gpu=-1)
    force_model = calculator.app.force_model

    graph_path = os.path.join('all_graphs_adsorption.bin')
    graph_list, graph_labels = load_graphs(graph_path)
    num_graphs = len(graph_list)
    selected_index = np.random.choice(list(range(num_graphs)), 1000, replace=False)

    true_force_surface, true_force_adsorbate, pred_force_surface, pred_force_adsorbate= [], [], [], []
    for i in selected_index:
        true_force = graph_list[i].ndata['forces_true']
        pred_force = force_model(graph_list[i])
        adsorbate_i = tf.cast(graph_list[i].ndata['adsorbate'], dtype='bool')
        clean_surface_i = tf.logical_not(adsorbate_i)
        true_force_surface.append(true_force[clean_surface_i])
        true_force_adsorbate.append(true_force[adsorbate_i])
        pred_force_surface.append(pred_force[clean_surface_i])
        pred_force_adsorbate.append(pred_force[adsorbate_i])

    true_force_surface = tf.concat(true_force_surface, 0)
    true_force_adsorbate = tf.concat(true_force_adsorbate, 0)
    pred_force_surface = tf.concat(pred_force_surface, 0)
    pred_force_adsorbate = tf.concat(pred_force_adsorbate, 0)

    true_force_surface = tf.reshape(true_force_surface, (-1)).numpy()
    true_force_adsorbate = tf.reshape(true_force_adsorbate, (-1)).numpy()
    pred_force_surface = tf.reshape(pred_force_surface, (-1)).numpy()
    pred_force_adsorbate = tf.reshape(pred_force_adsorbate, (-1)).numpy()

    np.savetxt('true_force_surface.txt', true_force_surface, fmt='%f')
    np.savetxt('true_force_adsorbate.txt', true_force_adsorbate, fmt='%f')
    np.savetxt('pred_force_surface.txt', pred_force_surface, fmt='%f')
    np.savetxt('pred_force_adsorbate.txt', pred_force_adsorbate, fmt='%f')

    mae_surf = mae(true_force_surface, pred_force_surface)
    mae_ads  = mae(true_force_adsorbate, pred_force_adsorbate)

    with open('mae.log', 'w') as f:
        print(f'Surface MAE: {mae_surf} eV/Å', file=f)
        print(f'Adsorbate MAE: {mae_ads} eV/Å', file=f)

