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
# from generate_adsorption_sites import AddAtoms
import sys
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
            perturb_amplitude=0.05,
            out=None):

    if isinstance(out, type(None)):
        logfile    = '-'
        trajectory = None
    else:
        logfile    = f'{out}.log'
        trajectory = f'{out}.traj'
    force_opt, energy_opt, atoms_list = [], [], []
    for i in range(perturb_steps+1):
        dyn = BFGS(atoms_with_calculator,
                   logfile=logfile,
                   trajectory=trajectory,
                   restart=restart,
                   maxstep=maxstep)

        return_code  = dyn.run(fmax=fmax, steps=steps)
        restart_step = 0
        while not return_code and restart_step < restart_steps:
            restart_step += 1
            maxstep      /= 2.0
            dyn = BFGS(atoms_with_calculator,
                       logfile=logfile,
                       trajectory=trajectory,
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

def ads_calc(formula, calculator, v_per_atom=14.045510416666668, fmax=0.05,
             calculation_index=None, fix_surface_atom=False, remove_bottom_atoms=True,
             save_trajectory=False, partial_fix_adsorbate=False):

    if save_trajectory:
        out_dir = f'{calculation_index}_th_calculation'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    else:
        out_dir     = '.'
        outbasename = None

    # generate bulk structure
    chemical_formula = formula
    atoms            = get_ase_atom_from_formula(chemical_formula, v_per_atom)
    atoms.set_calculator(calculator)

    if save_trajectory:
        write(os.path.join(out_dir, 'POSCAR_bulk.gat'), atoms, format='vasp')
        outbasename = os.path.join(out_dir, 'bulk_opt')

    energy_bulk, force_bulk, atoms_bulk = geo_opt(atoms, fmax=fmax, steps=200, maxstep=0.05,
                                                  restart=None, restart_steps=5,
                                                  perturb_steps=5, perturb_amplitude=0.05, out=outbasename)
    if save_trajectory:
        write(os.path.join(out_dir, 'CONTCAR_bulk.gat'), atoms_bulk)

    print('Bulk optimization done.')

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

    if remove_bottom_atoms:
        pop_list = np.where(atoms_bulk.positions[:,2] < 1.0)
        del atoms_bulk[pop_list]

    add_vacuum(atoms_bulk, 10.0)
    atoms_bulk.positions += 1.0 # avoid PBC error

    if save_trajectory:
        write(os.path.join(out_dir, 'POSCAR_surface.gat'), atoms_bulk)
        outbasename = os.path.join(out_dir, 'surface_opt')

    # surface optimization
    atoms_bulk.set_calculator(calculator)
    energy_surf, force_surf, atoms_surf = geo_opt(atoms_bulk, fmax=fmax, steps=200, maxstep=0.05,
                                                  restart=None, restart_steps=5,
                                                  perturb_steps=5, perturb_amplitude=0.05,out=outbasename)
    if save_trajectory:
        write(os.path.join(out_dir, 'CONTCAR_surface.gat'), atoms_surf)

    print('Surface optimization done.')

    force_max = np.linalg.norm(force_surf, axis=1).max()
    if force_max < fmax:
        if fix_surface_atom:
            c = FixAtoms(indices=[x for x in range(len(atoms_surf))])
            atoms_surf.set_constraint(c)
        write(f'POSCAR_surf_opt_{calculation_index}.gat', sort(atoms_surf))

        # generate adsorption configurations
        adder     = AddAtoms(f'POSCAR_surf_opt_{calculation_index}.gat', species='OH', sites='bridge', dist_from_surf=2.0, num_atomic_layer_along_Z=6)
        all_sites = adder.write_file_with_adsorption_sites(file_format='Cartesian', calculation_index=calculation_index, partial_fix=partial_fix_adsorbate)

        # adsorption optimization
        ase_atoms = [read(f'POSCAR_{calculation_index}_{x}') for x in range(all_sites)]
        [os.remove(f'POSCAR_{calculation_index}_{x}') for x in range(all_sites)]

        energy_ads_list, converge_stat = [], []
        for i, ads_atoms in enumerate(ase_atoms):
            if save_trajectory:
                write(os.path.join(out_dir, f'POSCAR_ads_{calculation_index}_{i}.gat'), ads_atoms)
                outbasename = os.path.join(out_dir, f'adsorption_opt_{i}')

            ads_atoms.set_calculator(calculator)
            energy_ads, force_ads, atoms_ads = geo_opt(ads_atoms, fmax=fmax, steps=200, maxstep=0.05,
                                                        restart=None, restart_steps=5,
                                                        perturb_steps=0, perturb_amplitude=0.05, out=outbasename)

            if save_trajectory:
                write(os.path.join(out_dir, f'CONTCAR_ads_{calculation_index}_{i}.gat'), atoms_ads)

            energy_ads_list.append(energy_ads)

            force_max = np.linalg.norm(force_ads, axis=1).max()
            if force_max > fmax:
                converge_stat.append(0.0)
            else:
                converge_stat.append(1.0)

        energy_surf = np.array([energy_surf] * len(energy_ads_list))
        energy_ads_list = np.array(energy_ads_list)

        out = np.vstack([energy_ads_list, energy_surf, converge_stat]).T
        np.savetxt(f'ads_surf_energy_{calculation_index}.txt', out, fmt='%f')
    else:
        energy_surf, energy_ads_list = None, None
    return energy_surf, energy_ads_list

def get_potential(ads_energy, surf_energy, length_of_atoms):
    adsorbate_dict = {81: 'O',
                      82: 'OH',
                      83: 'OOH'}
    adsorbate = adsorbate_dict[length_of_atoms]

    if adsorbate == 'O':
        e = ads_energy + 0.0488142 - surf_energy + -6.80545924 - -14.25416854
    elif adsorbate == 'OH':
        e = ads_energy + 0.3072701 - surf_energy + 0.5 * -6.80545924 - -14.25416854
    elif adsorbate == 'OOH':
        e = ads_energy + 0.341310571428571 - surf_energy + 1.5 * -6.80545924 - 2 * -14.25416854
    return e

if __name__ == '__main__':
    fmax = 0.1
    
    # model save path
    energy_model_save_dir = os.path.join('energy_ckpt')
    force_model_save_dir  = os.path.join('force_ckpt')

    # instantiate a calculator
    calculator=GatAseCalculator(energy_model_save_dir, force_model_save_dir, gpu=-1)
    
    surf_vs_energy   = np.loadtxt('surf_vs_energy.txt', dtype=str)
    surf_energy_dict = dict(zip(surf_vs_energy[:,0], [float(x) for x in surf_vs_energy[:,1]]))

    dirnames = os.listdir(os.path.join('..', 'DFT_opt'))
    # i, dft_potential, agat_potential = [], [], []

    out = open('result_dir_dft_agat.txt', 'w', buffering=1)
    for _dir in dirnames:
        calc_code = True
        try:
            atoms = read(os.path.join('..', 'DFT_opt', _dir, 'CONTCAR_dft'))

            # predict
            atoms.set_calculator(calculator)
            agat_energy, force, atoms = geo_opt(atoms,
                        fmax=fmax, steps=200, maxstep=0.05, 
                        restart=None,
                        restart_steps=5,
                        perturb_steps=0,
                        perturb_amplitude=0.05,
                        out=None)
        
            max_force = max(np.linalg.norm(force, axis=1))
        except:
            print(f'Exception raised in {os.path.join("..", "DFT_opt", _dir)}')
            calc_code = False

        if max_force < fmax and calc_code:
            lines = np.loadtxt(os.path.join('..', 'DFT_opt', _dir, 'dft_result.log'), dtype=str) # .split(',')
            surf_energy, dft_energy = surf_energy_dict[lines[1]], float(lines[8].strip(','))
            # dft_potential.append(get_potential(dft_energy, surf_energy, len(atoms)))
            # agat_potential.append(agat_energy)
            # i.append(int(_dir))
            dft_potential = get_potential(dft_energy, surf_energy, len(atoms))
            gat_potential = get_potential(agat_energy, surf_energy, len(atoms))
            print(_dir, dft_potential, gat_potential, file=out)
            
    # result = np.hstack([i, dft_potential, agat_potential])
    # np.savetxt('result_dir_dft_agat.txt', result, fmt='%f')
    
    out.close()
    
