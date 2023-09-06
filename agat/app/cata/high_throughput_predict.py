# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:59:30 2021

@author: ZHANG Jun
"""

from ase.optimize import BFGS
from ase.io import read, write
from ase.lattice.cubic import FaceCenteredCubic
from ase.visualize import view
from ase.data import covalent_radii, atomic_numbers # index of `ase.data` starts from ONE.
# from pymatgen.core.ion import Ion
import numpy as np
import os
from ase.build import add_vacuum, sort
from ase.constraints import FixAtoms, FixBondLength
from ase.formula import Formula
import json
import sys

from ..GatApp import GatAseCalculator
from .generate_adsorption_sites import AddAtoms
from ...lib.AgatException import file_exit
from ...lib.file import generate_file_name
from ...lib.GatLib import config_parser
from ...default_parameters import default_hp_config
from ...lib.adsorbate_poscar import adsorbate_poscar

class HpAds(object):
    def __init__(self, **hp_config):
        self.hp_config = {**default_hp_config, **config_parser(hp_config)}

    def perturb_positions(self, atoms, amplitude=0.1):
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

    def scale_atoms(self, atoms, scale_factor=1.0):
        calculator = atoms.get_calculator()
        new_atoms  = atoms.copy()
        new_atoms.set_calculator(calculator)
        cell        = new_atoms.get_cell()
        frac_coords = new_atoms.get_scaled_positions()
        new_cell    = cell * scale_factor
        new_atoms.set_cell(new_cell)
        new_atoms.set_scaled_positions(frac_coords)
        return new_atoms

    def get_concentration_from_ase_formula(self, formula):
        f_dict = Formula(formula).count()
        tot = np.sum(list(f_dict.values()))
        c_dict = {k: v/tot for k, v in f_dict.items()}
        c_dict = {k:c_dict[k] for k in c_dict if c_dict[k] > 0}
        return c_dict

    def get_v_per_atom(self, chemical_formula):
        # for now, use this for NiCoFePdPt only.
        # v_per_atom = -282.7957531391954 * c_NiCoFe(sum) - 278.79605077419797 * C_Pd - -278.6228860885035 * C_Pt + 293.66128761358624
        # comp             = Ion.from_formula(chemical_formula)
        # elements         = [x.name for x in comp.elements]
        # atomic_fracions  = [comp.get_atomic_fraction(x) for x in elements]
        # frac_dict        = dict(zip(elements, atomic_fracions))
        # frac_dict        = {**frac_dict, **{k:0 for k in ['Ni', 'Co', 'Fe', 'Pd', 'Pt']\
        #                                     if not frac_dict.__contains__(k)}}
        frac_dict = self.get_concentration_from_ase_formula(chemical_formula)
        return -282.7957531391954 * (frac_dict['Ni'] + frac_dict['Co'] + frac_dict['Fe'])\
            - 278.79605077419797 * frac_dict['Pd'] - 278.6228860885035 * frac_dict['Pt']\
                + 293.66128761358624

    def get_ase_atom_from_formula(self, chemical_formula, v_per_atom=None):
        # interpret formula
        atomic_fracions = self.get_concentration_from_ase_formula(chemical_formula)
        elements = [x for x in atomic_fracions]
        element_number = [atomic_numbers[x] for x in elements]
        mean_radii       = np.sum([covalent_radii[n] * atomic_fracions[e] for n, e in zip(element_number, elements)]) #

        Pt_radii           = covalent_radii[78]
        latticeconstant    = mean_radii / Pt_radii * 3.92
        atoms              = FaceCenteredCubic('Pt', directions=[[1,-1,0], [1,1,-2], [1,1,1]], size=(4, 3, 2), latticeconstant=latticeconstant, pbc=True)
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
            atoms = self.scale_atoms(atoms, scale_factor)
        return atoms

    def get_ase_atom_from_formula_template(self, chemical_formula, v_per_atom=None,
                                           template_file='POSCAR_temp'):
        # interpret formula
        # the template file should be a bulk structure
        atomic_fracions    = self.get_concentration_from_ase_formula(chemical_formula)
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
            atoms = self.scale_atoms(atoms, scale_factor)
        return atoms

    def geo_opt(self, atoms_with_calculator, **kwargs):
        calculator = atoms_with_calculator.get_calculator()
        atoms  = atoms_with_calculator.copy()
        atoms.set_calculator(calculator)

        config = {**self.hp_config['opt_config'], **kwargs}

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
                atoms = self.perturb_positions(atoms, amplitude=config["perturb_amplitude"])
        argmin = np.argmin(energy_opt)
        energy, force, atoms = energy_opt[argmin], force_opt[argmin], atoms_list[argmin]
        force_max = np.linalg.norm(force, axis=1).max()
        return energy, force, atoms, force_max

    def ads_calc(self, formula, calculator, **kwargs):
        hp_config =  {**self.hp_config, **kwargs}

        if hp_config['save_trajectory']:
            out_dir = f'{hp_config["calculation_index"]}_th_calculation'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
        else:
            out_dir     = '.'
            outbasename = None

        # generate bulk structure
        chemical_formula = formula

        if self.hp_config['using_template_bulk_structure']:
            atoms = self.get_ase_atom_from_formula_template(chemical_formula,
                                                            self.get_v_per_atom(chemical_formula),
                                                            template_file='POSCAR_temp')
        else:
            atoms = self.get_ase_atom_from_formula(chemical_formula, v_per_atom=self.get_v_per_atom(chemical_formula))

        atoms.set_calculator(calculator)

        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'POSCAR_bulk.gat'), atoms, format='vasp')
            outbasename = os.path.join(out_dir, 'bulk_opt')

        hp_config['out'] = outbasename

        energy_bulk, force_bulk, atoms_bulk, force_max_bulk = self.geo_opt(atoms, **hp_config)
        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'CONTCAR_bulk.gat'), atoms_bulk)

        print('Bulk optimization done.')

        # add vacuum space and fix bottom atoms
        len_z = atoms_bulk.cell.array[2][2]
        c     = FixAtoms(indices=np.where(atoms_bulk.positions[:,2] < len_z / 2 - 1.0)[0])
        atoms_bulk.set_constraint(c)

        if hp_config['remove_bottom_atoms']:
            pop_list = np.where(atoms_bulk.positions[:,2] < 1.0)
            del atoms_bulk[pop_list]

        atoms_bulk.positions += 1.3 # avoid PBC error
        atoms_bulk.wrap()
        add_vacuum(atoms_bulk, 10.0)

        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'POSCAR_surface.gat'), atoms_bulk)
            outbasename = os.path.join(out_dir, 'surface_opt')

        # surface optimization
        atoms_bulk.set_calculator(calculator)
        hp_config['out'] = outbasename

        energy_surf, force_surf, atoms_surf, force_max_surf = self.geo_opt(atoms_bulk, **hp_config)
        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'CONTCAR_surface.gat'), atoms_surf)

        print('Surface optimization done.')

        if force_max_surf < hp_config['opt_config']['fmax']:
            if hp_config['fix_all_surface_atom']:
                c = FixAtoms(indices=[x for x in range(len(atoms_surf))])
                atoms_surf.set_constraint(c)
            write(f'POSCAR_surf_opt_{hp_config["calculation_index"]}.gat', sort(atoms_surf))

            # adsorbate_shift = {'bridge': 0.0, 'ontop': 0.35, 'hollow': -0.1}

            for ads in hp_config['adsorbates']:
                # generate adsorption configurations: OH adsorption
                adder     = AddAtoms(f'POSCAR_surf_opt_{hp_config["calculation_index"]}.gat',
                                     species=ads,
                                     sites=hp_config['sites'],
                                     dist_from_surf=hp_config['dist_from_surf'],
                                     num_atomic_layer_along_Z=6)
                all_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar,
                                                                   calculation_index=hp_config['calculation_index'])

                # adsorption optimization
                ase_atoms = [read(f'POSCAR_{hp_config["calculation_index"]}_{x}') for x in range(all_sites)]
                [os.remove(f'POSCAR_{hp_config["calculation_index"]}_{x}') for x in range(all_sites)]
                # ase_atoms = ase_atoms[0:3] # !!!

                energy_ads_list, converge_stat = [], []
                for i, ads_atoms in enumerate(ase_atoms):
                    file_exit()
                    if hp_config['save_trajectory']:
                        write(os.path.join(out_dir, f'POSCAR_{ads}_ads_{hp_config["calculation_index"]}_{i}.gat'),
                              ads_atoms)
                        outbasename = os.path.join(out_dir, f'adsorption_{ads}_opt_{i}')

                    hp_config['out'] = outbasename
                    ads_atoms.set_calculator(calculator)
                    energy_ads, force_ads, atoms_ads, force_max_ads = self.geo_opt(ads_atoms, **hp_config)

                    if hp_config['save_trajectory']:
                        write(os.path.join(out_dir, f'CONTCAR_{ads}_ads_{hp_config["calculation_index"]}_{i}.gat'),
                              atoms_ads)

                    energy_ads_list.append(energy_ads)

                    if force_max_ads > hp_config['opt_config']['fmax']:
                        converge_stat.append(0.0)
                    else:
                        converge_stat.append(1.0)

                energy_surf_list = np.array([energy_surf] * len(energy_ads_list))
                energy_ads_list = np.array(energy_ads_list)

                out = np.vstack([energy_ads_list, energy_surf_list, converge_stat]).T
                np.savetxt(f'ads_surf_energy_{ads}_{hp_config["calculation_index"]}.txt',
                           out, fmt='%f')

    def run(self, formula, **kwargs):
        # model save path
        energy_model_save_dir = self.hp_config['energy_model_save_dir']
        force_model_save_dir  = self.hp_config['force_model_save_dir']

        # instantiate a calculator
        calculator=GatAseCalculator(energy_model_save_dir, force_model_save_dir,
                                    self.hp_config['graph_build_scheme_dir'],
                                    gpu=self.hp_config['gpu'])

        with open('high_throughput_config.json', 'w') as f:
            json.dump(self.hp_config, f, indent=4)

        self.ads_calc(formula, calculator, **kwargs)

if __name__ == '__main__':
    HA = HpAds()
    HA.run('NiCoFePdPt', fmax=1.0, steps=2, gpu=0) # debug only
