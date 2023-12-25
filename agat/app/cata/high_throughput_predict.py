# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:59:30 2021

@author: ZHANG Jun
"""

import json
import sys
import os
import numpy as np

from ase.optimize import BFGS
from ase.io import read, write
from ase.build import add_vacuum, sort
from ase.constraints import FixAtoms

from ..app import AgatCalculator
from .generate_adsorption_sites import AddAtoms
from ...lib.file_lib import file_exit
from ...lib.model_lib import config_parser
from ...default_parameters import default_high_throughput_config
from ...lib.adsorbate_poscar import adsorbate_poscar
from ...lib.high_throughput_lib import get_v_per_atom, get_ase_atom_from_formula, get_ase_atom_from_formula_template

class HtAds(object):
    def __init__(self, **hp_config):
        self.hp_config = {**default_high_throughput_config, **config_parser(hp_config)}

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
            restart_step = 1
            while not return_code and restart_step < config["restart_steps"]:
                restart_step += 1
                maxstep_tmp = config["maxstep"]/2**restart_step
                dyn = BFGS(atoms,
                           logfile=logfile,
                           trajectory=trajectory,
                           restart=config["restart"],
                           maxstep=maxstep_tmp)
                return_code  = dyn.run(fmax=config["fmax"], steps=config["steps"])
            force_opt.append(atoms.get_forces())
            energy_opt.append(atoms.get_total_energy())
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
            atoms = get_ase_atom_from_formula_template(chemical_formula,
                                                       get_v_per_atom(chemical_formula),
                                                       template_file='POSCAR_temp')
        else:
            atoms = get_ase_atom_from_formula(chemical_formula,
                                              v_per_atom=get_v_per_atom(chemical_formula))

        atoms.set_calculator(calculator)

        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'POSCAR_bulk.gat'), atoms, format='vasp')
            outbasename = os.path.join(out_dir, 'bulk_opt')

        hp_config['out'] = outbasename

        energy_bulk, force_bulk, atoms_bulk, force_max_bulk = self.geo_opt(atoms,
                                                                           **hp_config['opt_config'])
        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'CONTCAR_bulk.gat'), atoms_bulk)

        print('Bulk optimization done.')

        # add vacuum space and fix bottom atoms
        len_z = atoms_bulk.cell.array[2][2]
        atoms_bulk.positions += 1.3 # avoid PBC error
        atoms_bulk.wrap()
        c     = FixAtoms(indices=np.where(atoms_bulk.positions[:,2] < len_z / 2)[0])
        atoms_bulk.set_constraint(c)

        if hp_config['remove_bottom_atoms']:
            pop_list = np.where(atoms_bulk.positions[:,2] < 1.0)
            del atoms_bulk[pop_list]

        add_vacuum(atoms_bulk, 10.0)

        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'POSCAR_surface.gat'), atoms_bulk)
            outbasename = os.path.join(out_dir, 'surface_opt')

        # surface optimization
        atoms_bulk.set_calculator(calculator)
        hp_config['out'] = outbasename

        energy_surf, force_surf, atoms_surf, force_max_surf = self.geo_opt(atoms_bulk,
                                                                           **hp_config['opt_config'])
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
                    energy_ads, force_ads, atoms_ads, force_max_ads = self.geo_opt(ads_atoms, **hp_config['opt_config'])

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
        """

        :param formula: Input chemical formula
        :type formula: str
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        # model save path
        model_save_dir = self.hp_config['model_save_dir']

        # instantiate a calculator
        calculator=AgatCalculator(model_save_dir,
                                 self.hp_config['graph_build_scheme_dir'],
                                 device=self.hp_config['device'])

        with open('high_throughput_config.json', 'w') as f:
            json.dump(self.hp_config, f, indent=4)

        self.ads_calc(formula, calculator, **kwargs)

if __name__ == '__main__':
    HA = HtAds()
    HA.run('NiCoFePdPt') # debug only
