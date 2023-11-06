# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:12:52 2023

@author: 18326
"""

import numpy as np
import os
import shutil
import sys

from ase.build import add_vacuum, sort
from ase.constraints import FixAtoms
from ase.io import read

from .generate_adsorption_sites import AddAtoms
from ...lib.model_lib import config_parser
from ...lib.adsorbate_poscar import adsorbate_poscar
from ...default_parameters import default_hp_dft_config
from ...lib.file_lib import modify_INCAR
from ...lib.high_throughput_lib import get_ase_atom_from_formula, get_v_per_atom, run_vasp

class HtDftAds(object):
    """High-throughput DFT calculations for adsorption. See details in: https://jzhang-github.github.io/AGAT/Default%20parameters.html#default_hp_dft_config

    .. py:property:: root_dir

       The root working directory of this object.

    :param **hp_config: Configurations to control the process.
    :type **hp_config: dict

    Example::

        HA = HpDftAds(calculation_index=0)
        HA.run('NiCoFePdPt')
    """
    def __init__(self, **hp_config):
        self.hp_config = {**default_hp_dft_config, **config_parser(hp_config)}
        self.root_dir = os.getcwd()

    def bulk_opt(self, formula):
        """Structural optimization of the bulk structure with VASP.

        :param formula: Chemical formula
        :type formula: str
        """

        # generate bulk structure
        chemical_formula = formula
        v_per_atom = get_v_per_atom(formula)
        atoms = get_ase_atom_from_formula(chemical_formula, v_per_atom)
        atoms = sort(atoms)
        atoms.write('POSCAR')

        # get POTCAR
        os.system('getpotential_auto_zj.sh 6')

        # get INCAR
        with open('INCAR', 'w') as f:
            f.write(self.hp_config['INCAR_static'])
        os.system('modify_MAGMOM_auto.sh POSCAR')
        modify_INCAR('ISIF', '3')

        # get KPOINTS
        with open('KPOINTS', 'w') as f:
            f.write(self.hp_config['KPOINTS'])

        # run vasp
        run_vasp(vasp_bash_path=self.hp_config['vasp_bash_path'])
        print('Bulk static optimization done.')

    def surf_opt(self, bulk_structural_file='CONTCAR_bulk_opt'):
        """Structural optimization of the surface slab with VASP.

        :param bulk_structural_file: optimized bulk structure, defaults to 'CONTCAR_bulk_opt'
        :type bulk_structural_file: str, optional
        """

        atoms_bulk = read(bulk_structural_file)
        atoms_bulk.positions += 1.3 # avoid PBC error
        atoms_bulk.wrap() # avoid PBC error

        # add vacuum space and fix bottom atoms
        len_z = atoms_bulk.cell.array[2][2]
        c     = FixAtoms(indices=np.where(atoms_bulk.positions[:,2] < len_z / 2)[0])
        atoms_bulk.set_constraint(c)
        add_vacuum(atoms_bulk, 10.0)
        atoms_bulk.write('POSCAR') # Write POSCAR of the surface model.

        # get POTCAR
        os.system('getpotential_auto_zj.sh 6')

        # get INCAR
        with open('INCAR', 'w') as f:
            f.write(self.hp_config['INCAR_static'])
        os.system('modify_MAGMOM_auto.sh POSCAR')
        modify_INCAR('ISIF', '2')

        # get KPOINTS
        with open('KPOINTS', 'w') as f:
            f.write(self.hp_config['KPOINTS'])

        # run vasp
        run_vasp(vasp_bash_path=self.hp_config['vasp_bash_path'])
        print('Surface static optimization done.')

    def ads_opt(self, structural_file='CONTCAR_surf_opt', random_samples=5):
        """Structural optimization of the adsorption with VASP.

        :param structural_file: Structural file name of optimized clean surface, defaults to 'CONTCAR_surf_opt'
        :type structural_file: str, optional
        :param random_samples: On one surface, many surface sites can be detected, this number controls how many individual calculations will be performed on this surface, defaults to 5
        :type random_samples: int, optional

        .. Note::
            ``random_samples`` cannot be larger than the number of detected surface sites.
        """

        # generate structures
        for ads in self.hp_config['adsorbates']:
            # generate adsorption configurations
            adder     = AddAtoms(structural_file,
                                 species=ads,
                                 sites=self.hp_config['sites'],
                                 dist_from_surf=self.hp_config['dist_from_surf'],
                                 num_atomic_layer_along_Z=6)
            all_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar,
                                                               calculation_index=self.hp_config['calculation_index'])

            if not os.path.exists('POSCARs'):
                os.mkdir('POSCARs')
            for i in range(all_sites):
                os.rename(f'POSCAR_{self.hp_config["calculation_index"]}_{i}',
                          os.path.join('POSCARs', f'POSCAR_{self.hp_config["calculation_index"]}_{i}'))

            for i in np.random.choice(range(all_sites), random_samples, replace=False):
                if not os.path.exists(str(i)):
                    os.mkdir(str(i))
                shutil.copyfile(os.path.join('POSCARs', f'POSCAR_{self.hp_config["calculation_index"]}_{i}'),
                               os.path.join(str(i), 'POSCAR'))

                parent_dir = os.getcwd()
                os.chdir(str(i))
                # get POTCAR
                os.system('getpotential_auto_zj.sh 6')

                # get INCAR
                with open('INCAR', 'w') as f:
                    f.write(self.hp_config['INCAR_static'])
                os.system('modify_MAGMOM_auto.sh POSCAR')
                modify_INCAR('ISIF', '2')

                # get KPOINTS
                with open('KPOINTS', 'w') as f:
                    f.write(self.hp_config['KPOINTS'])

                # run vasp
                run_vasp(vasp_bash_path=self.hp_config['vasp_bash_path'])
                print(f'Adsorption static calculation for {ads}_{i} done.')
                os.chdir(parent_dir)

    def bulk_aimd(self, formula):
        """AIMD simulation for a bulk structure of given chemical formula.

        :param formula: The given chemical formula.
        :type formula: str
        """

        # generate bulk structure
        chemical_formula = formula
        v_per_atom = get_v_per_atom(formula)
        atoms = get_ase_atom_from_formula(chemical_formula, v_per_atom)
        atoms = sort(atoms)
        atoms.write('POSCAR')

        # get POTCAR
        os.system('getpotential_auto_zj.sh 6')

        # get INCAR
        with open('INCAR', 'w') as f:
            f.write(self.hp_config['INCAR_aimd'])
        os.system('modify_MAGMOM_auto.sh POSCAR')
        modify_INCAR('ISIF', '3')
        modify_INCAR('NSW', '100')

        # get KPOINTS
        with open('KPOINTS', 'w') as f:
            f.write(self.hp_config['KPOINTS'])

        # run vasp
        run_vasp(vasp_bash_path=self.hp_config['vasp_bash_path'])
        print('Bulk AIMD simulation done.')

    def surface_aimd(self, bulk_structural_file='CONTCAR_bulk_opt'):
        """AIMD simulation for the clean surface.

        :param bulk_structural_file: File name of the bulk structure, defaults to 'CONTCAR_bulk_opt'
        :type bulk_structural_file: str, optional
        """

        atoms_bulk = read(bulk_structural_file)
        atoms_bulk.positions += 1.3 # avoid PBC error
        atoms_bulk.wrap() # avoid PBC error

        # add vacuum space and fix bottom atoms
        len_z = atoms_bulk.cell.array[2][2]
        c     = FixAtoms(indices=np.where(atoms_bulk.positions[:,2] < len_z / 2)[0])
        atoms_bulk.set_constraint(c)
        add_vacuum(atoms_bulk, 10.0)
        atoms_bulk.write('POSCAR') # Write POSCAR of the surface model.

        # get POTCAR
        os.system('getpotential_auto_zj.sh 6')

        # get INCAR
        with open('INCAR', 'w') as f:
            f.write(self.hp_config['INCAR_aimd'])
        os.system('modify_MAGMOM_auto.sh POSCAR')
        modify_INCAR('ISIF', '2')
        modify_INCAR('NSW', '100')

        # get KPOINTS
        with open('KPOINTS', 'w') as f:
            f.write(self.hp_config['KPOINTS'])

        # run vasp
        run_vasp(vasp_bash_path=self.hp_config['vasp_bash_path'])
        print('Surface AIMD simulation done.')

    def ads_aimd(self, structural_file='CONTCAR_surf_opt', random_samples=2):
        """AIMD simulation for the adsorption.

        :param structural_file: File name of the clean surface, defaults to 'CONTCAR_surf_opt'
        :type structural_file: str, optional
        :param random_samples: Randomly select surface sites for the simulation, defaults to 2
        :type random_samples: int, optional

        .. Note::
            ``random_samples`` cannot be larger than the number of detected surface sites.
        """

        # generate structures
        for ads in self.hp_config['adsorbates']:
            # generate adsorption configurations
            adder     = AddAtoms(structural_file,
                                 species=ads,
                                 sites=self.hp_config['sites'],
                                 dist_from_surf=self.hp_config['dist_from_surf'],
                                 num_atomic_layer_along_Z=6)
            all_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar,
                                                               calculation_index=self.hp_config['calculation_index'])

            if not os.path.exists('POSCARs'):
                os.mkdir('POSCARs')
            for i in range(all_sites):
                os.rename(f'POSCAR_{self.hp_config["calculation_index"]}_{i}',
                          os.path.join('POSCARs', f'POSCAR_{self.hp_config["calculation_index"]}_{i}'))

            for i in np.random.choice(range(all_sites), random_samples, replace=False):
                if not os.path.exists(str(i)):
                    os.mkdir(str(i))
                shutil.copyfile(os.path.join('POSCARs', f'POSCAR_{self.hp_config["calculation_index"]}_{i}'),
                               os.path.join(str(i), 'POSCAR'))

                parent_dir = os.getcwd()
                os.chdir(str(i))
                # get POTCAR
                os.system('getpotential_auto_zj.sh 6')

                # get INCAR
                with open('INCAR', 'w') as f:
                    f.write(self.hp_config['INCAR_aimd'])
                os.system('modify_MAGMOM_auto.sh POSCAR')
                modify_INCAR('ISIF', '2')
                modify_INCAR('NSW', '100')

                # get KPOINTS
                with open('KPOINTS', 'w') as f:
                    f.write(self.hp_config['KPOINTS'])

                # run vasp
                run_vasp(vasp_bash_path=self.hp_config['vasp_bash_path'])
                print(f'Adsorption AIMD simulation for {ads}_{i} done.')
                os.chdir(parent_dir)

    def run(self, formula, **kwargs):
        """

        :param formula: Chemical formula.
        :type formula: str
        :param **kwargs: Configurations to control the process
        :type **kwargs: dict

        """

        self.hp_config = {**self.hp_config, **config_parser(kwargs)}

        # bulk static calculation
        if self.hp_config['include_bulk_static']:
            if not os.path.exists('bulk_static'):
                os.mkdir('bulk_static')
            os.chdir('bulk_static')
            self.bulk_opt(formula)
            os.chdir(self.root_dir)

        # surface static calculation
        if self.hp_config['include_surface_static']:
            if not os.path.exists('surface_static'):
                os.mkdir('surface_static')
            shutil.copyfile(os.path.join('bulk_static', 'CONTCAR'),
                            os.path.join('surface_static', 'CONTCAR_bulk_opt'))
            os.chdir('surface_static')
            self.surf_opt(bulk_structural_file='CONTCAR_bulk_opt')
            os.chdir(self.root_dir)

        # adsorption static calculation
        if self.hp_config['include_adsorption_static']:
            if not os.path.exists('adsorption_static'):
                os.mkdir('adsorption_static')
            shutil.copyfile(os.path.join('surface_static', 'CONTCAR'),
                            os.path.join('adsorption_static', 'CONTCAR_surf_opt'))
            os.chdir('adsorption_static')
            self.ads_opt(structural_file='CONTCAR_surf_opt', random_samples=self.hp_config['random_samples'])
            os.chdir(self.root_dir)

        # bulk aimd
        if self.hp_config['include_bulk_aimd']:
            if not os.path.exists('bulk_aimd'):
                os.mkdir('bulk_aimd')
            os.chdir('bulk_aimd')
            self.bulk_aimd(formula)
            os.chdir(self.root_dir)

        # surface aimd calculation
        if self.hp_config['include_surface_aimd']:
            if not os.path.exists('surface_aimd'):
                os.mkdir('surface_aimd')
            shutil.copyfile(os.path.join('bulk_static', 'CONTCAR'),
                            os.path.join('surface_aimd', 'CONTCAR_bulk_opt'))
            os.chdir('surface_aimd')
            self.surface_aimd(bulk_structural_file='CONTCAR_bulk_opt')
            os.chdir(self.root_dir)

        # adsorption static calculation
        if self.hp_config['include_adsorption_aimd']:
            if not os.path.exists('adsorption_aimd'):
                os.mkdir('adsorption_aimd')
            shutil.copyfile(os.path.join('surface_static', 'CONTCAR'),
                            os.path.join('adsorption_aimd', 'CONTCAR_surf_opt'))
            os.chdir('adsorption_aimd')
            self.ads_aimd(structural_file='CONTCAR_surf_opt', random_samples=self.hp_config['random_samples'])
            os.chdir(self.root_dir)

if __name__ == '__main__':
    HT = HtDftAds(calculation_index=0)
    HT.run(sys.argv[1]) # debug only
