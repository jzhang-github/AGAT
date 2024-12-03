# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 18:07:51 2023

@author: ZHANG Jun
"""

import os
from datetime import datetime
from shutil import copyfile, move, copytree, rmtree
import json

from ase import units
from ase.md import MDLogger
from ase.io import read
import pandas as pd

from agat.data import BuildDatabase, concat_graphs
from agat.model import Fit
from agat.default_parameters import default_train_config, default_graph_config, default_potential_generator_config
# Here, we use NVT by depressing the barostat
from agat.app.ensembles import ModifiedNPT as NPT # WARNING: Not specifying pfactor sets it to None, disabling the barostat.
from agat.app.calculators import OnTheFlyCalculator

class PotentialGenerator():
    def __init__(self, **config):
        self.config = {**default_potential_generator_config, **config}

        self.number_of_models = self.config['number_of_models']
        # self.generation = self.config['current_generation']
        self.factory_dir = self.config['current_generation']

        self.device = self.config['device']

        with open(os.path.join(self.factory_dir, 'generator_config.json'), 'w') as config_f:
            json.dump(self.config, config_f, indent=4)

        # check the file
        if not os.path.exists(self.factory_dir):
            os.makedirs(self.factory_dir)
        for site in ['0_prepare_vasp', '1_run_vasp', '2_build_graph', '3_train_agat']:
            if not os.path.exists(os.path.join(self.factory_dir, site)):
                os.makedirs(os.path.join(self.factory_dir, site))

    @property
    def time(self):
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def update_generator_status(self, Current_job, **kwargs):
        df = pd.DataFrame(columns=['Time', 'Generation', 'VASP_count', 'Current_work', 'Note'])
        df['Time'] = [0000] # self.time
        df['Generation'] = [self.generation]
        df['VASP_count'] = [None]
        df['Current_work'] = [Current_job]
        if os.path.exists(os.path.join(self.factory_dir, 'generator_status.csv')):
            df.to_csv(os.path.join(self.factory_dir, 'generator_status.csv'),
                      mode='a', header=False)
        else:
            df.to_csv(os.path.join(self.factory_dir, 'generator_status.csv')) # create a status file

    def file_force_action(self, func, src, dst):
        if os.path.exists(dst):
            print(f'Warning: {dst} already exists. Overwrite this file.',
                  file=self.logIO)
            if os.path.isdir(dst):
                rmtree(dst)
            else:
                os.remove(dst)
        func(src, dst)

    def build_graphs(self, **kwargs):
        # raw_dataset_dir = os.path.join(self.vasp_raw_data_dir,
        #                                f'generation_{self.generation}')

        # # generation = self.generation
        # assert os.path.isdir(raw_dataset_dir), f'{raw_dataset_dir} is not a directory.'
        config = {**default_graph_config, **self.config, **kwargs}

        # if not os.path.exists(f'generation_{self.generation}'):
        #     os.makedirs(f'generation_{self.generation}')
        # if not os.path.exists(self.graphs_dir):
        #     os.makedirs(self.graphs_dir)

        # build
        print(f'Starting time for building graphs (generation {self.generation}):',
              self.time, file=self.logIO)
        os.chdir(os.path.join(self.facotry_dir, '1_run_vasp'))
        os.system('get_paths.sh')
        # os.chdir(self.root_dir)
        dst = os.path.join(self.facotry_dir, '2_build_graph', 'paths.log')
        self.file_force_action(move,
                               os.path.join(self.facotry_dir, '1_run_vasp', 'paths.log'),
                               dst)

        # update settings
        config['path_file'] = dst
        config['dataset_path'] = os.path.join(f'generation_{self.generation}',
                                              'dataset')
        ad = BuildDatabase(**config)
        ad.build()
        dst_bin = os.path.join(self.graphs_dir, f'all_graphs_generation_{self.generation}.bin')
        self.file_force_action(move,
                               os.path.join(config['dataset_path'],
                                            'all_graphs.bin'),
                               dst_bin)
        self.file_force_action(copyfile,
                               os.path.join(config['dataset_path'],
                                            'graph_build_scheme.json'),
                               os.path.join(self.graphs_dir, 'graph_build_scheme.json'))
        print(f'Complete time for building graphs (generation {self.generation}):',
              self.time, file=self.logIO)

    def train_agat(self, **kwargs):
        config = {**default_train_config, **self.config, **kwargs}

        # check working directory
        if not os.path.exists(f'generation_{self.generation}'):
            os.makedirs(f'generation_{self.generation}')

        # prepare directory
        if not os.path.exists(os.path.join(self.agat_model_dir,
                                           'agat_model_latest')):
            os.makedirs(os.path.join(self.agat_model_dir, 'agat_model_latest'))
        if not os.path.exists(os.path.join(
                self.agat_model_dir,
                f'agat_model_generation_{self.generation}')):
            os.mkdir(os.path.join(self.agat_model_dir,
                                  f'agat_model_generation_{self.generation}'))

        print(f'Starting time for training AGAT model ensemble (generation {self.generation}):',
              self.time, file=self.logIO)
        # concat graphs
        graph_fnames = [x for x in os.listdir(self.graphs_dir) if x.split('.')[-1] == 'bin']
        concat_graphs(*[os.path.join(self.graphs_dir, x) for x in graph_fnames])
        self.file_force_action(move,
                               'concated_graphs.bin',
                               os.path.join(f'generation_{self.generation}',
                                            'concated_graphs.bin'))

        # update training configurations
        config['device'] = self.device
        config['dataset_path'] = os.path.join(f'generation_{self.generation}',
                                              'concated_graphs.bin')
        for n in range(self.number_of_models):
            # update training configurations
            config['model_save_dir'] = os.path.join(f'generation_{self.generation}',
                                                    f'agat_model_{n}')
            config['output_files'] = os.path.join(f'generation_{self.generation}',
                                                  f'agat_train_out_file_{n}')

            # fit
            f = Fit(**config)
            f.fit()
            self.file_force_action(
                copytree,
                config['model_save_dir'],
                os.path.join(self.agat_model_dir,
                             'agat_model_latest',
                             f'agat_model_{n}'))
            self.file_force_action(
                copytree,
                config['model_save_dir'],
                os.path.join(self.agat_model_dir,
                             f'agat_model_generation_{self.generation}',
                             f'agat_model_{n}'))
            self.file_force_action(
                copyfile,
                'fit.log',
                os.path.join(f'generation_{self.generation}', f'fit_{n}.log'))
        print(f'Complete time for training AGAT model ensemble (generation {self.generation}):',
              self.time, file=self.logIO)

    def npt_run(self, **kwargs):
        config = {**self.config, **kwargs}
        # if os.path.exists(f'md_NPT_{int(self.generation)-1}.traj'):
        #     atoms = read(f'md_NPT_{int(self.generation)-1}.traj')
        #     print(f'(Generation {self.generation}): Reading structure from md_NPT_{int(self.generation)-1}.traj.',
        #           self.time, file=self.logIO)
        # else:
        self.logIO = open('potential_generation_nvt.log', 'a+', buffering=1)
        atoms = read(config['structural_fname'])
        atoms.set_cell(atoms.cell * config['cell_scale_factor'],
                       scale_atoms=True)
        print(f'(Generation {self.generation}): Reading structure from {config["structural_fname"]}. Cell scale factor: {config["cell_scale_factor"]}',
              self.time, file=self.logIO)

        # start_step = config['start_step']
        print(f'Starting time for NPT run (generation {self.generation}):',
              self.time, file=self.logIO)

        otfc = OnTheFlyCalculator(
            os.path.join(config['agat_model_dir'], 'agat_model_latest'),
            config['graphs_dir'],
            use_vasp=config['use_vasp'],
            start_step=config['start_step'],
            vasp_work_dir=os.path.join(self.vasp_raw_data_dir,
                                       f'generation_{int(self.generation)+1}'),
            device = config['device'],
            energy_threshold = config['energy_threshold'],
            force_threshold = config['force_threshold'],
            stress_threshold = config['stress_threshold'],
            io=self.logIO)

        atoms.set_calculator(otfc)

        dyn = NPT(atoms,
                  timestep=config['timestep'] * units.fs,
                  temperature_K=config['temperature_K'],
                  ttime = 25 * units.fs,
                  pfactor = None, # 75 * units.fs, # Not specifying pfactor sets it to None, disabling the barostat.
                  externalstress = [0.0] * 6,
                  mask=[[1,0,0],
                        [0,1,0],
                        [0,0,1]],
                  max_collected_snapshot_num = config['collected_snapshot_num_in_each_gen'],
                  trajectory=f'md_NPT_{self.generation}.traj')

        dyn.attach(MDLogger(dyn, atoms, f'md_NPT_{self.generation}.log',
                            header=True,
                            stress=True,
                            peratom=False,
                            mode="a"),
                   interval=1)

        r_step = dyn.run(config['npt_steps']) # -config['start_step']

        config['start_step'] = r_step

        print(f'Complete time for NPT run (generation {self.generation}):',
              self.time, file=self.logIO)
        return r_step

    def one_loop(self, **kwargs):
        config = {**default_potential_generator_config,
                  **default_graph_config,
                  **self.config,
                  **kwargs}
        # build graphs
        self.build_graphs(**config)

        self.train_agat(**config)

        r_step = self.npt_run(**config)

        # clean WAVECAR files
        raw_dataset_dir = os.path.join(self.vasp_raw_data_dir,
                                   f'generation_{int(self.generation)+1}')
        dirs = os.listdir(raw_dataset_dir)
        for d in dirs:
            path_tmp = os.path.join(raw_dataset_dir, d, 'WAVECAR')
            if os.path.isfile(path_tmp):
                os.remove(path_tmp)
            path_tmp = os.path.join(raw_dataset_dir, d, 'CHGCAR')
            if os.path.isfile(path_tmp):
                os.remove(path_tmp)
            path_tmp = os.path.join(raw_dataset_dir, d, 'CHG')
            if os.path.isfile(path_tmp):
                os.remove(path_tmp)

        return r_step

    def generate(self, **kwargs):
        config = {**default_potential_generator_config,
                  **default_graph_config,
                  **self.config,
                  **kwargs}

        print('Starting time for potential generation:',
              self.time, file=self.logIO)

        start_step = config['start_step']

        while True:
            config['start_step'] = start_step
            start_step = self.one_loop(**config)

            # start structure for npt run in the next generation
            self.config['structural_fname'] = f'md_NPT_{int(self.generation)}.traj'
            self.config['cell_scale_factor'] = 1.0

            # calculate vasp contribution
            atoms_list = read(f'md_NPT_{self.generation}.traj', index=':')
            total_steps = len(atoms_list)
            vasp_steps = os.listdir(
                os.path.join(self.vasp_raw_data_dir,
                             f'generation_{int(self.generation)+1}'))

            vasp_contribution = len(vasp_steps) / total_steps

            print(f'VASP contribution in generation {self.generation}: {vasp_contribution}',
                  file=self.logIO)

            if vasp_contribution < config['vasp_contribution_convergence']:
                break

            self.generation += 1
        print('Complete time for potential generation:',
              self.time, file=self.logIO)

    def run_static(self, poscar='POSCAR', **kwargs):
        self.logIO = open('potential_generation_nvt.log', 'a+', buffering=1)

        # check dependent files
        ## VASP files
        ## Bash scripts
        pass

if __name__ == '__main__':
    pg = PotentialGenerator(
            structural_fname='POSCAR',
            cell_scale_factor=1.0,
            energy_threshold= 0.02,
            force_threshold= 0.10,
            stress_threshold= 0.003,
            collected_snapshot_num_in_each_gen=500,
            start_step=0,
            use_vasp=True,
            current_generation=0,
            vasp_contribution_convergence=0.001)
    pg.generate()

    current_generation = 0
    for csf in [0.97,0.98,0.99,1.01,1.02,1.03]:
        pg = PotentialGenerator(
                structural_fname='POSCAR',
                cell_scale_factor=csf,
                energy_threshold= 0.02,
                force_threshold= 0.10,
                stress_threshold= 0.003,
                collected_snapshot_num_in_each_gen=500,
                start_step=0,
                use_vasp=True,
                current_generation=current_generation, # auto?
                vasp_contribution_convergence=0.001)
        pg.generate()
        fnames = os.listdir()
        current_generation = max([int(f.split('_')[1]) for f in fnames if f.split('_')[0] == 'generation'])
