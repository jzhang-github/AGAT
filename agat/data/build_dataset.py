# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:54:47 2023

@author: ZHANG Jun
"""

import numpy as np
import os
import json
import multiprocessing

from ase.io import read
import torch
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm

from .build_graph import CrystalGraph
from ..default_parameters import default_data_config
from ..lib.model_lib import config_parser
from .dataset import Dataset

# for compatibility with old versions
from .dataset import concat_graphs, concat_dataset, select_graphs_random, select_graphs_from_dataset_random

class ReadGraphs(object):
    """
    .. py:class:: CrystalGraph(object)

        :param **data_config: Configuration file for building database.
        :type **data_config: str/dict
        :return: A ``DGL.graph``.
        :rtype: ``DGL.graph``.

    """
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}
        assert os.path.exists(self.data_config['dataset_dir']), str(self.data_config['dataset_dir']) + " not found."

        self.cg               = CrystalGraph(**self.data_config)
        fname_prop_data       = np.loadtxt(os.path.join(self.data_config['dataset_dir'],
                                                        'fname_prop.csv'),
                                           dtype=str, delimiter=',')
        self.name_list        = fname_prop_data[:,0]
        self.number_of_graphs = len(self.name_list)

    def read_batch_graphs(self, batch_index_list, batch_num):
        batch_fname  = [self.name_list[x] for x in batch_index_list]
        batch_g, batch_graph_info = [], []
        for fname in tqdm(batch_fname, desc='Reading ' + str(batch_num) + ' batch graphs', delay=batch_num):
            g, graph_info = self.cg.get_graph(os.path.join(self.data_config['dataset_dir'] ,fname))
            batch_g.append(g)
            batch_graph_info.append(graph_info)

        # batch_labels = {key: tf.stack([i[key] for i in batch_graph_info])\
        #                 for key in batch_graph_info[0] if key != 'path'}
        batch_labels = {}
        for key in batch_graph_info[0]:
            if key != 'path':
                graph_info_tmp = []
                for i in batch_graph_info:
                    graph_info_tmp.append(i[key].numpy())
                graph_info_tmp = np.array(graph_info_tmp)
                graph_info_tmp = torch.tensor(graph_info_tmp)
                batch_labels[key] = graph_info_tmp
        save_graphs(os.path.join(self.data_config['dataset_dir'], 'all_graphs_' + str(batch_num) + '.bin'), batch_g, batch_labels)

    def read_all_graphs(self, save_files=True): # prop_per_node=False Deprecated!
        """
        .. py:method:: read_all_graphs(self)

           Read all graphs specified in the csv file.

           .. Note:: The loaded graphs are saved under the attribute of :py:attr:`dataset_dir`.

           .. DANGER:: Do not scale the label if you don't know what are you doing.

           :param bool scale_prop: scale the label or not. DO NOT scale unless you know what you are doing.
           :param str ckpt_path: checkpoint directory of the well-trained model.
           :Returns:
              - graph_list： a list of ``DGL`` graph.
              - graph_labels： a list of labels.
        """
        if self.data_config['load_from_binary']:
            try:
                graph_path = os.readlink(os.path.join(self.data_config['dataset_dir'], 'all_graphs.bin'))
            except:
                graph_path = 'all_graphs.bin'
            cwd = os.getcwd()
            os.chdir(self.data_config['dataset_dir'])
            graph_list, graph_labels = load_graphs(graph_path)
            os.chdir(cwd)
        else:
            num_graph_per_core = self.number_of_graphs // self.data_config['num_of_cores'] + 1
            graph_index        = [x for x in range(self.number_of_graphs)]
            batch_index        = [graph_index[x: x + num_graph_per_core] for x in range(0, self.number_of_graphs, num_graph_per_core)]
            processes = []

            print('Waiting for all subprocesses...')
            for batch_num, batch_index_list in enumerate(batch_index):
                p = multiprocessing.Process(target=self.read_batch_graphs, args=[batch_index_list, batch_num])
                p.start()
                processes.append(p)
            print(processes)
            for process in processes:
                process.join()
            print('All subprocesses done.')
            graph_list = []
            graph_labels = {}
            for x in range(self.data_config['num_of_cores']):
                batch_g, batch_labels = load_graphs(os.path.join(self.data_config['dataset_dir'],
                                                                 'all_graphs_' + str(x) + '.bin'))
                graph_list.extend(batch_g)
                for key in batch_labels.keys():
                    try:
                        # graph_labels[key].extend(batch_labels[key])
                        graph_labels[key] = torch.cat([graph_labels[key],
                                                       batch_labels[key]], 0)
                    except KeyError:
                        graph_labels[key] = batch_labels[key]

                os.remove(os.path.join(self.data_config['dataset_dir'], 'all_graphs_' + str(x) + '.bin'))
            if save_files:
                save_graphs(os.path.join(self.data_config['dataset_dir'], 'all_graphs.bin'), graph_list, graph_labels)
                with open(os.path.join(self.data_config['dataset_dir'], 'graph_build_scheme.json'), 'w') as fjson:
                    json.dump(self.data_config, fjson, indent=4)
        return Dataset(dataset_path=None, from_file=False, graph_list=graph_list, props = graph_labels)

# class TrainValTestSplit(object):
#     """
#     Description:
#     ----------
#         Split the dataset.
#     Parameters
#     ----------
#     validation_size: int or float
#         int: number of samples of the validation set.
#         float: portion of samples of the validation set
#     test_size: int or float
#         int: number of samples of the validation set.
#         float: portion of samples of the validation set
#     csv_file: str
#         File name of a csv file that contains the filenames of crystals
#         with cif or VASP formate.
#     new_split: boolean
#         Split the dataset by `sklearn.model_selection.train_test_split` or
#         loaded from previously saved txt files.
#     Returns of `__call__` method
#     ----------------------------
#     train_index : list
#         A list of integers of training dataset.
#     validation_index : list
#         A list of integers of validation dataset.
#     test_index : list
#         A list of integers of test dataset.
#     """
#     def __init__(self, **data_config):
#         self.data_config = {**default_data_config, **config_parser(data_config)}

#         fname_prop_data       = np.loadtxt(os.path.join(self.data_config['dataset_path'],
#                                                         'fname_prop.csv'),
#                                            dtype=str, delimiter=',')
#         self.number_of_graphs = np.shape(fname_prop_data)[0]

#     def __call__(self):
#         if self.data_config['new_split']:
#             train_index,      validation_and_test_index = train_test_split([x for x in range(self.number_of_graphs)],
#                                                                            test_size=self.data_config['test_size']+self.data_config['validation_size'],
#                                                                            shuffle=True)
#             validation_index, test_index                = train_test_split(validation_and_test_index,
#                                                                            test_size=self.data_config['test_size']/(self.data_config['test_size']+self.data_config['validation_size']),
#                                                                            shuffle=True)
#             np.savetxt(os.path.join(self.data_config['dataset_path'], 'train.txt'),      train_index,      fmt='%.0f')
#             np.savetxt(os.path.join(self.data_config['dataset_path'], 'validation.txt'), validation_index, fmt='%.0f')
#             np.savetxt(os.path.join(self.data_config['dataset_path'], 'test.txt'),       test_index,       fmt='%.0f')
#         else:
#             try:
#                 train_index      = np.loadtxt(os.path.join(self.data_config['dataset_path'], 'train.txt'),      dtype=int)
#                 validation_index = np.loadtxt(os.path.join(self.data_config['dataset_path'], 'validation.txt'), dtype=int)
#                 test_index       = np.loadtxt(os.path.join(self.data_config['dataset_path'], 'test.txt'),       dtype=int)
#             except OSError:
#                 print('User: Index file not found, generate new files...')
#                 train_index,      validation_and_test_index = train_test_split([x for x in range(self.number_of_graphs)],
#                                                                                test_size=self.data_config['test_size']+self.data_config['validation_size'],
#                                                                                shuffle=True)
#                 validation_index, test_index                = train_test_split(validation_and_test_index,
#                                                                                test_size=self.data_config['test_size']/(self.data_config['test_size']+self.data_config['validation_size']),
#                                                                                shuffle=True)
#                 np.savetxt(os.path.join(self.data_config['dataset_path'], 'train.txt'),      train_index,      fmt='%.0f')
#                 np.savetxt(os.path.join(self.data_config['dataset_path'], 'validation.txt'), validation_index, fmt='%.0f')
#                 np.savetxt(os.path.join(self.data_config['dataset_path'], 'test.txt'),       test_index,       fmt='%.0f')
#         return train_index, validation_index, test_index

class ExtractVaspFiles(object):
    '''
    :param data_config['dataset_dir']: Absolute path where the collected data to save.
    :type data_config['dataset_dir']: str

    .. Note:: Always save the property per node as the label. For example: energy per atom (eV/atom).

    '''
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}
        if not os.path.exists(self.data_config['dataset_dir']):
            os.mkdir(self.data_config['dataset_dir'])

        self.in_path_list = np.loadtxt(self.data_config['path_file'], dtype=str)
        self.batch_index = np.array_split([x for x in range(len(self.in_path_list))], self.data_config['num_of_cores'])

        self.working_dir = os.getcwd()

    def read_oszicar(self,fname='OSZICAR'):
        """Get the electronic steps of a VASP run.

        :param fname: file name, defaults to 'OSZICAR'
        :type fname: str, optional
        :return: electronic steps of a VASP run.
        :rtype: list

        """

        ee_steps = []
        with open(fname, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if 'E0=' in line.split():
                ee_steps.append(int(lines[i-1].split()[1]))
        return ee_steps

    def read_incar(self, fname='INCAR'):
        """Get the NELM from INCAR. NELM: maximum electronic steps for each ionic step.

        :param fname: file name, defaults to 'INCAR'
        :type fname: str, optional
        :return: NELM tage in INCAR
        :rtype: int

        """

        with open(fname, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'NELM' in line.split():
                NELM = int(line.split()[2])
                break
        return NELM

    def split_output(self, process_index):
        '''
        :param process_index: A number to index the process.
        :type process_index: int.

        '''

        print('Mask similar frames:', self.data_config['mask_similar_frames'],
              'Mask reversed magnetic moments:', self.data_config['mask_reversed_magnetic_moments'])
        f_csv = open(os.path.join(self.data_config['dataset_dir'], f'fname_prop_{process_index}.csv'), 'w', buffering=1)
        in_path_index = self.batch_index[process_index]
        for path_index in tqdm(in_path_index, desc='Extracting ' + str(process_index) + ' VASP files.', delay=process_index): # tqdm(batch_fname, desc='Reading ' + str(batch_num) + ' batch graphs', delay=batch_num)
            in_path = self.in_path_list[path_index]
            in_path = in_path.strip("\n")
            os.chdir(in_path)

            if os.path.exists('OUTCAR') and os.path.exists('XDATCAR') and os.path.exists('OSZICAR') and os.path.exists('INCAR') and os.path.exists('CONTCAR'):

                read_good = True
                try:
                    # read frames
                    frame_contcar  = read('CONTCAR')
                    constraints    = frame_contcar.constraints
                    frames_outcar  = read('OUTCAR', index=':')   # coordinates in OUTCAR file are less accurate than that in XDATCAR. Energy in OUTCAR file is more accurate than that in OSZICAR file
                    frames_xdatcar = read('XDATCAR', index=':')
                    [x.set_constraint(constraints) for x in frames_xdatcar]

                    # pre processing
                    free_energy = [x.get_total_energy() for x in frames_outcar]
                    num_atoms = len(frames_outcar[0])
                    num_frames = len(frames_outcar)
                    ee_steps = self.read_oszicar()
                    NELM = self.read_incar()

                    # check magnetic moments (magmom).
                    if self.data_config['mask_reversed_magnetic_moments']:
                        frames_outcar[0].get_magnetic_moments()

                    assert len(frames_outcar) == len(frames_xdatcar), f'Inconsistent number of frames between OUTCAR and XDATCAR files. OUTCAR: {len(frames_outcar)}; XDATCAR {len(frames_xdatcar)}'
                    assert num_frames == len(free_energy), 'Number of frams does not equal to number of free energies.'
                    assert len(ee_steps) == len(frames_xdatcar), f'Inconsistent number of frames between OSZICAR and XDATCAR files. OSZICAR: {len(ee_steps)}; XDATCAR {len(frames_xdatcar)}'

                except:
                    print(f'Read OUTCAR, OSZICAR, INCAR, CONTCAR, and/or XDATCAR with exception in: {in_path}')
                    read_good = False

                if read_good:
                    output_mask = [True for x in range(num_frames)]
                    report_mask = False

                    # check similar frames
                    if self.data_config['mask_similar_frames']:
                        no_mask_list = [0]
                        for i, e in enumerate(free_energy):
                            if abs(e - free_energy[no_mask_list[-1]]) > self.data_config['energy_stride']:
                                no_mask_list.append(i)
                        if not no_mask_list[-1] == num_frames - 1: # keep the last frame
                            no_mask_list.append(num_frames - 1)
                        output_mask = [x if i in no_mask_list else False for i,x in enumerate(output_mask)]

                    # check electronic steps
                    for i, outcar in enumerate(frames_outcar):
                        if ee_steps[i] >= NELM:
                            output_mask[i] = False
                            report_mask = True

                    # check magnetic moments
                    if self.data_config['mask_reversed_magnetic_moments']:
                        for i, outcar in enumerate(frames_outcar):
                            magmoms = outcar.get_magnetic_moments()
                            if not (magmoms > self.data_config['mask_reversed_magnetic_moments']).all():
                                output_mask[i] = False
                                report_mask = True

                    no_mask_list   = [i for i,x in enumerate(output_mask) if x]
                    free_energy    = [free_energy[x] for x in no_mask_list]
                    frames_outcar  = [frames_outcar[x] for x in no_mask_list]
                    frames_xdatcar = [frames_xdatcar[x] for x in no_mask_list]
                    ee_steps       = [ee_steps[x] for x in no_mask_list]
                    free_energy_per_atom = [x / num_atoms for x in free_energy]

                    # save frames
                    for i in range(len(no_mask_list)):
                        fname = str(os.path.join(self.data_config['dataset_dir'], f'POSCAR_{process_index}_{path_index}_{i}'))
                        while os.path.exists(os.path.join(self.working_dir, fname)):
                            fname = fname + '_new'

                        frames_xdatcar[i].write(os.path.join(self.working_dir, fname))
                        forces = frames_outcar[i].get_forces(apply_constraint=False)
                        stress = frames_outcar[i].get_stress()
                        np.save(os.path.join(self.working_dir, f'{fname}_force.npy'), forces)
                        np.save(os.path.join(self.working_dir,f'{fname}_energy.npy'), free_energy_per_atom[i])
                        np.save(os.path.join(self.working_dir,f'{fname}_stress.npy'), stress)
                        f_csv.write(os.path.basename(fname) + ',  ')
                        f_csv.write(str(free_energy_per_atom[i]) + ',  ' + str(in_path) + '\n')

                    if report_mask:
                        print(f'Frame(s) in {in_path} are masked.')

            else:
                print(f'OUTCAR, OSZICAR, INCAR, CONTCAR, and/or XDATCAR files do not exist in {in_path}.')

            os.chdir(self.working_dir)
        f_csv.close()

    def __call__(self):
        """The __call__ function

        :return: DESCRIPTION
        :rtype: TYPE

        """

        processes = []
        for process_index in range(self.data_config['num_of_cores']):
            p = multiprocessing.Process(target=self.split_output, args=[process_index,])
            p.start()
            processes.append(p)
        print(processes)

        for process in processes:
            process.join()

        f = open(os.path.join(self.working_dir,
                              self.data_config['dataset_dir'],
                              'fname_prop.csv'), 'w')
        for job in range(self.data_config['num_of_cores']):
            lines = np.loadtxt(os.path.join(self.working_dir,
                                            self.data_config['dataset_dir'],
                                            f'fname_prop_{job}.csv'),
                               dtype=str)
            np.savetxt(f, lines, fmt='%s')
        f.close()

class BuildDatabase():
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}

    def build(self, save_files=True):
        # extract vasp files.
        evf = ExtractVaspFiles(**self.data_config)()

        # build binary DGL graphs.
        graph_reader = ReadGraphs(**self.data_config)
        dataset = graph_reader.read_all_graphs(save_files=save_files)

        # split the dataset.
        # train_index, validation_index, test_index = TrainValTestSplit(**self.data_config)()

        if not self.data_config['keep_readable_structural_files']:
            fname_prop_data = np.loadtxt(os.path.join(self.data_config['dataset_dir'],
                                                      'fname_prop.csv'),
                                         dtype=str, delimiter=',')
            fname_list      = fname_prop_data[:,0]
            for fname in fname_list:
                os.remove(os.path.join(self.data_config['dataset_dir'], fname))
                os.remove(os.path.join(self.data_config['dataset_dir'], f'{fname}_energy.npy'))
                os.remove(os.path.join(self.data_config['dataset_dir'], f'{fname}_force.npy'))
                os.remove(os.path.join(self.data_config['dataset_dir'], f'{fname}_stress.npy'))
            # os.remove(os.path.join(self.data_config['dataset_dir'], 'fname_prop.csv'))
            for i in range(self.data_config['num_of_cores']):
                os.remove(os.path.join(self.data_config['dataset_dir'], f'fname_prop_{i}.csv'))
        return dataset

# def concat_graphs(*list_of_bin, save_file=True, fname='concated_graphs.bin'):
#     """ Concat binary graph files.

#     :param *list_of_bin: input file names of binary graphs.
#     :type *list_of_bin: strings
#     :return: A new file is saved to the current directory: concated_graphs.bin.
#     :rtype: None. A new file.

#     Example::

#         concat_graphs('graphs1.bin', 'graphs2.bin', 'graphs3.bin')

#     """

#     warn("This object will be deprecated in the future. Please use `concat_dataset`.")

#     graph_list = []
#     graph_labels = {}
#     for file in list_of_bin:
#         batch_g, batch_labels = load_graphs(file)
#         graph_list.extend(batch_g)
#         for key in batch_labels.keys():
#             try:
#                 graph_labels[key] = torch.cat([graph_labels[key],
#                                                batch_labels[key]], 0)
#             except KeyError:
#                 graph_labels[key] = batch_labels[key]

#     if save_file:
#         save_graphs(fname, graph_list, graph_labels)
#     return Dataset(dataset_path=None, from_file=False, graph_list=graph_list, props=graph_labels)

# def concat_dataset(*list_of_datasets, save_file=False, fname='concated_graphs.bin'):
#     """ Concat binary graph files.

#     :param *list_of_bin: input file names of binary graphs.
#     :type *list_of_bin: strings
#     :return: A new file is saved to the current directory: concated_graphs.bin.
#     :rtype: None. A new file.

#     Example::

#         concat_graphs('graphs1.bin', 'graphs2.bin', 'graphs3.bin')

#     """

#     graph_list = []
#     graph_labels = {}
#     for d in list_of_datasets:
#         batch_g, batch_labels = d.graph_list, d.props
#         graph_list.extend(batch_g)
#         for key in batch_labels.keys():
#             try:
#                 graph_labels[key] = torch.cat([graph_labels[key],
#                                                batch_labels[key]], 0)
#             except KeyError:
#                 graph_labels[key] = batch_labels[key]

#     if save_file:
#         save_graphs(fname, graph_list, graph_labels)
#     return Dataset(dataset_path=None, from_file=False, graph_list=graph_list, props = graph_labels)

# def select_graphs_random(fname: str, num: int):
#     """ Randomly split graphs from a binary file.

#     :param fname: input file name.
#     :type fname: str
#     :param num: number of selected graphs (should be smaller than number of all graphs.
#     :type num: int
#     :return: A new file is saved to the current directory: Selected_graphs.bin.
#     :rtype: None. A new file.

#     Example::

#         select_graphs_random('graphs1.bin')

#     """
#     warn("This object will be deprecated in the future. Please use `select_graphs_from_dataset_random`")

#     bg, labels = load_graphs(fname)
#     num_graphs = len(bg)
#     assert num < num_graphs, f'The number of selected graphs should be lower than\
# the number of all graphs. Number of selected graphs: {num}. Number of all graphs: {num_graphs}.'
#     random_int = np.random.choice(range(num_graphs), size=num, replace=False)

#     selected_bg = [bg[x] for x in random_int]

#     graph_labels = {}
#     for key in labels.keys():
#         graph_labels[key] = labels[key][random_int]

#     save_graphs('selected_graphs.bin', selected_bg, graph_labels)

# def select_graphs_from_dataset_random(dataset, num: int, save_file=False,
#                                       fname='selected_graphs.bin'):
#     """ Randomly split graphs from a binary file.

#     :param fname: input file name.
#     :type fname: str
#     :param num: number of selected graphs (should be smaller than number of all graphs.
#     :type num: int
#     :return: A new file is saved to the current directory: Selected_graphs.bin.
#     :rtype: None. A new file.

#     Example::

#         select_graphs_random('graphs1.bin')

#     """

#     num_graphs = len(dataset)
#     assert num < num_graphs, f'The number of selected graphs should be lower than\
# the number of all graphs. Number of selected graphs: {num}. Number of all graphs: {num_graphs}.'
#     random_int = np.random.choice(range(num_graphs), size=num, replace=False)
#     dataset = dataset[list(random_int)]
#     if save_file:
#         save_graphs(fname, dataset.graph_list, dataset.props)
#     return dataset

# def save_dataset(dataset: Dataset, fname='graphs.bin'):
#     assert isinstance(dataset, Dataset), f'Wrong dataset type. Expect `LoadDataset`, but got {type(dataset)}'
#     save_graphs(fname, dataset.graph_list, dataset.props)

# build data
if __name__ == '__main__':
    ad = BuildDatabase(mode_of_NN='pymatgen_dist', num_of_cores=16)
    ad.build()

