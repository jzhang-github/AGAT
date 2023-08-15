# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:07:24 2023

@author: ZHANG Jun
"""

'''

.. py:data:: config

    Configurations for database construction, training process, and prediction behaviors.

'''

import os
import tensorflow as tf
import dgl
dgl.use_libxsmm(False)

default_elements = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B',  'Ba',
                    'Be', 'Bh', 'Bi', 'Bk', 'Br', 'C',  'Ca', 'Cd', 'Ce', 'Cf',
                    'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy',
                    'Er', 'Es', 'Eu', 'F',  'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd',
                    'Ge', 'H',  'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I',  'In', 'Ir',
                    'K',  'Kr', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg',
                    'Mn', 'Mo', 'Mt', 'N',  'Na', 'Nb', 'Nd', 'Ne', 'Nh', 'Ni',
                    'No', 'Np', 'O',  'Og', 'Os', 'P',  'Pa', 'Pb', 'Pd', 'Pm',
                    'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh',
                    'Rn', 'Ru', 'S',  'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn',
                    'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'Ts',
                    'U',  'V',  'W',  'Xe', 'Y',  'Yb', 'Zn', 'Zr']

default_data_config =  {
    'species': default_elements,
    'path_file': 'paths.log', # A file of absolute paths where OUTCAR and XDATCAR files exist.
    'build_properties': {'energy': True,
                         'forces': True,
                         'cell': True,
                         'cart_coords': True,
                         'frac_coords': True,
                         'constraints': True,
                         'distance': True,
                         'direction': True,
                         'path': False}, # Properties needed to be built into graph.
    'dataset_path': 'dataset', # Path where the collected data to save.
    'mode_of_NN': 'ase_natural_cutoffs', # How to identify connections between atoms. 'ase_natural_cutoffs', 'pymatgen_dist', 'ase_dist', 'voronoi'. Note that pymatgen is much faster than ase.
    'cutoff': 5.0, # Cutoff distance to identify connections between atoms. Deprecated if ``mode_of_NN`` is ``'ase_natural_cutoffs'``
    'load_from_binary': False, # Read graphs from binary graphs that are constructed before. If this variable is ``True``, these above variables will be depressed.
    'num_of_cores': 2,
    'super_cell': False,
    'has_adsorbate': False,
    'keep_readable_structural_files': False,
    'mask_similar_frames': False,
    'energy_stride': 0.05,
    'scale_prop': False,
    'validation_size': 0.15, # int or float. int: number of samples of the validation set. float: portion of samples of the validation set.
    'test_size': 0.15,
    'new_split': True
             }

default_train_config = {
    'dataset_path': 'dataset',
    'train_energy_model': True,
    'train_force_model': True,
    'epochs': 1000,
    'output_files': 'out_file',
    'new_energy_train': False,
    'new_force_train': False,
    'gpu_for_energy_train': 0,
    'gpu_for_force_train': 0,
    'load_graphs_on_gpu': False, # deprecated
    'validation_size': 0.15,
    'test_size': 0.15,
    'early_stop': True,
    'stop_patience': 300,
    'energy_GAT_out_list': [100, 100, 100],
    'force_GAT_out_list': [100, 100, 100],
    'energy_model_head_list': ['mul', 'div', 'free'],
    'force_model_head_list': ['mul', 'div', 'free'],
    'energy_model_readout_list': [200, 100, 50, 30, 10, 3, 1],
    'force_model_readout_list': [200, 100, 50, 30, 10, 3, 1],
    'bias': True,
    'negative_slope': 0.2,
    'embed_activation': 'LeakyReLU',
    'readout_activation': 'LeakyReLU',
    'energy_loss_fcn': tf.keras.losses.MeanSquaredError(),
    'force_loss_fcn': tf.keras.losses.MeanSquaredError(),
    'energy_optimizer': tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-8), # Default lr: 0.001
    'force_optimizer': tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-8), # Default lr: 0.001
    'weight_decay': 5e-5,
    'mae': tf.keras.losses.MeanAbsoluteError(),
    'batch_size': 64,
    'val_batch_size': 400,
    'L2_reg': False,
    'validation_freq': 20000,
    'validation_samples': 20000,
    # 'log_file': 'v7_energy.log',
    'transfer_learning': False,
    'trainable_layers': -4,
    'split_binary_graph': False,
    'batch_normalization': False,
    'mask_fixed': False,
    'tail_readout_noact': 3,
    'adsorbate': False, # indentify adsorbate or not when building graphs.
    'adsorbate_coeff': 20.0 # the importance of adsorbate atoms with respective to surface atoms.
    }

default_ase_calculator_config = {'fmax'             : 0.1,
                                 'steps'            : 200,
                                 'maxstep'          : 0.05,
                                 'restart'          : None,
                                 'restart_steps'    : 0,
                                 'perturb_steps'    : 0,
                                 'perturb_amplitude': 0.05,
                                 'out'              : None }

default_hp_config = {
        'volume_per_atom': 14.045510416666668,
        'energy_model_save_dir': os.path.join('out_file', 'energy_ckpt'),
        'force_model_save_dir': os.path.join('out_file', 'force_ckpt'),
        'opt_config': default_ase_calculator_config,
        'calculation_index'    : '0', # sys.argv[1],
        'fix_all_surface_atom' : False,
        'remove_bottom_atoms'  : False,
        'save_trajectory'      : False,
        'partial_fix_adsorbate': True,
        'adsorbates'           : ['H'],
        'sites'                : ['ontop'],
        'fmax'                 : 0.1,
        'dist_from_surf'       : 1.7,
        'using_template_bulk_structure': False,
        'graph_build_scheme_dir': os.path.join('dataset'),
        'gpu': 0 # in our test results, the A6000 is about 5 times faster than EPYC 7763.
        }

default_predict_config = {}

default_active_learning_config = {}
