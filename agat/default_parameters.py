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
import torch.nn as nn

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

default_build_properties = {'energy': True,
                         'forces': True,
                         'cell': True,
                         'cart_coords': True,
                         'frac_coords': True,
                         'constraints': True,
                         'stress': True,
                         'distance': True,
                         'direction': True,
                         'path': False}

default_data_config =  {
    'species': default_elements,
    'path_file': 'paths.log', # A file of absolute paths where OUTCAR and XDATCAR files exist.
    'build_properties': default_build_properties, # Properties needed to be built into graph.
    'topology_only': False,
    'dataset_path': 'dataset', # Path where the collected data to save.
    'mode_of_NN': 'ase_natural_cutoffs', # How to identify connections between atoms. 'ase_natural_cutoffs', 'pymatgen_dist', 'ase_dist', 'voronoi'. Note that pymatgen is much faster than ase.
    'cutoff': 5.0, # Cutoff distance to identify connections between atoms. Deprecated if ``mode_of_NN`` is ``'ase_natural_cutoffs'``
    'load_from_binary': False, # Read graphs from binary graphs that are constructed before. If this variable is ``True``, these above variables will be depressed.
    'num_of_cores': 8,
    'super_cell': False,
    'has_adsorbate': False,
    'keep_readable_structural_files': False,
    'mask_similar_frames': False,
    'mask_reversed_magnetic_moments': False, # or -0.5 # Frames with atomic magnetic moments lower than this value will be masked.
    'energy_stride': 0.05,
    'scale_prop': False
             }

FIX_VALUE = [1,3,6]

default_train_config = {
    'verbose': 1, # `0`: no train and validation output; `1`: Validation and test output; `2`: train, validation, and test output.
    'dataset_path': os.path.join('dataset', 'all_graphs.bin'),
    'model_save_dir': 'agat_model',
    'epochs': 1000,
    'output_files': 'out_file',
    'device': 'cuda:0',
    'validation_size': 0.15,
    'test_size': 0.15,
    'early_stop': True,
    'stop_patience': 300,
    'head_list': ['mul', 'div', 'free'],
    'gat_node_dim_list': [len(default_elements), 100, 100, 100],
    'energy_readout_node_list': [300, 100, 50, 30, 10, 3, FIX_VALUE[0]], # the first value should be: len(head_list)*gat_node_dim_list[-1]
    'force_readout_node_list': [300, 100, 50, 30, 10, FIX_VALUE[1]], # the first value should be: len(head_list)*gat_node_dim_list[-1]
    'stress_readout_node_list': [300, 100, 50, 30, 10, FIX_VALUE[2]], # the first value should be: len(head_list)*gat_node_dim_list[-1]
    'bias': True,
    'negative_slope': 0.2,
    'criterion': nn.MSELoss(),
    'a': 1.0,
    'b': 50.0,
    'c': 1000.0,
    # 'optimizer': 'adam',
    'learning_rate': 0.0001,
    'weight_decay': 0.0, # weight decay (L2 penalty)
    'batch_size': 64,
    'val_batch_size': 400,
    'transfer_learning': False,
    'trainable_layers': -4,
    'mask_fixed': False,
    'tail_readout_no_act': [3,3,3],
    # 'adsorbate': False, #  or not when building graphs.
    'adsorbate_coeff': 20.0, # indentify and specify the importance of adsorbate atoms with respective to surface atoms. zero for equal importance.
    'transfer_learning': False}

default_ase_calculator_config = {'fmax'             : 0.1,
                                 'steps'            : 200,
                                 'maxstep'          : 0.05,
                                 'restart'          : None,
                                 'restart_steps'    : 0,
                                 'perturb_steps'    : 0,
                                 'perturb_amplitude': 0.05,
                                 'out'              : None}


default_high_throughput_config = {
        'model_save_dir': 'agat_model',
        'opt_config': default_ase_calculator_config,
        'calculation_index'    : '0', # sys.argv[1],
        'fix_all_surface_atom' : False,
        'remove_bottom_atoms'  : False,
        'save_trajectory'      : False,
        'partial_fix_adsorbate': True,
        'adsorbates'           : ['H'],
        'sites'                : ['ontop'],
        'dist_from_surf'       : 1.7,
        'using_template_bulk_structure': False,
        'graph_build_scheme_dir': os.path.join('dataset'),
        'device': 'cuda' # in our test results, the A6000 is about * times faster than EPYC 7763.
        }

default_predict_config = {}

default_active_learning_config = {}

default_hp_dft_config = {'INCAR_static': '''
SYSTEM = ML

Start parameter for this Run:
  ISTART = 0
  ICHARG = 2
  INIWAV = 1

Electronic Relaxation:
  ENCUT = 500
  PREC = Accurate
  ALGO = Fast
  NELM = 300
  NELMIN = 4
  EDIFF = 1E-06
  GGA = PE
  LREAL = A

Ionic Relaxation:
  EDIFFG = -0.05
  NSW = 300
  IBRION = 2
  ISIF = 3
  POTIM = 0.5

DOS related values:
  SIGMA = 0.1
  ISMEAR = 1

Spin polarized:
  ISPIN = 2
  MAGMOM = placeholder

File writing
  LWAVE = .FALSE.
  LCHARG = .FALSE.

Calculation of DOS
  NPAR = 8
  LORBIT = 11
  NCORE = 1
  IVDW = 11

''',

'INCAR_aimd': '''
SYSTEM = ML

Start parameter for this Run:
  ISTART = 0
  ICHARG = 2
  INIWAV = 1

Electronic Relaxation:
  ENCUT = 500
  PREC = Accurate
  ALGO = Fast
  NELM = 300
  NELMIN = 4
  EDIFF = 1E-06
  GGA = PE
  LREAL = A

Ionic Relaxation:
  NSW = 100
  IBRION = 0
  ISIF = 3
  POTIM = 2

DOS related values:
  SIGMA = 0.1
  ISMEAR = 1

Spin polarized:
  ISPIN = 2
  MAGMOM = placeholder

File writing
  LWAVE = .FALSE.
  LCHARG = .FALSE.

Calculation of DOS
  NPAR = 8
  LORBIT = 11
  IVDW = 11


  MDALGO = 3
  SMASS = 0
  TEBEG = 300
  TEEND = 300

''',

  "KPOINTS": '''Automatic mesh
0
Gamma
1 1 1
0.0 0.0 0.0

''',

        'calculation_index'    : '0', # sys.argv[1],
        'adsorbates'           : ['H'],
        'sites'                : ['bridge'],
        'dist_from_surf'       : 1.7,
        'include_bulk_static': True, # This should be true for other calculations.
        'include_surface_static': True, # This should be true for other calculations.
        'include_adsorption_static': True,
        'include_bulk_aimd': True,
        'include_surface_aimd': True,
        'include_adsorption_aimd': True,
        'random_samples': 1, # number of surfaces
        'vasp_bash_path': 'vasp_run.sh'
    }
