# Default parameters
The default parameters that control the database construction, model training, high-throughput prediction ...

**You can import such parameters from `agat` module**.
```python
from agat.default_paramters import default_elements, default_build_properties, default_data_config, default_train_config, default_high_throughput_config
```
Or you can read the [source](https://github.com/jzhang-github/AGAT/blob/main/agat/default_parameters.py) code.

## `default_elements`
Elements used to build graph. A list of elements that are used to encode atomic features.
```python
['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B',  'Ba',
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
```

## `default_build_properties `
A dictionary defines which properties will be built into the graph.

| Parameter                  | Default value | Alternative(s) | Explanation |   
| -------------------------  | ------------- | -------------- | ----------- |   
| `energy`                   | `True`        | `False`        | Include total energy when building graphs. |   
| `forces`                   | `True`        | `False`        | Include atomic forces when building graphs. |   
| `cell`                     | `True`        | `False`        | Include structural cell when building graphs. |   
| `cart_coords`              | `True`        | `False`        | Include Cartesian coordinates when building graphs. |   
| `frac_coords`              | `True`        | `False`        | Include Fractional coordinates when building graphs. |   
| `constraints`              | `True`        | `False`        | Include constraint information when building graphs. |  
| `stress`                   | `True`        | `False`        | Include [Virial stress](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.get_stress) when building graphs. |  
| `distance`                 | `True`        | `False`        | Include distance between connected atoms when building graphs. |   
| `direction`                | `True`        | `False`        | Include unit vector between connected atoms when building graphs. |   
| `path`                     | `False`        | `True`        | Include file path of each graph corresponding to DFT calculations when building graphs. |   

## `default_data_config`
A dictionary defines how to build a database.

| Parameter                  | Default value | Alternative(s) | Explanation |
| -------------------------  | ------------- | ------------ | ----------- |
| `species`                  | `default_elements` above | A list of element symbols | A list of elements that are used to encode atomic features. |
| `path_file`                | 'paths.log'   | `str `         | A file of absolute paths where OUTCAR and XDATCAR files exist. | 
| `build_properties`         |  `default_build_properties` above | See [default_build_properties](#default_build_properties) | Properties needed to be built into graph. | 
| `dataset_path`             | 'dataset'     | A `str`      | A directory contains the database. |
| `mode_of_NN`               | 'ase_natural_cutoffs' | 'ase_natural_cutoffs', 'pymatgen_dist', 'ase_dist', and 'voronoi' | The mode of how to detect connection between atoms. Note that `pymatgen` is much faster than `ase`. |
| `cutoff`                   | 5.0           | A `float`    | Cutoff distance to identify connections between atoms. Deprecated if `mode_of_NN` is `'ase_natural_cutoffs'`|   
| `load_from_binary`         | `False`       | `True` | Read graphs from binary graphs that are constructed before. If this variable is `True`, these above variables will be depressed. | 
| `num_of_cores`             | 2             | `int`         | How many cores are used to extract vasp files and build graphs. |
| `super_cell`               | `False`       | `True`        | When building graphs, small cell may have problems to find neighbors. Specify this parameter as `True` to repeat cell to avoid such problems | 
| `has_adsorbate`            | `False`       | `True`        | Include adsorbate information when building graphs. For now, only `H` and `O` atoms are considered as adsorbate atoms. | |
| `keep_readable_structural_files` | `False`       | `True`        | Massive number of structural files (POSCARs) under `dataset_path` are generated when building graphs, you can choose to keep them or not. | 
| `mask_similar_frames`      | `False`       | `True`        | In VASP calculations, the energy optimization generate many frames that have similar geometry and total energies, you can extract only some of them by specifying this parameter and `energy_stride` below. |
| `mask_reversed_magnetic_moments`      | `False`       | `float`        | Frames with atomic magnetic moments lower than this value will be masked. |
| `scale_prop`               | `False`       | `True`        | Scale the properties. This function seems to be deprecated. I need to double-check the source code first, so do not use it. |


## `default_train_config`
A `dict` determines how to train the AGAT model.

| Parameter       | Default value | Alternative(s) | Explanation |
| --------------  | ------------- | -------------- | ----------- |
| `verbose`       | `1`           | `0`, `1`       | Output verbosity. `0`: test output; `1`: Validation and test output; `2`: train, validation, and test output. |
| `dataset_path`  | 'dataset'     | A `str`      | A directory contains the database. |
| `model_save_dir` | 'agat_model' | directory name, `str` | A directory to save the well-trained model. |
| `epochs`        | `1000`   | `int`      | Number of training epochs. | 
| `output_files`  | 'out_file'    | `str` | A directory to store ouputs of true and predicted properties. |
| `device`        | 'cuda:0'      | 'cpu' | Device to train the model. Use GPU cards to accerelate training. |
| `validation_size` | 0.15   | `float`, 0<`validation_size`<1 | Determines the proportion of the dataset to be included in the validation split.  |
| `test_size` | 0.15   | `float`, 0<`validation_size`<1 | Determines the proportion of the dataset to be included in the test split.  |
| `early_stop` | `True` | `False` | Implement early stop or not. If this is `True`, the training will be terminated after a specified number of epochs without model improvement. If this is `False`, the model weights will be saved every epoch. |
| `stop_patience` | `300` | `int` | Activated when `early_stop=True`. The training will be terminated after `stop_patience` epochs without model improvement. |
| `head_list` | ['mul', 'div', 'free'] | `list` | A list of attention mechanisms. See [agat/model/model.py](https://github.com/jzhang-github/AGAT/blob/a325dc1cad71a437dceeebdd9790efc35e8222b1/agat/model/model.py#L145-L152). |
| `gat_node_dim_list` | [len(default_elements), 100, 100, 100] | `list` | Node dimensions of AGAT `Layer`. |
| `energy_readout_node_list` | [len(head_list)*gat_node_dim_list[-1], 100, 50, 30, 10, 3, 1] | `list` | A list of node dimensions of energy readout layers. |
| `force_readout_node_list` | [len(head_list)*gat_node_dim_list[-1], 100, 50, 30, 10, 3, 3] | `list` | A list of node dimensions of force readout layers. | 
| `stress_readout_node_list` | [len(head_list)*gat_node_dim_list[-1], 100, 50, 30, 10, 3, 6] | `list` | A list of node dimensions of stress readout layers. | 
| `bias` | `True` | `False` | Add bias or not to the neural networks. |
| `negative_slope` | 0.2 | `float` | This specifies the negative slope of the [LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) activation function. |
| `criterion` | [nn.MSELoss()](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) | `torch.nn` loss functions | Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input `x` and target `y`. |
| `a` | 1.0 | `float` | The importance of energy loss in the total loss function. See [agat/model/fit.py](https://github.com/jzhang-github/AGAT/blob/a325dc1cad71a437dceeebdd9790efc35e8222b1/agat/model/fit.py#L212). | 
| `b` | 1.0 | `float` | The importance of force loss in the total loss function. See [agat/model/fit.py](https://github.com/jzhang-github/AGAT/blob/a325dc1cad71a437dceeebdd9790efc35e8222b1/agat/model/fit.py#L212). |
| `c` | 0.0 | `float` | The importance of stress loss in the total loss function. See [agat/model/fit.py](https://github.com/jzhang-github/AGAT/blob/a325dc1cad71a437dceeebdd9790efc35e8222b1/agat/model/fit.py#L212). |
| `learning_rate` | 0.0001 | `float` | The learning rate of [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer. |
| `weight_decay` | 0.0 | `float` | The weight decay of [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer. |
| `batch_size` | 64 | `int` | Training batch size. |
| `val_batch_size` | 400 | `int` | Batch size when validation and test. |
| `transfer_learning` | `False` | `True` | Turn on the transfer learning when `True`. (**Deprecated**) |
| `trainable_layers` | -4 | negative `int` | tail `trainable_layers` layers are trainable, other layers are freezed. (**Deprecated**) |
| `mask_fixed` | `False` | `True` | Mask fixed atoms or not. When `True`, the atomic forces of fixed atoms will not be included in the loss function. (**Deprecated**) |
| `tail_readout_no_act` | [3,3,3] | `list` | The tail `tail_readout_no_act` layers will have no activation functions. The first, second, and third elements are for energy, force, and stress readout layers, respectively. |
| `adsorbate_coeff` | 20.0 | `float` | Indentify and specify the importance of adsorbate atoms with respective to surface atoms. zero for equal importance. |

## `default_ase_calculator_config`.
See [bfgs](https://gitlab.com/ase/ase/-/blob/master/ase/optimize/bfgs.py) for more details.

| Parameter                  | Default value | Alternative(s) | Explanation |
| -------------------------  | ------------- | ------------ | ----------- |
| `fmax `                      | 0.1           | `float`      | Convergence criterion of atomic forces. Details: [ase optimizer](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#local-optimization) |
| `steps`                    | 200           | `int`        | Maximum iteration steps. |
| `maxstep`                  | 0.05          | `float`      | maximum distance an atom can move per iteration, unit is Å. |
| `restart`                  | `None`        |  `str`       | Pickle file used to store hessian matrix. | 
| `restart_steps`            | 0             | `int`        | Restart optimization if the optimization cannot converge. |
| `perturb_steps`            | 0             | `int`        | Number of perturbated steps. AGAT may have issues in converging BFGS, perturbating atomic positions may help the convergence. |
| `perturb_amplitude`        | 0.05          | `float`      | Perturbation amplitudes if `erturb_steps` larger than `1`. |



## `default_high_throughput_config`
Settings for the high-throughput predictions.

| Parameter      | Default value | Alternative(s) | Explanation |
| -------------- | ------------- | ------------ | ----------- |
| `model_save_dir` | `agat_model` | `str`, a directory name | A directory for loading the well-trained model from. |
| `opt_config` | `default_ase_calculator_config` | `dict` | Settings for [ase.optimize.BFGS](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.BFGS) structural optimizer. | 
| `calculation_index` | 0 | `str` | To label the calculation outputs. |
| `fix_all_surface_atom` | `False` | `True` | Fix all surface atoms or not. |
| `remove_bottom_atoms` | `False` | `True` | Remove the bottom atoms or not. |
| `save_trajectory` | `False` | `True` | Keep the optimization trajectory. |
| `partial_fix_adsorbate` | `False` | `True` | Partially fix the adsorbate freedom. |
| `adsorbates` | ['H'] | | `keys` of [agat/lib/adsorbate_poscar.py](https://github.com/jzhang-github/AGAT/blob/main/agat/lib/adsorbate_poscar.py) | A list of adsorbate names. |
| `sites` | ['ontop'] | `list` | A list of adsorption sites. See [agat/app/cata/generate_adsorption_sites.py](https://github.com/jzhang-github/AGAT/blob/a325dc1cad71a437dceeebdd9790efc35e8222b1/agat/app/cata/generate_adsorption_sites.py#L22C51-L22C51) |
| `dist_from_surf` | 1.7 | `float` | Distance between adosrbate and surface. Unit: angstrom. |
| `using_template_bulk_structure` | `False` | `True` | Using template to build the surface model. If this is `True`, you need to prepare a `POSCAR_temp` file in the working directory. |
| `graph_build_scheme_dir` | 'dataset' | A directory name. `str` | A directory storing the `graph_build_scheme.json` file. This file is generated when building the database, and is normally saved in the `default_data_config['dataset_path']`. |
| `device` | 'cuda' | `str` | Determines the device for the model prediction (forward). |

## `default_hp_dft_config`
Default settings for the high-throughput DFT calculations.

To be continued...