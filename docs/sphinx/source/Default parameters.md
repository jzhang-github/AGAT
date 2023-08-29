# Default parameters
The default parameters that control the database construction, model training, high-throughput prediction ...

**You can import such parameters from `agat` module**.
```python
from agat import default_paramters
```
Then you will get a python dictionary of all default parameters. Or you can read the [source](https://github.com/jzhang-github/AGAT/blob/main/agat/default_parameters.py) code.

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
| `distance`                 | `True`        | `False`        | Include distance between connected atoms when building graphs. |   
| `direction`                | `True`        | `False`        | Include unit vector between connected atoms when building graphs. |   
| `path`                     | `True`        | `False`        | Include file path of each graph corresponding to DFT calculations when building graphs. |   

## `default_data_config`
A dictionary defines how to build a database.

| Parameter                  | Default value | Alternative(s) | Explanation |
| -------------------------  | ------------- | ------------ | ----------- |
| `species`                  | `default_elements` above | For a system with `C` and `H`, this parameter can be `['C', 'H']` | A list of elements that are used to encode atomic features. |
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
| `scale_prop`               | `False`       | `True`        | Scale the properties. This function seems to be deprecated. I need to double-check the source code first, so do not use it. |
| `validation_size`          | 0.15          | `int`/`float`    | Size of the validation dataset. `int`: number of samples of the validation set. `float`: portion of samples of the validation set. | 
| `test_size`                | 0.15          | `int`/`float`    | Size of the test dataset. `int`: number of samples of the validation set. `float`: portion of samples of the validation set. |
| `new_split`                | `True`        | `False`          | Split the dataset according specified `validation_size` and `test_size` when building graphs.  |
| `gpu`                      | 0             | `int`: -1        | Specify device when building graphs. Negative values for cpu; Positive `int` for GPU. |

**Note**: More `num_of_cores` needs more memory. Weird results happen when you don't have enough memory.