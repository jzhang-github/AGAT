# Build database

### Collect paths of VASP calculations
- We provided examples of VASP outputs at [VASP_calculations_example](https://github.com/jzhang-github/AGAT/tree/v1.0.0/files/VASP_calculations_example).   
- Find all directories containing `OUTCAR` file:   
  ```
  find . -name OUTCAR > paths.log
  ```    
- Remove the string 'OUTCAR' in `paths.log`.   
  ```
  sed -i 's/OUTCAR$//g' paths.log
  ```   
- Specify the absolute paths in `paths.log`.   
  ```
  sed -i "s#^.#${PWD}#g" paths.log
  ``` 

### Python script
Modify `data_config` for your own purposes. **See [docs/sphinx/source/Default parameters.md#default_data_config](https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md#default_data_config) to know how to use the parameter settings.**
```python
from agat.data import BuildDatabase
data_config =  {
    'species': ['H', 'Ni', 'Co', 'Fe', 'Pd', 'Pt'],
    'path_file': 'paths.log', # A file of absolute paths where OUTCAR and XDATCAR files exist.
    'build_properties': default_build_properties, # Properties needed to be built into graph.
    'dataset_path': 'dataset', # Path where the collected data to save.
    'mode_of_NN': 'ase_dist', # How to identify connections between atoms. 'ase_natural_cutoffs', 'pymatgen_dist', 'ase_dist', 'voronoi'. Note that pymatgen is much faster than ase.
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

if __name__ == '__main__': # encapsulate the following line in '__main__' because of the `multiprocessing`
    database = BuildDatabase(**data_config)
    database.build()
```

### Outputs
A new folder is created, which is defined by the `data_config['dataset_path']`. The structure of this folder is:

```console
dataset
├── all_graphs.bin
├── fname_prop.csv
└── graph_build_scheme.json
```

| File name | Explanation |
| --------- | ----------- |
|`all_graphs.bin` | Binary file of the DGL graphs |
| `fname_prop.csv` | A file storing the structural file name, properties, and paths. This file will not be used in the training, but is useful for checking the raw data. |
| `graph_build_scheme.json` | An information file tells you how to build the database. When deploying the well-trained model, this file is useful to construct new graphs. |
