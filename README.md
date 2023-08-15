
# AGAT (Atomic Graph ATtention networks)
[![DOI](https://zenodo.org/badge/545430295.svg)](https://zenodo.org/badge/latestdoi/545430295) ![GitHub](https://img.shields.io/github/license/jzhang-github/AGAT) ![PyPI - Downloads](https://img.shields.io/pypi/dm/agat) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/agat) ![GitHub tag (with filter)](https://img.shields.io/github/v/tag/jzhang-github/AGAT)  


# Installation

### Install with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment
- Create a new environment   
```console
conda create -n agat python==3.10
```

- Activate the environment  
```console
conda activate agat
```

- Install package  
```console
pip install agat
```

- Install [dgl](https://www.dgl.ai/).   
Please navigate to the [Get Started](https://www.dgl.ai/pages/start.html) page of [dgl](https://www.dgl.ai/). For example:   
```console
conda install -c dglteam/label/cu118 dgl
```

- Change [dgl backend](https://docs.dgl.ai/en/1.1.x/install/#working-with-different-backends) to `tensorflow`.


# Using AGAT
The [documentation](https://jzhang-github.github.io/AGAT/) of AGAT API is available.


# Quick start
### Prepare VASP calculations


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

### Build database
```python
from agat.data import AgatDatabase
if __name__ == '__main__':
    ad = AgatDatabase(mode_of_NN='ase_dist', num_of_cores=2)
    ad.build()
```

### Train AGAT model
```python
from agat.model import Train
at = Train()
at.fit_energy_model()
at.fit_force_model()
```

### Model prediction
```python
from agat.app import GatApp
energy_model_save_dir = os.path.join('out_file', 'energy_ckpt')
force_model_save_dir = os.path.join('out_file', 'force_ckpt')
graph_build_scheme_dir = 'dataset'
app = GatApp(energy_model_save_dir, force_model_save_dir, graph_build_scheme_dir)
graph, info = app.get_graph('POSCAR')
energy = app.get_energy(graph)
forces = app.get_forces(graph)
```

### Geometry optimization
```python
from ase.io import read
from ase.optimize import BFGS
from agat.app import GatAseCalculator
from agat.default_parameters import default_hp_config
poscar = read('POSCAR')
calculator=GatAseCalculator(energy_model_save_dir,
                            force_model_save_dir,
                            graph_build_scheme_dir)
poscar = Atoms(poscar, calculator=calculator)
dyn = BFGS(poscar, trajectory='test.traj')
dyn.run(**default_hp_config['opt_config'])
```

# Change log  
Please check [Change_log.md](Change_log.md)