
# AGAT (Atomic Graph ATtention networks)
[![GitHub](https://img.shields.io/github/license/jzhang-github/AGAT)](https://github.com/jzhang-github/AGAT/blob/main/LICENSE)
[![Pypi](https://img.shields.io/pypi/v/agat.svg)](https://pypi.org/project/agat/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/agat)](https://pypi.org/project/agat/)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/agat)](https://pypi.org/project/agat/)
[![Documentation Status](https://readthedocs.org/projects/agat/badge/?version=latest)](https://jzhang-github.github.io/AGAT/)
 
  <br>  <br>  ![Model architecture](files/architecture.svg)

# Using AGAT
The [documentation](https://jzhang-github.github.io/AGAT/) of AGAT API is available.

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
Please navigate to the [Get Started](https://www.dgl.ai/pages/start.html) page of [dgl](https://www.dgl.ai/). For GPU version:   
  ```console
  conda install -c dglteam/label/cu118 dgl
  ```

  For now, the cpu version 1.1.2 of dgl has bugs. You can install the cpu version with `pip install dgl==1.1.1`.

- Change [dgl backend](https://docs.dgl.ai/en/1.1.x/install/#working-with-different-backends) to `tensorflow`.
  
  If you still cannot use `tensorflow` backend `dgl`, run the following on Linux OS:
  ```console
  wget https://data.dgl.ai/wheels/cu118/dgl-1.1.1%2Bcu118-cp310-cp310-manylinux1_x86_64.whl
  pip install ./dgl-1.1.1+cu118-cp310-cp310-manylinux1_x86_64.whl
  pip install numpy --upgrade
  ```

- For `tensorflow` of GPU version, if you don't have CUDA and CUDNN on your device, you need to run (Linux OS):
   ```console
   conda install -c conda-forge cudatoolkit=11.8.0
   pip install nvidia-cudnn-cu11==8.6.0.163
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   # Verify install:
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   
   Refer to [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#linux) and [Tensorflow_GPU](https://www.tensorflow.org/install/source#gpu) for more details (other OSs).


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

### Some default parameters
[agat/default_parameters.py](agat/default_parameters.py); Explanations: [docs/sphinx/source/Default parameters.md](docs/sphinx/source/Default parameters.md).


# Change log  
Please check [Change_log.md](https://github.com/jzhang-github/AGAT/blob/main/Change_log.md)