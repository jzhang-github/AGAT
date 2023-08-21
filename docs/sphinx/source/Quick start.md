# Quick start
### Prepare VASP calculations


### Collect paths of VASP calculations
  
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
