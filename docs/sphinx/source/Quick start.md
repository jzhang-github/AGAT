# Quick start
### Prepare VASP calculations
Run VASP calculations at this step.

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
from agat.data import BuildDatabase
if __name__ == '__main__':
    database = BuildDatabase(mode_of_NN='ase_dist', num_of_cores=16)
    dataset = database.build()
```

### Train AGAT model
```python
from agat.model import Fit
f = Fit()
f.fit()
```

### Application (geometry optimization)
```python
from ase.optimize import BFGS
from ase.io import read
from agat.app import AgatCalculator

model_save_dir = 'agat_model'
graph_build_scheme_dir = 'dataset'

atoms = read('POSCAR')
calculator=AgatCalculator(model_save_dir,
                          graph_build_scheme_dir)
atoms = Atoms(atoms, calculator=calculator)
dyn = BFGS(atoms, trajectory='test.traj')
dyn.run(fmax=0.05)
```
### Application (high-throughput prediction)
```python
from agat.app.cata import HtAds

model_save_dir = 'agat_model'
graph_build_scheme_dir = 'dataset'
formula='NiCoFePdPt'

ha = HtAds(model_save_dir=model_save_dir, graph_build_scheme_dir=graph_build_scheme_dir)
ha.run(formula=formula)
```



### Tips:

See [API doc](https://jzhang-github.github.io/AGAT/API_Docs.html) for more details. For example:

- Manipulating `agat.dataset`: 
- AGAT molecular dynamics simulations:
- More options for controlling the AGAT training process.



