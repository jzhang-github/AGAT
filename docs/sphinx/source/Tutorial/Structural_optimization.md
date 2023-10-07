# Application (structural optimization)

```python
from ase.optimize import BFGS
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

Navigate to [ase.optimize.BFGS](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.BFGS) for more details.