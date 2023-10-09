# Application (high-throughput prediction)

Modify the passed input dict to control the prediction.

### Python script
```python
import os
from agat.app import HtAds

ase_calculator_config = {'fmax'             : 0.1,
                         'steps'            : 200,
                         'maxstep'          : 0.05,
                         'restart'          : None,
                         'restart_steps'    : 0,
                         'perturb_steps'    : 0,
                         'perturb_amplitude': 0.05}

high_throughput_config = {
        'model_save_dir': 'agat_model',
        'opt_config': ase_calculator_config,
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


ha = HtAds(**high_throughput_config)
ha.run(formula='NiCoFePdPt')
```

**See [default_high_throughput_config](https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-high-throughput-config) to know how to use the parameter settings.**


### Output

```
.
├── ads_surf_energy_H_0.txt
└── POSCAR_surf_opt_0.gat
```

| File name | Explanation |
| --------- | ----------- |
|`ads_surf_energy_H_0.txt` | Predicted total energies. First column: Total energies of adsorption structure. Second column: Total energy of clean surface. Third column: convergence code: `1` for converge; `0` for ill converge. |
| `POSCAR_surf_opt_0.gat` | Optimized structure of clean surface. |