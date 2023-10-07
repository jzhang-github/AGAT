# Application (high-throughput prediction)

Modify the passed input dict to control the prediction.

```python
from agat.app import HpAds

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


ha = HpAds(**high_throughput_config)
ha.run(formula='NiCoFePdPt')
```

**See [docs/sphinx/source/Default parameters.md#default_high_throughput_config](https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md#default_high_throughput_config) to know how to use the parameter settings.**