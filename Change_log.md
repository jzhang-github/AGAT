# [main](https://github.com/jzhang-github/AGAT/tree/main)
- Fix bugs in high-throughput predict:
  - [agat/app/cata/generate_adsorption_sites.py](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/generate_adsorption_sites.py#L219)
  - [agat/app/cata/high_throughput_predict.py](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L208)
  - [agat/app/cata/high_throughput_predict.py](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L251)
  - [agat/app/cata/high_throughput_predict.py](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L292)
  - [agat/app/GatApp.py](https://github.com/jzhang-github/AGAT/tree/main/agat/app/GatApp.py#L69-L70)
  - [agat/default_parameters.py](https://github.com/jzhang-github/AGAT/tree/main/agat/default_parameters.py#L133)

- Deprecate redundant training configurations: train_energy_model, train_force_model, new_energy_train, new_force_train, load_graphs_on_gpu, 
- Modify high-throughput predictions:
	- [agat/app/cata/high_throughput_predict.py#L160](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L160), [L169](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L169), [L172](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L172), [L218](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L218), [L245](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L245), [L284](https://github.com/jzhang-github/AGAT/tree/main/agat/app/cata/high_throughput_predict.py#L284)

# [v7.13.1](https://github.com/jzhang-github/AGAT/tree/v7.13.1)
- Fix a bug here: [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.13.1/agat/model/ModelFit.py#L243)
- Load well-trained models: [agat/model/GatEnergyModel.py](https://github.com/jzhang-github/AGAT/tree/v7.13.1/agat/model/GatEnergyModel.py#L154-L197) and [agat/model/GatForceModel.py](https://github.com/jzhang-github/AGAT/tree/v7.13.1/agat/model/GatForceModel.py#L201-L246)
- Test with best model after training. [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.13.1/agat/model/ModelFit.py#L222) and [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.13.1/agat/model/ModelFit.py#L422).

# [v7.13](https://github.com/jzhang-github/AGAT/tree/v7.13)
- Raise exception if error occurs when parsing OUTCAR file. [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/data/data.py#L595-L599)
- Remove `os` from the root name space. [agat/__init__.py](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/__init__.py#L18)
- Fix a bug when build graphs. See [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/data/data.py#L400-L411) and [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/data/data.py#L444-L452). Specifically, one needs to cast `tf.tensor` as `np.array` before building graph properties with a very large tensor. [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/data/data.py#L408-L409).
- Debug at these lines of [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/data/data.py): [L553](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/data/data.py#L553) and [L585-L588](https://github.com/jzhang-github/AGAT/tree/v7.13/agat/data/data.py#L585-L588).


# [v7.12.2](https://github.com/jzhang-github/AGAT/tree/v7.12.2)
- Using relative import. For example: [agat/__init__.py](https://github.com/jzhang-github/AGAT/tree/v7.12.2/agat/__init__.py#L14-L16)
- Update [documentations](https://jzhang-github.github.io/AGAT/).
- Import useful objects only. For example: [agat/app__init__.py](https://github.com/jzhang-github/AGAT/tree/v7.12.2/agat/app/__init__.py#L11)
- Return test MAE after training. [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.12.2/agat/model/ModelFit.py#L247) and [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.12.2/agat/model/ModelFit.py#L442)


# [v7.12.1](https://github.com/jzhang-github/AGAT/tree/v7.12.1)
- Import `pymatgen` module when necessary. See [agat/data/AtomicFeatures.py](https://github.com/jzhang-github/AGAT/tree/v7.12.1/agat/data/AtomicFeatures.py#L11). This feature was changed back.
- Specify device when building graphs. See [agat/app/GatApp.py](https://github.com/jzhang-github/AGAT/tree/v7.12.1/agat/app/GatApp.py#L69), [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.12.1/agat/data/data.py#L79-L83)
- Add default gpu specification when building database. [agat/default_parameters.py](https://github.com/jzhang-github/AGAT/tree/v7.12.1/agat/default_parameters.py#L60)
- Attache distributions at [dist](https://github.com/jzhang-github/AGAT/tree/v7.12.1/dist).


# [v7.12](https://github.com/jzhang-github/AGAT/tree/v7.12)

- Release pip wheel.   
- Simplify packages. See [v1.0.0](https://github.com/jzhang-github/AGAT/tree/v1.0.0) for more details of the first release.

# [v1.0.0](https://github.com/jzhang-github/AGAT/tree/v1.0.0) [![DOI](https://zenodo.org/badge/545430295.svg)](https://zenodo.org/badge/latestdoi/545430295)  

First release to reproduce results and support conclusions of [***Design High-Entropy Electrocatalyst via Interpretable Deep Graph Attention Learning***](https://doi.org/10.1016/j.joule.2023.06.003).

