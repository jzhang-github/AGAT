# [main](https://github.com/jzhang-github/AGAT/tree/main)
- Fix warning when using `torch.load`. [agat/lib/model_lib.py](https://github.com/jzhang-github/AGAT/blob/main/agat/lib/model_lib.py#L42-L44)

- Fix bugs is `ase.atoms` has `ase.constraints.FixScaled` and `ase.constraints.FixedLine`.

- Add [`agat_linux_gpu_cu124.yml`](agat_linux_gpu_cu124.yml) file.

- Fix a bug: [agat\app\cata\generate_adsorption_sites.py](https://github.com/jzhang-github/AGAT/blob/main/agat/app/cata/generate_adsorption_sites.py#L129-L131); [agat\app\cata\generate_adsorption_sites.py](https://github.com/jzhang-github/AGAT/blob/main/agat/app/cata/generate_adsorption_sites.py#L164-L166); [agat\app\cata\generate_adsorption_sites.py](https://github.com/jzhang-github/AGAT/blob/main/agat/app/cata/generate_adsorption_sites.py#L209)

  

# [v9.0.0](https://github.com/jzhang-github/AGAT/tree/v9.0.0)
**Note: AGAT after this version (included) cannot load the well-trained model before.** If you need to do so, please use v8.0.5: https://pypi.org/project/agat/8.0.5/
- Fix bugs when traing model with voigt stress tensor.
    - Add a node to edge layer: [agat/model/model.py](https://github.com/jzhang-github/AGAT/blob/v9.0.0/agat/model/model.py#L137-L138).
    - Message passing: [agat/model/model.py](https://github.com/jzhang-github/AGAT/blob/v9.0.0/agat/model/model.py#L244-L247).
- Fix a bug when saving model [agat\lib\model_lib.py](https://github.com/jzhang-github/AGAT/blob/v9.0.0/agat/lib/model_lib.py#L161).
- Fix a bug when training model [agat\model\fit.py](https://github.com/jzhang-github/AGAT/blob/v9.0.0/agat/model/fit.py#L244-L252).

# [v8.0.3](https://github.com/jzhang-github/AGAT/tree/v8.0.4)
- Modify [agat/lib/model_lib.py#L40-L45](https://github.com/jzhang-github/AGAT/blob/v8.0.4/agat/lib/model_lib.py#L40-L45)
- Modify [agat/app/cata/high_throughput_predict.py#L208](https://github.com/jzhang-github/AGAT/blob/v8.0.4/agat/app/cata/high_throughput_predict.py#L208)

# [v8.0.3](https://github.com/jzhang-github/AGAT/tree/v8.0.3)
- Modify [agat/lib/model_lib.py#L178-L193](https://github.com/jzhang-github/AGAT/blob/v8.0.3/agat/lib/model_lib.py#L178-L193)
- Add default parameter: `vasp_bash_path`[high_throughput_dft_calculation.py#L71](https://github.com/jzhang-github/AGAT/blob/v8.0.3/agat/app/cata/high_throughput_dft_calculation.py#L71); [default_parameters.py#L242](https://github.com/jzhang-github/AGAT/blob/v8.0.3/agat/default_parameters.py#L242).
- Modify `run_vasp()` function: [high_throughput_lib.py#L124-L149](https://github.com/jzhang-github/AGAT/blob/v8.0.3/agat/lib/high_throughput_lib.py#L124-L149).
- Add transfer learning: [default_parameters.py#L97](https://github.com/jzhang-github/AGAT/blob/v8.0.3/agat/default_parameters.py#L97). [agat/model/fit.py#L169-L174](https://github.com/jzhang-github/AGAT/blob/v8.0.3/agat/model/fit.py#L169-L174)
- Add split graphs: [agat/data/build_dataset.py#L795-L824](https://github.com/jzhang-github/AGAT/blob/v8.0.3/agat/data/build_dataset.py#L795-L824)

# [v8.0.0](https://github.com/jzhang-github/AGAT/tree/v8.0.0)
- Convert TensorFlow to PyTorch backend.
- Updata docs.

# [v7.14.0](https://github.com/jzhang-github/AGAT/tree/v7.14.0)
- Add API for controling HP DFT calculation. [agat/default_parameters.py](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/default_parameters.py#L246-L251)
- Add `mask_reversed_magnetic_moments` in [agat/default_parameters.py](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/default_parameters.py#L58) and [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/data/data.py)
- Modify [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/data/data.py):
	- Include stress in the graph: [agat/data/data.py#L273-L275](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/data/data.py#L273-L275), [agat/data/data.py#L350-L352](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/data/data.py#L350-L352).
	- Update method of parsing the vasp data: [agat/data/data.py#L610](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/data/data.py#L610), [agat/data/data.py#L625-L656](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/data/data.py#L625-L656), [agat/data/data.py#L661-L675](https://github.com/jzhang-github/AGAT/tree/v7.14.0/agat/data/data.py#L661-L675).
- Update docs.

# [v7.13.4](https://github.com/jzhang-github/AGAT/tree/v7.13.4)
- Add [agat/app/cata/high_throughput_dft_calculation.py](https://github.com/jzhang-github/AGAT/tree/v7.13.4/agat/app/cata/high_throughput_dft_calculation.py).
- Shift atomic positions before fix bottom atoms: [agat/app/cata/high_throughput_predict.py#L225-L227](https://github.com/jzhang-github/AGAT/tree/v7.13.4/agat/app/cata/high_throughput_predict.py#L225-L227)
- Add `default_hp_dft_config` to [agat/default_parameters.py#L139-L246](https://github.com/jzhang-github/AGAT/tree/v7.13.4/agat/default_parameters.py#L139-L246).
- Add [agat/lib/HighThroughputLib.py](https://github.com/jzhang-github/AGAT/tree/v7.13.4/agat/lib/HighThroughputLib.py).
- Add [agat/lib/ModifyINCAR.py](https://github.com/jzhang-github/AGAT/tree/v7.13.4/agat/lib/ModifyINCAR.py).
- Upgrade docs.

# [v7.13.3](https://github.com/jzhang-github/AGAT/tree/v7.13.3)
- Using self-defined tf-based functions to calculate Pearson r: [agat/lib/GatLib.py#L248-L259](https://github.com/jzhang-github/AGAT/tree/v7.13.3/agat/lib/GatLib.py#L248-L259)

  This self-defined function can handle `ValueError: array must not contain infs or NaNs`.

- Fix a bug: [bug](https://github.com/jzhang-github/AGAT/tree/v7.13.3/agat/model/ModelFit.py#L280)
- Clip optimizer grads: [clipnorm=1.0](https://github.com/jzhang-github/AGAT/tree/v7.13.3/agat/default_parameters.py#L88-89)

# [v7.13.2](https://github.com/jzhang-github/AGAT/tree/v7.13.2)
- Fix bugs in high-throughput predict:
  - [agat/app/cata/generate_adsorption_sites.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/app/cata/generate_adsorption_sites.py#L218)
  - [agat/app/cata/high_throughput_predict.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/app/cata/high_throughput_predict.py#L207)
  - [agat/app/cata/high_throughput_predict.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/app/cata/high_throughput_predict.py#L250)
  - [agat/app/cata/high_throughput_predict.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/app/cata/high_throughput_predict.py#L291)
  - [agat/app/GatApp.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2ain/agat/app/GatApp.py#L69-L70)
  - [agat/default_parameters.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/default_parameters.py#L133)

- Deprecate redundant training configurations:
	- `train_energy_model`: [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/model/ModelFit.py#L96)  and [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/model/ModelFit.py#L198)
	- `train_force_model`: [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/model/ModelFit.py#L274) and [agat/model/ModelFit.py](https://github.com/jzhang-github/AGAT/tree/v7.13.2/agat/model/ModelFit.py#L398)
	- `new_energy_train`
	- `new_force_train`
	- `load_graphs_on_gpu`

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
