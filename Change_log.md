# [main](https://github.com/jzhang-github/AGAT/)
- Raise exception if error occurs when parsing OUTCAR file. [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/main/agat/data/data.py#L594-L598)
- Fix a bug when build graphs. See [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/main/agat/data/data.py#L400-L411) and [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/main/agat/data/data.py#L444-L452). Specifically, one needs to cast `tf.tensor` as `np.array` before building graph properties with a very large tensor. [agat/data/data.py](https://github.com/jzhang-github/AGAT/tree/main/agat/data/data.py#L407-L408).

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

