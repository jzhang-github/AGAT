# [main](https://github.com/jzhang-github/AGAT/)
- Using relative import. For example: [agat/__init__.py](agat/__init__.py#L14-L16)


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
