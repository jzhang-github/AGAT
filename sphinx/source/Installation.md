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
Please navigate to the [Get Started](https://www.dgl.ai/pages/start.html) page of [dgl](https://www.dgl.ai/). For example:   
```console
conda install -c dglteam/label/cu118 dgl
```
For now, the cpu version 1.1.2 of dgl has bugs. You can install the cpu version with `pip install dgl==1.1.1`.

- Change [dgl backend](https://docs.dgl.ai/en/1.1.x/install/#working-with-different-backends) to `tensorflow`.
