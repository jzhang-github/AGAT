
# Customized installation of AGAT environment

Examined dependency compatibility:

| OS          | Python    | DGL   | PyTorch | numpy | ASE    | CUDA | MKL    |
| ----------- | --------- | ----- | ------- | ----- | ------ | ---- | ------ |
| Windows CPU | 3.10/3.12 | 2.2.1 | 2.3.0   | 2.0.1 | 3.23.0 | -    |        |
| Linux GPU   | 3.12      | 2.4.0 | 2.4.1   | 2.1.3 | 3.23.0 | 12.4 | 2024.0 |



## Install with [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment on Linux.

Environmental variables may lead to errors, check your `~/.bashrc` file first.

- Create a new environment

  ```bash
  conda create -n agat python==3.12
  ```

  

- Activate the environment

  ```bash
  conda activate agat
  ```

  

- Install [PyTorch](https://pytorch.org/),
  Navigate to the [installation page](https://pytorch.org/get-started/locally/#start-locally) and choose your platform. You may need [previous versions](https://pytorch.org/get-started/previous-versions/). For example (GPU):

  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 # recommended 
  ```

  or

  ```bash
  conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia # not recommended.
  ```

  

- You may need to downgrade the `MKL` package [optional]:

  ```bash
  conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge mkl==2024.0
  ```

  

- Install [dgl](https://www.dgl.ai/).
  Please navigate to the [Get Started](https://www.dgl.ai/pages/start.html) page of [dgl](https://www.dgl.ai/). For example (GPU):

  ```bash
  conda install -c dglteam/label/th24_cu124 dgl # recommended
  ```

  or

  ```bash
  pip install  dgl -f https://data.dgl.ai/wheels/cu124/repo.html
  ```

  **Note:** `DGL` requires these packages: `pip install packaging pandas pydantic pyyaml`

  

- Install AGAT package

  ```bash
  pip install agat
  ```

  

- Install CUDA and CUDNN [**Optional**].

  - For HPC with Linux OS, you may load CUDA by checking `module av`, or you can contact your administrator for help.
  - Or download manually:
    - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN](https://developer.nvidia.com/cudnn)

  

- Install other dependencies, such as `pandas`.



### Install with [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment on Windows.

The `DGL` package has limited compatibility with Windows, particularly for CUDA versions. You are highly recommended to use the CPU version on Windows.

- Create a new environment   
  ```console
  conda create -n agat python==3.10
  ```

- Activate the environment  
  ```console
  conda activate agat
  ```

- Install [PyTorch](https://pytorch.org/),   
  Navigate to the [installation page](https://pytorch.org/get-started/locally/#start-locally) and choose your platform.
  For example (GPU):
  
  ```console
  conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch
  ```
  
- Install packaging: `pip install packaging`
  
- Install [dgl](https://www.dgl.ai/).   
  
  ```console
  
  ```
  
- Install ASE: `pip install ase`.

- Install AGAT package

  ```console
  pip install agat
  ```



### Check installation

```python
import torch
print('GPU avaliability:', torch.cuda.is_available())
print('Number of GPU cards:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Current GPU device:', torch.cuda.current_device())
    print('First device localtion:', torch.cuda.device(0))
    print('First device name:', torch.cuda.get_device_name(0))

import dgl
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
u, v = torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = torch.randn(5, 3)  # original feature is on CPU
print(g.device)
cuda_g = g.to(device)  # accepts any device objects from backend framework
print(cuda_g.device)
print(cuda_g.ndata['x'].device)       # feature data is copied to GPU too

# A graph constructed from GPU tensors is also on GPU
u, v = u.to(device), v.to(device)
g = dgl.graph((u, v))
print(g.device)
```

