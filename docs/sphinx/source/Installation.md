# Installation

### 1. Install with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment configuration file

- Download the [`agat_linux_gpu_cu124.yml`](https://github.com/jzhang-github/AGAT/blob/main/agat_linux_gpu_cu124.yml)  or [`agat_win_cpu.yml`](https://github.com/jzhang-github/AGAT/blob/main/agat_win_cpu.yml) file.

- Run

  ```shell
  conda env create -f agat_linux_gpu_cu124.yml
  ```

  or

  ```shell
  conda env create -f agat_win_cpu.yml
  ```




- Run `conda env list` to check installed environments.



### 2. Install manually with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

- Create a new environment   
  ```console
  conda create -n agat python==3.12
  ```

- Activate the environment  
  ```console
  conda activate agat
  ```

- Install [PyTorch](https://pytorch.org/)

  Navigate to the [installation page](https://pytorch.org/get-started/locally/#start-locally) and choose you platform.
  For example (GPU):

  ```console
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

- Install [dgl](https://www.dgl.ai/).   
  Please navigate to the [Get Started](https://www.dgl.ai/pages/start.html) page of [dgl](https://www.dgl.ai/). 
  For example (GPU):  

  ```console
  pip install  dgl -f https://data.dgl.ai/wheels/cu124/repo.html
  ```

- Install AGAT package  
  ```console
  pip install agat
  ```

- Install CUDA and CUDNN [**Optional**].
	- For HPC, you may load CUDA by checking `module av`, or you can contact your administrator for help.
	- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
	- [cuDNN](https://developer.nvidia.com/cudnn)
