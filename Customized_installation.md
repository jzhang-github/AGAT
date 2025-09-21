
# Customized installation of AGAT environment



Examined dependency compatibility:

| OS          | Python    | DGL   | PyTorch | numpy | ASE    | CUDA |
| ----------- | --------- | ----- | ------- | ----- | ------ | ---- |
| Windows CPU | 3.10/3.12 | 2.2.1 | 2.3.0   | 2.0.1 | 3.23.0 | -    |
| Linux GPU   | 3.12      | 2.4.0 | 2.4     | 2.0.1 | 3.23.0 | 12.4 |



## Install with [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment on Linux.

- Create a new environment

  ```
  conda create -n agat python==3.12
  ```

- Activate the environment

  ```
  conda activate agat
  ```

- Install [PyTorch](https://pytorch.org/),
  Navigate to the [installation page](https://pytorch.org/get-started/locally/#start-locally) and choose your platform. For example (GPU):

  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

- Install [dgl](https://www.dgl.ai/).
  Please navigate to the [Get Started](https://www.dgl.ai/pages/start.html) page of [dgl](https://www.dgl.ai/). For example (GPU):

  ```
  pip install  dgl -f https://data.dgl.ai/wheels/cu124/repo.html
  ```

- Install AGAT package

  ```
  pip install agat
  ```

- Install CUDA and CUDNN [**Optional**].

  - For HPC with Linux OS, you may load CUDA by checking `module av`, or you can contact your administrator for help.
  - Or download manually:
    - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN](https://developer.nvidia.com/cudnn)





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

