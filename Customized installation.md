
# Customized installation of AGAT environment


### Install with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment

- Create a new environment   
  ```console
  conda create -n agat python==3.12
  ```

- Activate the environment  
  ```console
  conda activate agat
  ```

- Install [PyTorch](https://pytorch.org/),   
  Navigate to the [installation page](https://pytorch.org/get-started/locally/#start-locally) and choose you platform.
  For example (GPU):
  ```console
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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
	- For HPC with Linux OS, you may load CUDA by checking `module av`, or you can contact your administrator for help.
	- Or download manually:
	  - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
	  - [cuDNN](https://developer.nvidia.com/cudnn)

