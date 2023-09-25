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
Please navigate to the [Get Started](https://www.dgl.ai/pages/start.html) page of [dgl](https://www.dgl.ai/). For GPU version:   
  ```console
  conda install -c dglteam/label/cu118 dgl
  ```

For now, the cpu version 1.1.2 of dgl has bugs. You can install the cpu version with `pip install dgl==1.1.1`.

- Change [dgl backend](https://docs.dgl.ai/en/1.1.x/install/#working-with-different-backends) to `tensorflow`.
  If you have cannot use `tensorflow` backend `dgl`, run the following on Linux OS:
  ```console
  wget https://data.dgl.ai/wheels/cu118/dgl-1.1.1%2Bcu118-cp310-cp310-manylinux1_x86_64.whl
  pip install ./dgl-1.1.1+cu118-cp310-cp310-manylinux1_x86_64.whl
  pip install numpy --upgrade
  
  ```

- For GPU, if you don't have CUDA and CUDNN on your device. You need to run (Linux OS):
   ```cousole
   conda install -c conda-forge cudatoolkit=11.8.0
   pip install nvidia-cudnn-cu11==8.6.0.163
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   # Verify install:
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Refer to [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#linux) and [Tensorflow_GPU](https://www.tensorflow.org/install/source#gpu) for more details (other OSs).

   **Attention: Please use the correct cuda and cudnn to avoid unnecessary issues.**