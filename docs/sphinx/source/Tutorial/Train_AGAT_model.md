# Train AGAT model

### Python script
```python
from agat.model import Fit

train_config = {
    'verbose': 1, # `0`: no train and validation output; `1`: Validation and test output; `2`: train, validation, and test output.
    'dataset_path': os.path.join('dataset', 'all_graphs.bin'),
    'model_save_dir': 'agat_model',
    'epochs': 1000,
    'output_files': 'out_file',
    'device': 'cuda:0',
    'validation_size': 0.15,
    'test_size': 0.15,
    'early_stop': True,
    'stop_patience': 300,
    'head_list': ['mul', 'div', 'free'],
    'gat_node_dim_list': [len(default_elements), 100, 100, 100],
    'energy_readout_node_list': [len(head_list)*gat_node_dim_list[-1], 100, 50, 30, 10, 3, FIX_VALUE[0]],
    'force_readout_node_list': [len(head_list)*gat_node_dim_list[-1], 100, 50, 30, 10, FIX_VALUE[1]],
    'stress_readout_node_list': [len(head_list)*gat_node_dim_list[-1], 100, 50, 30, 10, FIX_VALUE[2]],
    'bias': True,
    'negative_slope': 0.2,
    'criterion': nn.MSELoss(),
    'a': 1.0,
    'b': 1.0,
    'c': 0.0,
    # 'optimizer': 'adam',
    'learning_rate': 0.0001,
    'weight_decay': 0.0, # weight decay (L2 penalty)
    'batch_size': 64,
    'val_batch_size': 400,
    'transfer_learning': False,
    'trainable_layers': -4,
    'mask_fixed': False,
    'tail_readout_no_act': [3,3,3],
    # 'adsorbate': False, #  or not when building graphs.
    'adsorbate_coeff': 20.0 # indentify and specify the importance of adsorbate atoms with respective to surface atoms. zero for equal importance.
    }

f = Fit(**train_config)
f.fit()
```

**See [docs/sphinx/source/Default parameters.md#default_train_config](https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md#default_train_config) to know how to use the parameter settings.**

### Output

The file structure:

```console
.
├── agat_model
│   ├── agat_model.json
│   ├── agat.pth
│   └── agat_state_dict.pth
├── dataset
│   ├── all_graphs.bin
│   ├── fname_prop.csv
│   └── graph_build_scheme.json
├── fit.log
├── out_file
│   ├── energy_test_pred_true.txt
│   ├── energy_val_pred_true.txt
│   ├── force_test_pred_true.txt
│   ├── force_val_pred_true.txt
│   ├── stress_test_pred_true.txt
│   └── stress_val_pred_true.txt
└── train.py
```

| Folder/File | File | Explanation |
| ------ | ---- | ----------- |
| `agat_model` | ─── | A directory for saving well-trained model. |
|  ├──      | `agat_model.json` | An information file tells you how to build an AGAT model. | 
|  ├──      | `agat.pth` | The saved AGAT model including model structure and parameters. | 
|  └──      | `agat_state_dict.pth` | Model and optimizer state dict file including model parameters only. You will need to construct a model or optimizer before using this file. | 
| `dataset` | ─── | A directory for the database. |
| ├──       | `all_graphs.bin` | Binary file of the DGL graphs | 
| ├──       | `fname_prop.csv` | A file storing the structural file name, properties, and paths. This file will not be used in the training, but is useful for checking the raw data. | 
| └──       | `graph_build_scheme.json` | An information file tells you how to build the database. When deploying the well-trained model, this file is useful to construct new graphs. | 
| `fit.log`  |  | The training log file. The `train_config['verbose']` controls the verbosity. |
| `out_file` | ─── | A directory to store ouputs of true and predicted properties. Folder name specified by `train_config['output_files']`. | 
| ├── | `energy_test_pred_true.txt` | Predicted and true energy on the test dataset. |
| ├── | `energy_val_pred_true.txt` | Predicted and true energy on the validation dataset. |
| ├── | `force_test_pred_true.txt` | Predicted and true force on the test dataset. |
| ├── | `force_val_pred_true.txt` | Predicted and true force on the validation dataset. |
| ├── | `stress_test_pred_true.txt` | Predicted and true stress on the test dataset. |
| └── | `stress_val_pred_true.txt` | Predicted and true stress on the validation dataset. |
| `train.py`  |  | The training script. |  
