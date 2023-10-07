Tutorial
============

Default
---------------

By default, the database construction, model training, and model application are controled by the `Default parameters <https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md>`_ .


For example:: python

   # Build database
   from agat.data import BuildDatabase
   if __name__ == '__main__':
       database = BuildDatabase()
       database.build()

   # Train AGAT model
   from agat.model import Fit
   f = Fit()
   f.fit()

   # Application (high-throughput prediction)
   from agat.app.cata import HpAds
   ha = HpAds(model_save_dir=model_save_dir, graph_build_scheme_dir=graph_build_scheme_dir)
   ha.run(formula='NiCoFePdPt')


As you can see that, you only need to provide very few input arguments to instantiate the imported objects.


Customized
---------------

There are generally two ways to customize your AGAT training and deployment. Tacking the training as an example:

Customized input arguements
****************************


.. code-block:: python
   from agat.model import Fit
   f = Fit(verbose=2, gat_node_dim_list=[6, 10, 10], b=10.0)
   f.fit()



This will replace the seetings for ``verbose``, ``gat_node_dim_list``, ``b``. Other parameters are controled by the `Default parameters <https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md>`_ .


Customized input dict
****************************


.. code-block:: python
   import os
   import torch.nn as nn
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
       'gat_node_dim_list': [6, 100, 100, 100],
       'head_list': ['mul', 'div', 'free'],
       'energy_readout_node_list': [300, 300, 100, 50, 30, 10, 3, 1],
       'force_readout_node_list': [300, 300, 100, 50, 30, 10, 3],
       'stress_readout_node_list': [300, 300, 6],
       'bias': True,
       'negative_slope': 0.2,
       'criterion': nn.MSELoss(),
       'a': 1.0,
       'b': 1.0,
       'c': 0.0,
       'optimizer': 'adam', # Fix to sgd.
       'learning_rate': 0.0001,
       'weight_decay': 0.0, # weight decay (L2 penalty)
       'batch_size': 64,
       'val_batch_size': 400,
       'transfer_learning': False,
       'trainable_layers': -4,
       'mask_fixed': False,
       'tail_readout_no_act': [3,3,3],
       'adsorbate_coeff': 20.0 # indentify and specify the importance of adsorbate atoms with respective to surface atoms. zero for equal importance.
       }
   f = Fit(**train_config)
   f.fit()



The input ``train_config`` has higher priority than the `Default parameters <https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md>`_ .


.. toctree::
   :maxdepth: 6

   Tutorial/VASP calculations
   Tutorial/Build database
   Tutorial/Train AGAT model
   Tutorial/Structural optimization
   Tutorial/High-throughput prediction


