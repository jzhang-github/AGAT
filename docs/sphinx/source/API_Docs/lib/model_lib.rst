model_lib
#########


.. py:function:: save_model(model, model_save_dir='agat_model')

   Saving PyTorch model to the disk. Save PyTorch model, including parameters and structure. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

   :param model: A PyTorch-based model.
   :type model: PyTorch-based model.
   :param model_save_dir: A directory to store the model, defaults to 'agat_model'
   :type model_save_dir: str, optional
   :output: A file saved to the disk under ``model_save_dir``.
   :outputtype: A file.



.. py:function:: load_model(model_save_dir='agat_model', device='cuda')

   Loading PyTorch model from the disk.

   :param model_save_dir: A directory to store the model, defaults to 'agat_model'
   :type model_save_dir: str, optional
   :param device: Device for the loaded model, defaults to 'cuda'
   :type device: str, optional
   :return: A PyTorch-based model.
   :rtype: PyTorch-based model.









.. py:function:: save_state_dict(model, state_dict_save_dir='agat_model', **kwargs)

   Saving state dict (model weigths and other input info) to the disk. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

   :param model: A PyTorch-based model.
   :type model: PyTorch-based model.
   :param state_dict_save_dir: A directory to store the model state dict (model weigths and other input info), defaults to 'agat_model'
   :type state_dict_save_dir: str, optional
   :param **kwargs: More information you want to save.
   :type **kwargs: kwargs
   :output: A file saved to the disk under ``model_save_dir``.
   :outputtype: A file






.. py:function:: load_state_dict(state_dict_save_dir='agat_model')

   Loading state dict (model weigths and other info) from the disk. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

   :param state_dict_save_dir:  A directory to store the model state dict (model weigths and other info), defaults to 'agat_model'
   :type state_dict_save_dir: str, optional
   :return: State dict.
   :rtype: dict

   .. note::
      Reconstruct a model/optimizer before using the loaded state dict.

      Example::

         model = PotentialModel(...)
         model.load_state_dict(checkpoint['model_state_dict'])
         new_model.eval()
         model = model.to(device)
         model.device = device
         optimizer = ...
         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




.. py:function:: config_parser(config)

   Parse the input configurations/settings.

   :param config: configurations
   :type config: str/dict. if str, load from the json file.
   :raises TypeError: DESCRIPTION
   :return: TypeError('Wrong configuration type.')
   :rtype: TypeError



.. py:class:: EarlyStopping

   Stop training when model performance stop improving after some steps.

   .. py:method:: __init__(self, model, graph, logger, patience=10, folder='files')
   
      :param model: AGAT model
      :type model: torch.nn
      :param logger: I/O file
      :type logger: _io.TextIOWrapper
      :param patience: Stop patience, defaults to 10
      :type patience: int, optional
      :param model_save_dir: A directory to save the model, defaults to 'model_save_dir'
      :type model_save_dir: str, optional

   .. py:property:: model
   
      AGAT model.
   
   
   .. py:property:: patience
   
      Patience steps.
   
   
   .. py:property:: counter
   
      Number of steps since last improvement of model performance.
   
   
   .. py:property:: best_score
   
      Best model performance.
   
   
   
   .. py:property:: update 
   
      Update state.
   
   
   .. py:property:: early_stop
   
      Stop training if this variable is true.
   
   

   

   
   .. py:method:: step(self, score, model, optimizer)
   
      :param float score: metrics of model performance
      :param agat model: AGAT model object.
      :param optimizer optimizer: pytorch adam optimizer.
      
   
   .. py:method:: save_model(self, model)
   
      Saves model when validation loss decrease.
      
      :param agat model: AGAT model object.
      


.. py:function:: load_graph_build_method(path)

   Load graph building scheme. This file is normally saved when you build your dataset.

   :param path: Path to ``graph_build_scheme.json`` file.
   :type path: str
   :return: A dict denotes how to build the graph.
   :rtype: dict
   
   
   
   
   
   
.. py:function:: PearsonR(y_true, y_pred)

   Calculating the Pearson coefficient.

   :param y_true: The first torch.tensor.
   :type y_true: torch.Tensor
   :param y_pred: The second torch.tensor.
   :type y_pred: torch.Tensor
   :return: Pearson coefficient
   :rtype: torch.Tensor

   .. Note::

       It looks like the ``torch.jit.script`` decorator is not helping in comuputing large ``torch.tensor``, see ``agat/test/tesor_computation_test.py`` in the GitHub page for more details.

