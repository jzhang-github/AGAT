##############
model
##############

.. Note:: You can customize the :class:`PotentialModel` to train and predict atom and bond related properties. You need to store the labels on graph edges if you want to do so. This model has multiple attention heads.

.. py:class:: PotentialModel(nn.Module)

   A GAT model with multiple gat layers for predicting atomic energies, forces, and stress tensors.


   .. Important::

      The first value of ``gat_node_dim_list`` is the depth of atomic representation.

      The first value of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` is the input dimension and equals to last value of ``gat_node_list * num_heads``.

      The last values of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` are ``1``, ``3``, and ``6``, respectively.



   .. py:method:: __init__(self, gat_node_dim_list, energy_readout_node_list, force_readout_node_list, stress_readout_node_list, head_list=['div'], bias=True, negative_slope=0.2, device = 'cuda', tail_readout_no_act=[3,3,3])

      :param gat_node_dim_list: A list of node dimensions of the AGAT ``Layer``.
      :type gat_node_dim_list: list
      :param energy_readout_node_list: A list of node dimensions of the energy readout layers.
      :type energy_readout_node_list: list
      :param force_readout_node_list: A list of node dimensions of the force readout layers.
      :type force_readout_node_list: list
      :param stress_readout_node_list: A list of node dimensions of the stress readout layers.
      :type stress_readout_node_list: list
      :param head_list: A list of attention head names, defaults to ['div']
      :type head_list: list, optional
      :param bias: Add bias or not to the neural networks., defaults to True
      :type bias: TYPE, bool
      :param negative_slope: This specifies the negative slope of the LeakyReLU (see https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) activation function., defaults to 0.2
      :type negative_slope: float, optional
      :param device: Device to train the model. Use GPU cards to accerelate training., defaults to 'cuda'
      :type device: str, optional
      :param tail_readout_no_act: The tail ``tail_readout_no_act`` layers will have no activation functions. The first, second, and third elements are for energy, force, and stress readout layers, respectively., defaults to [3,3,3]
      :type tail_readout_no_act: list, optional
      :return: An AGAT model
      :rtype: agat.model.PotentialModel

   .. py:method:: mul(self, TorchTensor)

      Multiply head.

      :param TorchTensor: Input tensor
      :type TorchTensor: torch.tensor
      :return: Ouput tensor
      :rtype: torch.tensor



   .. py:method:: div(self, TorchTensor)

      Division head.

      :param TorchTensor: Input tensor
      :type TorchTensor: torch.tensor
      :return: Ouput tensor
      :rtype: torch.tensor



   .. py:method:: div(self, TorchTensor)
      :no-index:

      Free head.

      :param TorchTensor: Input tensor
      :type TorchTensor: torch.tensor
      :return: Ouput tensor all ones
      :rtype: torch.tensor




   
   .. py:method:: get_head_mechanism(self, fn_list, TorchTensor)
      
      Get attention heads
      
      :param fn_list: A list of head mechanisms. For example: ['mul', 'div', 'free']
      :type fn_list: list
      :param TorchTensor: A PyTorch tensor
      :type TorchTensor: torch.Tensor
      :return: A new tensor after the transformation.
      :rtype: torch.Tensor







   
   .. py:method:: forward(self, graph)
      
      The ``forward`` function of PotentialModel model.

      :param graph: ``DGL.Graph``
      :type graph: ``DGL.Graph``
      :return:
         - energy: atomic energy
         - force: atomic force
         - stress: cell stress tensor

      :rtype: tuple of torch.tensors



