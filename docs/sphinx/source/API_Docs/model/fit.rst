##############
fit
##############


.. py:class:: Fit(object)


   .. note::
   
      See https://jzhang-github.github.io/AGAT/Tutorial/Train_AGAT_model.html to know how to train an AGAT model.

      The first value of ``gat_node_dim_list`` is the depth of atomic representation, which will be modified if you specify a wrong value.

      The first value of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` is the input dimension and equals to last value of ``gat_node_list * num_heads``. The correct values will also be assigned.

      The last values of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` are ``1``, ``3``, and ``6``, respectively. The correct values will also be assigned.

   Example::
      
      from agat.model import Fit
      f = Fit()
      f.fit()



   .. py:method:: __init__(self, **train_config)
   
      Initialize parameters and model.
      
      :param dict train_config: Configuration file for building database. See https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-train-configfor more details.


   .. py:method:: fit(self, **train_config)
   
      Fit the :class:`PotentialModel`. You can modify some settings here by input keyword arguements or dictionary.
      
      .. note:: Some settings are already used in the :meth:`Fit.__init__` method, so input arguements here will not change them.