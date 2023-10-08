##############
layer
##############

Single graph attention network for predicting crystal properties.

.. Note:: Some abbreviations used in :class:`Layer` class:

   ===============  =================
   Abbreviations    Full name
   ===============  =================
   dist             distance matrix
   feat             features
   ft               features
   src              source node
   dst              destination node
   e                e_i_j: refer to: https://arxiv.org/abs/1710.10903
   a                alpha_i_j: refer to: https://arxiv.org/abs/1710.10903
   att              attention mechanism
   act              activation function
   ===============  =================


.. class:: Layer()

   .. method:: __init__(self, in_dim, out_dim, num_heads, device='cuda', bias=True, negative_slope=0.2)
   
      :param int in_dim: Depth of node representation in the input of this AGAT `Layer`.
      :param int out_dim: Depth of node representation in the output of this `GAT` layer.
      :param int num_heads: Number of attention heads.
      :param str device: Device to perform tensor calculations and store parameters.
      :param bool bias: Whether the dense layer uses a bias vector.
      :param float negative_slope: Negative slope coefficient of the LeakyReLU activation function.



   .. method:: forward(self, feat, dist, graph)
   
      Forward this AGAT `Layer`.
      
      :param torch.tensor feat: Input features of all nodes (atoms).
      :param torch.tensor dist: Distances between connected atoms.
      :param DGL.graph graph: A graph built with DGL.
      
      :Returns: dst: output features of all nodes.
      :rtype dst: torch.tensor



