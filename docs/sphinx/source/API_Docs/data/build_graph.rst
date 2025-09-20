#############
build_graph
#############



.. py:class:: CrystalGraph(object)

   Read structural file and return a graph.

   .. Caution:: The constructed crystal graph may be unreasonable for high-entropy materials, if the connections are analyzed by Voronoi method.

   Code example::

      from Crystal2raph import CrystalGraph
      cg = CrystalGraph(cutoff = 6.0, mode_of_NN='distance', adsorbate=True)
      cg.get_graph('POSCAR')

   .. Hint::
       Although we recommend representing atoms with one hot code, you can
       use the another way with: ``self.all_atom_feat = get_atomic_features()``

   .. Hint::
       In order to build a reasonable graph, a samll cell should be repeated.
       One can modify "self._cell_length_cutoff" for special needs.

   .. Hint::
       We encourage you to use ``ase`` module to build crystal graphs.
       The ``pymatgen`` module needs some dependencies that conflict with
       other modules.


   .. py:method:: __init__(self, **data_config)

      :param **data_config: Configuration file for building database. See https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-data-config for the detailed info.
      :type **data_config: str/dict
      :return: A ``DGL.graph``.
      :rtype: ``DGL.graph``.

      .. Hint::
         Mode of how to get the neighbors, which can be:

         - ``'voronoi'``: consider Voronoi neighbors only.
         - ``'pymatgen_dist'``: build graph based on a constant distance using ``pymatgen`` module.
         - ``'ase_dist'``: build graph based on a constant distance using ``ase`` module.
         - ``'ase_natural_cutoffs'``: build graph from ``ase`` which has a dynamic cutoff scheme. In this case, the ``cutoff`` is deprecated because ``ase`` will use the dynamic cutoffs in ``ase.neighborlist.natural_cutoffs()``.

     :param bool adsorbate: Identify the adsorbate or not.


   .. py:method:: get_adsorbate_bool(self, element_list)

      Identify adsorbates based on elements： H and O.

      :param list element_list: a list of element symbols.
      :return: a list of bool values.
      :rtype: torch.tensor


   .. py:method:: get_crystal(self, crystal_fpath, super_cell=True)

      Read structural file and return a pymatgen crystal object.

      :param str crystal_fpath: the path to the crystal structural.
      :param bool super_cell: repeat the cell or not.
      :return: a pymatgen structure object.
      :rtype: ``pymatgen.core.structure``.


   .. py:method:: get_1NN_pairs_voronoi(self, crystal)

      The ``get_connections_new()`` of ``VoronoiConnectivity`` object is modified.

      :param pymatgen.core.structure crystal: a pymatgen structure object.
      :Returns:
         - index of senders
         - index of receivers
         - a list of distance between senders and receivers


   .. py:method:: get_1NN_pairs_distance(self, crystal)

      Find the index of senders, receivers, and distance between them based on the ``distance_matrix`` of pymargen crystal object.

      :param pymargen.core.structure crystal: pymargen crystal object
      :Returns:
         - index of senders
         - index of receivers
         - a list of distance between senders and receivers

   .. py:method:: get_1NN_pairs_ase_distance(self, ase_atoms)

      :param ase.atoms ase_atoms: ``ase.atoms`` object.
      :Returns:
         - index of senders
         - index of receivers
         - a list of distance between senders and receivers


   .. py:method:: get_ndata(self, crystal)

      :param pymargen.core.structure crystal: a pymatgen crystal object.
      :return: ndata: the atomic representations of a crystal graph.
      :rtype: numpy.ndarray


   .. py:method:: get_graph_from_ase(self, fname, include_forces=False)

      Build graphs with ``ase``.

      :param str/``ase.Atoms`` fname: File name or ``ase.Atoms`` object.
      :param bool include_forces: Include forces into graphs or not.
      :return: A bidirectional graph with self-loop connection.


   .. py:method:: get_graph_from_pymatgen(self, crystal_fname, super_cell=True, include_forces=False)

      Build graphs with pymatgen.

      :param str crystal_fname: File name.
      :param bool super_cell: repeat small cell or not.
      :param bool include_forces: Include forces into graphs or not.
      :return: A bidirectional graph with self-loop connection.


   .. py:method:: get_graph(self, crystal_fname, super_cell=False, include_forces=True)

      This method can choose which graph-construction method is used, according to the ``mode_of_NN`` attribute.

      .. Hint:: You can call this method to build one graph.

      :param str crystal_fname: File name.
      :param bool super_cell: repeat small cell or not.
      :param bool include_forces: Include forces into graphs or not.
      :return: A bidirectional graph with self-loop connection.


.. py:class:: AseGraphTorch(object)

   Build ``DGL`` graphs with ``ASE`` and ``Torch`` module. A GPU card can be used for a large system, or for simulations of geometry optimization and molecular dynamics.

   .. Note:: During the geometry optimization or MD simulation, connections among atoms may not change with several ionic steps. Thus, calculating neighboring relationships for every ionic step is high-cost. This class aims to reduce such cost and improve speed.


   .. py:method:: __init__(self, **data_config)

      :param **data_config: Configuration file for building database. See https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-data-config for the detailed info.
      :type **data_config: str/dict


   .. py:method:: reset(self)

      Reset parameters for building graph from sctrach.

   .. py:method:: get_ndata(self, ase_atoms)

      Get node data (atomic features)

      :param ase_atoms: ``ase.Atoms``
      :param type: ``ase.Atoms``
      :return: atomic input features.
      :rtype: ``torch.Tensor``

   .. py:method:: get_adsorbate_bool(self, element_list)

      Identify adsorbates based on elements： H and O.

      :return: a list of bool values.
      :rtype: tf.constant

   .. py:method:: get_scaled_positions_wrap(self, cell_I_tensor, positions)

      Get the scaled atomic positions and wrap into the origional cell.

      :param torch.Tensor cell_I_tensor: cell array in the torch.Tensor format.
      :param torch.Tensor positions: atomic positions.

   .. py:method:: get_scaled_positions(self, cell_I_tensor, positions)

      Get the scaled atomic positions and DO NOT wrap into the origional cell.

      :param torch.Tensor cell_I_tensor: cell array in the torch.Tensor format.
      :param torch.Tensor positions: atomic positions.

   .. py:method:: fractional2cartesian(self, cell_tensor, scaled_positions)

      Convert scaled (fractional) coordinates to the cartesian coordinates.

      :param torch.Tensor cell_tensor: Simulation cell array.
      :param torch.Tensor scaled_positions: Scaled atomic positions.

   .. py:method:: safe_to_use(self, ase_atoms, critical=0.01)

      For small simulation cells or non-cubic cells, this class may lead to unwanted results.

      :param ase.Atoms ase_atoms: ASE atoms.
      :param float critical: Critical value for determing a cubic cell or not.
      :return: Safe or not.
      :rtype: bool


   .. py:method:: get_pair_distances(self, a, b, ase_atoms)

      Get distance between two atoms.

      :param int a: The index of first atom.
      :param int b: The index of second atom.
      :param ase.Atoms ase_atoms: ase atoms in RAM.
      :return: d
      :rtype: float
      :return: D
      :rtype: torch.Tensor

   .. py:method:: update_pair_distances(self, a, b, b_image, ase_atoms)

      Update distance between two atoms.

      :param int a: The index of first atom.
      :param int b: The index of second atom.
      :param int b_image: Clearly, I fogot things.
      :param ase.Atoms ase_atoms: ase atoms in RAM.
      :return: d
      :rtype: float
      :return: D
      :rtype: torch.Tensor

   .. py:method:: get_all_possible_distances(self, ase_atoms)

      .. Note:: Get senders and receivers, including inner and skin connections.
         Torch.from_numpy is memory effcient than torch.tensor, especially for large tensors.
         No self loop and reverse direction.

      Get all possible connections, not every pair of atoms will be calculated in order to improve the efficiency.

      :param ase.Atoms ase_atoms: ASE atoms in RAM.


   .. py:method:: get_init_connections(self, ase_atoms)

      Get connection from scratch.

      :param ase.Atoms ase_atoms: ASE atoms in RAM.

      :return: Return connections for both real neighbors and potential neighbors defined by the skin thickness.

   .. py:method:: update_connections(self, i_i, j_i, j_image_i, i_s, j_s, j_image_s, ase_atoms)

      Update connections instead of calculating every pair distance. Some skin aotm may become real neighbors, while neighbors can go to the skin region (potential neighbor).


   .. py:method:: build(self, ase_atoms)

      :return: a ``DGL`` graph.


      Build graph from scratch?

   .. py:method:: update(self, ase_atoms)

      Build graph from updated connections, not from scratch.

      :return: a ``DGL`` graph.



   .. py:method:: get_graph(self, ase_atoms)

      Get graph from ASE atoms object.

      .. Hint:: This is the high-level API of this class.

      :return: a ``DGL`` graph.

.. Warning:: The owner seems forgot some details. So use ``AseGraphTorch`` with caution.

..
 External links are list below:
.. _pymatgen.core.structure: https://pymatgen.org/pymatgen.core.structure.html
.. _ase.atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
.. _ase: https://wiki.fysik.dtu.dk/ase/
