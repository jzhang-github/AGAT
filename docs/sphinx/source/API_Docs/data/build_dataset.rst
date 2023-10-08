#############
build_dataset
#############

The high-level API in this script is :py:class:`BuildDatabase`.

For example::

   database = BuildDatabase()
   database.build()

See https://jzhang-github.github.io/AGAT/Tutorial/Build_database.html for more info.

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

      :param **data_config: Configuration file for building database. See https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md#default_data_config for the detailed info.
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
      




      
.. py:class:: ReadGraphs()

   This object is used to build a list of graphs.


   .. py:method:: __init__(self, **data_config)
   
      :param dict **data_config: Configuration file for building database. See https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md#default_data_config for the detailed info.

         .. Hint:: 
            Mode of how to get the neighbors, which can be:
            
            - ``'voronoi'``: consider Voronoi neighbors only.
            - ``'pymatgen_dist'``: build graph based on a constant distance using ``pymatgen`` module.
            - ``'ase_dist'``: build graph based on a constant distance using ``ase`` module.
            - ``'ase_natural_cutoffs'``: build graph from ``ase`` which has a dynamic cutoff scheme. In this case, the ``cutoff`` is deprecated because ``ase`` will use the dynamic cutoffs in ``ase.neighborlist.natural_cutoffs()``.
            
   
   .. py:method:: read_batch_graphs(self, batch_index_list, batch_num)
   
      Read graphs with batches. 
      
      .. Note:: The loaded graphs are saved under the attribute of :py:attr:`dataset_path`.
      
      :param list batch_index_list: a list of graph index.
      :param str batch_num: number the graph batches.
      
      
   .. py:method:: read_all_graphs(self, scale_prop=False, ckpt_path='.')
   
      Read all graphs specified in the csv file.
      
      .. Note:: The loaded graphs are saved under the attribute of :py:attr:`dataset_path`.
      
      .. DANGER:: Do not scale the label if you don't know what are you doing.
      
      :param bool scale_prop: scale the label or not. DO NOT scale unless you know what you are doing.
      :param str ckpt_path: checkpoint directory of the well-trained model.
      :Returns: 
         - graph_list： a list of ``DGL`` graph.
         - graph_labels： a list of labels.








.. py:class:: TrainValTestSplit(object)

   Split the dataset.

   .. Note:: This object is deprecated.





.. py:class:: ExtractVaspFiles(object)

   Extract VASP outputs for building AGAT database.
   
   :param data_config['dataset_path']: Absolute path where the collected data to save.
   :type data_config['dataset_path']: str

   .. Note:: Always save the property per node as the label. For example: energy per atom (eV/atom).

   .. method:: __init__(self, **data_config)
   
      :param dict **data_config: Configuration file for building database. See https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md#default_data_config for the detailed info.

   .. py:method:: read_oszicar(self,fname='OSZICAR')
   
      Get the electronic steps of a VASP run.

      :param fname: file name, defaults to 'OSZICAR'
      :type fname: str, optional
      :return: electronic steps of a VASP run.
      :rtype: list

   .. py:method: read_incar(self, fname='INCAR')
   
      Get the NELM from INCAR. NELM: maximum electronic steps for each ionic step.

      :param fname: file name, defaults to 'INCAR'
      :type fname: str, optional
      :return: NELM tage in INCAR
      :rtype: int


   .. py:method:: split_output(self, process_index)
   
      :param process_index: A number to index the process.
      :type process_index: int.


   .. py:method:: __call__(self)
   
      The __call__ function




.. py:class:: BuildDatabase()

   Build a database. Detailed information: https://jzhang-github.github.io/AGAT/Tutorial/Build_database.html
   
   .. method:: __init__(self, **data_config)
   
      :param dict **data_config: Configuration file for building database. See https://github.com/jzhang-github/AGAT/blob/main/docs/sphinx/source/Default%20parameters.md#default_data_config for the detailed info.

   .. py:method:: build(self)
   
      Run the construction process.



.. py:function:: concat_graphs(*list_of_bin)

   Concat binary graph files.

   :param *list_of_bin: input file names of binary graphs.
   :type *list_of_bin: strings
   :return: A new file is saved to the current directory: concated_graphs.bin.
   :rtype: None. A new file.
  
   Example::
  
       concat_graphs('graphs1.bin', 'graphs2.bin', 'graphs3.bin')





   
   
..
 External links are list below:
.. _pymatgen.core.structure: https://pymatgen.org/pymatgen.core.structure.html
.. _ase.atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
.. _ase: https://wiki.fysik.dtu.dk/ase/