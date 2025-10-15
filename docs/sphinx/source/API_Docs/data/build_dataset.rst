#############
build_dataset
#############

.. Hint::

   The high-level API in this script is :py:class:`BuildDatabase`.

   For example::

      database = BuildDatabase()
      database.build()

   See https://jzhang-github.github.io/AGAT/Tutorial/Build_database.html for more info.

.. Warning::

   Some functions on this page will be deprecated in the future. Including ``select_graphs_random`` and ``concat_graphs``. Use ``select_graphs_from_dataset_random`` and ``concat_dataset``, respectively.


.. py:class:: ReadGraphs()

   This object is used to build a list of graphs.


   .. py:method:: __init__(self, **data_config)

      :param dict \*\*data_config: Configuration file for building database. See https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-data-config for the detailed info.

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

      :param dict \*\*data_config: Configuration file for building database. See https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-data-config for the detailed info.

   .. py:method:: read_oszicar(self,fname='OSZICAR')

      Get the electronic steps of a VASP run.

      :param fname: file name, defaults to 'OSZICAR'
      :type fname: str, optional
      :return: electronic steps of a VASP run.
      :rtype: list.

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

      :param dict \*\*data_config: Configuration file for building database. See https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-data-config for the detailed info.

   .. py:method:: build(self)

      Run the construction process.


.. py:function:: concat_graphs(*list_of_bin)

   Concat binary graph files.

   :param \*list_of_bin: input file names of binary graphs.
   :type \*list_of_bin: strings
   :return: A new file is saved to the current directory: concated_graphs.bin.
   :rtype: None. A new file.

   Example::

       concat_graphs('graphs1.bin', 'graphs2.bin', 'graphs3.bin')



.. py:function:: concat_dataset(*list_of_datasets, save_file=False, fname='concated_graphs.bin')

   Concat ``agat.dataset.Dataset`` in the RAM.

   :param \*list_of_datasets: a list of ``agat.dataset.Dataset`` object.
   :type \*list_of_datasets: ``agat.dataset.Dataset``
   :return: A new file is saved to the current directory: concated_graphs.bin.
   :param save_file: save to a new file or not. Default: False
   :type save_file: bool
   :param fname: The saved file name if ``savefile=True``. Default: 'concated_graphs.bin'
   :type fname: str
   :rtype: ``agat.dataset.Dataset``



.. py:function:: select_graphs_random(fname: str, num: int)

   Randomly split graphs from a binary file.

   :param fname: input file name.
   :type fname: str
   :param num: number of selected graphs (should be smaller than number of all graphs.
   :type num: int
   :return: A new file is saved to the current directory: Selected_graphs.bin.
   :rtype: None. A new file.

   Example::

      select_graphs_random('graphs1.bin')

.. py:function:: select_graphs_from_dataset_random(dataset, num: int, save_file=False, fname='selected_graphs.bin')

   Randomly split graphs from a binary file.

   :param fname: input file name.
   :type fname: str
   :param num: number of selected graphs (should be smaller than number of all graphs.
   :type num: int
   :return: A new file is saved to the current directory: Selected_graphs.bin.
   :rtype: None. A new file.

   Example::

      select_graphs_random('graphs1.bin')


.. py:function:: save_dataset(dataset: Dataset, fname='graphs.bin')

   Save a ``agat.dataset.Dataset`` to a binary file.

   :param dataset: AGAT dataset in RAM.
   :type dataset: ``agat.dataset.Dataset``
   :param fname: output file name.
   :type fname: str



..
 External links are list below:
.. _pymatgen.core.structure: https://pymatgen.org/pymatgen.core.structure.html
.. _ase.atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
.. _ase: https://wiki.fysik.dtu.dk/ase/
