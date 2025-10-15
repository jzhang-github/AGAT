#############
dataset
#############

.. Hint:: The objects in this page help manipulating dataset(s).

Examples of manipulating dataset(s):

.. code-block:: python

   from agat.data import Dataset, concat_dataset
   from agat.data import select_graphs_from_dataset_random
   dataset = Dataset('graphs.bin')
   da = dataset[11]
   db = dataset[1:4]
   dc = concat_dataset(da, db)
   dd = select_graphs_from_dataset_random(dataset, 10, save_file=False,
                                          fname=None)
   print(dataset, da, db, dc, dd)
   
   from agat.data import select_graphs_from_dataset_random
   de = select_graphs_from_dataset_random(dd, 3, save_file=False, fname=None)
   print(de)
   
   from agat.data import save_dataset
   save_dataset(dd, fname='new_dataset.bin')



.. py:class:: Dataset(torch.utils.data.Dataset)

   This object is used to build a list of graphs.

   Load the binary graphs.

   Example::

       import os
       from agat.data import LoadDataset
       dataset=LoadDataset(os.path.join('dataset', 'all_graphs.bin'))

       # you can index or slice the dataset.
       g0, props0 = dataset[0]
       g_batch, props = dataset[0:100] # the g_batch is a batch collection of graphs. See https://docs.dgl.ai/en/1.1.x/generated/dgl.batch.html



   .. py:method:: __init__(self, dataset_path=None, from_file=True, graph_list=None, props=None)

      .. Tip::
         You can load a dataset from file or from the RAM.
         From file: ``dataset_path='string_example'`` and ``from_file=True``
         From RAM: Specify ``from_file=False``, and provide ``graph_list`` and ``props``.

      :param dataset_path: A paths leads to the binary DGL graph file.
      :type dataset_path: str
      :param bool from_file: Load from file or not.
      :param list graph_list: A list of graphs.
      :param list props: Properties tensor corresponding to a list of graphs.
      :return: a graph dataset.
      :rtype: list

.. py:method:: __getitem__(self, index)

   Index or slice the dataset.

      :param index: list index or slice
      :type index: int/slice
      :return: Dataset
      :rtype: agat.data.Dataset


.. py:method:: __repr__(self)

   Output if you ``print()`` a dataset.


.. py:method:: __len__(self)

   Output if you ``len()`` a dataset.


.. py:method:: save(self, file='graphs.bin')

   Save the dataset in RAM to the disk.

   :param str file: The output file name.
   :return: None. A file will be saved to the disk.



.. py:class:: Collater(object)

   The collate function used in torch.utils.data.DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

   The collate function determines how to merge the batch data.

   Example::

       import os
       from agat.data import LoadDataset, Collater
       from torch.utils.data import DataLoader

       dataset=LoadDataset(os.path.join('dataset', 'all_graphs.bin'))
       collate_fn = Collater(device='cuda')
       data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

   .. py:method:: __init__(self, device='cuda')

      :param str device: The device for manipulating Dataset(s)


   .. py:method:: __call__(self, data)

      Collate the data into batches.

      :param data: the output of :py:class:`Dataset`
      :type data: AGAT Dataset
      :return: AGAT Dataset with dgl batch graphs. See https://docs.dgl.ai/en/1.1.x/generated/dgl.batch.html
      :rtype: AGAT Dataset



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