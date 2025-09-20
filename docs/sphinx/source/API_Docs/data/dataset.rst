#############
dataset
#############

.. Hint:: The objects in this page help manipulating dataset(s).

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

      .. Tips::
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



.. py:class:: DCollater(object)

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
