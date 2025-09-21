#############
load_dataset
#############


.. Warning:: This page exists for compatibility with earlier versions. In the future, all manipulations of a dataset should by ``atoms.data.Dataset``


.. py:class:: LoadDataset(Dataset)

   Load the binary graphs.

   Example::

      import os
      from agat.data import LoadDataset
      dataset=LoadDataset(os.path.join('dataset', 'all_graphs.bin'))

      # you can index or slice the dataset.
      g0, props0 = dataset[0]
      g_batch, props = dataset[0:100] # the g_batch is a batch collection of graphs. See https://docs.dgl.ai/en/1.1.x/generated/dgl.batch.html


   .. py:method:: __init__(self, dataset_path)
      :no-index:

      :param dataset_path: A paths leads to the binary DGL graph file.
      :type dataset_path: str
      :return: a graph dataset.
      :rtype: list


   .. py:method:: __getitem__(self, index)

      Index or slice the dataset.

      :param index: list index or slice
      :type index: int/slice
      :return: graph or graph batch
      :rtype: dgl graph
      :return: props. Graph labels
      :rtype: A dict of torch.tensor


   .. py:method:: __len__(self)

      Get the length of the dataset.

      :return: the length of the dataset
      :rtype: int


.. py:class:: Collater(object)
   :no-index:

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
      :no-index:

      :param device: device to store the merged data, defaults to 'cuda'
      :type device: str, optional

   .. py:method:: __call__(self, data)
      :no-index:

      Collate the data into batches.

      :param data: the output of :py:class:`LoadDataset`
      :type data: tuple
      :return: dgl batch graphs. See https://docs.dgl.ai/en/1.1.x/generated/dgl.batch.html
      :rtype: DGLGraph
      :return: Graph labels
      :rtype: A dict of torch.tensor
