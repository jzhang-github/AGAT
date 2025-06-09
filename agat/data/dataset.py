# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:54:56 2023

@author: 18326
"""

import torch
# from torch.utils.data import Dataset
import dgl
from dgl.data.utils import load_graphs, save_graphs

class Dataset(torch.utils.data.Dataset):
    """Load the binary graphs.

    Example::

        import os
        from agat.data import LoadDataset
        dataset=LoadDataset(os.path.join('dataset', 'all_graphs.bin'))

        # you can index or slice the dataset.
        g0, props0 = dataset[0]
        g_batch, props = dataset[0:100] # the g_batch is a batch collection of graphs. See https://docs.dgl.ai/en/1.1.x/generated/dgl.batch.html


    :param dataset_path: A paths leads to the binary DGL graph file.
    :type dataset_path: str
    :return: a graph dataset.
    :rtype: list

    """
    def __init__(self, dataset_path=None, from_file=True, graph_list=None, props=None):
        # super(Dataset, self).__init__()
        super().__init__()
        if from_file:
            self.dataset_path = dataset_path
            self.graph_list, self.props = load_graphs(self.dataset_path) # `props`: properties.
        else:
            self.dataset_path = None
            self.graph_list = graph_list
            self.props = props

    def __getitem__(self, index):
        """Index or slice the dataset.

        :param index: list index or slice
        :type index: int/slice
        :return: graph or graph batch
        :rtype: dgl graph
        :return: props. Graph labels
        :rtype: A dict of torch.tensor

        """

        if isinstance(index, slice):
            graph_list = self.graph_list[index]
            props = {k:v[index] for k,v in self.props.items()}
        elif isinstance(index, (list, tuple)):
            graph_list = [self.graph_list[x] for x in index]
            props = {k:torch.stack([v[i] for i in index])\
                     for k,v in self.props.items()}
        elif isinstance(index, int):
            graph_list = [self.graph_list[index]]
            props = {k:v[index].unsqueeze(0) for k,v in self.props.items()}
        else:
            raise TypeError('Wrong index type.')

        return Dataset(dataset_path=None,
                       from_file=False,
                       graph_list=graph_list,
                       props=props)

    def __repr__(self):
        return f"AGAT_Dataset(num_graphs={len(self.graph_list)})"

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int

        """
        return len(self.graph_list)

    def save(self, file='graphs.bin'):
        save_graphs(file, self.graph_list, self.props)

class Collater(object):
    """The collate function used in torch.utils.data.DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    The collate function determines how to merge the batch data.

    Example::

        import os
        from agat.data import LoadDataset, Collater
        from torch.utils.data import DataLoader

        dataset=LoadDataset(os.path.join('dataset', 'all_graphs.bin'))
        collate_fn = Collater(device='cuda')
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    :param device: device to store the merged data, defaults to 'cuda'
    :type device: str, optional

    """
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, data):
        """Collate the data into batches.

        :param data: the output of :py:class:`Dataset`
        :type data: AGAT Dataset
        :return: AGAT Dataset with dgl batch graphs. See https://docs.dgl.ai/en/1.1.x/generated/dgl.batch.html
        :rtype: AGAT Dataset

        """
        data.graph_list = [dgl.batch(data.graph_list).to(self.device)]
        data.props = {k:v.to(self.device) for k,v in data.props.items()}
        return data
