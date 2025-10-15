# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:54:56 2023

@author: 18326
"""

import numpy as np
from warnings import warn
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

def concat_graphs(*list_of_bin, save_file=True, fname='concated_graphs.bin'):
    """ Concat binary graph files.

    :param *list_of_bin: input file names of binary graphs.
    :type *list_of_bin: strings
    :return: A new file is saved to the current directory: concated_graphs.bin.
    :rtype: None. A new file.

    Example::

        concat_graphs('graphs1.bin', 'graphs2.bin', 'graphs3.bin')

    """

    warn("This object will be deprecated in the future. Please use `concat_dataset`.")

    graph_list = []
    graph_labels = {}
    for file in list_of_bin:
        batch_g, batch_labels = load_graphs(file)
        graph_list.extend(batch_g)
        for key in batch_labels.keys():
            try:
                graph_labels[key] = torch.cat([graph_labels[key],
                                               batch_labels[key]], 0)
            except KeyError:
                graph_labels[key] = batch_labels[key]

    if save_file:
        save_graphs(fname, graph_list, graph_labels)
    return Dataset(dataset_path=None, from_file=False, graph_list=graph_list, props=graph_labels)

def concat_dataset(*list_of_datasets, save_file=False, fname='concated_graphs.bin'):
    """ Concat binary graph files.

    :param *list_of_bin: input file names of binary graphs.
    :type *list_of_bin: strings
    :return: A new file is saved to the current directory: concated_graphs.bin.
    :rtype: None. A new file.

    Example::

        concat_graphs('graphs1.bin', 'graphs2.bin', 'graphs3.bin')

    """

    graph_list = []
    graph_labels = {}
    for d in list_of_datasets:
        batch_g, batch_labels = d.graph_list, d.props
        graph_list.extend(batch_g)
        for key in batch_labels.keys():
            try:
                graph_labels[key] = torch.cat([graph_labels[key],
                                               batch_labels[key]], 0)
            except KeyError:
                graph_labels[key] = batch_labels[key]

    if save_file:
        save_graphs(fname, graph_list, graph_labels)
    return Dataset(dataset_path=None, from_file=False, graph_list=graph_list, props = graph_labels)

def select_graphs_random(fname: str, num: int):
    """ Randomly split graphs from a binary file.

    :param fname: input file name.
    :type fname: str
    :param num: number of selected graphs (should be smaller than number of all graphs.
    :type num: int
    :return: A new file is saved to the current directory: Selected_graphs.bin.
    :rtype: None. A new file.

    Example::

        select_graphs_random('graphs1.bin')

    """
    warn("This object will be deprecated in the future. Please use `select_graphs_from_dataset_random`")

    bg, labels = load_graphs(fname)
    num_graphs = len(bg)
    assert num < num_graphs, f'The number of selected graphs should be lower than\
the number of all graphs. Number of selected graphs: {num}. Number of all graphs: {num_graphs}.'
    random_int = np.random.choice(range(num_graphs), size=num, replace=False)

    selected_bg = [bg[x] for x in random_int]

    graph_labels = {}
    for key in labels.keys():
        graph_labels[key] = labels[key][random_int]

    save_graphs('selected_graphs.bin', selected_bg, graph_labels)

def select_graphs_from_dataset_random(dataset, num: int, save_file=False,
                                      fname='selected_graphs.bin'):
    """ Randomly split graphs from a binary file.

    :param fname: input file name.
    :type fname: str
    :param num: number of selected graphs (should be smaller than number of all graphs.
    :type num: int
    :return: A new file is saved to the current directory: Selected_graphs.bin.
    :rtype: None. A new file.

    Example::

        select_graphs_random('graphs1.bin')

    """

    num_graphs = len(dataset)
    assert num < num_graphs, f'The number of selected graphs should be lower than\
the number of all graphs. Number of selected graphs: {num}. Number of all graphs: {num_graphs}.'
    random_int = np.random.choice(range(num_graphs), size=num, replace=False)
    dataset = dataset[list(random_int)]
    if save_file:
        save_graphs(fname, dataset.graph_list, dataset.props)
    return dataset

def save_dataset(dataset: Dataset, fname='graphs.bin'):
    assert isinstance(dataset, Dataset), f'Wrong dataset type. Expect `LoadDataset`, but got {type(dataset)}'
    save_graphs(fname, dataset.graph_list, dataset.props)

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
        if isinstance(data, list):
            data = concat_dataset(*data)
        data.graph_list = [dgl.batch(data.graph_list).to(self.device)]
        data.props = {k:v.to(self.device) for k,v in data.props.items()}
        return data
