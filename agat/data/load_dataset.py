# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:54:56 2023

@author: 18326
"""

import torch
from torch.utils.data import Dataset
import dgl
from dgl.data.utils import load_graphs

class LoadDataset(Dataset):
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
    def __init__(self, dataset_path):
        super(LoadDataset, self).__init__()
        self.dataset_path = dataset_path
        self.graph_list, self.props = load_graphs(self.dataset_path) # `props`: properties.

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
            graph = dgl.batch(graph_list)
        else:
            graph = self.graph_list[index]
        props = {k:v[index] for k,v in self.props.items()}
        return graph, props

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int

        """

        return len(self.graph_list)

# class LoadDataset(Dataset):
#     def __init__(self, dataset_path):
#         super(LoadDataset, self).__init__()
#         self.dataset_path = dataset_path
#         self.graph_list, self.props = load_graphs(self.dataset_path) # `props`: properties.

#     def __getitem__(self, index):
#         graph = self.graph_list[index]
#         props = {k:v[index] for k,v in self.props.items()}
#         return graph, props

#     def __len__(self):
#         return len(self.graph_list)

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

        :param data: the output of :py:class:`LoadDataset`
        :type data: tuple
        :return: dgl batch graphs. See https://docs.dgl.ai/en/1.1.x/generated/dgl.batch.html
        :rtype: DGLGraph
        :return: Graph labels
        :rtype: A dict of torch.tensor

        """

        # print(data)
        graph_list = [x[0] for x in data]
        graph = dgl.batch(graph_list)
        props = [x[1] for x in data]
        props = {k:torch.stack([x[k] for x in props]) for k in props[0].keys()}
        props = {k:v.to(self.device) for k,v in props.items()}
        return graph.to(self.device), props

if __name__ == '__main__':
    dataset = LoadDataset('all_graphs.bin')
    g, p = dataset[10]
    from torch.utils.data import DataLoader
    collate_fn = Collater(device='cpu')
    import time
    start = time.time()
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
                        collate_fn=collate_fn)

    for i, sample in enumerate(loader):
        graph, props = sample
    print(time.time()-start)

    # test of num_workers larger than 0.
    start = time.time()
    for i, sample in enumerate(loader):
        graph, props = sample
        if i > 500:
            break
    dur1 = time.time()-start

    start = time.time()
    for i, sample in enumerate(loader):
        graph, props = sample
        time.sleep(0.01)
        if i > 500:
            break
    dur2 = time.time()-start
    print(dur2-5.0-dur1)

