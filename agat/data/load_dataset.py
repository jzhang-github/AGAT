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
    def __init__(self, dataset_path):
        super(LoadDataset, self).__init__()
        self.dataset_path = dataset_path
        self.graph_list, self.props = load_graphs(self.dataset_path) # `props`: properties.

    def __getitem__(self, index):
        if isinstance(index, slice):
            graph_list = self.graph_list[index]
            graph = dgl.batch(graph_list)
        else:
            graph = self.graph_list[index]
        props = {k:v[index] for k,v in self.props.items()}
        return graph, props

    def __len__(self):
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
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, data):
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

