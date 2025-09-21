# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 19:49:49 2023

@author: ZHANG Jun
"""

import time

from agat.data.loda_dataset import LoadDataset, Collater
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dataset = LoadDataset('all_graphs.bin')
    g, p = dataset[10]

    collate_fn = Collater(device='cpu')

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
