# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 01:33:11 2023

@author: ZHANG Jun
"""

import torch
import time

@torch.jit.script
def PearsonRjit(y_true, y_pred):
    ave_y_true = torch.mean(y_true)
    ave_y_pred = torch.mean(y_pred)

    y_true_diff = y_true - ave_y_true
    y_pred_diff = y_pred - ave_y_pred

    above = torch.sum(torch.mul(y_true_diff, y_pred_diff))
    below = torch.mul(torch.sqrt(torch.sum(torch.square(y_true_diff))),
                             torch.sqrt(torch.sum(torch.square(y_pred_diff))))
    return torch.divide(above, below)

def PearsonR(y_true, y_pred):
    ave_y_true = torch.mean(y_true)
    ave_y_pred = torch.mean(y_pred)

    y_true_diff = y_true - ave_y_true
    y_pred_diff = y_pred - ave_y_pred

    above = torch.sum(torch.mul(y_true_diff, y_pred_diff))
    below = torch.mul(torch.sqrt(torch.sum(torch.square(y_true_diff))),
                             torch.sqrt(torch.sum(torch.square(y_pred_diff))))
    return torch.divide(above, below)

if __name__ == '__main__':
    y_true = torch.randn(100000)
    y_pred = torch.randn(100000)
    start = time.time()
    for _ in range(10000):
        PearsonRjit(y_true, y_pred)
    print(time.time()-start)

    start = time.time()
    for _ in range(10000):
        PearsonR(y_true, y_pred)
    print(time.time()-start)
    y_true = torch.randn(100000, device='cuda')
    y_pred = torch.randn(100000, device='cuda')
    start = time.time()
    for _ in range(10000):
        PearsonRjit(y_true, y_pred)
    print(time.time()-start)

    start = time.time()
    for _ in range(10000):
        PearsonR(y_true, y_pred)
    print(time.time()-start)
    print('==================')

    """
    The output::
        15.30269169807434
        2.988755464553833
        2.4554362297058105
        1.8777987957000732
        ==================
    """

    y_true_numpy = y_true.cpu().numpy()
    y_pred_numpy = y_pred.cpu().numpy()
    start = time.time()
    for _ in range(10000):
        y_diff = y_true_numpy - y_pred_numpy
        y_div = y_true_numpy / y_pred_numpy
        y_mul = y_true_numpy * y_pred_numpy
    print(time.time()-start)
    print('==================')
