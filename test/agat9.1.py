# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 10:52:44 2025

@author: JXZ
"""

# GPU card
import torch
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available.")

import dgl
u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
print(f'DGL graph device: {g.device}')

# torch calculators


# dataset / database