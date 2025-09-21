# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:01:10 2023

@author: ZHANG Jun
"""

import os
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
graph_list, props = load_graphs('all_graphs_generation_0.bin')

stresses = props['stress_true'].numpy()


plt.hist(stresses[:,0], bins=100, range=[-0.01, 0.01])
