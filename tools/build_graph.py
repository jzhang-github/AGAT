# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:33:42 2021

@author: ZHANG Jun

This script is used to build graphs from sctratch (cif or VASP files).
"""

# !!! This script should be run under the `tools` directory

# build graphs based on distance cutoff is much more faster than based on Voronoi neighbors. However, the distance method will include more edges.

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from modules.Crystal2Graph import ReadGraphs

if __name__ == '__main__':
    project_path = os.path.join('..', 'project') # a folder contains the project folders, which are `dataset`, `ckpt`, and `files`.
    num_cores    = 3

    graph_reader = ReadGraphs(os.path.join(project_path, 'files', 'fname_prop.csv'),
                              os.path.join(project_path, 'dataset'),
                              cutoff       = 6.5,
                              mode_of_NN   = 'distance',
                              from_binary  = False,
                              num_of_cores = num_cores,
                              super_cell   = False)

    print(f"Reading graphs from: {os.path.join(project_path, 'dataset')}")

if __name__ == '__main__':
   ckpt_path = os.path.join(project_path, 'ckpt')
   if not os.path.exists(ckpt_path):
       os.mkdir(ckpt_path)
    graph_list, graph_labels = graph_reader.read_all_graphs(scale_prop=False,
                                                            ckpt_path=ckpt_path)
