import numpy as np
import os
from dgl.data.utils import save_graphs, load_graphs
import tensorflow as tf
import json

# graph_dict = {}

binary_files  = ['all_graphs_bulk.bin',
                 'all_graphs_surface_static.bin']
csv_files     = ['fname_prop_bulk.csv',
                 'fname_prop_surface_static.csv']
sample_factor = [2,1]

assert len(binary_files) == len(csv_files) == len(sample_factor), 'Lengths of binary_files, csv_files, and sample_factor are not equal.'

num = len(binary_files)

all_graphs, all_csvs, all_labels, num_graphs = [], [], [], []

for i in range(num):
    graph_list, graph_labels = load_graphs(binary_files[i])
    num_graphs.append(len(graph_list))
    with open(csv_files[i], 'r') as f:
        lines = f.readlines()
    for j in range(sample_factor[i]):
        all_graphs += graph_list
        all_csvs   += lines
        all_labels.append(graph_labels['prop'])

all_labels = {'prop': tf.concat(all_labels, axis=0)}
save_graphs('all_graphs_all.bin', all_graphs, all_labels)
with open('fname_prop_all.csv', 'w') as f:
    for i in all_csvs:
        f.write(i)


graph_dict = {'binary_files' : binary_files,
              'csv_files'    : csv_files,
              'sample_factor': sample_factor,
              'num_graphs'   : num_graphs}

with open('graph_concat.log', 'w') as f:
    json.dump(graph_dict, f)

# graph_list_1, graph_labels_1 = load_graphs('all_graphs_bulk.bin')
# graph_list_2, graph_labels_2 = load_graphs('all_graphs_surfaces.bin')
# graph_list_new =  graph_list_1 +  graph_list_2
# graph_labels_new = {}
# graph_labels_new['prop'] = tf.concat((graph_labels_1['prop'], graph_labels_2['prop']), axis=0)
# save_graphs('all_graphs_Direction_Constraints_Forces_AseGrph_Surfaces.bin', graph_list_new, graph_labels_new)
