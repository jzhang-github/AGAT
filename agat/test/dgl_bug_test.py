# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:01:12 2023

@author: ZHANG Jun
"""

import dgl
import dgl.function as fn
import tensorflow as tf

if __name__ == '__main__':
    # Source nodes for edges (2, 1), (3, 2), (4, 3)
    src_ids = tf.constant([2, 3, 4])
    # Destination nodes for edges (2, 1), (3, 2), (4, 3)
    dst_ids = tf.constant([1, 2, 3])
    g = dgl.graph((src_ids, dst_ids))
    # g_test = ... # create a DGLGraph
    g.ndata['h'] = tf.random.uniform((g.num_nodes(), 10)) # each node has feature size 10
    g.edata['w'] = tf.random.uniform((g.num_edges(), 1))  # each edge has feature size 1
    # collect features from source nodes and aggregate them in destination nodes
    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))
    # multiply source node features with edge weights and aggregate them in destination nodes
    g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.max('m', 'h_max'))
    # compute edge embedding by multiplying source and destination node embeddings

    for i in range(20):
        g.apply_edges(fn.u_mul_v('h', 'h', 'w_new'))
        print('========================')
        print(g.edata['w_new'])

    '''
    I raise this issue on: https://github.com/dmlc/dgl/issues/6378
    '''
