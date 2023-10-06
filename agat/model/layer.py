# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:04:06 2023

@author: ZHANG Jun
"""

import torch
import torch.nn as nn
from dgl import function as fn
from dgl.ops import edge_softmax

class Layer(nn.Module):
    """
    Description:
    ----------
        Single graph attention network for crystal.
    Parameters
    ----------
    in_dim: int
        Depth of node representation in the input of each head of this `GAT` layer.
    out_dim: int
        Depth of node representation in the output of each head of this `GAT` layer.
    bias: Boolean
        Whether the dense layer uses a bias vector.
    negative_slope: float
        Negative slope coefficient


    Meaning of abbreviations:
    ----------
    dist:
        distance matrix
    feat:
        features
    ft:
        features
    src:
        source node
    dst:
        destination node
    e:
        raw e_i_j: refer to: https://arxiv.org/abs/1710.10903
    a:
        alpha_i_j: refer to: https://arxiv.org/abs/1710.10903
    att:
        attention mechanism
	act:
        activation function
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 device='cuda',
                 bias=True,
                 negative_slope=0.2):
        super(Layer, self).__init__()
        # input args
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.num_heads      = num_heads
        self.device         = device
        self.bias           = bias
        self.negative_slope = negative_slope

        # initialize trainable parameters
        self.w_att_src = nn.Parameter(torch.randn(1, self.num_heads, self.out_dim, device=self.device))
        self.w_att_dst = nn.Parameter(torch.randn(1, self.num_heads, self.out_dim, device=self.device))

        # dense layers
        self.layer = nn.Linear(self.in_dim, self.out_dim*self.num_heads, bias=self.bias, device=self.device)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=self.negative_slope)

        # leaky relu function
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, feat, dist, graph): # feat with shape of (number of nodes, number of features of each node). The graph can have no values of nodes.
        """
        Description:
        ----------
            Forward this `GAT` layer.
        Parameters
        ----------
        feat: `torch.Tensor`
            Input features of all nodes (atoms).
        graph: `DGLGraph`
            A graph built with DGL.
        """
        feat_src = feat_dst = self.leaky_relu1(self.layer(feat))

        # attention
        feat_src = torch.reshape(feat_src, (-1, self.num_heads, self.out_dim)) # shape: (number of nodes, number of heads, num out)
        feat_dst = torch.reshape(feat_dst, (-1, self.num_heads, self.out_dim)) # shape: (number of nodes, number of heads, num out)

        # shape of e_src and e_dst after `torch.sum`: (number of nodes, number of heads, 1)
        e_src = torch.sum(feat_src * self.w_att_src, axis=-1, keepdim=True)
        e_dst = torch.sum(feat_dst * self.w_att_dst, axis=-1, keepdim=True)

        # save on nodes
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        graph.dstdata.update({'e_dst': e_dst})

        # save on edges
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))   # shape of e: (number of edges, number of heads, 1) # similar to the original paper, but feed to dense first, summation is the second.
        e = self.leaky_relu2(graph.edata.pop('e'))  # shape of e: (number of edges, number of heads, 1)

        dist = torch.reshape(dist, (-1, self.num_heads, 1))
        graph.edata['a']  = edge_softmax(graph, e) * dist # shape of a: (number of edges, number of heads, 1) # shape of dist: (number of edges, number of heads, 1)

        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft')) # shape of ft: (number of nodes, number of heads, number of out)
        dst = torch.reshape(graph.ndata['ft'], (-1, self.num_heads * self.out_dim)) # shape of `en`: (number of nodes, number of heads * number of out)
        return dst # node_energy, node_force

if __name__ == '__main__':
    import dgl
    g_list, l_list = dgl.load_graphs('all_graphs.bin')
    graph = g_list[1].to('cuda')
    feat = graph.ndata['h']
    dist = graph.edata['dist']

    GL = Layer(6, 10, 2, device='cuda')
    out = GL.forward(feat, dist, graph)
