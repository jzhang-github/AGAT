# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:41:00 2021

@author: ZHANG Jun
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from dgl import function as fn
from dgl.ops import edge_softmax

class GATLayer(layers.Layer):
    """
    Description:
    ----------
        Single graph attention network for crystal.
    Parameters
    ----------
    head_list: list
        A list of head mechanisms of this `GAT` layer.
    num_out: int
        Depth of node representation in the output of each head of this `GAT` layer.
    bias: Boolean
        Whether the dense layer uses a bias vector.
    negative_slope: float
        Negative slope coefficient
    activation: `tf.keras.activations`
        Activation function to use
    layer_type: str
        type of this gat layer.
        'input': input will be mapped into higher space.
        'hidden': dimension of feature representation will be kept.
        'output': output will be reduced into lower space.

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
                 num_out,
                 num_heads,
                 bias=False,
                 negative_slope=0.2,
                 activation=None,
                 batch_normalization=False
                 ): # act_att='tanh'
        # The activation function of attention score on edges are deprecated.
        # See: https://docs.dgl.ai/en/0.6.x/generated/dgl.nn.functional.edge_softmax.html and https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax

        super(GATLayer, self).__init__()

        self.num_out        = num_out
        self.num_heads      = num_heads
        self.bias           = bias
        self.negative_slope = negative_slope
        self.activation     = activation
        # self.act_att        = act_att
        self.leaky_relu     = layers.LeakyReLU(alpha=negative_slope)
        self.xinit          = tf.keras.initializers.VarianceScaling(scale=np.sqrt(2),
                                                                     mode="fan_avg",
                                                                     distribution="untruncated_normal")
        self.batch_normalization = batch_normalization

    def build(self, input_shape):
        """
        The `build` method is thoroughly introduced by `TensorFlow`.
        Details: https://www.tensorflow.org/guide/keras/custom_layers_and_models
        """
        # print("Run build. ================") # for debug use
        self.w_att_src = tf.Variable(initial_value=self.xinit(shape=(1, self.num_heads, self.num_out),
                                                         dtype='float32'),
                                     trainable=True, name='w_att_src')
        self.w_att_dst = tf.Variable(initial_value=self.xinit(shape=(1, self.num_heads, self.num_out),
                                                         dtype='float32'),
                                     trainable=True, name='w_att_dst')

        self.dense = tf.keras.layers.Dense(self.num_out * self.num_heads,
                                                self.activation,
                                                self.bias,
                                                self.xinit,
                                                self.xinit)
        if self.batch_normalization:
            self.bn = tf.keras.layers.BatchNormalization()

    def call(self, feat, dist, graph): # feat with shape of (number of nodes, number of features of each node). The graph can have no values of nodes.
        """
        Description:
        ----------
            Forward this `GAT` layer.
        Parameters
        ----------
        feat: `tf.Tensor`
            Input features of all nodes (atoms).
        graph: `DGLGraph`
            A graph built with DGL.
        """

        # with graph.local_scope():
        feat_src = feat_dst = self.dense(feat)

        # attention
        feat_src = tf.reshape(feat_src, (-1, self.num_heads, self.num_out)) # shape: (number of nodes, number of heads, num out)
        feat_dst = tf.reshape(feat_dst, (-1, self.num_heads, self.num_out)) # shape: (number of nodes, number of heads, num out)

        # shape of e_src and e_dst after `tf.reduce_sum()`: (number of edges, number of heads, 1)
        # print(self.w_att_src)
        e_src = tf.reduce_sum(feat_src * self.w_att_src, axis=-1, keepdims=True)
        e_dst = tf.reduce_sum(feat_dst * self.w_att_dst, axis=-1, keepdims=True)

        # save on nodes
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        graph.dstdata.update({'e_dst': e_dst})

        # save on edges
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))   # shape of e: (number of edges, number of heads, 1) # similar to the original paper, but feed to dense first, summation is the second.
        e = self.leaky_relu(graph.edata.pop('e'))  # shape of e: (number of edges, number of heads, 1)

        graph.edata['a']  = edge_softmax(graph, e) * dist # shape of a: (number of edges, number of heads, 1) # shape of dist: (number of edges, number of heads, 1)

        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft')) # shape of ft: (number of nodes, number of heads, number of out)
        # print(tf.shape(graph.ndata['ft']))
        # print(tf.shape(graph.edata['a']))
        dst = tf.reshape(graph.ndata['ft'], (-1, self.num_heads * self.num_out)) # shape of `en`: (number of nodes, number of heads * number of out)
        if self.batch_normalization:
            dst = self.bn(dst)
        return dst # node_energy, node_force

# debug
if __name__ == '__main__':
    from modules.Crystal2Graph import CrystalGraph
    cg = CrystalGraph()
    bg = cg.get_graph('POSCAR.txt')
    feat = bg.ndata['h']
    dist = bg.edata['dist']
    dist = tf.reshape(dist, (-1,1,1))

    SGL = GATLayer(50, 1)

    a = SGL(feat, dist, bg)

    # SGL1 = GATLayer(50, head_list=['mul', 'div', 'free'], layer_type='input')
    # bg = cg.get_graph('mp-1444.cif')
    # feat = bg.ndata['h']
    # SGL1(a, bg)

    # SGL2 = GATLayer(5, 50, layer_type='hidden')
    # bg = cg.get_graph('mp-1444.cif')
    # # feat = bg.ndata['h']
    # SGL2(a, bg)
