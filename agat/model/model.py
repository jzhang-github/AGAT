# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:30:20 2023

@author: ZHANG Jun
"""

import os
import json
import torch
import torch.nn as nn

from dgl.ops import edge_softmax
from dgl import function as fn
from dgl.data.utils import load_graphs

from .layer import Layer

class PotentialModel(nn.Module):
    """A GAT model with multiple gat layers for predicting atomic energies, forces, and stress tensors.


    .. Note:: You can also use this model to train and predict atom and bond related properties. You need to store the labels on graph edges if you want to do so. This model has multiple attention heads.


    .. Important::

        The first value of ``gat_node_dim_list`` is the depth of atomic representation.

        The first value of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` is the input dimension and equals to last value of `gat_node_list * num_heads`.

        The last values of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` are ``1``, ``3``, and ``6``, respectively.


    :param gat_node_dim_list: A list of node dimensions of the AGAT ``Layer``s.
    :type gat_node_dim_list: list
    :param energy_readout_node_list: A list of node dimensions of the energy readout layers.
    :type energy_readout_node_list: list
    :param force_readout_node_list: A list of node dimensions of the force readout layers.
    :type force_readout_node_list: list
    :param stress_readout_node_list: A list of node dimensions of the stress readout layers.
    :type stress_readout_node_list: list
    :param head_list: A list of attention head names, defaults to ['div']
    :type head_list: list, optional
    :param bias: Add bias or not to the neural networks., defaults to True
    :type bias: TYPE, bool
    :param negative_slope: This specifies the negative slope of the LeakyReLU (see https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) activation function., defaults to 0.2
    :type negative_slope: float, optional
    :param device: Device to train the model. Use GPU cards to accerelate training., defaults to 'cuda'
    :type device: str, optional
    :param tail_readout_no_act: The tail ``tail_readout_no_act`` layers will have no activation functions. The first, second, and third elements are for energy, force, and stress readout layers, respectively., defaults to [3,3,3]
    :type tail_readout_no_act: list, optional
    :return: An AGAT model
    :rtype: agat.model.PotentialModel

    """
    def __init__(self,
                 gat_node_dim_list,
                 energy_readout_node_list,
                 force_readout_node_list,
                 stress_readout_node_list,
                 head_list=['div'],
                 bias=True,
                 negative_slope=0.2,
                 device = 'cuda',
                 tail_readout_no_act=[3,3,3]):
        super(PotentialModel, self).__init__()

        # args
        self.gat_node_dim_list = gat_node_dim_list
        self.energy_readout_node_list = energy_readout_node_list
        self.force_readout_node_list = force_readout_node_list
        self.stress_readout_node_list = stress_readout_node_list
        self.head_list = head_list
        self.bias = bias
        self.device = device
        self.negative_slope = negative_slope
        self.tail_readout_no_act = tail_readout_no_act

        self.num_gat_layers = len(self.gat_node_dim_list)-1
        self.num_energy_readout_layers = len(self.energy_readout_node_list)-1
        self.num_force_readout_layers = len(self.force_readout_node_list)-1
        self.num_stress_readout_layers = len(self.stress_readout_node_list)-1
        self.num_heads = len(self.head_list)

        self.__gat_real_node_dim_list = [x*self.num_heads for x in self.gat_node_dim_list[1:]]
        self.__gat_real_node_dim_list.insert(0,self.gat_node_dim_list[0])
        # self.energy_readout_node_list.insert(0, self.__gat_real_node_dim_list[-1]) # calculate and insert input dimension.
        # self.force_readout_node_list.insert(0, self.__gat_real_node_dim_list[-1])
        # self.stress_readout_node_list.insert(0, self.__gat_real_node_dim_list[-1])

        # register layers and parameters.
        self.gat_layers = nn.ModuleList([])
        self.energy_readout_layers = nn.ModuleList([])
        self.force_readout_layers = nn.ModuleList([])
        self.stress_readout_layers = nn.ModuleList([])

        for l in range(self.num_gat_layers):
            self.gat_layers.append(Layer(self.__gat_real_node_dim_list[l],
                                            self.gat_node_dim_list[l+1],
                                            self.num_heads,
                                            device=self.device,
                                            bias=self.bias,
                                            negative_slope=self.negative_slope))

        # energy readout layer
        for l in range(self.num_energy_readout_layers-self.tail_readout_no_act[0]):
            self.energy_readout_layers.append(nn.Linear(self.energy_readout_node_list[l],
                                                         self.energy_readout_node_list[l+1],
                                                         self.bias, self.device))
            self.energy_readout_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        for l in range(self.tail_readout_no_act[0]):
            self.energy_readout_layers.append(nn.Linear(self.energy_readout_node_list[l-self.tail_readout_no_act[0]-1],
                                                         self.energy_readout_node_list[l-self.tail_readout_no_act[0]],
                                                         self.bias, self.device))
        # force readout layer
        for l in range(self.num_force_readout_layers-self.tail_readout_no_act[1]):
            self.force_readout_layers.append(nn.Linear(self.force_readout_node_list[l],
                                                        self.force_readout_node_list[l+1],
                                                        self.bias, self.device))
            self.force_readout_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        for l in range(self.tail_readout_no_act[1]):
            self.force_readout_layers.append(nn.Linear(self.force_readout_node_list[l-self.tail_readout_no_act[1]-1],
                                                        self.force_readout_node_list[l-self.tail_readout_no_act[1]],
                                                        self.bias, self.device))
        # stress readout layer
        for l in range(self.num_stress_readout_layers-self.tail_readout_no_act[2]):
            self.stress_readout_layers.append(nn.Linear(self.stress_readout_node_list[l],
                                                         self.stress_readout_node_list[l+1],
                                                         self.bias, self.device))
            self.stress_readout_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        for l in range(self.tail_readout_no_act[2]):
            self.stress_readout_layers.append(nn.Linear(self.stress_readout_node_list[l-self.tail_readout_no_act[2]-1],
                                                         self.stress_readout_node_list[l-self.tail_readout_no_act[2]],
                                                         self.bias, self.device))

        self.u2e = nn.Linear(self.gat_node_dim_list[0], self.stress_readout_node_list[0],
                             False, self.device) # propogate source nodes to edges.

        self.__real_num_energy_readout_layers = len(self.energy_readout_layers)
        self.__real_num_force_readout_layers = len(self.force_readout_layers)
        self.__real_num_stress_readout_layers = len(self.stress_readout_layers)

        # attention heads
        self.head_fn = {'mul' : self.mul,
                        'div' : self.div,
                        'free': self.free}

    def mul(self, TorchTensor):
        """Multiply head.

        :param TorchTensor: Input tensor
        :type TorchTensor: torch.tensor
        :return: Ouput tensor
        :rtype: torch.tensor

        """

        return TorchTensor

    def div(self, TorchTensor):
        """Division head.

        :param TorchTensor: Input tensor
        :type TorchTensor: torch.tensor
        :return: Ouput tensor
        :rtype: torch.tensor

        """
        return 1/TorchTensor

    def free(self, TorchTensor):
        """Free head.

        :param TorchTensor: Input tensor
        :type TorchTensor: torch.tensor
        :return: Ouput tensor all ones
        :rtype: torch.tensor

        """
        return torch.ones(TorchTensor.size(), device=self.device)

    def get_head_mechanism(self, fn_list, TorchTensor):
        """Get attention heads

        :param fn_list: A list of head mechanisms. For example: ['mul', 'div', 'free']
        :type fn_list: list
        :param TorchTensor: A PyTorch tensor
        :type TorchTensor: torch.Tensor
        :return: A new tensor after the transformation.
        :rtype: torch.Tensor

        """
        TorchTensor_list = []
        for func in fn_list:
            TorchTensor_list.append(self.head_fn[func](TorchTensor))
        return torch.cat(TorchTensor_list, 1)

    def forward(self, graph):
        """The ``forward`` function of PotentialModel model.

        :param graph: ``DGL.Graph``
        :type graph: ``DGL.Graph``
        :return:
            - energy: atomic energy
            - force: atomic force
            - stress: cell stress tensor

        :rtype: tuple of torch.tensors

        """
        with graph.local_scope():
            h    = graph.ndata['h']                                    # shape: (number of nodes, dimension of one-hot code representation)
            dist = torch.reshape(graph.edata['dist'], (-1, 1, 1))      # shape: (number of edges, 1, 1)
            dist = torch.where(dist < 0.01, 0.01, dist)                  # This will creat a new `dist` variable, insted of modifying the original memory.
            dist = self.get_head_mechanism(self.head_list, dist) # shape of dist: (number of edges, number of heads, 1)

            for l in range(self.num_gat_layers):
                h = self.gat_layers[l](h, dist, graph)                 # shape of h: (number of nodes, number of heads * num_out)

            # predict energy
            energy = h
            for l in range(self.__real_num_energy_readout_layers):
                energy = self.energy_readout_layers[l](energy)
            batch_nodes = graph.batch_num_nodes().tolist()
            energy = torch.split(energy, batch_nodes)
            energy = torch.stack([torch.mean(e) for e in energy])

            # Predict force
            graph.ndata['node_force'] = h
            graph.apply_edges(fn.u_add_v('node_force', 'node_force', 'force_score'))   # shape of score: (number of edges, ***, 1)
            force_score = torch.reshape(graph.edata['force_score'],(-1, self.num_heads, self.gat_node_dim_list[-1])) / dist
            force_score = torch.reshape(force_score, (-1, self.num_heads * self.gat_node_dim_list[-1]))

            stress_score = torch.clone(force_score)

            for l in range(self.__real_num_force_readout_layers):
                force_score = self.force_readout_layers[l](force_score)
            graph.edata['force_score_vector'] = force_score * graph.edata['direction']      # shape (number of edges, 1)
            graph.update_all(fn.copy_e('force_score_vector', 'm'), fn.sum('m', 'force_pred'))        # shape of graph.ndata['force_pred']: (number of nodes, 3)
            force = graph.ndata['force_pred']

            # Predict stress
            graph.edata['stress_score'] = stress_score
            graph.ndata['atom_code'] = self.u2e(graph.ndata['h'])
            graph.apply_edges(fn.u_add_e('atom_code', 'stress_score',  'stress_score'))
            stress_score = graph.edata['stress_score']

            for l in range(self.__real_num_stress_readout_layers):
                stress_score = self.stress_readout_layers[l](stress_score)
            graph.edata['stress_score_vector'] = stress_score * torch.cat((graph.edata['direction'],
                                                                           graph.edata['direction']), dim=1)      # shape (number of edges, 2)
            batch_edges = graph.batch_num_edges().tolist()
            stress = torch.split(graph.edata['stress_score_vector'], batch_edges)
            stress = torch.stack([torch.mean(s, dim=0) for s in stress])

            # graph.update_all(fn.copy_e('stress_score_vector', 'm'), fn.sum('m', 'stress_pred'))        # shape of graph.ndata['force_pred']: (number of nodes, 3)
            # # stress = torch.sum(graph.ndata['stress_pred'], dim=0)
            # stress = torch.split(graph.ndata['stress_pred'], batch_nodes)
            # stress = torch.stack([torch.sum(s, dim=0) for s in stress])
            return energy, force, stress

class CrystalPropertyModel(nn.Module):
    pass

class AtomicPropertyModel(nn.Module):
    pass

class AtomicVectorModel(nn.Module):
    pass

if __name__ == '__main__':
    import dgl
    g_list, l_list = dgl.load_graphs('all_graphs.bin')
    graph = g_list[1] #.to('cuda')
    graph = g_list[1].to('cuda')
    feat = graph.ndata['h']
    dist = graph.edata['dist']

    PM = PotentialModel([6,30,20,10],
                        [10,10,10,5,1],
                        [10,10,10,5,3],
                        [10,10,10,5,6],
                        head_list=['div', 'mul'],
                        bias=True,
                        negative_slope=0.2,
                        device = 'cuda',
                        tail_readout_no_act=[2,2,2]
                        )

    energy, force, stress = PM.forward(graph)

    model = PM
