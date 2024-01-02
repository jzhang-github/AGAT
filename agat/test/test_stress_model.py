# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:34:12 2023

@author: ZHANG Jun
"""
import os
# import json
import torch
import torch.nn as nn

# from dgl.ops import edge_softmax
from dgl import function as fn
# from dgl.data.utils import load_graphs

from agat.model.layer import Layer

import numpy as np
import time
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# from .model import PotentialModel
from agat.lib.model_lib import EarlyStopping, PearsonR, config_parser, save_model, load_state_dict, save_state_dict
from agat.data.load_dataset import LoadDataset, Collater
from agat.default_parameters import default_train_config
from agat.lib.file_lib import file_exit


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

        # self.stress_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        # self.stress_act = nn.LeakyReLU(negative_slope=self.negative_slope)
        # self.stress_outlayer = nn.Linear(6, 6, False, self.device)
        # self.stress_outlayer = nn.BatchNorm1d(6, device=self.device)
        self.u2e = nn.Linear(self.gat_node_dim_list[0], self.stress_readout_node_list[0],
                             False, self.device) # propogate source nodes to edges.
        # self.skip_connect = nn.Linear(1, self.stress_readout_node_list[0],
        #                      False, self.device)

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

        # Input dim: (number of nodes, number of heads * number of out)
        # stress readout layer
        for l in range(self.num_stress_readout_layers-self.tail_readout_no_act[2]):
            self.stress_readout_layers.append(nn.Linear(self.stress_readout_node_list[l],
                                                         self.stress_readout_node_list[l+1],
                                                         self.bias, self.device))
            # self.stress_readout_layers.append(nn.BatchNorm1d(
            #     self.stress_readout_node_list[l+1], device=self.device))
            self.stress_readout_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        for l in range(self.tail_readout_no_act[2]):
            self.stress_readout_layers.append(nn.Linear(self.stress_readout_node_list[l-self.tail_readout_no_act[2]-1],
                                                         self.stress_readout_node_list[l-self.tail_readout_no_act[2]],
                                                         self.bias, self.device))
            # self.stress_readout_layers.append(nn.BatchNorm1d(
            #     self.stress_readout_node_list[l-self.tail_readout_no_act[2]],
            #     device=self.device))

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

    # def get_unit_vector(self, direction):
    #     return direction/torch.norm(direction, dim=0)

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
            print(torch.flatten(graph.edata['dist']))
            h    = graph.ndata['h']                                    # shape: (number of nodes, dimension of one-hot code representation)
            dist = torch.reshape(graph.edata['dist'], (-1, 1, 1))      # shape: (number of edges, 1, 1)
            dist = torch.where(dist < 0.01, 0.01, dist)                  # This will creat a new `dist` variable, insted of modifying the original memory.
            dist = self.get_head_mechanism(self.head_list, dist) # shape of dist: (number of edges, number of heads, 1)

            for l in range(self.num_gat_layers):
                h = self.gat_layers[l](h, dist, graph)                 # shape of h: (number of nodes, number of heads * num_out)
            # print(h)
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
            stress_score = graph.edata['stress_score']  # + self.skip_connect(graph.edata['dist'])

            # graph.edata['stress_score_test'][6]
            # graph.edata['stress_score_test'][690]

            # torch.mean(graph.edata['stress_score_test'],dim=0)
            # fn.copy_u('atom_code', 'm')

            for l in range(self.__real_num_stress_readout_layers):
                stress_score = self.stress_readout_layers[l](stress_score)

            # unit_vector = graph.edata['direction']/torch.norm(graph.edata['direction'], dim=0)
            graph.edata['stress_score_vector'] = stress_score * torch.cat((graph.edata['direction'],
                                                                           graph.edata['direction']), dim=1)      # shape (number of edges, 2)

            batch_edges = graph.batch_num_edges().tolist()
            stress = torch.split(graph.edata['stress_score_vector'], batch_edges)
            stress = torch.stack([torch.mean(s, dim=0) for s in stress])

            # graph.edata['stress_score_vector'] = self.stress_act(graph.edata['stress_score_vector'])
            # graph.edata['stress_score_vector'] = self.stress_outlayer(graph.edata['stress_score_vector'])
            # graph.update_all(fn.copy_e('stress_score_vector', 'm'), fn.sum('m', 'stress_pred'))        # shape of graph.ndata['force_pred']: (number of nodes, 3)
            # stress = torch.sum(graph.ndata['stress_pred'], dim=0)
            # stress = torch.split(graph.ndata['stress_pred'], batch_nodes) # shape of stress: number of atoms * 6
            # stress = torch.stack([torch.sum(s, dim=0) for s in stress]) # + self.stress_bias
            return energy, force, stress

class Fit(object):
    def __init__(self, **train_config):
        self.train_config = {**default_train_config, **config_parser(train_config)}
        self.log = open('fit.log', 'w', buffering=1)

        # check device
        self.device = self.train_config['device']
        self.verbose = self.train_config['verbose']
        if torch.cuda.is_available() and self.device == 'cpu':
            print('User warning: `CUDA` device is available, but you choosed `cpu`.', file=self.log)
        elif not torch.cuda.is_available() and self.device.split(':')[0] == 'cuda':
            print('User warning: `CUDA` device is not available, but you choosed \
`cuda:0`. Change the device to `cpu`.', file=self.log)
            self.device = 'cpu'
        print('User info: Specified device for potential model:', self.device, file=self.log)

        # read dataset
        print(f'User info: Loading dataset from {self.train_config["dataset_path"]}',
              file=self.log)
        self._dataset=LoadDataset(self.train_config['dataset_path'])

        # split dataset
        self._val_size = int(len(self._dataset)*self.train_config['validation_size'])
        self._test_size = int(len(self._dataset)*self.train_config['test_size'])
        self._train_size = len(self._dataset) - self._val_size - self._test_size
        train_dataset, val_dataset, test_dataset = random_split(self._dataset,
                                                                [self._train_size,
                                                                 self._val_size,
                                                                 self._test_size])
        # check batch size
        self.train_config['batch_size'] = min(self.train_config['batch_size'],
                                              self._train_size)
        self.train_config['val_batch_size'] = min(self.train_config['val_batch_size'],
                                                  self._val_size,
                                                  self._test_size)

        # instantiate data loader
        collate_fn = Collater(device=self.device)
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.train_config['batch_size'],
                                       shuffle=True,
                                       num_workers=0,
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.train_config['val_batch_size'],
                                     shuffle=True,
                                     num_workers=0,
                                     collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.train_config['val_batch_size'],
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=collate_fn)

        # check the existence of adsorbates.
        _graph, _ = self._dataset[0]
        self._has_adsorbate = _graph.ndata.__contains__('adsorbate')

        # check neural nodes dimensions, and modify them if necessary.
        num_heads = len(self.train_config['head_list'])
        atomic_depth = self._dataset[0][0].ndata['h'].size()[1]
        if self.train_config['gat_node_dim_list'][0] != atomic_depth:
            print(f'User warning: Input dimension of the first AGAT `Layer` \
(the first element of `gat_node_dim_list`) should equal to the dimension of \
atomic representation depth. Your input is: {self.train_config["gat_node_dim_list"][0]}, \
which is changed to be: `{atomic_depth}`.', file=self.log)
            self.train_config['gat_node_dim_list'][0] = atomic_depth
        in_dim = num_heads * self.train_config['gat_node_dim_list'][-1]
        if self.train_config['energy_readout_node_list'][0] != in_dim:
            print(f"User warning: Input dimension of the first energy readout layer \
(the first element of `energy_readout_node_list`) should equal to \
`len(self.train_config['head_list']) * self.train_config['gat_node_dim_list'][-1]`. \
Your input is: {self.train_config['energy_readout_node_list'][0]}, which is \
changed to be: `{in_dim}`.", file=self.log)
            self.train_config['energy_readout_node_list'][0] = in_dim
        if self.train_config['energy_readout_node_list'][-1] != 1:
            print(f"User warning: Output dimension of the last energy readout layer \
(the last element of `energy_readout_node_list`) should equal to \
`1`. Your input is: {self.train_config['energy_readout_node_list'][-1]}, which is \
changed to be: `1`.", file=self.log)
            self.train_config['energy_readout_node_list'][-1] = 1
        if self.train_config['force_readout_node_list'][0] != in_dim:
            print(f"User warning: Input dimension of the first force readout layer \
(the first element of `force_readout_node_list`) should equal to \
`len(self.train_config['head_list']) * self.train_config['gat_node_dim_list'][-1]`. \
Your input is: {self.train_config['force_readout_node_list'][0]}, which is \
changed to be: `{in_dim}`.", file=self.log)
            self.train_config['force_readout_node_list'][0] = in_dim
        if self.train_config['force_readout_node_list'][-1] != 3:
            print(f"User warning: Output dimension of the last force readout layer \
(the last element of `force_readout_node_list`) should equal to \
`3`. Your input is: {self.train_config['force_readout_node_list'][-1]}, which is \
changed to be: `3`.", file=self.log)
            self.train_config['force_readout_node_list'][-1] = 3
        if self.train_config['stress_readout_node_list'][0] != in_dim:
            print(f"User warning: Input dimension of the first stress readout layer \
(the first element of `stress_readout_node_list`) should equal to \
`len(self.train_config['head_list']) * self.train_config['gat_node_dim_list'][-1]`. \
Your input is: {self.train_config['stress_readout_node_list'][0]}, which is \
changed to be: `{in_dim}`.", file=self.log)
            self.train_config['stress_readout_node_list'][0] = in_dim
        if self.train_config['stress_readout_node_list'][-1] != 6:
            print(f"User warning: Output dimension of the last stress readout layer \
(the last element of `stress_readout_node_list`) should equal to \
`6`. Your input is: {self.train_config['stress_readout_node_list'][-1]}, which is \
changed to be: `6`.", file=self.log)
            self.train_config['stress_readout_node_list'][-1] = 6

        # prepare out file
        if not os.path.exists(self.train_config['output_files']):
            os.mkdir(self.train_config['output_files'])

        # debug
        TIMESTAMP = "{0:%Y-%m-%d--%H-%M-%S}".format(datetime.now())
        self.writer = SummaryWriter(os.path.join('fit_debug', TIMESTAMP),
                                    flush_secs=10)

    def fit(self, **train_config):
        # update config if needed.
        self.train_config = {**self.train_config, **config_parser(train_config)}

        # construct a model and an optimizer.
        model = PotentialModel(self.train_config['gat_node_dim_list'],
                               self.train_config['energy_readout_node_list'],
                               self.train_config['force_readout_node_list'],
                               self.train_config['stress_readout_node_list'],
                               self.train_config['head_list'],
                               self.train_config['bias'],
                               self.train_config['negative_slope'],
                               self.device,
                               self.train_config['tail_readout_no_act']
                               )

        optimizer = optim.Adam(model.parameters(),
                              lr=self.train_config['learning_rate'],
                              weight_decay=self.train_config['weight_decay'])

        # # reset parameters
        # nn.init.orthogonal_(model.stress_outlayer.weight)
        # for l in model.stress_readout_layers:
        #     if hasattr(l, 'weight'):
        #         nn.init.orthogonal_(l.weight)

        # self.writer.add_graph(model, self._dataset[0][0], )
        # load stat dict if there exists.
        if os.path.exists(os.path.join(self.train_config['model_save_dir'],
                                       'agat_state_dict.pth')):
            try:
                checkpoint = load_state_dict(self.train_config['model_save_dir'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model = model.to(self.device)
                model.device = self.device
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f'User info: Model and optimizer state dict loaded successfully from {self.train_config["model_save_dir"]}.', file=self.log)
            except:
                print('User warning: Exception catched when loading model and optimizer state dict.', file=self.log)
        else:
            print('User info: Checkpoint not detected', file=self.log)

        # transfer learning.
        if self.train_config['transfer_learning']:
            for param in model.gat_layers.parameters():
                param.requires_grad = False

            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100,
                                                   gamma=0.1, verbose=True)

        # early stop
        if self.train_config['early_stop']:
            stopper = EarlyStopping(model, self.log,
                                    patience=self.train_config['stop_patience'],
                                    model_save_dir=self.train_config['model_save_dir'])

        # loss function
        criterion = self.train_config['criterion']
        a, b, c = self.train_config['a'], self.train_config['b'], self.train_config['c']
        mae = nn.L1Loss()
        r = PearsonR

        # log file
        print('========================================================================', file=self.log)
        print(model, file=self.log)
        print('========================================================================', file=self.log)
        if self.verbose > 1:
            print("Epoch Batch Energy_Loss Force_Loss Stress_Loss Total_Loss Dur_(s) Train_info",
                  file=self.log)
        if self.verbose > 0:
            print("Epoch Energy_Loss Force_Loss Stress_Loss Total_Loss \
Energy_MAE Force_MAE Stress_MAE Energy_R Force_R Stress_R Dur_(s) Validation_info",
              file=self.log)


        batch_step = 0
        # start the training
        start_time= time.time()
        for epoch in range(self.train_config['epochs']):
            # release GPU memory
            # torch.cuda.empty_cache()

            for i, (graph, props) in enumerate(self.train_loader):
                batch_step += 1
                energy_true = props['energy_true']
                force_true = graph.ndata['forces_true']
                stress_true = props['stress_true1000']

                # self.writer.add_scalar('stress_true_xx', stress_true[0].item(), epoch)
                self.writer.add_histogram('stress_true1000', torch.flatten(stress_true), batch_step)

                optimizer.zero_grad()
                energy_pred, force_pred, stress_pred = model.forward(graph)
                self.writer.add_histogram('stress_pred1000', stress_pred, batch_step)
                energy_loss = criterion(energy_pred, energy_true)
                if self._has_adsorbate:
                    force_true *= torch.reshape(graph.ndata['adsorbate']*self.train_config['adsorbate_coeff']+1.,
                                            (-1,1))
                    force_pred *= torch.reshape(graph.ndata['adsorbate']*self.train_config['adsorbate_coeff']+1.,
                                            (-1,1))

                # if self._has_adsorbate:
                #     force_loss = ...
                force_loss = criterion(force_pred, force_true)
                stress_loss = criterion(stress_pred, stress_true)
                self.writer.add_scalar('stress_loss1000', torch.flatten(stress_loss), batch_step)
                self.writer.add_scalar('force_loss', torch.flatten(force_loss), batch_step)
                total_loss = a*energy_loss + b*force_loss + c*stress_loss
                total_loss.backward()
                self.writer.add_histogram('u2e_weight_grad',
                                          torch.flatten(model.u2e.weight.grad),
                                          batch_step)
                self.writer.add_histogram('u2e_weight',
                                          torch.flatten(model.u2e.weight),
                                          batch_step)
                self.writer.add_histogram('skip_weight_grad',
                                          torch.flatten(model.skip_connect.weight.grad),
                                          batch_step)
                # self.writer.add_histogram('stress_outlayer_bias_grad',
                #                           torch.flatten(model.stress_outlayer.bias.grad),
                #                           batch_step)
                self.writer.add_histogram('last_force_readout_layer_weight_grad',
                                          torch.flatten(model.force_readout_layers[5].weight.grad),
                                          batch_step)
                self.writer.add_histogram('last_force_readout_layer_weight',
                                          torch.flatten(model.force_readout_layers[5].weight),
                                          batch_step)
                optimizer.step()
                dur = time.time() - start_time
                if self.verbose > 1:
                    print("{:0>5d} {:0>5d} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Train_info".format(
                          epoch, i, energy_loss.item(), force_loss.item(),
                          stress_loss.item(), total_loss.item(), dur),
                          file=self.log)

            # validation every epoch
            with torch.no_grad():
                energy_true_all, force_true_all, stress_true_all = [], [], []
                energy_pred_all, force_pred_all, stress_pred_all = [], [], []
                for i, (graph, props) in enumerate(self.val_loader):
                    energy_true_all.append(props['energy_true'])
                    force_true = graph.ndata['forces_true']
                    stress_true_all.append(props['stress_true1000'])
                    energy_pred, force_pred, stress_pred = model.forward(graph)
                    energy_pred_all.append(energy_pred)
                    if self._has_adsorbate:
                        force_true *= torch.reshape(graph.ndata['adsorbate']*self.train_config['adsorbate_coeff']+1.,
                                                    (-1,1))
                        force_pred *= torch.reshape(graph.ndata['adsorbate']*self.train_config['adsorbate_coeff']+1.,
                                                    (-1,1))
                    force_true_all.append(force_true)
                    force_pred_all.append(force_pred)
                    stress_pred_all.append(stress_pred)

                energy_true_all = torch.cat(energy_true_all)
                energy_pred_all = torch.cat(energy_pred_all)
                force_true_all = torch.cat(force_true_all)
                force_pred_all = torch.cat(force_pred_all)
                stress_true_all = torch.cat(stress_true_all)
                stress_pred_all = torch.cat(stress_pred_all)
                energy_loss = criterion(energy_pred_all, energy_true_all)
                force_loss = criterion(force_pred_all, force_true_all)
                stress_loss = criterion(stress_pred_all, stress_true_all)
                total_loss = a*energy_loss + b*force_loss + c*stress_loss

                energy_mae = mae(energy_pred_all, energy_true_all)
                force_mae = mae(force_pred_all, force_true_all)
                stress_mae = mae(stress_pred_all, stress_true_all)

                energy_r = r(energy_pred_all, energy_true_all)
                force_r = r(force_pred_all, force_true_all)
                stress_r = r(stress_pred_all, stress_true_all)
                self.writer.add_scalar('energy_r', energy_r, epoch)
                self.writer.add_scalar('force_r', force_r, epoch)
                self.writer.add_scalar('stress_r', stress_r, epoch)
                self.writer.add_scalar('stress_mae', stress_mae, epoch)
                # self.writer.add_scalar('stress_loss', stress_loss, epoch)
                if self.verbose > 0:
                    print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Validation_info".format(
                          epoch, energy_loss.item(), force_loss.item(), stress_loss.item(),
                          total_loss.item(), energy_mae.item(), force_mae.item(),
                          stress_mae.item(), energy_r.item(), force_r.item(),
                          stress_r.item(), dur),
                          file=self.log)

            if self.train_config['early_stop']:
                if stopper.step(total_loss, epoch, model, optimizer):
                    break
                if stopper.update:
                    energy = torch.cat([torch.reshape(energy_pred_all, (-1,1)),
                                        torch.reshape(energy_true_all, (-1,1))],
                                       dim=1).cpu().numpy()
                    force = torch.cat([torch.reshape(force_pred_all, (-1,1)),
                                       torch.reshape(force_true_all, (-1,1))],
                                      dim=1).cpu().numpy()
                    stress = torch.cat([torch.reshape(stress_pred_all, (-1,1)),
                                        torch.reshape(stress_true_all, (-1,1))],
                                       dim=1).cpu().numpy()
                    np.savetxt(os.path.join(self.train_config['output_files'],
                                            'energy_val_pred_true.txt'),
                               energy, fmt='%.8f')
                    np.savetxt(os.path.join(self.train_config['output_files'],
                                            'force_val_pred_true.txt'),
                               force, fmt='%.8f')
                    np.savetxt(os.path.join(self.train_config['output_files'],
                                            'stress_val_pred_true.txt'),
                               stress, fmt='%.8f')
            else:
                save_model(model, model_save_dir=self.train_config['model_save_dir'])
                save_state_dict(model, state_dict_save_dir=self.train_config['model_save_dir'])

            if self.train_config['transfer_learning']:
                exp_lr_scheduler.step()
            file_exit()

        # test with the best model
        try:
            checkpoint = load_state_dict(self.train_config['model_save_dir'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model = model.to(self.device)
            model.device = self.device
            print(f'User info: Test model state dict loaded successfully from {self.train_config["model_save_dir"]}.', file=self.log)
        except:
            print('User warning: Exception catched when loading test model state dict. \nUsing train model instead.', file=self.log)

        with torch.no_grad():
            energy_true_all, force_true_all, stress_true_all = [], [], []
            energy_pred_all, force_pred_all, stress_pred_all = [], [], []
            for i, (graph, props) in enumerate(self.test_loader):
                energy_true_all.append(props['energy_true'])
                force_true_all.append(graph.ndata['forces_true'])
                stress_true_all.append(props['stress_true1000'])
                energy_pred, force_pred, stress_pred = model.forward(graph)
                energy_pred_all.append(energy_pred)
                force_pred_all.append(force_pred)
                stress_pred_all.append(stress_pred)

            energy_true_all = torch.cat(energy_true_all)
            energy_pred_all = torch.cat(energy_pred_all)
            force_true_all = torch.cat(force_true_all)
            force_pred_all = torch.cat(force_pred_all)
            stress_true_all = torch.cat(stress_true_all)
            stress_pred_all = torch.cat(stress_pred_all)

            energy_loss = criterion(energy_pred_all, energy_true_all)
            force_loss = criterion(force_pred_all, force_true_all)
            stress_loss = criterion(stress_pred_all, stress_true_all)
            total_loss = a*energy_loss + b*force_loss + c*stress_loss

            energy_mae = mae(energy_pred_all, energy_true_all)
            force_mae = mae(force_pred_all, force_true_all)
            stress_mae = mae(stress_pred_all, stress_true_all)

            energy_r = r(energy_pred_all, energy_true_all)
            force_r = r(force_pred_all, force_true_all)
            stress_r = r(stress_pred_all, stress_true_all)

            print(f'''User info, model performance on testset: (No sample weight on the loss)
    Epoch      : {epoch}
    Energy loss: {energy_loss.item()}
    Force_Loss : {force_loss.item()}
    Stress_Loss: {stress_loss.item()}
    Total_Loss : {total_loss.item()}
    Energy_MAE : {energy_mae.item()}
    Force_MAE  : {force_mae.item()}
    Stress_MAE : {stress_mae.item()}
    Energy_R   : {energy_r.item()}
    Force_R    : {force_r.item()}
    Stress_R   : {stress_r.item()}
    Dur (s)    : {dur}''',
            file=self.log)

            energy = torch.cat([torch.reshape(energy_pred_all, (-1,1)),
                                torch.reshape(energy_true_all, (-1,1))],
                               dim=1).cpu().numpy()
            force = torch.cat([torch.reshape(force_pred_all, (-1,1)),
                               torch.reshape(force_true_all, (-1,1))],
                              dim=1).cpu().numpy()
            stress = torch.cat([torch.reshape(stress_pred_all, (-1,1)),
                                torch.reshape(stress_true_all, (-1,1))],
                               dim=1).cpu().numpy()
            np.savetxt(os.path.join(self.train_config['output_files'],
                                    'energy_test_pred_true.txt'),
                       energy, fmt='%.8f')
            np.savetxt(os.path.join(self.train_config['output_files'],
                                    'force_test_pred_true.txt'),
                       force, fmt='%.8f')
            np.savetxt(os.path.join(self.train_config['output_files'],
                                    'stress_test_pred_true.txt'),
                       stress, fmt='%.8f')

        self.log.close()
        self.writer.close()
        return total_loss

if __name__ == '__main__':
    from agat.lib import load_model
    from agat.data import CrystalGraph
    from agat.default_parameters import default_data_config
    from ase.io import read
    # import dgl
    # g_list, l_list = dgl.load_graphs(os.path.join(
    #     '..', 'all_graphs_generation_0_aimd_only.bin'))
    # graph = g_list[1] #.to('cuda')

    fname = os.path.join('agat', 'test', 'POSCAR')
    default_data_config['topology_only'] = True
    default_data_config['mode_of_NN'] = 'ase_dist'
    default_data_config['species'] = ['Ni', 'Co', 'Fe', 'Pd', 'Pt', 'H']
    atoms = read(fname)
    atoms.wrap()
    cg = CrystalGraph(**default_data_config)
    # model = load_model(model_save_dir='agat_model', device='cpu')

    PM = PotentialModel([6,30,20,10],
                        [20,10,10,5,1],
                        [20,10,10,5,3],
                        [20,10,10,5,6],
                        head_list=['free', 'mul'],
                        bias=True,
                        negative_slope=0.2,
                        device = 'cpu',
                        tail_readout_no_act=[2,2,2]
                        )

    atoms_r = atoms.copy()
    atoms_r.rotate('z', 180)
    atoms_r.wrap()
    graph, _ = cg.get_graph(atoms)
    graph_r, _ = cg.get_graph(atoms_r)

    with torch.no_grad():
        e, f, s = PM(graph)
        er, fr, sr = PM(graph_r)


    # compare edge distances
    for i, d in enumerate(graph.edata['dist']):
        src = graph.edges()[0][i]
        dst = graph.edges()[1][i]
        j = torch.where((graph_r.edges()[0] == src) & (graph_r.edges()[1] == dst))
        dr = graph_r.edata['dist'][j]
        diff = d-dr
        if torch.abs(diff) > 0.0:
            print(i, j)

        # graph.edata['dist'][1]
        # graph_r.edata['dist'][2]

    # energy, force, stress = PM.forward(graph)

    # # self = PM


    # import shutil
    # if os.path.isdir('agat_model'):
    #     shutil.rmtree('agat_model')
    #     print('remove agat_model')

    # FIX_VALUE = [1,3,6]
    # train_config = {
    # 'verbose': 2,
    # 'dataset_path': os.path.join('..', 'all_graphs_generation_0_aimd_only.bin'),
    # # 'dataset_path': os.path.join('..', 'all_graphs_generation_0.bin'),
    # 'model_save_dir': 'agat_model',
    # 'epochs': 200,
    # 'output_files': 'out_file',
    # 'device': 'cuda',
    # 'validation_size': 0.15,
    # 'test_size': 0.15,
    # 'early_stop': True,
    # 'stop_patience': 300,
    # 'gat_node_dim_list': [6, 100, 100, 100],
    # 'head_list': ['mul', 'div', 'free'],
    # 'energy_readout_node_list': [300, 100, 50, 30, 10, 3, FIX_VALUE[0]],
    # 'force_readout_node_list': [300, 100, 50, 30, 10, FIX_VALUE[1]],
    # 'stress_readout_node_list': [300, 100, 50, 30, 10, FIX_VALUE[2]],
    # 'bias': True,
    # 'negative_slope': 0.2,
    # 'criterion': nn.MSELoss(),
    # 'a': 1.0,
    # 'b': 1.0,
    # 'c': 1.0,
    # 'optimizer': 'adam', # Fix to sgd.
    # 'learning_rate': 0.0001,
    # 'weight_decay': 0.0, # weight decay (L2 penalty)
    # 'batch_size': 64,
    # 'val_batch_size': 400,
    # # 'validation_batches': 150,
    # 'transfer_learning': False,
    # 'trainable_layers': -4,
    # 'mask_fixed': False,
    # 'tail_readout_no_act': [3,3,3],
    # 'adsorbate': False, # indentify adsorbate or not when building graphs.
    # 'adsorbate_coeff': 20.0, # the importance of adsorbate atoms with respective to surface atoms.
    # 'transfer_learning': False}

    # f = Fit(**train_config)
    # f.fit()
    # # self = f
