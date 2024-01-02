# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:49:14 2023

@author: ZHANG Jun
"""

import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler

from .model import PotentialModel
from ..lib.model_lib import EarlyStopping, PearsonR, config_parser, save_model, load_state_dict, save_state_dict
from ..data.load_dataset import LoadDataset, Collater
from ..default_parameters import default_train_config
from ..lib.file_lib import file_exit

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

        # start the training
        start_time= time.time()
        for epoch in range(self.train_config['epochs']):
            # release GPU memory
            # torch.cuda.empty_cache()

            for i, (graph, props) in enumerate(self.train_loader):
                energy_true = props['energy_true']
                force_true = graph.ndata['forces_true']
                stress_true = props['stress_true']
                optimizer.zero_grad()
                energy_pred, force_pred, stress_pred = model.forward(graph)
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
                total_loss = a*energy_loss + b*force_loss + c*stress_loss
                total_loss.backward()
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
                    stress_true_all.append(props['stress_true'])
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
                stress_true_all.append(props['stress_true'])
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
        return total_loss

if __name__ == '__main__':
    FIX_VALUE = [1,3,6]
    train_config = {
        'verbose': 1,
    'dataset_path': os.path.join('dataset', 'concated_graphs.bin'),
    'model_save_dir': 'agat_model',
    'epochs': 1000,
    'output_files': 'out_file',
    'device': 'cuda:0',
    'validation_size': 0.15,
    'test_size': 0.15,
    'early_stop': True,
    'stop_patience': 300,
    'gat_node_dim_list': [6, 100, 100, 100],
    'head_list': ['mul', 'div', 'free'],
    'energy_readout_node_list': [300, 100, 50, 30, 10, 3, FIX_VALUE[0]],
    'force_readout_node_list': [300, 100, 50, 30, 10, FIX_VALUE[1]],
    'stress_readout_node_list': [300, FIX_VALUE[2]],
    'bias': True,
    'negative_slope': 0.2,
    'criterion': nn.MSELoss(),
    'a': 1.0,
    'b': 1.0,
    'c': 0.0,
    'optimizer': 'adam', # Fix to sgd.
    'learning_rate': 0.0005,
    'weight_decay': 0.0, # weight decay (L2 penalty)
    'batch_size': 64,
    'val_batch_size': 400,
    # 'validation_batches': 150,
    'transfer_learning': False,
    'trainable_layers': -4,
    'mask_fixed': False,
    'tail_readout_no_act': [3,3,1],
    'adsorbate': False, # indentify adsorbate or not when building graphs.
    'adsorbate_coeff': 20.0, # the importance of adsorbate atoms with respective to surface atoms.
    'transfer_learning': False}

    f = Fit(**train_config)
    f.fit()
