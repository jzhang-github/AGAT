# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:28:54 2021

@author: ZHANG Jun
"""

import numpy as np
import tensorflow as tf
import os
import time
import sys
import dgl

from .GatEnergyModel import EnergyGat
from .GatForceModel import ForceGat
# from modules.Crystal2Graph import ReadGraphs, TrainValTestSplit
from ..lib.GatLib import EarlyStopping, PearsonR, load_gat_weights, config_parser # get_src_dst_data #, evaluate
from ..default_parameters import default_train_config
from ..data.data import ReadGraphs

class Train():
    def __init__(self, **train_config):
        self.train_config = {**default_train_config, **config_parser(train_config)}

        data_config = config_parser(os.path.join(self.train_config['dataset_path'], 'graph_build_scheme.json'))
        graph_reader = ReadGraphs(**{**data_config, **{'load_from_binary':True}})
        print(f'Reading graphs from: {self.train_config["dataset_path"]} ......')
        self.graph_list, self.graph_labels = graph_reader.read_all_graphs()

        # read previously subset index
        train_index, validation_index, test_index = (np.loadtxt(os.path.join(self.train_config['dataset_path'], 'train.txt'), dtype=int),
                                                     np.loadtxt(os.path.join(self.train_config['dataset_path'], 'validation.txt'), dtype=int),
                                                     np.loadtxt(os.path.join(self.train_config['dataset_path'], 'test.txt'), dtype=int))

        self.train_index, self.validation_index, self.test_index = train_index, validation_index, test_index
        self.graph_eg = self.graph_list[0]

        # # train dataset
        # self.validation_graphs = [self.graph_list[x] for x in self.validation_index]
        # self.validation_prop   = [self.graph_labels['energy_true'][x] for x in self.validation_index]

        # validation dataset
        self.validation_graphs = [self.graph_list[x] for x in self.validation_index]
        self.validation_prop   = [self.graph_labels['energy_true'][x] for x in self.validation_index]

        # test dataset
        self.test_graphs = [self.graph_list[x] for x in self.test_index]
        self.test_prop   = [self.graph_labels['energy_true'][x] for x in self.test_index]

        # check inputs
        self.train_config['batch_size'] = min(self.train_config['batch_size'],
                                              len(self.train_index),
                                              len(self.validation_index),
                                              len(self.test_index))
        self.train_config['validation_freq'] = min(self.train_config['validation_freq'],
                                                   len(self.train_index))
        self.train_config['validation_samples'] = min(self.train_config['validation_freq'],
                                                   len(self.validation_index))
        self.train_config['val_batch_size'] = min(self.train_config['val_batch_size'],
                                                  len(self.validation_index))

        # if self.train_config['load_graphs_on_gpu']:
        # # if True:
        #     self.graph_list = [x.to("/gpu:0") for x in self.graph_list]
        #     self.validation_graphs = [x.to("/gpu:0") for x in self.validation_graphs]
        #     self.test_graphs = [x.to("/gpu:0") for x in self.test_graphs]

        # prepare out file
        if not os.path.exists(self.train_config['output_files']):
            os.mkdir(self.train_config['output_files'])
        if not os.path.exists(os.path.join(self.train_config['output_files'], 'energy_ckpt')):
            os.mkdir(os.path.join(self.train_config['output_files'], 'energy_ckpt'))
        if not os.path.exists(os.path.join(self.train_config['output_files'], 'force_ckpt')):
            os.mkdir(os.path.join(self.train_config['output_files'], 'force_ckpt'))

    def fit_energy_model(self):
        # open log file
        energy_log = open('energy_train.log', 'w', buffering=1)

        # define computing device.
        if self.train_config['gpu_for_energy_train'] < 0:
            energy_model_device = "/cpu:0"
        else:
            energy_model_device = "/gpu:{}".format(self.train_config['gpu_for_energy_train'])
        print('User log: specified device for energy model:', energy_model_device, file=energy_log)

        # initialize a model
        energy_model = EnergyGat(self.train_config['energy_GAT_out_list'],
                     num_readout_out_list=self.train_config['energy_model_readout_list'],
                     head_list_en=self.train_config['energy_model_head_list'],
                     embed_activation=self.train_config['embed_activation'],
                     readout_activation=self.train_config['readout_activation'],
                     bias=self.train_config['bias'],
                     negative_slope=self.train_config['negative_slope'])

        # early stop
        if self.train_config['early_stop'] and self.train_config['train_energy_model']:
            stopper = EarlyStopping(energy_model, self.graph_eg, energy_log,
                                    patience=self.train_config['stop_patience'],
                                    folder=os.path.join(self.train_config['output_files'], 'energy_ckpt'))

        # detect the saved weights. Load them if possible.
        model_save_path = os.path.join(self.train_config['output_files'], 'energy_ckpt', 'gat.ckpt')
        print("User log: Model save path:", model_save_path, file=energy_log)
        if os.path.exists(model_save_path + '.index'):    # If the `checkpoint` file is saved before, try to load it.
            print("User log: Loading saved weights......", file=energy_log)
            g_init = self.graph_eg.clone() # Initialize trainable variables first.
            load_gat_weights(energy_model, g_init, model_save_path, energy_log, energy_model_device)
        else:
            print("User log: Weights not detected.", file=energy_log)

        if self.train_config['transfer_learning']:
            nontrainable_layers = [x for x in range(len(energy_model.layers))][1:self.train_config['trainable_layers']]
            for layer_i in nontrainable_layers:
                energy_model.layers[layer_i].trainable = False
# =============
        # start the training
        start_time = time.time()
        step       = 1

        # train
        flag = False
        for epoch in range(self.train_config['epochs']):
            np.random.shuffle(self.train_index) # shuffle training set every epoch

            # Split the data in batchs.
            train_index_batch = np.array_split(self.train_index, int(len(self.train_index) / self.train_config['batch_size'] + 1))
            en_pred_all, en_true_all = [], []

            for i_index, i in enumerate(train_index_batch):
                with tf.device("/cpu:0"):
                    g_batch = [self.graph_list[x] for x in i]
                    g_batch = dgl.batch(g_batch)
                    en_true = tf.stack([self.graph_labels['energy_true'][x] for x in i], axis=0)

                with tf.device(energy_model_device):
                    en_true = tf.identity(en_true)
                    g_batch = g_batch.to(energy_model_device)
                    with tf.GradientTape() as tape:
                        en_pred     = energy_model(g_batch)
                        batch_nodes = g_batch.batch_num_nodes()
                        en_pred     = tf.split(en_pred, batch_nodes)
                        en_pred     = tf.convert_to_tensor([tf.reduce_mean(x) for x in en_pred]) #tf.reduce_mean(en_pred, axis=1)

                        en_pred_all.append(en_pred)
                        en_true_all.append(en_true)
                        step += len(i)
                        loss_value = self.train_config['energy_loss_fcn'](en_true, en_pred) # calculate loss

                        # regularization. Specified in `Settings`
                        if self.train_config['L2_reg']:
                            for weight in energy_model.trainable_weights:
                                loss_value += self.train_config['weight_decay']*tf.nn.l2_loss(weight)
                    grads = tape.gradient(loss_value, energy_model.trainable_weights)
                    self.train_config['energy_optimizer'].apply_gradients(zip(grads, energy_model.trainable_weights))

                    batch_mae = self.train_config['mae'](en_true, en_pred)  # 不使用tf.keras.metrics.MeanAbsoluteError()。使用metrics，同样的输入，连续计算，输出不同。

                print("Epoch: {:0>4d} | Batch: {:0>4d} | Train_Loss (mse): {:1.4f} | Batch_MAE: {:1.4f} | Dur: {:.1f} s"
                      .format(epoch, i_index, loss_value, batch_mae, time.time() - start_time), file=energy_log)

                # evaluate every `validation_freq` epochs. `val` refers to validation.
                # `pred` refers to predict
                if step % self.train_config['validation_freq'] < self.train_config['batch_size'] and step > self.train_config['batch_size']:
                    print("========================================================================", file=energy_log)
                    en_val_true, en_val_pred = [], []

                    # with tf.device(energy_model_device):
                    validation_list = np.random.choice(range(len(self.validation_index)),
                                                       size=min(self.train_config['validation_samples'], len(self.validation_index)), replace=False)
                    validation_batch = np.array_split(validation_list, len(validation_list) / self.train_config['val_batch_size'])
                    for i in validation_batch:
                        with tf.device("/cpu:0"):
                            g_batch = [self.validation_graphs[x] for x in i]
                            g_batch = dgl.batch(g_batch)
                            en_true = tf.stack([self.validation_prop[x] for x in i], axis=0)

                        with tf.device(energy_model_device):
                            g_batch = g_batch.to(energy_model_device)
                            en_pred = energy_model(g_batch)
                            batch_nodes = g_batch.batch_num_nodes()
                            en_pred = tf.split(en_pred, batch_nodes)
                            en_pred = tf.convert_to_tensor([tf.reduce_mean(x) for x in en_pred])

                        with tf.device("/cpu:0"):
                            en_val_pred.append(tf.identity(en_pred))
                            en_val_true.append(en_true)

                    en_val_true, en_val_pred = tf.reshape(tf.concat(en_val_true, axis=0), shape=(-1)), tf.reshape(tf.concat(en_val_pred, axis=0), shape=(-1))
                    val_mae = self.train_config['mae'](en_val_true, en_val_pred)

                    dur = time.time() - start_time
                    en_true_all, en_pred_all = tf.reshape(tf.concat(en_true_all, axis=0), shape=(-1)), tf.reshape(tf.concat(en_pred_all, axis=0), shape=(-1))
                    print("Epoch: {:0>4d} | Train_MAE : {:1.4f} | Train_PearsonR: {:1.4f} | Val_MAE: {:1.4f} | Val_PearsonR: {:1.4f} | Dur: {:.1f} s".
                          format(epoch, self.train_config['mae'](en_true_all, en_pred_all),
                                 PearsonR(en_true_all, en_pred_all), val_mae,
                                 PearsonR(en_val_true, en_val_pred), dur),file=energy_log)

                    if self.train_config['early_stop'] and self.train_config['train_energy_model']:
                        if stopper.step(val_mae, energy_model):
                            print('User log: model summary:', file=energy_log)
                            energy_model.summary()
                            print('User log: model summary done.', file=energy_log)
                            flag = True
                            break
                        if stopper.update:
                            np.savetxt(os.path.join(self.train_config['output_files'], 'en_train_true.txt'), en_true_all, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                            np.savetxt(os.path.join(self.train_config['output_files'], 'en_train_pred.txt'), en_pred_all, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                            np.savetxt(os.path.join(self.train_config['output_files'], 'en_val_true.txt'),   en_val_true, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                            np.savetxt(os.path.join(self.train_config['output_files'], 'en_val_pred.txt'),   en_val_pred, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                        en_true_all, en_pred_all = [], []
                    else:
                        energy_model.save_weights(model_save_path)
                if flag:
                    break

        print('User log: model summary:', file=energy_log)
        energy_model.summary()
        print('User log: model summary done.', file=energy_log)
# =============
        # predict test set.
        y_true, y_pred = self.test_prop, [] # saved on GPU.
        # with tf.device(energy_model_device):
        batch_index = np.array_split(range(len(self.test_graphs)), len(self.test_graphs) / self.train_config['val_batch_size'])
        for i in batch_index:
            with tf.device("/cpu:0"):
                g_batch = [self.test_graphs[x] for x in i]
                g_batch = dgl.batch(g_batch)

            with tf.device(energy_model_device):
                g_batch = g_batch.to(energy_model_device)
                en_pred = energy_model(g_batch)
                batch_nodes = g_batch.batch_num_nodes()
                en_pred = tf.split(en_pred, batch_nodes)
                en_pred = tf.convert_to_tensor([tf.reduce_mean(x) for x in en_pred])

            with tf.device("/cpu:0"):
                y_pred.append(tf.identity(en_pred))
                mae_value = self.train_config['mae'](tf.convert_to_tensor(y_true), tf.concat(y_pred, axis=0))
        print("User log: predict MAE: {:.4f}".format(mae_value), file=energy_log)

        np.savetxt(os.path.join(self.train_config['output_files'], 'en_test_final_true.txt'), tf.convert_to_tensor(y_true).numpy(), fmt='%.8f')
        np.savetxt(os.path.join(self.train_config['output_files'], 'en_test_final_pred.txt'), tf.concat(y_pred, axis=0).numpy(), fmt='%.8f')

        energy_log.close()

    def fit_force_model(self):
        # open log file
        logf = open('force_train.log', 'w', buffering=1)

        # define computing device.
        if self.train_config['gpu_for_force_train'] < 0:
            device = "/cpu:0"
        else:
            device = "/gpu:{}".format(self.train_config['gpu_for_force_train'])
        print('User log: specified device for force model:', device, file=logf)

        model = ForceGat(self.train_config['force_GAT_out_list'],
                     num_readout_out_list=self.train_config['force_model_readout_list'],
                     head_list_force=self.train_config['force_model_head_list'],
                     embed_activation=self.train_config['embed_activation'],
                     readout_activation=self.train_config['readout_activation'],
                     bias=self.train_config['bias'],
                     negative_slope=self.train_config['negative_slope'],
                     batch_normalization=self.train_config['batch_normalization'],
                     tail_readout_no_act=self.train_config['tail_readout_noact'])

        # early stop
        if self.train_config['early_stop'] and self.train_config['train_force_model']:
            stopper = EarlyStopping(model, self.graph_eg, logf,
                                    patience=self.train_config['stop_patience'],
                                    folder=os.path.join(self.train_config['output_files'], 'force_ckpt'))

        # detect the saved weights. Load them if possible.
        model_save_path = os.path.join(self.train_config['output_files'], 'foce_ckpt', 'gat.ckpt')
        print("User log: Model save path:", model_save_path, file=logf)
        if os.path.exists(model_save_path + '.index'):    # If the `checkpoint` file is saved before, try to load it.
            print("User log: Loading saved weights......", file=logf)
            g_init = self.graph_eg.clone() # Initialize trainable variables first.
            load_gat_weights(model, g_init, model_save_path, logf, device)
            # if batch_normalization:
            #     with open(os.path.join(Project, CKPT, 'Non_trainable_variables.npy'), 'rb') as f:
            #         model.bn.moving_mean     = tf.Variable(np.load(f), trainable=False)
            #         model.bn.moving_variance = tf.Variable(np.load(f), trainable=False)
        else:
            print("User log: Weights not detected.", file=logf)

        if self.train_config['transfer_learning']:
            nontrainable_layers = [x for x in range(len(model.layers))][1:self.train_config['trainable_layers']]
            for layer_i in nontrainable_layers:
                model.layers[layer_i].trainable = False

        start_time     = time.time()
        step           = 1

        # train
        flag = False
        for epoch in range(self.train_config['epochs']):
            np.random.shuffle(self.train_index) # shuffle training set every epoch

            # Split the data in batchs.
            train_index_batch = np.array_split(self.train_index, int(len(self.train_index) / self.train_config['batch_size'] + 1))
            force_pred_all, force_true_all = [], []

            for i_index, i in enumerate(train_index_batch):
                with tf.device("/cpu:0"):
                    g_batch = [self.graph_list[x] for x in i]
                    g_batch = dgl.batch(g_batch)
                    force_true = g_batch.ndata['forces_true']

                with tf.device(device):
                    g_batch = g_batch.to(device)
                    force_true = tf.identity(force_true)
                    with tf.GradientTape() as tape:
                        force_pred = model(g_batch, training=True)

                        # mask fixed atoms. Comment the following lines for HECC.
                        ######################################################
                        if self.train_config['mask_fixed']:
                            constraints = tf.cast(g_batch.ndata['constraints'], dtype='bool')
                            force_pred  = tf.reshape(force_pred[constraints], [-1, 3]) # g.ndata['constraints'] can be considered as the `tf.boolean_mask`, the output will be flattened to a 1D tensor.
                            force_true  = tf.reshape(force_true[constraints], [-1, 3])
                        ######################################################

                        force_pred_all.append(force_pred)
                        force_true_all.append(force_true)
                        step += len(i)
                        if self.train_config['adsorbate']:
                            sample_weight = tf.where(tf.cast(g_batch.ndata['adsorbate'], dtype='bool'), self.train_config['adsorbate_coeff'], 1.0)
                            # sample_weight = tf.transpose(tf.stack([sample_weight, sample_weight, sample_weight], axis=0))
                            if self.train_config['mask_fixed']:
                                sample_weight = sample_weight[tf.math.reduce_any(constraints, axis=1)]
                        else:
                            sample_weight = None

                        loss_value = self.train_config['force_loss_fcn'](force_true, force_pred, sample_weight)

                        # regularization. Specified in `Settings`
                        if self.train_config['L2_reg']:
                            for weight in model.trainable_weights:
                                loss_value += self.train_config['weight_decay']*tf.nn.l2_loss(weight)
                    grads = tape.gradient(loss_value, model.trainable_weights)
                    self.train_config['force_optimizer'].apply_gradients(zip(grads, model.trainable_weights))

                    force_batch_mae = self.train_config['mae'](force_true, force_pred)  # 不使用tf.keras.metrics.MeanAbsoluteError()。若使用metrics，同样的输入，连续计算，输出不同。

                    print("Epoch: {:0>4d} | Batch: {:0>4d} | Loss (mse): {:1.4f} | MAE: {:1.4f} | Dur: {:.1f} s"
                          .format(epoch, i_index, loss_value, force_batch_mae, time.time() - start_time), file=logf)

                # evaluate every `validation_freq` epochs/graphs. `val` refers to validation.
                # `pred` refers to predict
                if step % self.train_config['validation_freq'] < self.train_config['batch_size'] and step > self.train_config['batch_size']:
                    print("========================================================================", file=logf)
                    force_val_true, force_val_pred = [], []
                    # with tf.device(device):
                    # save time if not forward all validation samples.
                    validation_list = np.random.choice(range(len(self.validation_index)),
                                                       size=min(self.train_config['validation_samples'], len(self.validation_index)), replace=False)
                    validation_batch = np.array_split(validation_list, len(validation_list) / self.train_config['val_batch_size'])
                    for i in validation_batch:
                        with tf.device("/cpu:0"):
                            g_batch    = [self.validation_graphs[x] for x in i]
                            g_batch    = dgl.batch(g_batch)
                            force_true = g_batch.ndata['forces_true']

                        with tf.device(device):
                            g_batch = g_batch.to(device)
                            force_pred = model(g_batch, training=False)

                            # mask fixed atoms. Comment the following lines for HECC.
                            ######################################################
                            if self.train_config['mask_fixed']:
                                constraints = tf.cast(g_batch.ndata['constraints'], dtype='bool')
                                force_pred  = force_pred[constraints] # g.ndata['constraints'] can be considered as the `tf.boolean_mask`, the output will be flattened to a 1D tensor.
                                force_true  = force_true[constraints]
                            ######################################################

                        with tf.device("/cpu:0"):
                            force_val_pred.append(tf.identity(force_pred))
                            force_val_true.append(force_true)

                    force_val_true, force_val_pred = tf.concat(force_val_true, axis=0), tf.concat(force_val_pred, axis=0)
                    force_val_mae = self.train_config['mae'](force_val_true, force_val_pred)

                    force_true_all, force_pred_all = tf.concat(force_true_all, axis=0), tf.concat(force_pred_all, axis=0)
                    dur = time.time() - start_time
                    print("Epoch: {:0>4d} | Train_Force_MAE : {:1.4f} | Train_Force_PearsonR: {:1.4f} | Force_Val_MAE: {:1.4f} | Force_Val_PearsonR: {:1.4f} | Dur: {:.1f} s".
                          format(epoch,
                                 self.train_config['mae'](force_true_all, force_pred_all), PearsonR(tf.reshape(force_true_all, [-1]), tf.reshape(force_pred_all, [-1])),
                                 force_val_mae, PearsonR(tf.reshape(force_val_true, [-1]), tf.reshape(force_val_pred, [-1])), dur),
                                 file=logf)

                    if self.train_config['early_stop'] and self.train_config['train_force_model']:
                        if stopper.step(force_val_mae, model):
                            print('User log: model summary:', file=logf)
                            model.summary()
                            print('User log: model summary done.', file=logf)
                            flag = True
                            break
                        if stopper.update:
                            np.savetxt(os.path.join(self.train_config['output_files'], 'force_train_true.txt'), tf.reshape(force_true_all, [-1]), fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                            np.savetxt(os.path.join(self.train_config['output_files'], 'force_train_pred.txt'), tf.reshape(force_pred_all, [-1]), fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                            np.savetxt(os.path.join(self.train_config['output_files'], 'force_val_true.txt'),   tf.reshape(force_val_true, [-1]), fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                            np.savetxt(os.path.join(self.train_config['output_files'], 'force_val_pred.txt'),   tf.reshape(force_val_pred, [-1]), fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                        force_true_all, force_pred_all = [], []

                    else:
                        model.save_weights(model_save_path)
                if flag:
                    break

        print('User log: model summary:', file=logf)
        model.summary()
        print('User log: model summary done.', file=logf)

        # predict test set.
        y_true, y_pred = [g.ndata['forces_true'] for g in self.test_graphs], [] # saved on GPU

        # with tf.device(device):
        batch_index = np.array_split(range(len(self.test_graphs)), len(self.test_graphs) / self.train_config['batch_size'])
        for i in batch_index:
            with tf.device("/cpu:0"):
                g_batch = [self.test_graphs[x] for x in i]
                g_batch = dgl.batch(g_batch)

            with tf.device(device):
                g_batch = g_batch.to(device)
                y_pred_tmp = model(g_batch)

            with tf.device("/cpu:0"):
                y_pred.append(tf.identity(y_pred_tmp))

        y_true, y_pred = tf.concat(y_true, axis=0), tf.concat(y_pred, axis=0)
        mae_value      = self.train_config['mae'](y_true, y_pred)

        print("User log: Final predict MAE of testset is: {:.4f}".format(mae_value), file=logf)
        np.savetxt(os.path.join(self.train_config['output_files'], 'force_test_final_true.txt'), tf.reshape(y_true, [-1]).numpy(), fmt='%.8f')
        np.savetxt(os.path.join(self.train_config['output_files'], 'force_test_final_pred.txt'), tf.reshape(y_pred, [-1]).numpy(), fmt='%.8f')
        logf.close()

if __name__ == '__main__':
    at = Train()
    at.fit_energy_model()
    at.fit_force_model()
