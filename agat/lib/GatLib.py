# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:34:59 2021

@author: ZHANG Jun
"""

import tensorflow as tf
from scipy.stats import pearsonr
import os
import json
from dgl.data.utils import save_graphs
import multiprocessing
import time
import numpy as np
# from GatApp import GatApp # cannot import functions from each other

def config_parser(config):
    if isinstance(config, dict):
        return config
    elif isinstance(config, str):
        with open(config, 'r') as config_f:
            return json.load(config_f)
    elif isinstance(config, type(None)):
        return {}
    else:
        raise TypeError('Wrong configuration type.')

class EarlyStopping:
    def __init__(self, model, graph, logger, patience=10, folder='files'):
        self.model      = model
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.update     = None
        self.early_stop = False
        self.folder     = folder
        self._graph     = graph
        self.save_structure()



        # save a graph
        save_graphs(os.path.join(self.folder, 'graph_tmp.bin'), [graph],
                    {'prop': tf.constant([0.0])})

        self.logger     = logger

    def step(self, score, model):
        # score = mae
        if self.best_score is None:
            self.best_score = score
            self.update     = True
            self.save_checkpoint(model)
        elif score > self.best_score:
            self.counter += 1
            print(f'User log: EarlyStopping counter: {self.counter} out of {self.patience}',
                  file=self.logger)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update     = True
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        model.save_weights(os.path.join(self.folder, 'gat.ckpt'))
        if model.batch_normalization:
            non_trainable = [x.numpy() for x in model.non_trainable_variables]
            with open(os.path.join(self.folder, 'Non_trainable_variables.npy'), 'wb') as f:
                for var in non_trainable:
                    np.save(f, var)

        print(f'User log: Save model weights with the best score: {self.best_score}',
              file=self.logger)

    def save_structure(self):
        gat_model = {}
        gat_model['Description']          = 'This is the architecture of a GAT model.'
        gat_model['num_gat_out_list']     = self.model.num_gat_out_list
        gat_model['num_readout_out_list'] = self.model.num_readout_out_list
        try:
            gat_model['head_list_en']     = self.model.head_list_en
        except AttributeError:
            pass
        try:
            gat_model['head_list_force']  = self.model.head_list_force
            gat_model['tail_readout_no_act'] = self.model.tail_readout_no_act
        except AttributeError:
            pass

        gat_model['att_activation']       = 'dgl.ops.edge_softmax'
        gat_model['embed_activation']     = self.model.embed_act_str
        gat_model['readout_activation']   = self.model.readout_act_str
        gat_model['bias']                 = self.model.bias
        gat_model['negative_slope']       = self.model.negative_slope
        gat_model['batch_normalization']  = self.model.batch_normalization

        node_depth                        = self._graph.ndata['h'].shape[1]
        gat_model['depth_of_node_representation'] = node_depth

        with open(os.path.join(self.folder, 'gat_model.json'), 'w') as f:
            json.dump(gat_model, f, indent=4)

def forward(model, graph): # , mean_prop=False): mean_prp: deprecated!
    """
    Parameters
    ----------
    model: a GAT model for crystal

    graph: a DGL graph

    mean_prop: whether to divide the graph output by the number of nodes.

    Return
    ----------
    `tf.reduce_mean(logit)`: `tf tensor`
        Prediction of the input `graph`
    or
    `tf.reduce_sum(logit)`: `tf tensor`
        Prediction of the input `graph`

    Note
    ----------
    The input graph will be modified by this function. So `clone()` it first.

    Modify this function when necessary.
        Using `tf.reduce_mean(logit)` when predict property per atom.

        Using `tf.reduce_sum(logit)` when predict the total property,
        for example, total energy.

    """
    features = graph.ndata['h']
    logit = model(features, graph)
    return tf.reduce_mean(logit)
    # if mean_prop:
    #     return tf.reduce_mean(logit)
    # else:
    #     return tf.reduce_sum(logit)

def forward_serial(model, graph, energy_list, forces_list, calculator): # clone graph before calling this function
    energy_per_atom, force_mat = calculator.get_force_energy_new(model, graph)
    energy_list.append(energy_per_atom)
    forces_list.append(force_mat)
    # print(energy_per_atom, energy_list)

def forward_parallel(model, graph_list, calculator):
    energy_list = multiprocessing.Manager().list()
    forces_list = multiprocessing.Manager().list()

    p           = multiprocessing.Pool(10)
    features = graph_list[1].ndata['h']
    p.apply_async(model, args=(features, graph_list[1],))
    # result2 = p.apply_async(model, args=(features, graph_list[1],))
    # print(result1.get())
    # p.apply_async(forward_serial, args=(model, graph_list[1], energy_list, forces_list, calculator,))
    # p.apply_async(forward_serial, args=(model, graph_list[2], energy_list, forces_list, calculator,))
    p.close()
    p.join()


    # energy_list, forces_list = [], []
    # # p           = multiprocessing.Pool(10)
    # with multiprocessing.Pool(processes = 10) as pool:
    #     # print('Here!!!')
    #     # for graph in graph_list:
    #         # forward_serial(model, graph, energy_list, forces_list, calculator)
    #         # p.apply_async(time.sleep, args=(1))
    #         # print(time.time())
    #         # args=(model, graph, energy_list, forces_list, calculator,)
    #         # print(type(args))
    #         # temp = pool.apply_async(func=forward_serial, args=(model, graph, energy_list, forces_list, calculator,),)
    #     temp = [pool.apply_async(func=forward_serial, args=(model, graph, energy_list, forces_list, calculator,),) for graph in graph_batch]
    #     # results = [t.get() for t in temp]
    #         # print(result)
    #         # result.get()
    #         # p.apply_async(forward_serial, args=(model, graph, energy_list, forces_list, calculator,))
    # # pool.close()
    # # pool.join()

    # # return list(energy_list), list(forces_list)
    # # return tf.convert_to_tensor(energy_list, dtype='float32'), tf.convert_to_tensor(forces_list, dtype='float32')
    # # return tf.convert_to_tensor(list(energy_list), dtype='float32'), tf.convert_to_tensor(list(forces_list), dtype='float32'), result
    # # return results

def load_gat_weights(model, graph, ckpt_path, logger, device): # clone graph before calling this function
    """
    Parameters
    ----------
    model: a fresh GAT model without trainable `variables`
    graph: a DGL graph
    ckpt_path: str
        Path to the saved checkpoint files

    Return
    ----------
    model: GAT model
        A GAT model with trainable `variables`
    """
    assert os.path.exists(ckpt_path + '.index'), "Checkpoint file not found when loading weights."
    # Varibles in model should be instantized and built first. See: https://github.com/tensorflow/tensorflow/issues/27937
    # forward(model, graph)    # build a gat model with varibles.
    # print(f'graph device: {graph.device}')
    # print(f'device passed into this function: {device}')

    graph = graph.to(device)
    # print(f'graph device: {graph.device}')
    with tf.device(device):
        model(graph)
    try:
        load_status = model.load_weights(ckpt_path)
        load_status.assert_consumed()  # check the load status
    except:
        print("User log: Weights detected but incompatible.", file=logger)

def accuracy(metrics, y_true, y_pred):
    metrics.update_state([y_true], [y_pred])
    return metrics.result().numpy()

def evaluate(metrics, model, g, y_true): # for now, this function is deprecated
    features = g.ndata['h']
    logits   = model(features, g)
    y_pred   = tf.reduce_mean(logits)
    return accuracy(metrics, y_true, y_pred)

def PearsonR(y_pred, y_true):
    return pearsonr(y_pred, y_true)[0]

def get_src_dst_data(graph):
    src, dst = graph.edges()
    src_data = tf.convert_to_tensor([graph.ndata['h'][x] for x in src])
    dst_data = tf.convert_to_tensor([graph.ndata['h'][x] for x in dst])
    return src_data, dst_data



# debug
if __name__ == '__main__':
    calculator = GatApp('1000_test\ckpt')
    graph_batch = [x.clone() for x in graph_list[0:10]]
    model = calculator.model
    start = time.time()
    # energy_list, forces_list, result = forward_parallel(model, graph_batch, calculator)
    forward_parallel(model, graph_batch, calculator)
    print(time.time() - start)
