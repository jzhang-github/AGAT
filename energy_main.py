# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:28:54 2021

@author: ZHANG Jun
"""

import numpy as np
import tensorflow as tf
import os
from modules.Crystal2Graph import ReadGraphs, TrainValTestSplit
from modules.GatLib import EarlyStopping, PearsonR, load_gat_weights # get_src_dst_data #, evaluate
import time
import sys
from modules.GatEnergyModel import GAT
import json
from dgl.data.utils import save_graphs, load_graphs
import dgl

# =============================================================================
# Settings
# =============================================================================
if __name__ == '__main__':
    Train              = True
    epochs             = 3000
    Project            = 'project'           # file name of user-defined project. It contains dataset, file, ckpt, log... and so on.
    new_training       = False # If true, the graphs are loaded from scratch. The training, validation, and test dataset are split, rather than loaded.
    gpu                = 0
    validation_size    = 0.15  # float or int. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
    test_size          = 0.15  # float or int. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
    early_stop         = True  # if False, the model weights will be saved every epoch.
    stop_patience      = 300   # early stop patience
    GAT_out_list       = [100, 100, 100]
    head_list_en       = ['mul', 'div', 'free']
    readout_list       = [200, 100, 50, 30, 10, 3, 1]
    bias               = True
    negative_slope     = 0.2
    embed_activation   ='LeakyReLU' # Alternatives: LeakyReLU, relu, tanh, softmax, edge_softmax
    readout_activation ='LeakyReLU' # Alternatives: LeakyReLU, relu, tanh, softmax, edge_softmax
    loss_fcn           = tf.keras.losses.MeanSquaredError()              # loss function
    optimizer          = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8) # Default lr: 0.001
    weight_decay       = 5e-5
    mae                = tf.keras.losses.MeanAbsoluteError()
    batch_size         = 256
    val_batch_size     = 400  # number of graphs for validation.
    L2_reg             = False # L2 regularization
    validation_freq    = 20000    # validation frequency. e.g. validation_freq=2 runs validation every 2 graphs.
    validation_samples = 20000 # validation samples that are randomly selected in the validation set.
    log_file           = 'v7_energy.log' #'energy.log' #str(Project)+'_1_.log' #'testv2.0.log' #'test.log' # The outputs of `print()` function. Default: None
    num_core_read      = 3    # specify the number of cores for reading graphs.
    num_core_train     = 2    # number of cores of training
    super_cell         = False
    mode_of_NN         = 'ase_natural_cutoffs' # 'pymatgen_dist', 'ase_dist', 'ase_natural_cutoffs', or 'voronoi'
    NN_cutoff          = 3.2
    transfer_learning  = False
    trainable_layers   = -4 # tail layers
    adsorbate          = True # indentify adsorbate or not when building graphs.
    split_binary_graph = False # split the binary graph file into small graphs (according to the `batch_size`). To solve the out-of-memory error.
# =============================================================================
# Setting done
# =============================================================================

if __name__ == '__main__':
    """
    One can run multiple projects under the same directory. The path of the
    project should be defined first.

    The structure of a project should be:
        project
        ├── ckpt
        │   ├── checkpoint
        │   ├── gat.ckpt.data-00000-of-00001
        │   └── gat.ckpt.index
        ├── dataset
        │   ├── all_graphs.bin -> $PATH/all_graphs.bin
        │   ├── file1.cif
        │   ├── POSCAR2
        │   └── ......
        ├── files
        │   ├── fname_prop.csv
        │   ├── test.txt
        │   ├── train.txt
        │   └── validation.txt
        └── log
            └── log.dat

    Note:
        "fname_prop.csv" is mandatory under "files".
        "dataset" should contain the DGLGraph binary file or structural files
        with "cif" or "VASP" formate.
        "ckpt" and "log" can be empty for a new running.
    """

    # time
    Very_start   = time.time()     # start time
    assert os.path.exists(Project), str(Project) + " not found."

    """ You can modify the following line for your own structure of file. """
    UsrFolder, DataSet, CKPT, LOG, CsvFile = 'files', 'dataset', 'energy_ckpt', 'log', 'fname_prop.csv' # you can define your own file system here.

    # define a logger
    if log_file is None:
        logf = sys.stdout
    else:
        logf = open(log_file, 'w', buffering=1)

    # define computing device. The Gpu device is deprecated for now.
    if gpu < 0:
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(gpu)
    print('User log: specified device:', device, file=logf)

    # check the dependent files
    for i in [UsrFolder, DataSet, CKPT, LOG]:
        if not os.path.exists(os.path.join(Project, i)):
            os.mkdir(os.path.join(Project, i))

    # initialize a graph reader.
    """
    If "new_training" is "True", graphs are loaded from *.cif or VASP files.
    Otherwise, graphs will be loaded from dgl binary file. The later way is
    much faster than loading graphs from scratch. Also, the dataset will be
    split if "new_training" is "True".
    """
    if new_training:
        from_binary = False # load graphs from binary files or not.
        new_split   = True  # split the dataset index or load them from files.
    else:
        from_binary = True
        new_split   = False

    # modify `cutoff` for your own purpose. For a slab model with vacuum
    # space, the `cutoff` should be less than the thickness of the vacuum
    # space. But anyway, it depends on you.
    graph_reader = ReadGraphs(os.path.join(Project, UsrFolder, CsvFile),
                                  os.path.join(Project, DataSet),
                                  cutoff = NN_cutoff,
                                  mode_of_NN=mode_of_NN,
                                  from_binary=from_binary,
                                  num_of_cores=num_core_read,
                                  super_cell=super_cell,
                                  adsorbate=adsorbate)

    # read graphs
    graph_list, graph_labels = graph_reader.read_all_graphs(
        scale_prop=False,
        ckpt_path=os.path.join(Project, CKPT))

    # split the dataset
    train_index, validation_index, test_index = TrainValTestSplit(
        validation_size,
        test_size,
        os.path.join(Project, UsrFolder, CsvFile),
        new_split)()


    graph_eg = graph_list[0]

    # validation dataset
    validation_graphs = [graph_list[x] for x in validation_index]
    validation_prop   = [graph_labels['prop'][x] for x in validation_index]

    # test dataset
    test_graphs = [graph_list[x] for x in test_index]
    test_prop   = [graph_labels['prop'][x] for x in test_index]

    if split_binary_graph:
        '''
        Split binary file if tensorflow raised out-of-memory error on GPU.
        Normally, memory for CPU is much larger than that for GPU.
        '''
        # train dataset
        print(f'User log: {time.time()} split dataset into small binary files......', file=logf)
        train_index_batch = np.array_split(train_index, int(len(train_index) /\
                                                            batch_size + 1))

        with tf.device("/cpu:0"):
            for i_index, i in enumerate(train_index_batch):
                g_batch = [graph_list[x].to("/cpu:0") for x in i]
                # g_batch = dgl.batch(g_batch)
                en_true = {'prop': tf.concat([graph_labels['prop'][x] for x in i], axis=0)}
                save_graphs(os.path.join(Project, DataSet, f'train_graphs_{i_index}.bin'),
                            g_batch, en_true)

        # remove variables from memory
        del graph_list
        del graph_labels
        del train_index
        # del validation_index
        del test_index
        del g_batch
        del en_true
        print(f'User log: {time.time()} split dataset finished.', file=logf)
    else:
        train_index_batch=None

    # # calculate the distance distribution
    # selected_index = np.random.choice([x for x in range(len(graph_list))], 1000)
    # dists          = [graph_list[x].edata['dist'] for x in selected_index]
    # dists          = tf.concat(dists, axis=0)
    # dists          = dists.numpy()
    # dists          = dists[np.where(dists != 0.0)]
    # np.savetxt('dists.txt', dists, fmt='%10.5f')

    # initialize a model
    model = GAT(GAT_out_list,
                 num_readout_out_list=readout_list,
                 head_list_en=head_list_en,
                 embed_activation=embed_activation,
                 readout_activation=readout_activation,
                 bias=bias,
                 negative_slope=negative_slope)

    # early stop
    if early_stop and Train:
        stopper = EarlyStopping(model, graph_eg, logf,
                                patience=stop_patience,
                                folder=os.path.join(Project, CKPT))

    # detect the saved weights. Load them if possible.
    model_save_path = os.path.join(Project, CKPT, 'gat.ckpt')
    print("User log: Model save path:", model_save_path, file=logf)
    if os.path.exists(model_save_path + '.index'):    # If the `checkpoint` file is saved before, try to load it.
        print("User log: Loading saved weights......", file=logf)
        g_init = graph_eg.clone() # Initialize trainable variables first.
        load_gat_weights(model, g_init, model_save_path, logf, device)
    else:
        print("User log: Weights not detected.", file=logf)

    if transfer_learning:
        nontrainable_layers = [x for x in range(len(model.layers))][1:trainable_layers]
        for layer_i in nontrainable_layers:
            model.layers[layer_i].trainable = False

def train(train_index_batch=train_index_batch):
# if __name__ == '__main__':
    start_time = time.time()
    step       = 1

    # train
    flag = False
    for epoch in range(epochs):
        if not split_binary_graph:
            np.random.shuffle(train_index) # shuffle training set every epoch
            # Split the data in batchs.
            train_index_batch = np.array_split(train_index, int(len(train_index) / batch_size + 1))
        en_pred_all, en_true_all = [], []

        for i_index, i in enumerate(train_index_batch):
            with tf.device(device):
                with tf.GradientTape() as tape:
                    if split_binary_graph:
                        g_batch, en_true = load_graphs(os.path.join(Project,
                                                                    DataSet,
                                                                    f'train_graphs_{i_index}.bin'))
                        en_true = en_true['prop']
                        g_batch = [g.to(device) for g in g_batch]
                        g_batch = dgl.batch(g_batch)
                    else:
                        g_batch = [graph_list[x].to(device) for x in i]
                        g_batch = dgl.batch(g_batch)
                        en_true = tf.concat([graph_labels['prop'][x] for x in i], axis=0)
                    en_pred     = model(g_batch)
                    batch_nodes = g_batch.batch_num_nodes()
                    en_pred     = tf.split(en_pred, batch_nodes)
                    en_pred     = tf.convert_to_tensor([tf.reduce_mean(x) for x in en_pred]) #tf.reduce_mean(en_pred, axis=1)

                    en_pred_all.append(en_pred)
                    en_true_all.append(en_true)
                    step += len(i)
                    loss_value = loss_fcn(en_true, en_pred) # calculate loss

                    # regularization. Specified in `Settings`
                    if L2_reg:
                        for weight in model.trainable_weights:
                            loss_value += weight_decay*tf.nn.l2_loss(weight)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                batch_mae = mae(en_true, en_pred)  # 不使用tf.keras.metrics.MeanAbsoluteError()。使用metrics，同样的输入，连续计算，输出不同。

                print("Epoch: {:0>4d} | Batch: {:0>4d} | Train_Loss (mse): {:1.4f} | Batch_MAE: {:1.4f} | Dur: {:.1f} s"
                      .format(epoch, i_index, loss_value, batch_mae, time.time() - start_time), file=logf)

            # evaluate every `validation_freq` epochs. `val` refers to validation.
            # `pred` refers to predict
            if step % validation_freq < batch_size and step > batch_size:
                print("========================================================================", file=logf)
                en_val_true, en_val_pred = [], []

                with tf.device(device):
                    validation_list = np.random.choice(range(len(validation_index)),
                                                       size=min(validation_samples, len(validation_index)), replace=False)
                    validation_batch = np.array_split(validation_list, len(validation_list) / val_batch_size)
                    for i in validation_batch:
                        g_batch = [validation_graphs[x].to(device) for x in i]
                        g_batch = dgl.batch(g_batch)
                        en_pred = model(g_batch)
                        batch_nodes = g_batch.batch_num_nodes()
                        en_pred = tf.split(en_pred, batch_nodes)
                        en_pred = tf.convert_to_tensor([tf.reduce_mean(x) for x in en_pred])
                        en_true = tf.concat([validation_prop[x] for x in i], axis=0)
                        en_val_pred.append(en_pred)
                        en_val_true.append(en_true)

                en_val_true, en_val_pred = tf.reshape(tf.concat(en_val_true, axis=0), shape=(-1)), tf.reshape(tf.concat(en_val_pred, axis=0), shape=(-1))
                val_mae = mae(en_val_true, en_val_pred)

                dur = time.time() - start_time
                en_true_all, en_pred_all = tf.reshape(tf.concat(en_true_all, axis=0), shape=(-1)), tf.reshape(tf.concat(en_pred_all, axis=0), shape=(-1))
                print("Epoch: {:0>4d} | Train_MAE : {:1.4f} | Train_PearsonR: {:1.4f} | Val_MAE: {:1.4f} | Val_PearsonR: {:1.4f} | Dur: {:.1f} s".
                      format(epoch, mae(en_true_all, en_pred_all),
                             PearsonR(en_true_all, en_pred_all), val_mae,
                             PearsonR(en_val_true, en_val_pred), dur),file=logf)

                if early_stop and Train:
                    if stopper.step(val_mae, model):
                        print('User log: model summary:', file=logf)
                        model.summary()
                        print('User log: model summary done.', file=logf)
                        flag = True
                        break
                    if stopper.update:
                        np.savetxt(os.path.join(Project, LOG, 'en_train_true.txt'), en_true_all, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                        np.savetxt(os.path.join(Project, LOG, 'en_train_pred.txt'), en_pred_all, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                        np.savetxt(os.path.join(Project, LOG, 'en_val_true.txt'),   en_val_true, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                        np.savetxt(os.path.join(Project, LOG, 'en_val_pred.txt'),   en_val_pred, fmt='%.8f')    # if scale or mean prop is true, the saved data is not the original data
                    en_true_all, en_pred_all = [], []
                else:
                    model.save_weights(os.path.join(Project, CKPT, 'gat.ckpt'))
            if flag:
                break

    print('User log: model summary:', file=logf)
    model.summary()
    print('User log: model summary done.', file=logf)

def predict(graph_list, label_list, model_save_path, batch_size=200):
    y_true, y_pred = label_list, [] # saved on GPU.
    json_file      = os.path.join(model_save_path, 'gat_model.json')
    graph_file     = os.path.join(model_save_path, 'graph_tmp.bin')
    ckpt_file      = os.path.join(model_save_path, 'gat.ckpt')
    for f in [json_file, graph_file, ckpt_file + '.index']:
        assert os.path.exists(f), str(f) + " file dose not exist."

    # load json file
    with open(json_file, 'r') as jsonf:
        model_config = json.load(jsonf)

    # build a model
    model =  GAT(model_config['num_gat_out_list'],
                 num_readout_out_list = model_config['num_readout_out_list'],
                 head_list_en         = model_config['head_list_en'],
                 embed_activation     = model_config['embed_activation'],
                 readout_activation   = model_config['readout_activation'],
                 bias                 = model_config['bias'],
                 negative_slope       = model_config['negative_slope'])

    # load weights
    graph_tmp, label_tmp = load_graphs(graph_file)
    graph_tmp = graph_tmp[0].to(device)
    with tf.device(device):
        model(graph_tmp)
    load_status = model.load_weights(ckpt_file)
    load_status.assert_consumed()               # check the load status
    print(f'Load weights from {ckpt_file} successfully.')

    with tf.device(device):
        batch_index = np.array_split(range(len(graph_list)), len(graph_list) / batch_size)
        for i in batch_index:
            g_batch = [graph_list[x].to(device) for x in i]
            g_batch = dgl.batch(g_batch)
            en_pred = model(g_batch)
            batch_nodes = g_batch.batch_num_nodes()
            en_pred = tf.split(en_pred, batch_nodes)
            en_pred = tf.convert_to_tensor([tf.reduce_mean(x) for x in en_pred])
            y_pred.append(en_pred)
    mae_value = mae(tf.convert_to_tensor(y_true), tf.concat(y_pred, axis=0))
    print("User log: predict MAE: {:.4f}".format(mae_value), file=logf)
    return y_true, tf.concat(y_pred, axis=0).numpy()

if __name__ == '__main__':
    if Train:
        train()
    y_test_true, y_test_pred = predict(test_graphs, test_prop, os.path.join(Project, CKPT), batch_size=batch_size)
    np.savetxt(os.path.join(Project, LOG, 'en_test_final_true.txt'), y_test_true, fmt='%.8f')
    np.savetxt(os.path.join(Project, LOG, 'en_test_final_pred.txt'), y_test_pred, fmt='%.8f')

    y_val_true, y_val_pred = predict(validation_graphs, validation_prop, os.path.join(Project, CKPT), batch_size=batch_size)
    np.savetxt(os.path.join(Project, LOG, 'en_val_final_true.txt'), y_val_true, fmt='%.8f')
    np.savetxt(os.path.join(Project, LOG, 'en_val_final_pred.txt'), y_val_pred, fmt='%.8f')

    Very_end = time.time()
    print("User log: Total time: {:.2f}".format(Very_end - Very_start),
          file=logf)

    if log_file is not None:
        logf.close()

# # debug
# reader = CrystalGraph()
# graph1 = reader.get_graph('POSCAR_Ni3Al.txt')
# graph2 = reader.get_graph('POSCAR_Ni3Al_APB.txt')
# graph_list = []
# model  = get_model()
# predict(model, [graph1, graph2], 'import_graph/ckpt/gat.ckpt')
# crystal1 = reader.get_crystal('POSCAR_Ni3Al.txt')
# crystal2 = reader.get_crystal('POSCAR_Ni3Al_APB.txt')
