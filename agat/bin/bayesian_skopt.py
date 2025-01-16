# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:33:26 2025

@author: ZHANGJUN

For more info, please refer to: https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-train-config

skopt: https://scikit-optimize.github.io/stable/auto_examples/hyperparameter-optimization.html#sphx-glr-auto-examples-hyperparameter-optimization-py
skopt: https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html#sphx-glr-auto-examples-bayesian-optimization-py
"""

import os
import shutil
from datetime import datetime
from agat.model import Fit
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import joblib
import matplotlib.pyplot as plt

search_space = [
    # `gat_node_dim_list`
    Integer(2, 7, name="gat_layers"),
    Integer(50, 200, name="gat_layer_2"),
    Integer(50, 200, name="gat_layer_3"),
    Integer(50, 200, name="gat_layer_4"),
    Integer(50, 200, name="gat_layer_5"),
    Integer(50, 200, name="gat_layer_6"),
    Integer(50, 200, name="gat_layer_7"),
    Integer(50, 200, name="gat_layer_8"),

    # `energy_readout_node_list`
    Integer(3, 12, name="energy_readout_layers"),
    Integer(50, 200,  name="energy_readout_layer_2"),
    Integer(50, 200,  name="energy_readout_layer_3"),
    Integer(50, 200,  name="energy_readout_layer_4"),
    Integer(50, 200,  name="energy_readout_layer_5"),
    Integer(50, 200,  name="energy_readout_layer_6"),
    Integer(50, 200,  name="energy_readout_layer_7"),
    Integer(50, 200,  name="energy_readout_layer_8"),
    Integer(50, 200,  name="energy_readout_layer_9"),
    Integer(50, 200,  name="energy_readout_layer_10"),
    Integer(50, 200,  name="energy_readout_layer_11"),
    Integer(50, 200,  name="energy_readout_layer_12"),

    # `force_readout_node_list`
    Integer(3, 12, name="force_readout_layers"),
    Integer(50, 200,  name="force_readout_layer_2"),
    Integer(50, 200,  name="force_readout_layer_3"),
    Integer(50, 200,  name="force_readout_layer_4"),
    Integer(50, 200,  name="force_readout_layer_5"),
    Integer(50, 200,  name="force_readout_layer_6"),
    Integer(50, 200,  name="force_readout_layer_7"),
    Integer(50, 200,  name="force_readout_layer_8"),
    Integer(50, 200,  name="force_readout_layer_9"),
    Integer(50, 200,  name="force_readout_layer_10"),
    Integer(50, 200,  name="force_readout_layer_11"),
    Integer(50, 200,  name="force_readout_layer_12"),

    # `negative_slope`
    Real(0.1, 0.4, prior='uniform', name='negative_slope'),

    # `a` & `b`. weights of energy and forces
    Real(1e0, 1e4, prior='log-uniform', name='a'),
    Real(1e0, 1e4, prior='log-uniform', name='b'),

    # `learning_rate`
    Real(1e-6, 1e-3, prior='log-uniform', name='learning_rate'),

    # `weight_decay`
    Real(1e-5, 1e-2, prior='log-uniform', name='weight_decay'),

    # `batch_size`
    Integer(2, 256,  prior='log-uniform', base=2, name="batch_size"),

    # `head_list`
    Categorical([('mul',), ('div',), ('free',),
                 ('mul', 'div'), ('mul', 'free'), ('div', 'free'),
                 ('mul', 'div', 'free')], name="head_list")
]

@use_named_args(search_space)
def objective(gat_layers, gat_layer_2, gat_layer_3, gat_layer_4, gat_layer_5, gat_layer_6, gat_layer_7, gat_layer_8,
              energy_readout_layers, energy_readout_layer_2, energy_readout_layer_3, energy_readout_layer_4, energy_readout_layer_5, energy_readout_layer_6, energy_readout_layer_7, energy_readout_layer_8, energy_readout_layer_9, energy_readout_layer_10, energy_readout_layer_11, energy_readout_layer_12,
              force_readout_layers, force_readout_layer_2, force_readout_layer_3, force_readout_layer_4, force_readout_layer_5, force_readout_layer_6, force_readout_layer_7, force_readout_layer_8, force_readout_layer_9, force_readout_layer_10, force_readout_layer_11, force_readout_layer_12,
              negative_slope,
              a,
              b,
              learning_rate,
              weight_decay,
              batch_size,
              head_list
              ):

    gat_node_dim_list = [
        118, gat_layer_2, gat_layer_3, gat_layer_4, gat_layer_5,
        gat_layer_6, gat_layer_7, gat_layer_8][:gat_layers]
    energy_readout_node_list = [gat_node_dim_list[-1]*len(head_list),
        energy_readout_layer_2, energy_readout_layer_3, energy_readout_layer_4,
        energy_readout_layer_5, energy_readout_layer_6, energy_readout_layer_7,
        energy_readout_layer_8, energy_readout_layer_9, energy_readout_layer_10,
        energy_readout_layer_11, energy_readout_layer_12][:energy_readout_layers]
    force_readout_node_list = [gat_node_dim_list[-1]*len(head_list),
        force_readout_layer_2, force_readout_layer_3, force_readout_layer_4,
        force_readout_layer_5, force_readout_layer_6, force_readout_layer_7,
        force_readout_layer_8, force_readout_layer_9, force_readout_layer_10,
        force_readout_layer_11, force_readout_layer_12,][:force_readout_layers]
    gat_node_dim_list = [int(x) for x in gat_node_dim_list]
    energy_readout_node_list = [int(x) for x in energy_readout_node_list]
    force_readout_node_list = [int(x) for x in force_readout_node_list]
    negative_slope = float(negative_slope)
    a, b = float(a), float(b)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    batch_size=int(batch_size)
    energy_readout_node_list.append(1)
    force_readout_node_list.append(3)

    f = Fit(
            gat_node_dim_list=gat_node_dim_list,
            energy_readout_node_list=energy_readout_node_list,
            force_readout_node_list=force_readout_node_list,
            negative_slope=negative_slope,
            a=a,
            b=b,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            head_list=head_list,

            # some parameters cannot be optimized.
            epochs=1500, device='cuda', c=0,
            tail_readout_no_act=[3,3,1],
            stress_readout_node_list=[gat_node_dim_list[-1]*len(head_list),6]
            )

    loss = f.fit()
    loss = loss.item()

    with open('skopt_parameters.txt', mode='a') as f:
        print('###################################################')
        print(f' loss: {loss}\n',
              f'gat_node_dim_list: {gat_node_dim_list}\n',
              f'energy_readout_node_list: {energy_readout_node_list}\n',
              f'force_readout_node_list: {force_readout_node_list}\n',
              f'negative_slope: {negative_slope}\n',
              f'a: {a}\n',
              f'b: {b}\n',
              f'learning_rate: {learning_rate}\n',
              f'weight_decay: {weight_decay}\n',
              f'batch_size: {batch_size}\n',
              f'head_list: {head_list}\n',
              end='\n', file=f)

    # move agat model.
    if not os.path.exists('all_agat_models'): os.mkdir('all_agat_models')
    dst_dir = os.path.join('all_agat_models',
                           f'agat_model_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    shutil.move('agat_model', dst_dir)
    shutil.move('fit.log', os.path.join(dst_dir, 'fit.log'))
    return loss

def skopt_run():
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=300,
        n_random_starts=5,
        random_state=0,
        verbose=True,
    )

    joblib.dump(result, "optimization_result.pkl")
    print("Optimization result saved to 'optimization_result.pkl'")

    print("Best validation loss:", result.fun)

    best_params = dict(zip([dim.name for dim in search_space], result.x))
    print("Best parameters:", best_params)

    fig = plot_convergence(result)
    fig.savefig("plot_convergence.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    skopt_run()
