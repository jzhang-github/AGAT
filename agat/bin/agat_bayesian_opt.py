# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:56:42 2023

@author: ZHANG Jun
"""

import os

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from agat.model import Fit

def train(head1, head2, head3,
          GAT_out1, GAT_out2, GAT_out3,
          ereadout2, ereadout3, ereadout4, ereadout5, ereadout6,
          freadout2, freadout3, freadout4, freadout5, freadout6):

    all_heads = ['mul', 'div', 'free']

    f = Fit(epochs=100, device='cuda', c=0.0, learning_rate=0.0001, tail_readout_no_act=[3,3,1],
            gat_node_dim_list=[6,
                               round(GAT_out1),
                               round(GAT_out2),
                               round(GAT_out3)],
            energy_readout_node_list=[3*round(GAT_out3),
                                      round(ereadout2),
                                      round(ereadout3),
                                      round(ereadout4),
                                      round(ereadout5),
                                      round(ereadout6),
                                      1],
            force_readout_node_list=[3*round(GAT_out3),
                                      round(freadout2),
                                      round(freadout3),
                                      round(freadout4),
                                      round(freadout5),
                                      round(freadout6),
                                      3],
            stress_readout_node_list=[3*round(GAT_out3),6],
            head_list=[all_heads[round(head1)],
                       all_heads[round(head2)],
                       all_heads[round(head3)]]
            )

    loss = f.fit()
    loss = loss.item()

    with open('parameters_force.txt', mode='a') as f:
        print(loss,
              head1, head2, head3,
              GAT_out1, GAT_out2, GAT_out3,
              ereadout2, ereadout3, ereadout4, ereadout5, ereadout6,
              freadout2, freadout3, freadout4, freadout5, freadout6,
              file=f)
    return -loss

pbounds = {
    'head1':(0,2),
    'head2':(0,2),
    'head3':(0,2),
    'GAT_out1': (10,200),
    'GAT_out2': (10,200),
    'GAT_out3': (10,200),
    'ereadout2': (50,400),
    'ereadout3': (50,400),
    'ereadout4': (50,400),
    'ereadout5': (50,400),
    'ereadout6': (50,400),
    'freadout2': (50,400),
    'freadout3': (50,400),
    'freadout4': (50,400),
    'freadout5': (50,400),
    'freadout6': (50,400),
    }

optimizer = BayesianOptimization(
            f=train,
            # constraint=constraint,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
            )

# load model if necessary
if os.path.exists('logs.log.json'):
    load_logs(optimizer, logs=["./logs.log"])
    print('Bayesian progress loaded.')
else:
    print('Bayesian not detected.')

# save model
logger = JSONLogger(path="./logs.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=30, # How many steps of **random** exploration you want to perform. Random exploration can help by diversifying the exploration space.
    n_iter=1000, # How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    )

print(optimizer.max)
