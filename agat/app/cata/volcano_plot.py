# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:57:33 2022

@author: 18326
"""

import numpy as np

def volcano_plot():
    # RuRhPdIrPt: miu_OH = 0.8791826719834506 * miu_O + -0.21093560896977448 R-squared: 0.814781
    # NiCoFePdPt: miu_OH = 0.7783951605505598 * miu_O + -0.5136024752937013 R-squared: 0.663885

    # combine 2 systems:
    # R-squared: 0.731274 0.600242 0.804015
    # R        : 0.855146 0.774753 0.896669
    # Equation: miu_O = 0.9217868230710621 * miu_OH + 0.6303784975379894
    # Equation: miu_OOH = 0.7815803282809595 * miu_O + 3.133163731293882
    # Equation: miu_OOH = 0.9760033168772717 * miu_OH + 3.5297309325053163

    ###############################################################################
    # O vs OOH
    intervals = 201
    eta_list, arg_list = np.zeros((intervals, intervals)), np.zeros((intervals, intervals))
    y_list = np.linspace(0.5, 3.5, intervals)
    x_list = np.linspace(2.5, 5.5, intervals)
    for i, O in enumerate(y_list):
        for j, OOH in enumerate(x_list):
            OH = 0.793 * O + -0.399

            g1 = 0.0
            g2 = OOH - 3 * 1.23
            g3 = O   - 2 * 1.23
            g4 = OH  - 1.23
            g5 = 0.0

            G1 = g2 - g1 # ΔG1
            G2 = g3 - g2 # ΔG2
            G3 = g4 - g3 # ΔG3
            G4 = g5 - g4 # ΔG4

            eta           = max(G1,G2,G3,G4)
            eta_list[i,j] = eta
            arg           = np.argmax([G1,G2,G3,G4])
            arg_list[i,j] = arg
    arg_list += 1

    eta_list = np.vstack((x_list, eta_list))
    eta_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          eta_list))
    arg_list = np.vstack((x_list, arg_list))
    arg_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          arg_list))

    np.savetxt('eta_O_vs_OOH.txt', eta_list, fmt='%f')
    np.savetxt('arg_O_vs_OOH.txt', arg_list, fmt='%f')
    ###############################################################################

    ###############################################################################
    # O vs OH
    intervals = 201
    eta_list, arg_list = np.zeros((intervals, intervals)), np.zeros((intervals, intervals))
    y_list = np.linspace(-0.5, 4.5, intervals)
    x_list = np.linspace(-1.5, 3.5, intervals)
    for i, O in enumerate(y_list):
        for j, OH in enumerate(x_list):
            OOH = 0.9760033168772717 * OH + 3.5297309325053163

            g1 = 0.0
            g2 = OOH - 3 * 1.23
            g3 = O - 2 * 1.23
            g4 = OH - 1.23
            g5 = 0.0

            G1 = g2 - g1 # ΔG1
            G2 = g3 - g2 # ΔG2
            G3 = g4 - g3 # ΔG3
            G4 = g5 - g4 # ΔG4

            eta = max(G1,G2,G3,G4)
            eta_list[i,j] = eta
            arg = np.argmax([G1,G2,G3,G4])
            arg_list[i,j] = arg
    arg_list += 1

    eta_list = np.vstack((x_list, eta_list))
    eta_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          eta_list))
    arg_list = np.vstack((x_list, arg_list))
    arg_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          arg_list))

    np.savetxt('eta_O_vs_OH.txt', eta_list, fmt='%f')
    np.savetxt('arg_O_vs_OH.txt', arg_list, fmt='%f')
    ###############################################################################

    ###############################################################################
    # 1D plot (OH)
    eta_list = []
    x_list = np.linspace(-2, 4, 300)
    for OH in x_list:
        O = 0.9217868230710621 * OH + 0.6303784975379894
        OOH = 0.9760033168772717 * OH + 3.5297309325053163

        g1 = 0.0
        g2 = OOH - 3 * 1.23
        g3 = O - 2 * 1.23
        g4 = OH - 1.23
        g5 = 0.0

        G1 = g2 - g1 # ΔG1
        G2 = g3 - g2 # ΔG2
        G3 = g4 - g3 # ΔG3
        G4 = g5 - g4 # ΔG4

        eta = max(G1,G2,G3,G4)
        eta_list.append(eta)
        # print(np.argmax([G1,G2,G3,G4]))

    ###############################################################################
    # O vs OH 2-electron mechanism
    intervals = 201
    eta_list, arg_list = np.zeros((intervals, intervals)), np.zeros((intervals, intervals))
    y_list = np.linspace(-2, 3.0, intervals)
    x_list = np.linspace(-1.5, 2.0, intervals)
    for i, O in enumerate(y_list):
        for j, OH in enumerate(x_list):

            g1 = 2.46 - 2 * 1.23
            g2 = O - 2 * 1.23
            g3 = OH - 1.23
            g4 = 0.0

            G1 = g2 - g1 # ΔG1
            G2 = g3 - g2 # ΔG2
            G3 = g4 - g3 # ΔG3

            eta = max(G1,G2,G3)
            eta_list[i,j] = eta
            arg = np.argmax([G1,G2,G3])
            arg_list[i,j] = arg
    arg_list += 1

    eta_list = np.vstack((x_list, eta_list))
    eta_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          eta_list))
    arg_list = np.vstack((x_list, arg_list))
    arg_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          arg_list))

    np.savetxt('eta_O_vs_OH_2e.txt', eta_list, fmt='%f')
    np.savetxt('arg_O_vs_OH_2e.txt', arg_list, fmt='%f')
    ###############################################################################
