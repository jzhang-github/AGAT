# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 22:24:24 2022

@author: 18326
"""
import numpy as np

def volcano_2e():

    k = 8.617333262e-5 # unit: eV/K
    T = 300.0 # unit K
    k0 = 200.0 # unit: S^-1*site^-1
    vi = 21362619.98986784
    h = 4.1357 * 10**-15 # unit: eV * s
    vi = k*T/h  # This value will not affect the trend.

    # dissociative mechanism (2 e transfer)
    intervals = 201
    A_list, arg_list = np.zeros((intervals, intervals)), np.zeros((intervals, intervals))
    y_list = np.linspace(-2, 3.0, intervals)
    x_list = np.linspace(-1.5, 2.0, intervals)
    for i, O in enumerate(y_list):
        for j, OH in enumerate(x_list):

            g1 = 2.46 - 2 * 1.23
            g2 = O - 2 * 1.23
            g3 = OH - 1.23
            g4 = 0.0

            Ea = 1.8*O-2.95

            G1 = g2 - g1 # ΔG1
            G2 = g3 - g2 # ΔG2
            G3 = g4 - g3 # ΔG3

            eta = max([G1,G2,G3])

            k1 = vi*np.exp(-Ea/(k*T))
            k2 = vi*np.exp(-G1/(k*T))
            k3 = vi*np.exp(-G2/(k*T))
            k4 = vi*np.exp(-G3/(k*T))

            A1 = k*T*np.log10(k1/k0)
            A2 = k*T*np.log10(k2/k0)
            A3 = k*T*np.log10(k3/k0)
            A4 = k*T*np.log10(k4/k0)

            A = min(A1,A2,A3,A4)
            A_list[i,j] = A
            arg = np.argmax([A1,A2, A3])
            arg_list[i,j] = arg
    arg_list += 1

    A_list = np.vstack((x_list, A_list))
    A_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          A_list))
    arg_list = np.vstack((x_list, arg_list))
    arg_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          arg_list))

    np.savetxt('A_O_vs_OH_2e.txt', A_list, fmt='%f')
    np.savetxt('arg_O_vs_OH_2e.txt', arg_list, fmt='%f')

    # repeat the paper: J. Phys. Chem. B, Vol. 108, No. 46, 2004
    intervals = 201
    A_list, arg_list = np.zeros((intervals, intervals)), np.zeros((intervals, intervals))
    y_list = np.linspace(-2, 3.0, intervals)
    x_list = np.linspace(-1.5, 2.0, intervals)
    for i, O in enumerate(y_list):
        for j, OH in enumerate(x_list):

            Ea = 1.8*O-2.89    # From table 2

            G0 = O - 2.45      # From table 2
            G1 = OH - O + 0.97 # ΔG1 From table 2
            G2 = -OH + 1.48    # ΔG2 From table 2

            k1 = vi*np.exp(-Ea/(k*T))
            k2 = vi*np.exp(-G1/(k*T))
            k3 = vi*np.exp(-G2/(k*T))

            A1 = k*T*np.log10(k1/k0)
            A2 = k*T*np.log10(k2/k0)
            A3 = k*T*np.log10(k3/k0)

            A = min(A1,A2,A3)
            A_list[i,j] = A

    A_list = np.vstack((x_list, A_list))
    A_list = np.hstack((np.vstack(([[0.0]],
                                     y_list.reshape(-1,1))),
                          A_list))

    np.savetxt('A_O_vs_OH_2e.txt', A_list, fmt='%f')
