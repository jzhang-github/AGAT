# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:32:45 2022

@author: 18326
"""

import sys
import numpy as np


def get_potential(ads_energy, surf_energy, adsorbate):
    if adsorbate == 'O':
        e = ads_energy + 0.0488142 - surf_energy + -6.80545924 - -14.25416854
    elif adsorbate == 'OH':
        e = ads_energy + 0.3072701 - surf_energy + 0.5 * -6.80545924 - -14.25416854
    elif adsorbate == 'OOH':
        e = ads_energy + 0.341310571428571 - surf_energy + 1.5 * -6.80545924 - 2 * -14.25416854
    return e

if __name__ == '__main__':
    assert len(sys.argv) > 2, 'Usage: command + file name + adsorbate name.'
    fname = sys.argv[1]
    adsorbate = sys.argv[2]
    data = np.loadtxt(fname)
    ill_conv = np.where(data[:,2]==0.0)
    data = np.delete(data, ill_conv, axis=0)
    e = get_potential(data[:,0], data[:,1], adsorbate)
    mean = np.mean(e)
    std = np.std(e)
    print(f'Mean: {mean}; std: {std}')
