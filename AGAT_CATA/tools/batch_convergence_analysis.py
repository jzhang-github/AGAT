# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:23:14 2022

@author: ZHANG Jun
"""


# 下面是DFT的结果
import numpy as np
import os

# surf = np.loadtxt('surface_sort.txt')
# all_miu = np.loadtxt('miu_sort.txt')

surf_e = np.loadtxt('surf_vs_energy.txt', dtype=float)
surf_e = dict(zip([int(x) for x in surf_e[:,0]], surf_e[:,1]))
energies = np.loadtxt('OH_bridge_surf_and_energy.txt')
miu = energies[:,1] + 0.3072701 - np.array([surf_e[int(x)] for x in energies[:,0]]) + 0.5 * -6.80545924 - -14.25416854
np.where(miu < 2)
miu_new = miu[np.where(miu < 1.5)]
np.mean(miu_new)
# miu_dict = {}


# for i in range(1, 31):
#     pos = np.where(surf == i)
#     miu_dict[i] = all_miu[pos]

# values = np.array([])
# for i in miu_dict:
#     values = np.hstack([values, miu_dict[i]])
#     print(np.mean(values))


# 下面是AGAT的结果
fnames = os.listdir('batch_convergence')

miu_dict = {}
for key in fnames:
    if os.path.splitext(key)[-1] == '.txt':
        val = np.loadtxt(os.path.join('batch_convergence', key))
        miu = val[:,0] + 0.3072701 - val[:,1] + 0.5 * -6.80545924 - -14.25416854
        # miu = miu[np.where(miu<2)]
        miu_dict[key] = miu
        # print(np.where(miu>2))

values = np.array([])
for i in miu_dict:
    values = np.hstack([values, miu_dict[i]])
    print(np.mean(values))

print(f'Standard deviation: {np.std(values)}')
