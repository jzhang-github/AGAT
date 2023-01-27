# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:10:03 2022

@author: 18326
"""

'''
How to generate input files with linux shell:
for i in $(seq 1 40); do cat ads_surf_energy_O_$i.txt >> ads_O.txt; cat ads_surf_energy_OH_$i.txt >> ads_OH.txt; cat ads_surf_energy_OOH_$i.txt >> ads_OOH.txt; done
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_potential(ads_energy, surf_energy, adsorbate):
    # adsorbate_dict = {81: 'O',
    #                   82: 'OH',
    #                   83: 'OOH'}
    # adsorbate = adsorbate_dict[length_of_atoms]
    # adsorbate = adsorbate_dict[length_of_atoms]

    if adsorbate == 'O':
        e = ads_energy + 0.0488142 - surf_energy + -6.80545924 - -14.25416854
    elif adsorbate == 'OH':
        e = ads_energy + 0.3072701 - surf_energy + 0.5 * -6.80545924 - -14.25416854
    elif adsorbate == 'OOH':
        e = ads_energy + 0.341310571428571 - surf_energy + 1.5 * -6.80545924 - 2 * -14.25416854
    return e

O_data = np.loadtxt('ads_O.txt')
OH_data = np.loadtxt('ads_OH.txt')
OOH_data = np.loadtxt('ads_OOH.txt')
num_data = min(len(O_data), len(OH_data), len(OOH_data))
O_data = O_data[0:num_data]
OH_data = OH_data[0:num_data]
OOH_data = OOH_data[0:num_data]


ill_conv = np.concatenate((np.where(O_data[:,2]==0.0),
                           np.where(OH_data[:,2]==0.0),
                           np.where(OOH_data[:,2]==0.0)), axis=1)[0]

O_data = np.delete(O_data, ill_conv, axis=0)
OH_data = np.delete(OH_data, ill_conv, axis=0)
OOH_data = np.delete(OOH_data, ill_conv, axis=0)

miu_O = get_potential(O_data[:,0], O_data[:,1], 'O')
miu_OH = get_potential(OH_data[:,0], OH_data[:,1], 'OH')
miu_OOH = get_potential(OOH_data[:,0], OOH_data[:,1], 'OOH')
np.savetxt('miu_O.txt', miu_O)
np.savetxt('miu_OH.txt', miu_OH)
np.savetxt('miu_OOH.txt', miu_OOH)

res = stats.linregress(miu_O, miu_OH)
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Equation: miu_OH = {res.slope} * miu_O + {res.intercept}")

plt.scatter(miu_O, miu_OH, s=1)
# plt.plot(miu_O, res.intercept + res.slope*miu_O, 'r', label='fitted line')
plt.scatter(miu_O, miu_OOH, s=1)
plt.scatter(miu_OH, miu_OOH, s=1)
plt.show()

miu_O_OH = np.vstack((miu_O, miu_OH)).T
miu_O_OOH = np.vstack((miu_O, miu_OOH)).T
miu_OH_OOH = np.vstack((miu_OH, miu_OOH)).T

miu_O_OH = miu_O_OH[np.random.choice(range(len(miu_O_OH)), size=2000, replace=False)]
miu_O_OOH = miu_O_OOH[np.random.choice(range(len(miu_O_OOH)), size=2000, replace=False)]
miu_OH_OOH = miu_OH_OOH[np.random.choice(range(len(miu_OH_OOH)), size=2000, replace=False)]

plt.scatter(miu_O_OH[:,0], miu_O_OH[:,1], s=1)
plt.scatter(miu_O_OOH[:,0], miu_O_OOH[:,1], s=1)
plt.scatter(miu_OH_OOH[:,0], miu_OH_OOH[:,1], s=1)
plt.show()

np.savetxt('miu_O_OH_selected.txt', miu_O_OH, fmt='%f')
np.savetxt('miu_O_OOH_selected.txt', miu_O_OOH, fmt='%f')
np.savetxt('miu_OH_OOH_selected.txt', miu_OH_OOH, fmt='%f')
