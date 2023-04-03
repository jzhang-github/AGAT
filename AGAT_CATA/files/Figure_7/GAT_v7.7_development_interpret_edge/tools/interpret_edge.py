# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:54:36 2022

@author: 18326
"""

from GatApp import GatApp, GatAseCalculator
import os
from ase.io import read, write
from high_throughput_predict import geo_opt
import numpy as np
from generate_adsorption_sites import AddAtoms
from GatApp import GatAseCalculator
from dgl.data.utils import save_graphs, load_graphs

# import potential
energy_model_save_dir = os.path.join('..', 'NiCoFePdPt_potential_20220806_bak', 'energy_ckpt')
force_model_save_dir  = os.path.join('..', 'NiCoFePdPt_potential_20220806_bak', 'force_ckpt')
app = GatApp(energy_model_save_dir, force_model_save_dir)
calculator=GatAseCalculator(energy_model_save_dir, force_model_save_dir, gpu=-1)

files = os.listdir('clean_surfaces')
for i, surf in enumerate(files):
    surf_fname = os.path.join('clean_surfaces', surf)
    adder      = AddAtoms(os.path.join('clean_surfaces', surf),
                         species='O',
                         sites='hollow',
                         dist_from_surf=1.7,
                         num_atomic_layer_along_Z=5)
    all_sites = adder.write_file_with_adsorption_sites(file_format='F',
                                                       calculation_index=i,
                                                       partial_fix=False)
    for j in range(all_sites):
        ads_name = os.path.join('adsorption', f'POSCAR_{i}_{j}')
        atoms = read(ads_name)
        atoms.set_calculator(calculator)
        _energy, _force, atoms_opt, _force_max = geo_opt(atoms, fmax=0.03)
        outname = os.path.join('contcars', f'CONTCAR_{i}_{j}.gat')
        write(outname, atoms_opt)
        graph = app.get_graph(outname)
        _ = app.get_energy(graph, fname=f'surf_{i}_ads_{j}')
        _ = app.get_forces(graph, fname=f'surf_{i}_ads_{j}')


# analyse graphs
def flatten(arr:np.array, points=50):
    x = arr[:,0]
    y = arr[:,1]
    x_min = min(x)
    x_max = max(x)
    x_span = x_max - x_min
    step = x_span/points
    bins = np.linspace(x_min+step, x_max+step, points)
    new_x = bins - step/2
    dig = np.digitize(x, bins)
    new_y = [[] for x in range(points)]
    [new_y[d].append(y[i]) for i, d in enumerate(dig)]
    new_y = [np.mean(x) for x in new_y]
    nanarg = np.argwhere(np.isnan(new_y))
    new_x = np.delete(new_x, nanarg)
    new_y = np.delete(new_y, nanarg)
    return np.array([new_x, new_y]).T

for ele in ['Ni', 'Co', 'Fe', 'Pd']:
    fnames = os.listdir('graphs')
    results =[]
    element_name = ele
    for f in fnames:
        # f = fnames[0]
        f_list = f.split('_')
        contcar = f'CONTCAR_{f_list[2]}_{f_list[4]}.gat'
        atoms = read(os.path.join('contcars', contcar))
        symbols = atoms.get_chemical_symbols()

        if len(f_list) > 7:
            if f_list[7]=='energy.bin':
                g = load_graphs(os.path.join('graphs', f))[0][0]
                src, dst = g.all_edges()
                src, dst = src.numpy(), dst.numpy()
                edge_index = np.where(src==80)
                edge_index = [x for x in edge_index[0] if symbols[dst[x]] == element_name]
                dist = g.edata['dist'].numpy()[edge_index]
                score = g.edata['a'].numpy()[edge_index].reshape((-1,3))
                score = np.mean(score, axis=1)
                t = np.array([dist,score]).T
                results.append(t)

    results = np.vstack(results)
    flatten_results = flatten(results,points=20)
    np.savetxt(f'all_score_vs_dist_O-{element_name}_energy.txt',
                results, fmt='%.8f')
    np.savetxt(f'all_score_vs_dist_O-{element_name}_energy_flatten.txt',
                flatten_results, fmt='%.8f')

import matplotlib.pyplot as plt
plt.scatter(results[:,0], results[:,1], s=0.5)
plt.plot(flatten_results[:,0], flatten_results[:,1])
plt.scatter(results[:,0], results[:,1], s=0.5)
plt.xlabel("Distance")
plt.ylabel("Edge score")
