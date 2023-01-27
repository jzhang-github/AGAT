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

# import potential
fname = os.path.join('interpret_model', 'POSCAR_surf.txt')
fname_ads = os.path.join('interpret_model', 'POSCAR_ads.txt')
# fname_ads = os.path.join('interpret_model', 'CONTCAR_ads')
energy_model_save_dir = os.path.join('..', 'NiCoFePdPt_potential_20220806_bak', 'energy_ckpt')
force_model_save_dir  = os.path.join('..', 'NiCoFePdPt_potential_20220806_bak', 'force_ckpt')
app = GatApp(energy_model_save_dir, force_model_save_dir)

graph  = app.get_graph(fname)
energy_surf = app.get_energies(graph)
forces_surf = app.get_forces(graph)
energy_surf = np.append(energy_surf, [[0.0]], axis=0)
forces_surf = np.append(forces_surf, [[0.0,0.0,0.0]], axis=0)
forces_surf = np.sum(forces_surf ** 2, axis=1).reshape((-1,1))**0.5

# atoms  = read(fname)
# write('poscar.cif', atoms, format='cif')

# import structure with O ads
atoms_ads = read(fname_ads)

# export atomic energy and forces
graph_ads  = app.get_graph(fname_ads)
energy_ads = app.get_energies(graph_ads)
forces_ads = app.get_forces(graph_ads)
forces_ads = np.sum(forces_ads ** 2, axis=1).reshape((-1,1))**0.5

# export ovito files
delta_e = energy_ads - energy_surf
delta_f = forces_ads - forces_surf
delta_e[-1][0] = 0.0
delta_f[-1][0] = 0.0
pos = atoms_ads.get_positions()
# with open(os.path.join('interpret_model', 'ads.lmp'), 'a+') as f:
#     for i, e in enumerate(delta_e):
#         f.write(f'{i} 1 {pos[i][0]} {pos[i][1]} {pos[i][2]} {e[0]} {delta_f[i][0]}\n')

# optimize adsorption structure
atoms_ads = read(fname_ads)
calculator=GatAseCalculator(energy_model_save_dir, force_model_save_dir, gpu=-1)
atoms_ads.set_calculator(calculator)
energy, force, atoms_geo_opt, force_max = geo_opt(atoms_ads, fmax=0.01)
write(os.path.join('interpret_model', 'CONTCAR_ads'),
      atoms)

# change z coordinates
pos = atoms_geo_opt.get_positions()
d_zs = np.linspace(-0.5, 1.0, 16)
es, fs = [], []
for dz in d_zs:
    atoms_ads_new = atoms_geo_opt.copy()
    pos_new = pos.copy()
    pos_new[-1][2] += dz
    atoms_ads_new.set_positions(pos_new)
    write('CONTCAR_tmp_z', atoms_ads_new)
    graph_tmp  = app.get_graph('CONTCAR_tmp_z')
    e = app.get_energy(graph_tmp)
    f = app.get_forces(graph_tmp)
    f = np.sum(f ** 2, axis=1).reshape((-1,1))**0.5

    es.append(e)
    fs.append(f[-1][0])

np.savetxt(os.path.join('interpret_model','energy_variation_along_z'),
           np.vstack((d_zs, es)).T, fmt='%.8f')
np.savetxt(os.path.join('interpret_model','force_variation_along_z'),
           np.vstack((d_zs, fs)).T, fmt='%.8f')


# interpret edge information
