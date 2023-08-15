# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:41:20 2023

@author: ZHANG Jun
"""

def main(formula):
    from agat.data.data import AgatDatabase
    from agat.model.ModelFit import Train
    from agat.app.cata import high_throughput_predict
    
    ad = AgatDatabase(mode_of_NN='ase_dist', num_of_cores=16)
    ad.build()
    
    at = Train()
    at.fit_energy_model()
    at.fit_force_model()
    
    HA = HpAds()
    HA.run(formula, fmax=0.05, steps=200, gpu=0)
