# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:41:20 2023

@author: ZHANG Jun
"""

def main(formula):
    from agat.data.build_dataset import BuildDatabase
    from agat.model.fit import Fit
    from agat.app.cata import high_throughput_predict
    from agat.app.cata.high_throughput_predict import HtAds

    ad = BuildDatabase(mode_of_NN='ase_dist', num_of_cores=16)
    ad.build()

    f = Fit()
    f.fit()

    HA = HtAds()
    HA.run(formula, device='cuda')
