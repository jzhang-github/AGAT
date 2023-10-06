# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:41:20 2023

@author: ZHANG Jun
"""

__version__ = '8.0.0'

import os
# os.environ['DGLBACKEND']="pytorch"

# import important object.
from .data.build_dataset import AgatDatabase
from .model.fit import Fit
from .app.cata.high_throughput_predict import HpAds

del os
