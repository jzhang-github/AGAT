# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:41:20 2023

@author: ZHANG Jun
"""

__version__ = '9.0.0'

import os
# os.environ['DGLBACKEND']="pytorch"

# import important object.
from .data.build_dataset import BuildDatabase
from .model.fit import Fit
from .app.cata.high_throughput_predict import HtAds

del os
