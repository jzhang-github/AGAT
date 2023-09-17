# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:41:20 2023

@author: ZHANG Jun
"""

__version__ = '7.13.3'

import os
os.environ['DGLBACKEND']="tensorflow"

# import important object.
from .data.data import AgatDatabase
from .model.ModelFit import Train
from .app.cata import high_throughput_predict

del os
