# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:41:20 2023

@author: ZHANG Jun
"""

__version__ = '7.12'

import os
os.environ['DGLBACKEND']="tensorflow"

# import important object.
from agat.data.data import AgatDatabase
from agat.model.ModelFit import Train
from agat.app.cata import high_throughput_predict
