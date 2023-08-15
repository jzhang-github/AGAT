# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:08:41 2023

@author: ZHANG Jun
"""

import os

def generate_file_name(fname):
    while os.path.exists(fname):
        fname = fname + '_new'
    return fname
