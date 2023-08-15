# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:08:41 2023

@author: ZHANG Jun
"""
import os
class FileExit(Exception):
    pass

def file_exit():
    if os.path.exists('StopPython'):
        os.remove('StopPython')
        raise FileExit('Exit because `StopPython` file is found.')

