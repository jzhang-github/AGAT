# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:08:41 2023

@author: ZHANG Jun
"""

import os
from .exceptions import FileExit
from .incar_tag import INCAR_TAG

def generate_file_name(fname):
    while os.path.exists(fname):
        fname = fname + '_new'
    return fname

def file_exit():
    if os.path.exists('StopPython'):
        os.remove('StopPython')
        raise FileExit('Exit because `StopPython` file is found.')

def modify_INCAR(key='NSW', value='300', s=''):
    """Modify the INCAR file.

    :param key: The INCAR tag, defaults to 'NSW'
    :type key: str, optional
    :param value: Value of the INCAR tag, defaults to '300'
    :type value: str, optional
    :param s: Comment string, defaults to ''
    :type s: str, optional
    :return: Modified INCAR file

    """

    if not key in INCAR_TAG:
        print('Input key not avaliable, please check.')
        return 1

    new_incar, discover_code = [], False
    with open('INCAR', 'r') as f:
        for line in f:
            str_list = line.split()
            if len(str_list) == 0:
                new_incar.append('\n')
            elif str_list[0] == key:
                str_list[2] = value
                new_incar.append(f'  {str_list[0]} = {str_list[2]}\n')
                discover_code = True
            else:
                new_incar.append(line)

    if s:
        new_incar.append(f'\n{s}\n')

    if not discover_code:
        new_incar.append(f'  {key} = {value}\n')

    with open('INCAR', 'w') as f:
        for line in new_incar:
            f.write(line)
    return 0
