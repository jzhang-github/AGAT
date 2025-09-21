# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:08:41 2023

@author: ZHANG Jun
"""

import sys
import platform
import os
from .exceptions import FileExit
from .incar_tag import INCAR_TAG
import shutil

def generate_file_name(fname):
    while os.path.exists(fname):
        fname = fname + '_new'
    return fname

def file_exit():
    if os.path.exists('StopPython'):
        os.remove('StopPython')
        raise FileExit('Exit because `StopPython` file is found.')

def modify_INCAR(working_dir='.', key='NSW', value='300', s=''):
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
    with open(os.path.join(working_dir, 'INCAR'), 'r') as f:
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

    with open(os.path.join(working_dir, 'INCAR'), 'w') as f:
        for line in new_incar:
            f.write(line)
    return 0

def modify_KPOINTS(working_dir='.', k='3 3 3'):
    with open(os.path.join(working_dir, 'KPOINTS'), 'r') as f:
        lines = f.readlines()

    lines[3] = k

    with open(os.path.join(working_dir, 'KPOINTS'), 'r') as f:
        for l in f:
            f.write(l)

def get_INCAR(src='INCAR', dst='INCAR'):
    # path = dst
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    else:
        print(f'INCAR file already exists in {dst}. Skipping ...')

def get_KPOINTS(src='KPOINTS', dst='KPOINTS'):
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    else:
        print(f'KPOINTS file already exists in {dst}. Skipping ...')

def get_KPOINTS_gamma(dst='KPOINTS'):

    with open(dst, 'w') as f:
        f.write('''Automatic mesh
0
Gamma
1 1 1
0.0 0.0 0.0
''')

def get_POTCAR(cmd='getpotential.sh', line=1, working_dir='.'):
    if sys.platform != 'linux':
        print('The POTCAR file can only be generated on a Linux OS.')
        return None
    path = os.path.join(working_dir, 'POTCAR')
    if not os.path.exists(path):
        curdir = os.getcwd()
        os.chdir(working_dir)
        os.system(f"{cmd} {str(line)}")
        os.chdir(curdir)
    else:
        print(f'POTCAR file already exists in {path}. Skipping ...')

def run_vasp(cmd='vasp_run.sh'):
    assert platform.system() == 'Linux', 'The VASP code can only be executed on a Linux OS.'
    # The POSCAR file should exist already.
    r = os.system(f'bash {cmd}')
    return r # `0` for good; `1` for bad execution.

def file_force_action(func, src, dst):
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    func(src, dst)
