#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 18:12:17 2022

@author: ZHANG Jun
"""

from ase.io import read, write
import sys

def increase_vacuum(fname):
    """

    :param fname: Structural file name.
    :type fname: str
    :return: A new file with thicker vacuum.
    :rtype: A new file

    """

    # fname = sys.argv[1]

    poscar = read(fname)
    # frac   = poscar.get_scaled_positions()
    cart   = poscar.get_positions()
    cell   = poscar.cell.array
    cell[2][2] += 5

    poscar.set_cell(cell)
    poscar.set_positions(cart)

    write(f'{fname}_more_vacuum', poscar)
