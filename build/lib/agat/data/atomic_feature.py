# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:51:55 2021

@author: ZHANG Jun
"""


import numpy as np

# from pymatgen.core.periodic_table import Element
from ase.data import atomic_numbers

all_elements = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B',  'Ba',
                'Be', 'Bh', 'Bi', 'Bk', 'Br', 'C',  'Ca', 'Cd', 'Ce', 'Cf',
                'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy',
                'Er', 'Es', 'Eu', 'F',  'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd',
                'Ge', 'H',  'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I',  'In', 'Ir',
                'K',  'Kr', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg',
                'Mn', 'Mo', 'Mt', 'N',  'Na', 'Nb', 'Nd', 'Ne', 'Nh', 'Ni',
                'No', 'Np', 'O',  'Og', 'Os', 'P',  'Pa', 'Pb', 'Pd', 'Pm',
                'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh',
                'Rn', 'Ru', 'S',  'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn',
                'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'Ts',
                'U',  'V',  'W',  'Xe', 'Y',  'Yb', 'Zn', 'Zr']

elements = ['Ac', 'Ag', 'Al', 'Am', 'As', 'Au', 'B',  'Ba', 'Be',
            'Bi', 'Br', 'C',  'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs',
            'Cu', 'Dy', 'Er', 'Eu', 'F',  'Fe', 'Ga', 'Gd', 'Ge', 'H',
            'Hf', 'Hg', 'Ho', 'I',  'In', 'Ir', 'K',  'La', 'Li', 'Lu',
            'Mg', 'Mn', 'Mo', 'N',  'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O',
            'Os', 'P',  'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra',
            'Rb', 'Re', 'Rh', 'Ru', 'S',  'Sb', 'Sc', 'Se', 'Si', 'Sm',
            'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm',
            'U',  'V',  'W',  'Y',  'Yb', 'Zn', 'Zr']

selected_elements = ['Ni', 'Co', 'Fe', 'Al', 'Ti']

# elements_not_used = ['Ar', 'Bh', 'Cn', 'Db', 'Ds', 'Fl', 'Hs', 'Lv', 'Mc', 'Mt', 'Nh', 'Og', 'Rf', 'Rg', 'Sg', 'Ts']

NiFe_ele = ['Fe', 'Ni']

Elements_used = elements

# def get_atomic_features(Elements_used): # deprecated

#     """
#     For more input features, please refer to: https://pymatgen.org/pymatgen.core.periodic_table.html

#     Returns
#     -------
#     None.

#     """
#     atom_feat_data, atom_feat = [], {}
#     for i, element in enumerate(Elements_used):
#         atom_feat_tmp = []
#         atom_feat_tmp.append(Element(element).Z)                # atomic number
#         atom_feat_tmp.append(Element(element).atomic_radius)
#         atom_feat_tmp.append(Element(element).molar_volume)
#         atom_feat_tmp.append(Element(element).atomic_mass)
#         atom_feat_tmp.append(Element(element).mendeleev_no)
#         atom_feat_tmp.append(Element(element).X)                # Electronegativity of element
#         atom_feat_tmp.append(Element(element).boiling_point)
#         atom_feat_tmp.append(Element(element).melting_point)
#         atom_feat_tmp.append(Element(element).row)
#         atom_feat_tmp.append(Element(element).group)
#         atom_feat_tmp.append(Element(element).max_oxidation_state)
#         atom_feat_tmp.append(Element(element).min_oxidation_state)
#         atom_feat_data.append(atom_feat_tmp)

#     # scale the features
#     atom_feat_data = np.array(atom_feat_data)
#     max_list = np.max(atom_feat_data, axis=0)
#     min_list = np.min(atom_feat_data, axis=0)

#     for i in range(np.shape(atom_feat_data)[0]):
#         for j in range(np.shape(atom_feat_data)[1]):
#             atom_feat_data[i][j] = (atom_feat_data[i][j] - min_list[j]) / (max_list[j] - min_list[j])
#     for i_index, i in enumerate(Elements_used):
#         atom_feat[i] = atom_feat_data[i_index,:]
#     return atom_feat

def get_atomic_feature_onehot(Elements_used):
    """
    Description
    ----------
    Get the atomic number of all considered elements.

    Returns
    -------
    atom_feat : dict
        atomic number of all considered elements
    """

    number_of_elements = len(Elements_used)
    atomic_number = [atomic_numbers[x] for x in Elements_used]
    atoms_dict = dict(zip(atomic_number, Elements_used))

    atomic_number.sort()

    Elements_sorted = [atoms_dict[x] for x in atomic_number]

    keys               = Elements_sorted
    element_index      = [i for i, ele in enumerate(keys)]
    values             = np.eye(number_of_elements)[element_index]
    atom_feat          = dict(zip(keys, values))
    return atom_feat
