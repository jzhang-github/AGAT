###################
atomic_feature
###################

Atomic features are one-hot codes.

.. py:data:: all_elements 

   All element symbols in the periodic table. Some radioactive elements are not included::

      ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B',  'Ba',
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
   
.. py:data:: elements

   Common elements::
   
      ['Ac', 'Ag', 'Al', 'Am', 'As', 'Au', 'B',  'Ba', 'Be',
       'Bi', 'Br', 'C',  'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs',
       'Cu', 'Dy', 'Er', 'Eu', 'F',  'Fe', 'Ga', 'Gd', 'Ge', 'H',
       'Hf', 'Hg', 'Ho', 'I',  'In', 'Ir', 'K',  'La', 'Li', 'Lu',
       'Mg', 'Mn', 'Mo', 'N',  'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O',
       'Os', 'P',  'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra',
       'Rb', 'Re', 'Rh', 'Ru', 'S',  'Sb', 'Sc', 'Se', 'Si', 'Sm',
       'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm',
       'U',  'V',  'W',  'Y',  'Yb', 'Zn', 'Zr']
   

.. py:data:: selected_elements

   Selected elements for a specific system. For example::
   
      ['H', 'He', 'Li', 'Be', 'B']
   

.. py:data:: Elements_used

   Elements used for one-hot coding.
   
   .. Important:: This variable defines the elements used for a well-trained model. Make sure the atomistic representation for the training and prediction are same. Normally, ``Elements_used = elements`` is robust.
   
   
.. py:function:: get_atomic_features()

   Encode atomistic features with physical properties. For more input features, please refer to: https://pymatgen.org/pymatgen.core.periodic_table.html. For now, the following properties are included.
   
   .. note:: This function will be deprecated in the short future.
      
      ====================  =============
      Property Code         Property
      ====================  =============
      Z                     atomic number
      atomic_radius         atomic radius
      molar_volume          molar volume
      atomic_mass           atomic mass
      mendeleev_no          mendeleev number
      X                     electronegativity
      boiling_point         boiling point
      melting_point         melting point
      row                   row number in the periodic table
      group                 group number in the periodic table
      max_oxidation_state   maximum oxidation state
      min_oxidation_state   minimum oxidation state
      ====================  =============
      

   :Returns: atom_feat: atomic features of all elements.
   

.. py:function:: get_atomic_feature_onehot(Elements_used)

   Get the one-hot code of all considered elements. The elements are sorted by their atomic number::
   
     atomic_number = [atomic_numbers[x] for x in Elements_used]
     atoms_dict = dict(zip(atomic_number, Elements_used))
     
     atomic_number.sort()
     
     Elements_sorted = [atoms_dict[x] for x in atomic_number]

   :param list Elements_used: Elements used to encode atomic representations.

   :Returns: atom_feat: one-hot code representation of all considered elements.



