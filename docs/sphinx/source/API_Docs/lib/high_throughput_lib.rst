high_throughput_lib
###################




.. function:: perturb_positions(atoms, amplitude=0.1)

   Perturbate atomic positions. This method will not change positions of fixed atoms.
   
   :param ase.atoms atoms: input atoms structure.
   :param float amplitude: Perturbate amplitude.
   :returns: new atoms.


.. function:: scale_atoms(atoms, scale_factor=1.0)

   Scale cell volume and atomic positions.
   
   :param ase.atoms atoms: input atoms structure.
   :param float scale_factor: scale factor.
   :returns: new atoms.


.. function:: get_concentration_from_ase_formula(formula)

   Calculate elemental concentrations of given chemical formula.
   
   :param str formula: chemical formula.
   :returns: a dictionary of concentrations of all elements.


.. function:: get_v_per_atom(chemical_formula)

   Get volume per atom of Ni-Co-Fe-Pd-Pt system.
   
   .. Note: v_per_atom = -282.7957531391954 * c_NiCoFe(sum) - 278.79605077419797 * C_Pd - -278.6228860885035 * C_Pt + 293.66128761358624
   
   :param str chemical_formula: chemical formula.
   :return: volume per atom of given chemical formula.


.. function:: get_ase_atom_from_formula(chemical_formula, v_per_atom=None)

   Build bulk structure according to given chemical formula.
   
   .. Note:: Cell orientation: 
      
      - ``x``: <1 -1 0>
      - ``y``: <1 1 -2>
      - ``z``: <1 1 1>
   
   :param str chemical_formula: chemical formula
   :param float v_per_atom: volume per atom
   :returns: bulk structure of a ``ase.atoms`` object.


.. function:: get_ase_atom_from_formula_template(chemical_formula, v_per_atom=None, template_file='POSCAR_temp')
   
   Build structure based on a template file.
   
   :param str chemical_formula: chemical formula
   :param v_per_atom float/NoneType/int/bool: volume per atom. Scale the template structure to fit the given volume per atom.
   :param str template_file: file name of the template structure.
   :returns: bulk structure of a ``ase.atoms`` object.

