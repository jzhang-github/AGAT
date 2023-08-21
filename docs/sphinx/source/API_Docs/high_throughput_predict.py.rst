##############################
high_throughput_predict.py
##############################


.. Note:: Example of using this code: `example`_


.. function:: file_exit()

   Stop high-throughput prediction if ``StopPython`` is detected.
   
   :raises FileExit: Exit because `StopPython` file is found.
   

.. function:: generate_file_name(fname)

   Generate a new file name according to the input.

   :param str fname: file name.
   :Returns: fname. A new file name by appending `_new`
   :rtype: str


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


.. function:: geo_opt(atoms_with_calculator, **kwargs)

   Geometrical optimization of given structure.
   
   .. Note:: `BFGS`_ optimizer is adopted.
   
   :param ase.atoms atoms_with_calculator: structure with ``ase.calculators`` attribute.
   :param dict/args **kwargs: relaxation configuration.
   
      .. Hint:: You can find detailed settings of `BFGS`_. Here is the summary:

         ================= ============ ============
         Setting           Default      Description
         ================= ============ ============
         fmax              0.05         Maximum force acting on every atom should be lower than this.
         steps             200          Stop optimization if iteration step is reached.
         maxstep           0.1          Used to set the maximum distance an atom can move per iteration
         restart           None         Pickle file used to store hessian matrix. If set, file with such a name will be searched and hessian matrix stored will be used, if the file exists.
         restart_steps     5            In some cases, it can be difficult for `BFGS`_ to converge. If `BFGS`_ cannot coverge after ``steps``, this code halves ``maxstep`` and rerun. But this code will not rerun more than ``restart_steps`` times.
         perturb_steps     0            Defines how many perturbated structures (perturb free atoms only) are optimized. Only structure with lowest energy is adopted.
         perturb_amplitude 0.05         perturbation amplitude
         out               None         file name for outputs.
         ================= ============ ============
   
   :Returns: - energy: energy after relaxation
      - force: atomic forces after relaxation.
      - force_max: max atomic force after relaxation.


.. function:: ads_calc(formula, calculator, **kwargs)

   `BFGS`_ calculations including geometrical optimizations of bulk, clean surface, and adsorption strucures.
   
   :param str formula: chemical formula.
   :param ase.calculators calculator: calculator.
   :param dict/args **kwargs: relaxation configuration.
   
      .. Hint:: Details of relaxation configuration:
         
         ===================== ================================== ============
         Setting               Default                             Description
         ===================== ================================== ============
         v_per_atom            14.045510416666668                  Volume per atom of bulk structure.
         calculation_index     None                                Calculation index. You can use this parameter to differentiate multiple calculations.
         fix_surface_atom      False                               Fix all surface atoms if this is True.
         remove_bottom_atoms   False                               Remove the bottom atomic plane if this is True.
         save_trajectory       False                               Save relaxation trajectory or not.
         partial_fix_adsorbate False                               Partially fix adsorbate positions.
         adsorbates            ['O', 'OH', 'OOH']                  Adsorbates placed on the surface.
         sites                 ['ontop', 'bridge', 'hollow']       Sites for placing adsorbates.
         fmax                  0.1                                 Maximum force acting on every atom should be lower than this.
         dist_from_surf        2.0                                 Distance between adsorbate and surface.
         ===================== ================================== ============
   




.. _example: https://github.com/jzhang-github/AGAT/tree/main/AGAT_CATA#high-throughput-predict
.. _BFGS: https://wiki.fysik.dtu.dk/ase/ase/optimize.html#bfgs




