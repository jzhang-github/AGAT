##############################
high_throughput_predict
##############################


.. class:: HpAds(object)

   High-throughput predictions. 

   .. function:: __init__(self, **hp_config)
   
      See https://jzhang-github.github.io/AGAT/Default%20parameters.html#default-high-throughput-config for more details.


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
   
   .. function:: run(self, formula, **kwargs)

      :param formula: Input chemical formula
      :type formula: str


.. _BFGS: https://wiki.fysik.dtu.dk/ase/ase/optimize.html#bfgs




