#############
HpDftAds
#############


.. py:class:: HpDftAds

   High-throughput DFT calculations for adsorption.

   .. py:property:: root_dir

      The root working directory of this object.

   .. py:method:: __init__(self, **hp_config)

      :param **hp_config: Configurations to control the process.
      :type **hp_config: dict

      Example::

          HA = HpDftAds(calculation_index=0)
          HA.run('NiCoFePdPt')

   .. py:method:: bulk_opt(self, formula)

      Structural optimization of the bulk structure with VASP.

      :param formula: Chemical formula
      :type formula: str

   .. py:method:: surf_opt(self, bulk_structural_file='CONTCAR_bulk_opt')

      Structural optimization of the surface slab with VASP.

      :param bulk_structural_file: optimized bulk structure, defaults to 'CONTCAR_bulk_opt'
      :type bulk_structural_file: str, optional


   .. py:method:: ads_opt(self, structural_file='CONTCAR_surf_opt', random_samples=5)

      Structural optimization of the adsorption with VASP.

      :param structural_file: Structural file name of optimized clean surface, defaults to 'CONTCAR_surf_opt'
      :type structural_file: str, optional
      :param random_samples: On one surface, many surface sites can be detected, this number controls how many individual calculations will be performed on this surface, defaults to 5
      :type random_samples: int, optional

      .. Note::
         ``random_samples`` cannot be larger than the number of detected surface sites.

   .. py:method:: bulk_aimd(self, formula)

      AIMD simulation for a bulk structure of given chemical formula

      :param formula: The given chemical formula.
      :type formula: str


   .. py:method:: surface_aimd(self, bulk_structural_file='CONTCAR_bulk_opt')

      AIMD simulation for the clean surface.

      :param bulk_structural_file: File name of the bulk structure, defaults to 'CONTCAR_bulk_opt'
      :type bulk_structural_file: str, optional

   .. py:method:: ads_aimd(self, structural_file='CONTCAR_surf_opt', random_samples=2)

      AIMD simulation for the adsorption.

      :param structural_file: File name of the clean surface, defaults to 'CONTCAR_surf_opt'
      :type structural_file: str, optional
      :param random_samples: Randomly select surface sites for the simulation, defaults to 2
      :type random_samples: int, optional

      .. Note::
          ``random_samples`` cannot be larger than the number of detected surface sites.

   .. py:method:: run(self, formula, **kwargs)

      :param formula: Chemical formula.
      :type formula: str
      :param **kwargs: Configurations to control the process.
      :type **kwargs: dict

