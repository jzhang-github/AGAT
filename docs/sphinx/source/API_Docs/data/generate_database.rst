#################
generate_database
#################

Generate database for AGAT, including strain, perturbation, vacancy, static, NPT, and NVT simulations. The function names are quite intuitive, so allow me finish this page later.

.. py:class:: DataGenerator(object)


   .. py:method:: __init__(self, bulk_fname)

   .. py:method:: set_structure(self, fname)

   .. py:method:: get_INCAR(self, dst='.')

   .. py:method:: get_KPOINTS(self, dst='.')

   .. py:method:: get_POTCAR(self, line=1, working_dir='.')

   .. py:method:: apply_strain(self, ase_atoms, strain: float)

   .. py:method:: apply_perturbation(self, ase_atoms, amplitude: float = 0.2)

   .. py:method:: create_vacancy(self, ase_atoms, index='random')


   .. py:method:: create_species_vacancy(self, ase_atoms, species='Li', num=1)


   .. py:method:: relocate_atoms(self, ase_atoms, displacement=[0.0, 0.0, 0.1], species='Li')


   .. py:method:: static(self, dst='.', strain=[-0.02, -0.01, 0.0, 0.01, 0.02], perturbation_num=10)



   .. py:method:: aimd(self, dst='.', strain=[-0.02, -0.01, 0.0, 0.01, 0.02], start_T=[100, 300, 500], end_T=[400, 600, 800])
