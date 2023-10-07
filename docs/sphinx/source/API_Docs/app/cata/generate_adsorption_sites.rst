################################
generate_adsorption_sites_ase
################################

Generate adsorption configurations on [111] surface of FCC metals.


.. class:: AddAtoms(object)

   Place atoms (adsorbate) on surface.
   
   .. code-block:: python
   
      adder = AddAtoms('POSCAR_surf_opt_1.gat',   # input file name.
                 species='C3H6_di',               # name of adsorbates. O, H, OH, OOH, C3H6_di, C3H6_pi, C3H7
                 sites='disigma',            # name of sites. ontop, bridge, hollow, disigma, or all
                 dist_from_surf=1.7,        # distance between adsorbate and the surface. Decrease this variable if the adsorbate is too high.
                 num_atomic_layer_along_Z=6) # number of atomic layers along z direction of the slab model.
       from agat.lib.adsorbate_poscar import adsorbate_poscar
       num_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar)



   .. method:: __init__(self, fname, sites='all', species='O', num_atomic_layer_along_Z=3, dist_from_surf=2.2, crystal_type='fcc')
   
      :param str fname: file name of a surface structure.
      :param str/list sites: a list of sites for placing adsorbate.
      :param str species: name of adsorbate.
      :param int num_atomic_layer_along_Z: input number of atomic layers along z of input structural file.
      :param float dist_from_surf: distance between adsorbate and surface.
      :param str crystal_type: FCC, BCC, HCP, cubic.
      
      
   .. property:: _support_sites
      
      :type: list
      :value: ['ontop', 'bridge', 'hollow']
      
   .. method:: shift_fcoords(self, fcoord1, fcoord2)
   
      Identify closest periodic image between two scaled coordinates.
      
      :param numpy.ndarray fcoord1: the first coordinate.
      :param numpy.ndarray fcoord2: the second coordinate.
      :returns: - fcoord1_new: new coordinate of first input.
         - fcoord2_new: new coordinate of second input.
         
      .. Note:: This method treats periodic boundary conditions. Two returned coordinates show shortest distance among periodic cells.


   .. method: get_middle_fcoord(self, fcoords)
      
      Get the centroid of the input coordinates
      
      :param numpy.ndarray fcoords: an array of coordinates.
      :returns: centroid
      :rtype: numpy.ndarray
      

   .. method:: get_1NN_cutoff(self)
   
      Get cutoff for identifying the first-nearest neighbor.
      
      :returns: 1NN cutoff
      :rtype: float


   .. method:: get_atomic_diameter_for_slab(self, layer_sapcing, atomic_layers, structure)
   
      Get average atomic diameter in a slab model.
      
      :param float layer_sapcing: Layer spacing along z.
      :param int atomic_layers: Number of atomic layers along z.
      :param ase.atoms structure: ase atoms object.
      :returns: average atomic diameter.


   .. method:: get_1NN_of_top_layer(self, structure=None)
      
      Neighbor analysis of the top atomic layer.
      
      :param ase.atoms structure: input structure.
      :returns: 1NN relation of top atomic layer.
      :rtype: numpy.ndarray


   .. method:: get_ontop_sites(self, x_shift=0.0, y_shift=0.0)
   
      Find ontop sites for placing adsorbate.
      
      :param float x_shift: move adsorbate along x.
      :param float y_shift: move adsorbate along y.
      :returns: - ccoords: Cartesian coordinates
         - an array of ``None`` with the same length of ``ccords``.


   .. method:: get_bridge_sites(self, x_shift=0.0, y_shift=0.0)
   
      Find bridge sites for placing adsorbate.
      
      :param float x_shift: move adsorbate along x.
      :param float y_shift: move adsorbate along y.
      :returns: - ccoords: Cartesian coordinates
         - an array of ``None`` with the same length of ``ccords``.


   .. method:: get_disigma_sites(self, x_shift=0.0, y_shift=0.0)
      
      Find bridge sites for placing adsorbate.
      
      .. Note:: Not used for now.
      
      :param float x_shift: move adsorbate along x.
      :param float y_shift: move adsorbate along y.
      :returns: - src: source binding site.
         - vectors: direction of adsorbate.


   .. method:: get_hollow_sites(self, x_shift=0.0, y_shift=0.0)
   
      Find hollow sites for placing adsorbate.
      
      :param float x_shift: move adsorbate along x.
      :param float y_shift: move adsorbate along y.
      :returns: - ccoords: Cartesian coordinates
         - an array of ``None`` with the same length of ``ccords``.


   .. method:: fractional2cartesian(self, vector_tmp, D_coord_tmp)
   
      Convert fractional coordinates to Cartesian coordinates. Source code:
      
      .. code-block::
         
         def fractional2cartesian(self, vector_tmp, D_coord_tmp):
             C_coord_tmp = np.dot(D_coord_tmp, vector_tmp)
             return C_coord_tmp
      
      :param numpy.ndarray vector_tmp: cell vectors.
      :param numpy.ndarray D_coord_tmp: direct (fractional or scaled) coordinates.
      :returns: Cartesian coordinates.
      :rtype: numpy.ndarray


   .. method:: cartesian2fractional(self, vector_tmp, C_coord_tmp)

      Convert Cartesian coordinates to fractional coordinates. Source code:
      
      .. code-block::
         
         def cartesian2fractional(self, vector_tmp, C_coord_tmp):
             vector_tmp = np.mat(vector_tmp)
             D_coord_tmp = np.dot(C_coord_tmp, vector_tmp.I)
             D_coord_tmp = np.array(D_coord_tmp, dtype=float)
             return D_coord_tmp

      :param numpy.ndarray vector_tmp: cell vectors.
      :param numpy.ndarray C_coord_tmp: Cartesian coordinates.
      :returns: direct (fractional or scaled) coordinates.
      :rtype: numpy.ndarray


   .. method:: write_file_with_adsorption_sites(self, adsorbate_poscar, calculation_index=0)
   
      Write adsorption structures to disk.
      
      :param str adsorbate_poscar: `POSCAR`_ of adsorbate.
         
         .. Note:: the `POSCAR`_ of adsorbate is read by ``StringIO``.
      
      :param int/str calculation_index: index of calculations. Use this to differentiate outputs of high-throughput predictions.

      




.. data:: adsorbate_poscar

   Structural positions with `VASP`_ format.
   
   :type: dict
   :value: 
   
      .. code-block::
      
         {'O':
         '''O
            1.000000000000000
          10.0000000000000000  0.0000000000000000  0.0000000000000000
           0.0000000000000000 10.0000000000000000  0.0000000000000000
           0.0000000000000000  0.0000000000000000 10.0000000000000000
            1
         Direct
         0.0 0.0 0.0
         ''',
         
         'OH':
         '''O H
            1.000000000000000
          10.0000000000000000  0.0000000000000000  0.0000000000000000
           0.0000000000000000 10.0000000000000000  0.0000000000000000
           0.0000000000000000  0.0000000000000000 10.0000000000000000
            1 1
         C
         0.0 0.0 0.0
         0.6 0.0 0.7
         ''',
         
         'H':
         '''H
            1.000000000000000
          10.0000000000000000  0.0000000000000000  0.0000000000000000
           0.0000000000000000 10.0000000000000000  0.0000000000000000
           0.0000000000000000  0.0000000000000000 10.0000000000000000
            1
         C
         0.0 0.0 0.0
         ''',
         
         'OOH':
         '''O H
            1.000000000000000
          10.0000000000000000  0.0000000000000000  0.0000000000000000
           0.0000000000000000 10.0000000000000000  0.0000000000000000
           0.0000000000000000  0.0000000000000000 10.0000000000000000
            2 1
         C
         0.0 0.0 0.0
         1.290 0.0  0.733
         1.290 0.985 0.733
         '''
         }
         
      .. Hint:: You can add your own adsorbate.











.. _VASP: https://www.vasp.at/
.. _POSCAR: https://www.vasp.at/wiki/index.php/POSCAR




