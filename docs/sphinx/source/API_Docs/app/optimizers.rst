###
optimizers
###

AGAT model applications

.. class:: AgatCalculatorb(Calculator)

   Deploy AGAT model on ``ase.calculators``, which can relax and find the ground state energy of a structure.

   
   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'forces']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
         

   .. method:: __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda', **kwargs)

      :param model_save_dir: Directory storing the well-trained model.
      :type model_save_dir: str
      :param graph_build_scheme_dir: Direcotry storing the ``graph_build_scheme.json`` file.
      :type graph_build_scheme_dir: str
      :param device: model device, defaults to 'cuda'
      :type device: str, optional
      :param **kwargs: other input arguments
      :type **kwargs: dict
      :return: Calculated properties.
      :rtype: dict
    
      Example::
    
          model_save_dir = 'agat_model'
          graph_build_scheme_dir = 'dataset'
          atoms = read('CONTCAR')
          calculator=AgatCalculator(model_save_dir,
                                    graph_build_scheme_dir)
          atoms = Atoms(atoms, calculator=calculator)
          dyn = BFGS(atoms, trajectory='test.traj')
          dyn.run(fmax=0.005)
    
          traj = read('test.traj', index=':')
          write("XDATCAR.gat", traj)
    
   .. method:: load_graph_build_scheme(self, path)

        Load graph building scheme. 
        
        .. note:: This file is normally saved to the disk when you build your dataset, under the same directory containing ``all_graphs.bin``.

        :param path: Directory for storing ``graph_build_scheme.json`` file.
        :type path: str
        :return: A dict denotes how to build the graph.
        :rtype: dict

   .. method:: calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc'])

        :param atoms: ase.atoms object, defaults to None
        :type atoms: ase.atoms, optional
        :param properties: calculated properties, defaults to None
        :type properties: none, optional
        :param system_changes: DESCRIPTION, defaults to ['positions', 'numbers', 'cell', 'pbc']
        :type system_changes: TYPE, optional
        :return: calculated results
        :rtype: dict