############
calculators
############

AGAT model applications

.. class:: AgatCalculator(Calculator)

   Deploy AGAT model on ``ase.calculators`` for geometry optimization and molecular dynamics simulations.

   
   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'forces', 'stress']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
         
   .. attribute:: ignored_changes
   
      .. code-block::
      
         set()
		 
   .. method:: __init__(self, model_save_dir, graph_build_scheme_dir, graph_build_scheme, device = 'cuda', \**kwargs)

      :param model_save_dir: Directory storing the well-trained model.
      :type model_save_dir: str
      :param graph_build_scheme_dir: Direcotry storing the ``graph_build_scheme.json`` file.
      :type graph_build_scheme_dir: str
	  :param graph_build_scheme: Direcotry storing the ``graph_build_scheme.json`` file or parse the input dict. Note that this argument has higher priority than ``graph_build_scheme_dir``.
      :type graph_build_scheme: str / dict
      :param device: model device, defaults to 'cuda'
      :type device: str, optional
      :param \**kwargs: other input arguments
      :type \**kwargs: dict
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
		

.. class:: AgatCalculatorAseGraphTorch(Calculator)

   Deploy AGAT model on ``ase.calculators`` for geometry optimization and molecular dynamics simulations.

   .. Hint:: This object builds ``dgl`` graphs with modified ase codes that leverage GPU resources: ``AseGraphTorch``, and much faster than original ase method on CPU. See https://github.com/jzhang-github/AGAT/blob/main/agat/data/build_graph.py#L383
   
   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'forces', 'stress']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
         
   .. attribute:: ignored_changes
   
      .. code-block::
      
         set()
		 
   .. method:: __init__(self, model_save_dir, graph_build_scheme_dir, graph_build_scheme, device = 'cuda', \**kwargs)

      :param model_save_dir: Directory storing the well-trained model.
      :type model_save_dir: str
      :param graph_build_scheme_dir: Direcotry storing the ``graph_build_scheme.json`` file.
      :type graph_build_scheme_dir: str
	  :param graph_build_scheme: Direcotry storing the ``graph_build_scheme.json`` file or parse the input dict. Note that this argument has higher priority than ``graph_build_scheme_dir``.
      :type graph_build_scheme: str / dict
      :param device: model device, defaults to 'cuda'
      :type device: str, optional
      :param \**kwargs: other input arguments
      :type \**kwargs: dict
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
		
		.. Note::  The outputs are ``torch.Tensor`` s.

.. class:: AgatCalculatorAseGraphTorchNumpy(Calculator)

   Deploy AGAT model on ``ase.calculators`` for geometry optimization and molecular dynamics simulations.

   .. Hint:: This object builds ``dgl`` graphs with modified ase codes that leverage GPU resources: ``AseGraphTorch``, and much faster than original ase method on CPU. See https://github.com/jzhang-github/AGAT/blob/main/agat/data/build_graph.py#L383
   
   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'forces', 'stress']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
         
   .. attribute:: ignored_changes
   
      .. code-block::
      
         set()
		 
   .. method:: __init__(self, model_save_dir, graph_build_scheme_dir, graph_build_scheme, device = 'cuda', \**kwargs)

      :param model_save_dir: Directory storing the well-trained model.
      :type model_save_dir: str
      :param graph_build_scheme_dir: Direcotry storing the ``graph_build_scheme.json`` file.
      :type graph_build_scheme_dir: str
	  :param graph_build_scheme: Direcotry storing the ``graph_build_scheme.json`` file or parse the input dict. Note that this argument has higher priority than ``graph_build_scheme_dir``.
      :type graph_build_scheme: str / dict
      :param device: model device, defaults to 'cuda'
      :type device: str, optional
      :param \**kwargs: other input arguments
      :type \**kwargs: dict
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
		
		.. Note::  The outputs are ``numpy.array`` s.


.. class:: AgatEnsembleCalculator(Calculator)

   Deploy AGAT model on ``ase.calculators`` for geometry optimization and molecular dynamics simulations.

   .. Hint:: This object is used to calculate atomic energy, forces, and cell stresses with multiples models.
   
   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'forces', 'stress']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
         
   .. attribute:: ignored_changes
   
      .. code-block::
      
         set()
		 
   .. method:: __init__(self, model_save_dir, graph_build_scheme_dir, graph_build_scheme, start_step, device = 'cuda', io, \**kwargs)

      :param model_save_dir: Directory storing the well-trained model.
      :type model_save_dir: str
      :param graph_build_scheme_dir: Direcotry storing the ``graph_build_scheme.json`` file.
      :type graph_build_scheme_dir: str
	  :param graph_build_scheme: Direcotry storing the ``graph_build_scheme.json`` file or parse the input dict. Note that this argument has higher priority than ``graph_build_scheme_dir``.
      :type graph_build_scheme: str / dict
	  :param start_step: log the calculation steps.
      :type start_step: int
      :param device: model device, defaults to 'cuda'
      :type device: str, optional
	  :param io: Unknown.
      :type io: int
      :param \**kwargs: other input arguments
      :type \**kwargs: dict
      :return: Calculated properties.
      :rtype: dict
    
   .. Note:: ``graph_build_scheme`` has higher priority than ``graph_build_scheme_dir``.
   
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
		

.. class:: OnTheFlyCalculator(Calculator)

   Deploy AGAT model on ``ase.calculators`` for geometry optimization and molecular dynamics simulations.
   
   For the on-the-fly training of a ``agat.model.PotentialModel``.

   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'forces', 'stress']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
         
   .. attribute:: ignored_changes
   
      .. code-block::
      
         set()
		 
   .. method:: __init__(self, model_save_dir, graph_build_scheme, use_vasp=False, start_step=0, vasp_work_dir='.', vasp_inputs_dir='.', gamma_only=False, vasp_potential_generator='getpotential.sh', vasp_script='vasp_run.sh', device = 'cuda', energy_threshold = 0.5, force_threshold = 0.5, stress_threshold = 0.5, io=None, \**kwargs)

      :param model_save_dir: Directory storing the well-trained model.
      :type model_save_dir: str
	  :param use_vasp: TEST
	  
	  :param graph_build_scheme: Direcotry storing the ``graph_build_scheme.json`` file or parse the input dict. Note that this argument has higher priority than ``graph_build_scheme_dir``.
      :type graph_build_scheme: str / dict
	  :param start_step: log the calculation steps.
      :type start_step: int
      :param device: model device, defaults to 'cuda'
      :type device: str, optional
	  :param io: Unknown.
      :type io: int
      :param \**kwargs: other input arguments
      :type \**kwargs: dict
      :return: Calculated properties.
      :rtype: dict
    
   .. Note:: ``graph_build_scheme`` has higher priority than ``graph_build_scheme_dir``.
   
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
		
		.. Note::  The outputs are ``numpy.array`` s.
	