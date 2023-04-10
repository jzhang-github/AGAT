############
GatApp
############


Applications for AGAT model.


.. class:: GatApp

   Apply well-trained model.
   
   .. method:: __init__(self, energy_model_save_path, force_model_save_path, load_model=True, gpu=-1)
   :param str energy_model_save_path: directory storing well-trained energy model.
   :param str force_model_save_path: directory storing well-trained force model.
   :param bool load_model: load energy model and force model.
   :param int gpu: specify calculations on device. ``/cpu:0``: CPU cores; ``"/gpu:{}".format(gpu)``: GPU card.


   .. method:: load_graph_build_scheme(self, model_save_path)
   
      :param str model_save_path: load scheme for building graphs.
      
      :Returns: graph_build_scheme: a json file.
   
   
   .. method:: load_energy_model(self, energy_model_save_path)
   
      :param str energy_model_save_path: directory storing well-trained energy model.
      
      :Returns: an energy AGAT model.
      
      
   .. method:: load_force_model(self, force_model_save_path)
   
      :param str force_model_save_path: directory storing well-trained force model.
      
      :Returns: a force AGAT model.
      
      
   .. method:: get_graph(self,  fname, super_cell=self.graph_build_scheme['read_super_cell'], include_forces=False)
   
      :param str fname: load graph from a file.
      
         .. Note:: In some cases, the ``fname`` can also be ``ase.atoms`` object.
         
      :param bool super_cell: repeat small cell.
      :param bool include_forces: ``False``; No need to read True forces for application.
   

   .. method:: get_energy(self, graph)
   
      Predict energy of the input graph.
      
      :param DGL.graph graph: a graph.
      :Returns: crystal energy.


   .. method:: get_energies(self, graph)
   
      Predict atomic energies of the input graph.
      
      :param DGL.graph graph: a graph.
      :Returns: atomic energies.


   .. method:: get_forces(self, graph)
   
      Predict forces acting on atoms in real space.
      
      :param DGL.graph graph: a graph.
      :Returns: atomic forces.


   .. method:: get_stress(self,)
   
      Predict cell stress.
      
      .. Note:: Not applicable.


.. class:: GatAseCalculator(Calculator)

   Deploy AGAT model on ``ase.calculators``, which can relax and find the ground state energy of a structure.
   
   .. Note:: This calculator is more efficient than :class:`GatCalculator`
   
   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'energies', 'free_energy', 'forces', 'stress', 'stresses']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
   
   .. method:: __init__(self, energy_model_save_path, force_model_save_path, load_model=True, gpu = -1, **kwargs)
   
      :param str energy_model_save_path: directory storing well-trained energy model.
      :param str force_model_save_path: directory storing well-trained force model.
      :param bool load_model: whether to load the model.
      :param int gpu: - ``-1``: predict energy and forces on CPU cores.
         - ``0``: predict energy and forces on GPU card.
      :param args **kwargs: configurations.
      
   .. method:: calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc'])
   
      :param ase.atoms atoms: ase.atoms object. Input structure.
      :param list/NoneType properties: a list of properties calculated by this calculator.
      :param list system_changes: a list of properties that can change during relaxation.


.. class:: GatCalculator(Calculator)

   Deploy AGAT model on ``ase.calculators``, which can relax and find the ground state energy of a structure.
   
   .. Note:: This calculator saves intermediate structural file to disk, and loads graph by pymatgen. Not memory and storage efficient. More efficient option: :class:`GatAseCalculator`
   
   .. Note:: Go to https://wiki.fysik.dtu.dk/ase/development/calculators.html#adding-new-calculators for more information about ``ase.calculators``
   
   .. attribute:: implemented_properties
   
      .. code-block::
      
         ['energy', 'energies', 'free_energy', 'forces', 'stress', 'stresses']
      
   .. attribute:: default_parameters
   
      .. code-block::
      
         {}
   
   .. method:: __init__(self, energy_model_save_path, force_model_save_path, load_model=True, **kwargs)
   
      :param str energy_model_save_path: directory storing well-trained energy model.
      :param str force_model_save_path: directory storing well-trained force model.
      :param bool load_model: whether to load the model.
      :param args **kwargs: configurations.
      
      
   .. method:: calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc'])
   
      :param ase.atoms atoms: ase.atoms object. Input structure.
      :param list/NoneType properties: a list of properties calculated by this calculator.
      :param list system_changes: a list of properties that can change during relaxation.

