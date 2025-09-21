##########
optimizers
##########

.. Hint:: Modified ase optimizers by implementing Torch tensor calculation.

Examples of using optimizers:

.. code-block:: python

   ###############################################################################
   # Detect GPU card
   ###############################################################################
   import torch
   if torch.cuda.is_available():
       device='cuda'
       print("CUDA is available.")
       print(f"Number of GPUs: {torch.cuda.device_count()}")
       print(f"GPU Name: {torch.cuda.get_device_name(0)}")
   else:
       device='cpu'
       print("CUDA is NOT available.")
   
   import dgl
   u, v = torch.tensor([0, 0, 0, 1], device=device), torch.tensor([1, 2, 3, 3], 
                                                                  device=device)
   g = dgl.graph((u, v), device=device)
   print(f'DGL graph device: {g.device}')

   ###############################################################################
   # Optimizers
   ###############################################################################
   import os
   from ase.io import read, write
   from ase import Atoms
   from agat.app.optimizers import BFGSTorch, MDMinTorch
   from agat.app.calculators import AgatCalculator
   # --- 1 --- BFGSTorch
   model_save_dir = os.path.join('potential_models', 'agat_model_1')
   graph_build_scheme_dir = os.path.join('potential_models')
   atoms = read(os.path.join('potential_models', 'POSCAR'))
   calculator=AgatCalculator(model_save_dir,
                             graph_build_scheme_dir,
                             device=device)
   atoms = Atoms(atoms, calculator=calculator)
   dyn = BFGSTorch(atoms, trajectory='test.traj', device=device)
   dyn.run(fmax=1.0)
   traj = read('test.traj', index=':')
   write("XDATCAR.gat", traj)

   # --- 2 --- MDMinTorch
   model_save_dir = os.path.join('potential_models', 'agat_model_1')
   graph_build_scheme_dir = os.path.join('potential_models')
   atoms = read(os.path.join('potential_models', 'POSCAR'))
   calculator=AgatCalculator(model_save_dir,
                             graph_build_scheme_dir,
                             device=device)
   atoms = Atoms(atoms, calculator=calculator)
   dyn = MDMinTorch(atoms, trajectory='test.traj', device=device)
   dyn.run(fmax=1.0)
   
   # --- 3 --- ase.optimize.BFGS
   from ase.optimize import BFGS
   model_save_dir = os.path.join('potential_models', 'agat_model_1')
   graph_build_scheme_dir = os.path.join('potential_models')
   atoms = read(os.path.join('potential_models', 'POSCAR'))
   calculator=AgatCalculator(model_save_dir,
                             graph_build_scheme_dir,
                             device=device)
   atoms = Atoms(atoms, calculator=calculator)
   dyn = BFGS(atoms, trajectory='test.traj')
   dyn.run(fmax=1.0)



.. attribute:: BFGSTorch
   :no-index:
   
   .. method:: __init__(self, atoms: Atoms, restart: Optional[str] = None, logfile: Optional[Union[IO, str]] = '-', trajectory: Optional[str] = None, append_trajectory: bool = False, maxstep: Optional[float] = None, master: Optional[bool] = None, alpha: Optional[float] = None, device = torch.device('cuda'))
		
      :param str/torch.device device: Device to run the simulation.
	  
	  .. Other parameters:: atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.




.. attribute:: MDMinTorch
   :no-index:
   
   .. method:: __init__(self, atoms: Atoms, restart: Optional[str] = None, logfile: Union[IO, str] = '-', trajectory: Optional[str] = None, dt: Optional[float] = None, maxstep: Optional[float] = None, master: Optional[bool] = None, 'cuda')
      :no-index:
	  
      :param str/torch.device device: Device to run the simulation.
	  
	  .. Other parameters:: atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: string
            Text file used to write summary information.

        dt: float
            Time step for integrating the equation of motion.

        maxstep: float
            Spatial step limit in Angstrom. This allows larger values of dt
            while being more robust to instabilities in the optimization.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
   
