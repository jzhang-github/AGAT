#########
ensembles
#########

.. Note:: Modified ``ase.md`` ensembles to deploy the AGAT model.

Examples of using ensembles:

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
   
   ###############################################################################
   # Modified ASE ensembles for AGAT
   ###############################################################################
   import dgl
   u, v = torch.tensor([0, 0, 0, 1], device=device), torch.tensor([1, 2, 3, 3], 
                                                                  device=device)
   g = dgl.graph((u, v), device=device)
   print(f'DGL graph device: {g.device}')
   
   import os
   from ase.io import read
   from ase import units
   from ase.md.npt import NPT
   from ase.md import MDLogger
   from agat.app.ensembles import ModifiedNPT
   from agat.app.calculators import AgatEnsembleCalculator
   model_ensemble_dir = os.path.join('potential_models')
   graph_build_scheme_dir = os.path.join('potential_models')
   atoms = read(os.path.join('potential_models', 'POSCAR'))
   calculator=AgatEnsembleCalculator(model_ensemble_dir, graph_build_scheme_dir, 
                                     device=device)
   atoms.set_calculator(calculator)
   
   dyn = ModifiedNPT(atoms,
             timestep=1.0 * units.fs,
             temperature_K=300,
             ttime = 25 * units.fs,
             pfactor = 75 * units.fs,
             externalstress = [0.0] * 6,
             mask=[[1,0,0],
                   [0,1,0],
                   [0,0,1]],
             trajectory=os.path.join('md_NPT.traj'))
   
   dyn.attach(MDLogger(dyn, atoms, os.path.join('md_NPT.log'),
                       header=True,
                       stress=True,
                       peratom=False,
                       mode="a"),
              interval=1)
   
   dyn.run(200)


.. class:: ModifiedNPT(NPT)

   Modified ``ase.md.npt.NPT`` ensemble, which is used for the on-the-fly training of a AGAT ``PotentialModel``.


   .. Note:: Go to https://ase-lib.org/ase/md.html for more information.

   .. attribute:: classname

      .. code-block::

         'ModifiedNPT'


   .. method:: __init__(self, atoms, timestep, temperature, externalstress, ttime, pfactor, *arg, temperature_K, mask, trajectory, logfile, loginterval, append_trajectory, max_collected_snapshot_num = 500)

      Most arguments can be found at https://ase-lib.org/ase/md.html#constant-npt-simulations-the-isothermal-isobaric-ensemble

      : param max_collected_snapshot_num: The maximum number of collected snashots in a on-the-fly training. Defaults to ``500``.
      : type max_collected_snapshot_num: int

   .. method:: run(self, steps)

        Run NPT simulation.

        :param steps: Steps for the MD simulation.
        :type steps: int



