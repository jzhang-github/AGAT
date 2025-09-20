#########
ensembles
#########

Modified ``ase.md`` ensembles to deploy the AGAT model.

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



