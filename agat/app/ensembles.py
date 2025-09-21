# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 22:37:52 2024

@author: ZHANG Jun
"""

from typing import IO, Optional, Tuple, Union
import numpy as np

from ase import Atoms
from ase.md.npt import NPT

class ModifiedNPT(NPT):
    classname = 'ModifiedNPT'

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: Optional[float] = None,
        externalstress: Optional[float] = None,
        ttime: Optional[float] = None,
        pfactor: Optional[float] = None,
        *arg,
        temperature_K: Optional[float] = None,
        mask: Optional[Union[Tuple[int], np.ndarray]] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        append_trajectory: bool = False,

        # max_collected_snapshot_num = 500
    ):

        super(ModifiedNPT, self).__init__(
                     atoms,
                     timestep,
                     temperature,
                     externalstress,
                     ttime,
                     pfactor,
                     *arg,
                     temperature_K=temperature_K,
                     mask=mask,
                     trajectory=trajectory,
                     logfile=logfile,
                     loginterval=loginterval,
                     append_trajectory=append_trajectory)


        # self.max_collected_snapshot_num = max_collected_snapshot_num

    def run(self, steps):
        """Perform a number of time steps."""
        if not self.initialized:
            self.initialize()
        else:
            if self.have_the_atoms_been_changed():
                raise NotImplementedError(
                    "You have modified the atoms since the last timestep.")

        for i in range(steps):
            self.step()
            self.nsteps += 1
            self.call_observers()

            # if self.atoms.calc.collected_snapshot_num > self.max_collected_snapshot_num:
            #     return i
        else:
            return i
