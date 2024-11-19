# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:22:46 2023

@author: ZHANG Jun
"""

import sys
# import pickle
# import time
# from math import sqrt
# from os.path import isfile

# import numpy as np

# from ase.parallel import rank, barrier
# import ase.parallel.world.rank as rank
# from ase.parallel import world
import ase
from ase.io.trajectory import PickleTrajectory

import torch

class Dynamics:
    """Base-class for all MD and structure optimization classes.

    Dynamics(atoms, logfile)

    atoms: Atoms object
        The Atoms object to operate on
    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.
    trajectory: Trajectory object or str
        Attach trajectory object.  If *trajectory* is a string a
        PickleTrajectory will be constructed.  Use *None* for no
        trajectory.
    """
    def __init__(self, atoms, logfile, trajectory):
        self.atoms = atoms

        if ase.parallel.world.rank != 0:
            logfile = None
        elif isinstance(logfile, str):
            if logfile == '-':
                logfile = sys.stdout
            else:
                logfile = open(logfile, 'a')
        self.logfile = logfile

        self.observers = []
        self.nsteps = 0

        if trajectory is not None:
            if isinstance(trajectory, str):
                trajectory = PickleTrajectory(trajectory, 'w', atoms)
            self.attach(trajectory)

    def get_number_of_steps(self):
        return self.nsteps

    def insert_observer(self, function, position=0, interval=1, 
                        *args, **kwargs):
        """Insert an observer."""
        if not callable(function):
            function = function.write
        self.observers.insert(position, (function, interval, args, kwargs))

    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function.

        At every *interval* steps, call *function* with arguments
        *args* and keyword arguments *kwargs*."""

        if not hasattr(function, '__call__'):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def call_observers(self):
        for function, interval, args, kwargs in self.observers:
            if self.nsteps % interval == 0:
                function(*args, **kwargs)

# class Optimizer(Dynamics):
#     """Base-class for all structure optimization classes."""
#     def __init__(self, atoms, restart, logfile, trajectory):
#         """Structure optimizer object.

#         atoms: Atoms object
#             The Atoms object to relax.
#         restart: str
#             Filename for restart file.  Default value is *None*.
#         logfile: file object or str
#             If *logfile* is a string, a file with that name will be opened.
#             Use '-' for stdout.
#         trajectory: Trajectory object or str
#             Attach trajectory object.  If *trajectory* is a string a
#             PickleTrajectory will be constructed.  Use *None* for no
#             trajectory.
#         """
#         Dynamics.__init__(self, atoms, logfile, trajectory)
#         self.restart = restart

#         if restart is None or not isfile(restart):
#             self.initialize()
#         else:
#             self.read()
#             ase.parallel.world.barrier()
#     def initialize(self):
#         pass

#     def run(self, fmax=0.05, steps=100000000):
#         """Run structure optimization algorithm.

#         This method will return when the forces on all individual
#         atoms are less than *fmax* or when the number of steps exceeds
#         *steps*."""

#         self.fmax = fmax
#         step = 0
#         while step < steps:
#             f = self.atoms.get_forces()
#             self.log(f)
#             self.call_observers()
#             if self.converged(f):
#                 return
#             self.step(f)
#             self.nsteps += 1
#             step += 1

#     def converged(self, forces=None):
#         """Did the optimization converge?"""
#         if forces is None:
#             forces = self.atoms.get_forces()
#         if hasattr(self.atoms, 'get_curvature'):
#             return (forces**2).sum(axis=1).max() < self.fmax**2 and \
#                    self.atoms.get_curvature() < 0.0
#         return (forces**2).sum(axis=1).max() < self.fmax**2

#     def log(self, forces):
#         fmax = sqrt((forces**2).sum(axis=1).max())
#         e = self.atoms.get_potential_energy()
#         T = time.localtime()
#         if self.logfile is not None:
#             name = self.__class__.__name__
#             self.logfile.write('%s: %3d  %02d:%02d:%02d %15.6f %12.4f\n' %
#                                (name, self.nsteps, T[3], T[4], T[5], e, fmax))
#             self.logfile.flush()
        
#     def dump(self, data):
#         if world.rank == 0 and self.restart is not None:
#             pickle.dump(data, open(self.restart, 'wb'), protocol=2)

#     def load(self):
#         return pickle.load(open(self.restart))
    
from acatal.ase_torch.optimize_torch import Optimizer
from typing import IO, Optional, Union

# import numpy as np

from ase import Atoms
# from ase.optimize.optimize import Optimizer


class MDMinTorch(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'dt': 0.2}

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        dt: Optional[float] = None,
        maxstep: Optional[float] = None,
        master: Optional[bool] = None,
        device = torch.device('cuda'),
    ):
        """Parameters:

        atoms: Atoms object
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
        """
        self.device = torch.device(device)
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master,
                           device=self.device)

        self.dt = dt or self.defaults['dt']
        self.maxstep = maxstep or self.defaults['maxstep']
        

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, forces=None):
        optimizable = self.optimizable

        if forces is None:
            forces = optimizable.get_forces()

        if self.v is None:
            # self.v = np.zeros((len(optimizable), 3))
            self.v = torch.zeros((len(optimizable), 3), device=self.device)
        else:
            self.v += 0.5 * self.dt * forces
            # Correct velocities:
            # vf = np.vdot(self.v, forces)
            vf = torch.vdot(self.v.flatten(), forces.flatten())
            if vf < 0.0:
                self.v[:] = 0.0
            else:
                # self.v[:] = forces * vf / np.vdot(forces, forces)
                self.v[:] = forces * vf / torch.vdot(forces.flatten(),
                                                     forces.flatten())

        self.v += 0.5 * self.dt * forces
        # pos = torch.tensor(optimizable.get_positions(),
        #                    dtype=torch.float32,
        #                    device=self.device)
        pos = optimizable.get_positions()
        dpos = self.dt * self.v

        # For any dpos magnitude larger than maxstep, scaling
        # is <1. We add a small float to prevent overflows/zero-div errors.
        # All displacement vectors (rows) of dpos which have a norm larger
        # than self.maxstep are scaled to it.
        # scaling = self.maxstep / (1e-6 + np.max(np.linalg.norm(dpos, axis=1)))
        scaling = self.maxstep / (1e-6 + torch.max(torch.linalg.norm(dpos, dim=1)))
        dpos *= torch.clip(scaling, 0.0, 1.0)
        pos_new = pos + dpos
        optimizable.set_positions(pos_new)
        # self.dump((self.v, self.dt))
        self.dump((self.v.cpu().numpy(), self.dt))