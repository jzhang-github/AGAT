"""Structure optimization. """
import time
from collections.abc import Callable
# from math import sqrt
from os.path import isfile
from typing import IO, Any, Dict, List, Optional, Tuple, Union
import warnings

from ase import Atoms
# from ase.filters import UnitCellFilter
from acatal.ase_torch.ase_filters import UnitCellFilter
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import barrier, world
from ase.utils import IOContext, lazyproperty
# from ase.utils.abc import Optimizable
from acatal.ase_torch.ase_abc import Optimizable

import torch

DEFAULT_MAX_STEPS = 100_000_000


class RestartError(RuntimeError):
    pass


class OptimizableAtoms(Optimizable):
    def __init__(self, atoms, device=torch.device('cuda')):
        self.atoms = atoms
        self.device = torch.device(device)

    def get_positions(self):
        # return self.atoms.get_positions()
        return torch.tensor(self.atoms.get_positions(),
                            dtype=torch.float32,
                            device=self.device)

    def set_positions(self, positions):
        # self.atoms.set_positions(positions)
        self.atoms.set_positions(positions.cpu().numpy())

    def get_forces(self):
        return self.atoms.get_forces()

    @lazyproperty
    def _use_force_consistent_energy(self):
        # This boolean is in principle invalidated if the
        # calculator changes.  This can lead to weird things
        # in multi-step optimizations.
        try:
            self.atoms.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            # warnings.warn(
            #     'Could not get force consistent energy (\'free_energy\').  '
            #     'Please make sure calculator provides \'free_energy\', even '
            #     'if equal to the ordinary energy.  '
            #     'This will raise an error in future versions of ASE.',
            #     FutureWarning)
            return False
        else:
            return True

    def get_potential_energy(self):
        force_consistent = self._use_force_consistent_energy
        return self.atoms.get_potential_energy(
            force_consistent=force_consistent)

    def iterimages(self):
        # XXX document purpose of iterimages
        return self.atoms.iterimages()

    def __len__(self):
        # TODO: return 3 * len(self.atoms), because we want the length
        # of this to be the number of DOFs
        return len(self.atoms)


class Dynamics(IOContext):
    """Base-class for all MD and structure optimization classes."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: Optional[Union[IO, str]] = None,
        trajectory: Optional[str] = None,
        append_trajectory: bool = False,
        master: Optional[bool] = None,
        device=torch.device('cuda'),
    ):
        """Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        append_trajectory: boolean
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
        """
        self.device = torch.device(device)
        self.atoms = atoms
        # self.optimizable = atoms.__ase_optimizable__()
        self.optimizable = OptimizableAtoms(atoms, device=self.device)
        self.logfile = self.openfile(logfile, mode='a', comm=world)
        self.observers: List[Tuple[Callable, int, Tuple, Dict[str, Any]]] = []
        self.nsteps = 0
        self.max_steps = 0  # to be updated in run or irun

        if trajectory is not None:
            if isinstance(trajectory, str):
                from ase.io.trajectory import Trajectory
                mode = "a" if append_trajectory else "w"
                trajectory = self.closelater(Trajectory(
                    trajectory, mode=mode, master=master
                ))
            self.attach(trajectory, atoms=self.optimizable)

        self.trajectory = trajectory

    def todict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_number_of_steps(self):
        return self.nsteps

    def insert_observer(
        self, function, position=0, interval=1, *args, **kwargs
    ):
        """Insert an observer.

        This can be used for pre-processing before logging and dumping.

        Examples
        --------
        >>> from ase.build import bulk
        >>> from ase.calculators.emt import EMT
        >>> from ase.optimize import BFGS
        ...
        ...
        >>> def update_info(atoms, opt):
        ...     atoms.info["nsteps"] = opt.nsteps
        ...
        ...
        >>> atoms = bulk("Cu", cubic=True) * 2
        >>> atoms.rattle()
        >>> atoms.calc = EMT()
        >>> with BFGS(atoms, logfile=None, trajectory="opt.traj") as opt:
        ...     opt.insert_observer(update_info, atoms=atoms, opt=opt)
        ...     opt.run(fmax=0.05, steps=10)
        True
        """
        if not isinstance(function, Callable):
            function = function.write
        self.observers.insert(position, (function, interval, args, kwargs))

    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function.

        If *interval > 0*, at every *interval* steps, call *function* with
        arguments *args* and keyword arguments *kwargs*.

        If *interval <= 0*, after step *interval*, call *function* with
        arguments *args* and keyword arguments *kwargs*.  This is
        currently zero indexed."""

        if hasattr(function, "set_description"):
            d = self.todict()
            d.update(interval=interval)
            function.set_description(d)
        if not isinstance(function, Callable):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def call_observers(self):
        for function, interval, args, kwargs in self.observers:
            call = False
            # Call every interval iterations
            if interval > 0:
                if (self.nsteps % interval) == 0:
                    call = True
            # Call only on iteration interval
            elif interval <= 0:
                if self.nsteps == abs(interval):
                    call = True
            if call:
                function(*args, **kwargs)

    def irun(self, steps=DEFAULT_MAX_STEPS):
        """Run dynamics algorithm as generator.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of dynamics steps to be run.

        Yields
        ------
        converged : bool
            True if the forces on atoms are converged.

        Examples
        --------
        This method allows, e.g., to run two optimizers or MD thermostats at
        the same time.
        >>> opt1 = BFGS(atoms)
        >>> opt2 = BFGS(StrainFilter(atoms)).irun()
        >>> for _ in opt2:
        ...     opt1.run()
        """

        # update the maximum number of steps
        self.max_steps = self.nsteps + steps

        # compute the initial step
        self.optimizable.get_forces()

        # log the initial step
        if self.nsteps == 0:
            self.log()

            # we write a trajectory file if it is None
            if self.trajectory is None:
                self.call_observers()
            # We do not write on restart w/ an existing trajectory file
            # present. This duplicates the same entry twice
            elif len(self.trajectory) == 0:
                self.call_observers()

        # check convergence
        is_converged = self.converged()
        yield is_converged

        # run the algorithm until converged or max_steps reached
        while not is_converged and self.nsteps < self.max_steps:
            # compute the next step
            self.step()
            self.nsteps += 1

            # log the step
            self.log()
            self.call_observers()

            # check convergence
            is_converged = self.converged()
            yield is_converged

    def run(self, steps=DEFAULT_MAX_STEPS):
        """Run dynamics algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of dynamics steps to be run.

        Returns
        -------
        converged : bool
            True if the forces on atoms are converged.
        """

        for converged in Dynamics.irun(self, steps=steps):
            pass
        return converged

    def converged(self):
        """" a dummy function as placeholder for a real criterion, e.g. in
        Optimizer """
        return False

    def log(self, *args):
        """ a dummy function as placeholder for a real logger, e.g. in
        Optimizer """
        return True

    def step(self):
        """this needs to be implemented by subclasses"""
        raise RuntimeError("step not implemented.")


class Optimizer(Dynamics):
    """Base-class for all structure optimization classes."""

    # default maxstep for all optimizers
    defaults = {'maxstep': 0.2}
    _deprecated = object()

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        trajectory: Optional[str] = None,
        master: Optional[bool] = None,
        append_trajectory: bool = False,
        force_consistent=_deprecated,
        device = torch.device('cuda'),
    ):
        """Structure optimizer object.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: str
            Filename for restart file.  Default value is *None*.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        append_trajectory: boolean
            Appended to the trajectory file instead of overwriting it.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  If force_consistent=None, uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        self.check_deprecated(force_consistent)

        super().__init__(
            atoms=atoms,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            master=master,
            device=self.device)

        self.restart = restart

        self.fmax = None

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            barrier()
        self.device = torch.device(device)

    @classmethod
    def check_deprecated(cls, force_consistent):
        if force_consistent is cls._deprecated:
            return False

        warnings.warn(
            'force_consistent keyword is deprecated and will '
            'be ignored.  This will raise an error in future versions '
            'of ASE.',
            FutureWarning)

    def read(self):
        raise NotImplementedError

    def todict(self):
        description = {
            "type": "optimization",
            "optimizer": self.__class__.__name__,
        }
        # add custom attributes from subclasses
        for attr in ('maxstep', 'alpha', 'max_steps', 'restart'):
            if hasattr(self, attr):
                description.update({attr: getattr(self, attr)})
        return description

    def initialize(self):
        pass

    def irun(self, fmax=0.05, steps=DEFAULT_MAX_STEPS):
        """Run optimizer as generator.

        Parameters
        ----------
        fmax : float
            Convergence criterion of the forces on atoms.
        steps : int, default=DEFAULT_MAX_STEPS
            Number of optimizer steps to be run.

        Yields
        ------
        converged : bool
            True if the forces on atoms are converged.
        """
        self.fmax = fmax
        return Dynamics.irun(self, steps=steps)

    def run(self, fmax=0.05, steps=DEFAULT_MAX_STEPS):
        """Run optimizer.

        Parameters
        ----------
        fmax : float
            Convergence criterion of the forces on atoms.
        steps : int, default=DEFAULT_MAX_STEPS
            Number of optimizer steps to be run.

        Returns
        -------
        converged : bool
            True if the forces on atoms are converged.
        """
        self.fmax = fmax
        return Dynamics.run(self, steps=steps)

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.optimizable.get_forces()
        return self.optimizable.converged(forces, self.fmax)

    def log(self, forces=None):
        if forces is None:
            forces = self.optimizable.get_forces()
        # fmax = sqrt((forces ** 2).sum(axis=1).max())
        # print(forces)
        fmax = torch.max(torch.norm(forces, dim=1))
        e = self.optimizable.get_potential_energy()
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Time", "Energy", "fmax")
                msg = "%s  %4s %8s %15s  %12s\n" % args
                self.logfile.write(msg)

            args = (name, self.nsteps, T[3], T[4], T[5], e, fmax)
            msg = "%s:  %3d %02d:%02d:%02d %15.6f %15.6f\n" % args
            self.logfile.write(msg)
            self.logfile.flush()

    def dump(self, data):
        from ase.io.jsonio import write_json
        if world.rank == 0 and self.restart is not None:
            with open(self.restart, 'w') as fd:
                write_json(fd, data)

    def load(self):
        from ase.io.jsonio import read_json
        with open(self.restart) as fd:
            try:
                from ase.optimize import BFGS
                if not isinstance(self, BFGS) and isinstance(
                    self.atoms, UnitCellFilter
                ):
                    warnings.warn(
                        "WARNING: restart function is untested and may result "
                        "in unintended behavior. Namely orig_cell is not "
                        "loaded in the UnitCellFilter. Please test on your own"
                        " to ensure consistent results."
                    )
                return read_json(fd, always_array=False)
            except Exception as ex:
                msg = ('Could not decode restart file as JSON.  '
                       'You may need to delete the restart file '
                       f'{self.restart}')
                raise RestartError(msg) from ex
