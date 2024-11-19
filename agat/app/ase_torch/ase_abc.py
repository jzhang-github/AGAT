import collections
from abc import abstractmethod

# import numpy as np
import torch

# Due to the high prevalence of cyclic imports surrounding ase.optimize,
# we define the Optimizable ABC here in utils.
# Can we find a better way?


class Optimizable(collections.abc.Sized):
    @abstractmethod
    def get_positions(self):
        ...

    @abstractmethod
    def set_positions(self, positions):
        ...

    @abstractmethod
    def get_forces(self):
        ...

    @abstractmethod
    def get_potential_energy(self):
        ...

    @abstractmethod
    def iterimages(self):
        ...

    def converged(self, forces, fmax):
        # return np.linalg.norm(forces, axis=1).max() < fmax
        return torch.linalg.norm(forces, dim=1).max() < fmax

    def is_neb(self):
        return False

    def __ase_optimizable__(self):
        return self
