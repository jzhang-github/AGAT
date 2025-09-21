##########
ase_abc
##########

.. class:: Optimizable(collections.abc.Sized)

   Modified object to fit the Torch tensor calculation.

   .. method:: converged(self, forces, fmax)

      .. Note:: Replace ``return np.linalg.norm(forces, axis=1).max() < fmax`` with ``return torch.linalg.norm(forces, dim=1).max() < fmax``
