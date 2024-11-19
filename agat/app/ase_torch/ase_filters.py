"""Filters"""
from itertools import product
from warnings import warn

import numpy as np

from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from ase.utils import deprecated, lazyproperty
# from ase.utils.abc import Optimizable
from acatal.ase_torch.ase_abc import Optimizable

__all__ = [
    'Filter', 'StrainFilter', 'UnitCellFilter', 'FrechetCellFilter',
    'ExpCellFilter'
]


class OptimizableFilter(Optimizable):
    def __init__(self, filterobj):
        self.filterobj = filterobj

    def get_positions(self):
        return self.filterobj.get_positions()

    def set_positions(self, positions):
        self.filterobj.set_positions(positions)

    def get_forces(self):
        return self.filterobj.get_forces()

    @lazyproperty
    def _use_force_consistent_energy(self):
        # This boolean is in principle invalidated if the
        # calculator changes.  This can lead to weird things
        # in multi-step optimizations.
        try:
            self.filterobj.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            return False
        else:
            return True

    def get_potential_energy(self):
        force_consistent = self._use_force_consistent_energy
        return self.filterobj.get_potential_energy(
            force_consistent=force_consistent)

    def __len__(self):
        return len(self.filterobj)

    def iterimages(self):
        return self.filterobj.iterimages()


class Filter:
    """Subset filter class."""

    def __init__(self, atoms, indices=None, mask=None):
        """Filter atoms.

        This filter can be used to hide degrees of freedom in an Atoms
        object.

        Parameters
        ----------
        indices : list of int
           Indices for those atoms that should remain visible.
        mask : list of bool
           One boolean per atom indicating if the atom should remain
           visible or not.

        If a Trajectory tries to save this object, it will instead
        save the underlying Atoms object.  To prevent this, override
        the iterimages method.
        """

        self.atoms = atoms
        self.constraints = []
        # Make self.info a reference to the underlying atoms' info dictionary.
        self.info = self.atoms.info

        if indices is None and mask is None:
            raise ValueError('Use "indices" or "mask".')
        if indices is not None and mask is not None:
            raise ValueError('Use only one of "indices" and "mask".')

        if mask is not None:
            self.index = np.asarray(mask, bool)
            self.n = self.index.sum()
        else:
            self.index = np.asarray(indices, int)
            self.n = len(self.index)

    def iterimages(self):
        # Present the real atoms object to Trajectory and friends
        return self.atoms.iterimages()

    def get_cell(self):
        """Returns the computational cell.

        The computational cell is the same as for the original system.
        """
        return self.atoms.get_cell()

    def get_pbc(self):
        """Returns the periodic boundary conditions.

        The boundary conditions are the same as for the original system.
        """
        return self.atoms.get_pbc()

    def get_positions(self):
        'Return the positions of the visible atoms.'
        return self.atoms.get_positions()[self.index]

    def set_positions(self, positions, **kwargs):
        'Set the positions of the visible atoms.'
        pos = self.atoms.get_positions()
        pos[self.index] = positions
        self.atoms.set_positions(pos, **kwargs)

    positions = property(get_positions, set_positions,
                         doc='Positions of the atoms')

    def get_momenta(self):
        'Return the momenta of the visible atoms.'
        return self.atoms.get_momenta()[self.index]

    def set_momenta(self, momenta, **kwargs):
        'Set the momenta of the visible atoms.'
        mom = self.atoms.get_momenta()
        mom[self.index] = momenta
        self.atoms.set_momenta(mom, **kwargs)

    def get_atomic_numbers(self):
        'Return the atomic numbers of the visible atoms.'
        return self.atoms.get_atomic_numbers()[self.index]

    def set_atomic_numbers(self, atomic_numbers):
        'Set the atomic numbers of the visible atoms.'
        z = self.atoms.get_atomic_numbers()
        z[self.index] = atomic_numbers
        self.atoms.set_atomic_numbers(z)

    def get_tags(self):
        'Return the tags of the visible atoms.'
        return self.atoms.get_tags()[self.index]

    def set_tags(self, tags):
        'Set the tags of the visible atoms.'
        tg = self.atoms.get_tags()
        tg[self.index] = tags
        self.atoms.set_tags(tg)

    def get_forces(self, *args, **kwargs):
        return self.atoms.get_forces(*args, **kwargs)[self.index]

    def get_stress(self, *args, **kwargs):
        return self.atoms.get_stress(*args, **kwargs)

    def get_stresses(self, *args, **kwargs):
        return self.atoms.get_stresses(*args, **kwargs)[self.index]

    def get_masses(self):
        return self.atoms.get_masses()[self.index]

    def get_potential_energy(self, **kwargs):
        """Calculate potential energy.

        Returns the potential energy of the full system.
        """
        return self.atoms.get_potential_energy(**kwargs)

    def get_chemical_symbols(self):
        return self.atoms.get_chemical_symbols()

    def get_initial_magnetic_moments(self):
        return self.atoms.get_initial_magnetic_moments()

    def get_calculator(self):
        """Returns the calculator.

        WARNING: The calculator is unaware of this filter, and sees a
        different number of atoms.
        """
        return self.atoms.calc

    @property
    def calc(self):
        return self.atoms.calc

    def get_celldisp(self):
        return self.atoms.get_celldisp()

    def has(self, name):
        'Check for existence of array.'
        return self.atoms.has(name)

    def __len__(self):
        'Return the number of movable atoms.'
        return self.n

    def __getitem__(self, i):
        'Return an atom.'
        return self.atoms[self.index[i]]

    def __ase_optimizable__(self):
        return OptimizableFilter(self)


class StrainFilter(Filter):
    """Modify the supercell while keeping the scaled positions fixed.

    Presents the strain of the supercell as the generalized positions,
    and the global stress tensor (times the volume) as the generalized
    force.

    This filter can be used to relax the unit cell until the stress is
    zero.  If MDMin is used for this, the timestep (dt) to be used
    depends on the system size. 0.01/x where x is a typical dimension
    seems like a good choice.

    The stress and strain are presented as 6-vectors, the order of the
    components follow the standard engingeering practice: xx, yy, zz,
    yz, xz, xy.

    """

    def __init__(self, atoms, mask=None, include_ideal_gas=False):
        """Create a filter applying a homogeneous strain to a list of atoms.

        The first argument, atoms, is the atoms object.

        The optional second argument, mask, is a list of six booleans,
        indicating which of the six independent components of the
        strain that are allowed to become non-zero.  It defaults to
        [1,1,1,1,1,1].

        """

        self.strain = np.zeros(6)
        self.include_ideal_gas = include_ideal_gas

        if mask is None:
            mask = np.ones(6)
        else:
            mask = np.array(mask)

        Filter.__init__(self, atoms=atoms, mask=mask)
        self.mask = mask
        self.origcell = atoms.get_cell()

    def get_positions(self):
        return self.strain.reshape((2, 3)).copy()

    def set_positions(self, new):
        new = new.ravel() * self.mask
        eps = np.array([[1.0 + new[0], 0.5 * new[5], 0.5 * new[4]],
                        [0.5 * new[5], 1.0 + new[1], 0.5 * new[3]],
                        [0.5 * new[4], 0.5 * new[3], 1.0 + new[2]]])

        self.atoms.set_cell(np.dot(self.origcell, eps), scale_atoms=True)
        self.strain[:] = new

    def get_forces(self, **kwargs):
        stress = self.atoms.get_stress(include_ideal_gas=self.include_ideal_gas)
        return -self.atoms.get_volume() * (stress * self.mask).reshape((2, 3))

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return 2


class UnitCellFilter(Filter):
    """Modify the supercell and the atom positions. """

    def __init__(self, atoms, mask=None,
                 cell_factor=None,
                 hydrostatic_strain=False,
                 constant_volume=False,
                 orig_cell=None,
                 scalar_pressure=0.0):
        """Create a filter that returns the atomic forces and unit cell
        stresses together, so they can simultaneously be minimized.

        The first argument, atoms, is the atoms object. The optional second
        argument, mask, is a list of booleans, indicating which of the six
        independent components of the strain are relaxed.

        - True = relax to zero
        - False = fixed, ignore this component

        Degrees of freedom are the positions in the original undeformed cell,
        plus the deformation tensor (extra 3 "atoms"). This gives forces
        consistent with numerical derivatives of the potential energy
        with respect to the cell degreees of freedom.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        You can still use constraints on the atoms, e.g. FixAtoms, to control
        the relaxation of the atoms.

        >>> # this should be equivalent to the StrainFilter
        >>> atoms = Atoms(...)
        >>> atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
        >>> ucf = UnitCellFilter(atoms)

        You should not attach this UnitCellFilter object to a
        trajectory. Instead, create a trajectory for the atoms, and
        attach it to an optimizer like this:

        >>> atoms = Atoms(...)
        >>> ucf = UnitCellFilter(atoms)
        >>> qn = QuasiNewton(ucf)
        >>> traj = Trajectory('TiO2.traj', 'w', atoms)
        >>> qn.attach(traj)
        >>> qn.run(fmax=0.05)

        Helpful conversion table:

        - 0.05 eV/A^3   = 8 GPA
        - 0.003 eV/A^3  = 0.48 GPa
        - 0.0006 eV/A^3 = 0.096 GPa
        - 0.0003 eV/A^3 = 0.048 GPa
        - 0.0001 eV/A^3 = 0.02 GPa

        Additional optional arguments:

        cell_factor: float (default float(len(atoms)))
            Factor by which deformation gradient is multiplied to put
            it on the same scale as the positions when assembling
            the combined position/cell vector. The stress contribution to
            the forces is scaled down by the same factor. This can be thought
            of as a very simple preconditioners. Default is number of atoms
            which gives approximately the correct scaling.

        hydrostatic_strain: bool (default False)
            Constrain the cell by only allowing hydrostatic deformation.
            The virial tensor is replaced by np.diag([np.trace(virial)]*3).

        constant_volume: bool (default False)
            Project out the diagonal elements of the virial tensor to allow
            relaxations at constant volume, e.g. for mapping out an
            energy-volume curve. Note: this only approximately conserves
            the volume and breaks energy/force consistency so can only be
            used with optimizers that do require do a line minimisation
            (e.g. FIRE).

        scalar_pressure: float (default 0.0)
            Applied pressure to use for enthalpy pV term. As above, this
            breaks energy/force consistency.
        """

        Filter.__init__(self, atoms=atoms, indices=range(len(atoms)))
        self.atoms = atoms
        if orig_cell is None:
            self.orig_cell = atoms.get_cell()
        else:
            self.orig_cell = orig_cell
        self.stress = None

        if mask is None:
            mask = np.ones(6)
        mask = np.asarray(mask)
        if mask.shape == (6,):
            self.mask = voigt_6_to_full_3x3_stress(mask)
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError('shape of mask should be (3,3) or (6,)')

        if cell_factor is None:
            cell_factor = float(len(atoms))
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.scalar_pressure = scalar_pressure
        self.cell_factor = cell_factor
        self.copy = self.atoms.copy
        self.arrays = self.atoms.arrays

    def deform_grad(self):
        return np.linalg.solve(self.orig_cell, self.atoms.cell).T

    def get_positions(self):
        """
        this returns an array with shape (natoms + 3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor associated with the unit cell,
        scaled by self.cell_factor.
        """

        cur_deform_grad = self.deform_grad()
        natoms = len(self.atoms)
        pos = np.zeros((natoms + 3, 3))
        # UnitCellFilter's positions are the self.atoms.positions but without
        # the applied deformation gradient
        pos[:natoms] = np.linalg.solve(cur_deform_grad,
                                       self.atoms.positions.T).T
        # UnitCellFilter's cell DOFs are the deformation gradient times a
        # scaling factor
        pos[natoms:] = self.cell_factor * cur_deform_grad
        return pos

    def set_positions(self, new, **kwargs):
        """
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the new cell is first set from original cell transformed by the new
        deformation gradient, then the positions are set with respect to the
        current cell by transforming them with the same deformation gradient
        """

        natoms = len(self.atoms)
        new_atom_positions = new[:natoms]
        new_deform_grad = new[natoms:] / self.cell_factor
        # Set the new cell from the original cell and the new
        # deformation gradient.  Both current and final structures should
        # preserve symmetry, so if set_cell() calls FixSymmetry.adjust_cell(),
        # it should be OK
        self.atoms.set_cell(self.orig_cell @ new_deform_grad.T,
                            scale_atoms=True)
        # Set the positions from the ones passed in (which are without the
        # deformation gradient applied) and the new deformation gradient.
        # This should also preserve symmetry, so if set_positions() calls
        # FixSymmetyr.adjust_positions(), it should be OK
        self.atoms.set_positions(new_atom_positions @ new_deform_grad.T,
                                 **kwargs)

    def get_potential_energy(self, force_consistent=True):
        """
        returns potential energy including enthalpy PV term.
        """
        atoms_energy = self.atoms.get_potential_energy(
            force_consistent=force_consistent)
        return atoms_energy + self.scalar_pressure * self.atoms.get_volume()

    def get_forces(self, **kwargs):
        """
        returns an array with shape (natoms+3,3) of the atomic forces
        and unit cell stresses.

        the first natoms rows are the forces on the atoms, the last
        three rows are the forces on the unit cell, which are
        computed from the stress tensor.
        """

        stress = self.atoms.get_stress(**kwargs)
        atoms_forces = self.atoms.get_forces(**kwargs)

        volume = self.atoms.get_volume()
        virial = -volume * (voigt_6_to_full_3x3_stress(stress) +
                            np.diag([self.scalar_pressure] * 3))
        cur_deform_grad = self.deform_grad()
        atoms_forces = atoms_forces @ cur_deform_grad
        virial = np.linalg.solve(cur_deform_grad, virial.T).T

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        if self.constant_volume:
            vtr = virial.trace()
            np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = virial / self.cell_factor

        self.stress = -full_3x3_to_voigt_6_stress(virial) / volume
        return forces

    def get_stress(self):
        raise PropertyNotImplementedError

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return (len(self.atoms) + 3)


class FrechetCellFilter(UnitCellFilter):
    """Modify the supercell and the atom positions."""

    def __init__(self, atoms, mask=None,
                 exp_cell_factor=None,
                 hydrostatic_strain=False,
                 constant_volume=False,
                 scalar_pressure=0.0):
        r"""Create a filter that returns the atomic forces and unit cell
        stresses together, so they can simultaneously be minimized.

        The first argument, atoms, is the atoms object. The optional second
        argument, mask, is a list of booleans, indicating which of the six
        independent components of the strain are relaxed.

        - True = relax to zero
        - False = fixed, ignore this component

        Degrees of freedom are the positions in the original undeformed cell,
        plus the log of the deformation tensor (extra 3 "atoms"). This gives
        forces consistent with numerical derivatives of the potential energy
        with respect to the cell degrees of freedom.

        You can still use constraints on the atoms, e.g. FixAtoms, to control
        the relaxation of the atoms.

        >>> # this should be equivalent to the StrainFilter
        >>> atoms = Atoms(...)
        >>> atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
        >>> ecf = FrechetCellFilter(atoms)

        You should not attach this FrechetCellFilter object to a
        trajectory. Instead, create a trajectory for the atoms, and
        attach it to an optimizer like this:

        >>> atoms = Atoms(...)
        >>> ecf = FrechetCellFilter(atoms)
        >>> qn = QuasiNewton(ecf)
        >>> traj = Trajectory('TiO2.traj', 'w', atoms)
        >>> qn.attach(traj)
        >>> qn.run(fmax=0.05)

        Helpful conversion table:

        - 0.05 eV/A^3   = 8 GPA
        - 0.003 eV/A^3  = 0.48 GPa
        - 0.0006 eV/A^3 = 0.096 GPa
        - 0.0003 eV/A^3 = 0.048 GPa
        - 0.0001 eV/A^3 = 0.02 GPa

        Additional optional arguments:

        exp_cell_factor: float (default float(len(atoms)))
            Scaling factor for cell variables. The cell gradients in
            FrechetCellFilter.get_forces() is divided by exp_cell_factor.
            By default, set the number of atoms. We recommend to set
            an extensive value for this parameter.

        hydrostatic_strain: bool (default False)
            Constrain the cell by only allowing hydrostatic deformation.
            The virial tensor is replaced by np.diag([np.trace(virial)]*3).

        constant_volume: bool (default False)
            Project out the diagonal elements of the virial tensor to allow
            relaxations at constant volume, e.g. for mapping out an
            energy-volume curve.

        scalar_pressure: float (default 0.0)
            Applied pressure to use for enthalpy pV term. As above, this
            breaks energy/force consistency.

        Implementation note:

        The implementation is based on that of Christoph Ortner in JuLIP.jl:
        https://github.com/JuliaMolSim/JuLIP.jl/blob/master/src/expcell.jl

        The initial implementation of ExpCellFilter gave inconsistent gradients
        for cell variables (matrix log of the deformation tensor). If you would
        like to keep the previous behavior, please use ExpCellFilter.

        The derivation of gradients of energy w.r.t positions and the log of the
        deformation tensor is given in
        https://github.com/lan496/lan496.github.io/blob/main/notes/cell_grad.pdf
        """

        Filter.__init__(self, atoms=atoms, indices=range(len(atoms)))
        UnitCellFilter.__init__(self, atoms=atoms, mask=mask,
                                hydrostatic_strain=hydrostatic_strain,
                                constant_volume=constant_volume,
                                scalar_pressure=scalar_pressure)

        # We defer the scipy import to avoid high immediate import overhead
        from scipy.linalg import expm, expm_frechet, logm
        self.expm = expm
        self.logm = logm
        self.expm_frechet = expm_frechet

        # Scaling factor for cell gradients
        if exp_cell_factor is None:
            exp_cell_factor = float(len(atoms))
        self.exp_cell_factor = exp_cell_factor

    def get_positions(self):
        pos = UnitCellFilter.get_positions(self)
        natoms = len(self.atoms)
        pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
        return pos

    def set_positions(self, new, **kwargs):
        natoms = len(self.atoms)
        new2 = new.copy()
        new2[natoms:] = self.expm(new[natoms:] / self.exp_cell_factor)
        UnitCellFilter.set_positions(self, new2, **kwargs)

    def get_forces(self, **kwargs):
        # forces on atoms are same as UnitCellFilter, we just
        # need to modify the stress contribution
        stress = self.atoms.get_stress(**kwargs)
        volume = self.atoms.get_volume()
        virial = -volume * (voigt_6_to_full_3x3_stress(stress) +
                            np.diag([self.scalar_pressure] * 3))

        cur_deform_grad = self.deform_grad()
        cur_deform_grad_log = self.logm(cur_deform_grad)

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        # Cell gradient for UnitCellFilter
        ucf_cell_grad = virial @ np.linalg.inv(cur_deform_grad.T)

        # Cell gradient for FrechetCellFilter
        deform_grad_log_force = np.zeros((3, 3))
        for mu, nu in product(range(3), repeat=2):
            dir = np.zeros((3, 3))
            dir[mu, nu] = 1.0
            # Directional derivative of deformation to (mu, nu) strain direction
            expm_der = self.expm_frechet(
                cur_deform_grad_log,
                dir,
                compute_expm=False
            )
            deform_grad_log_force[mu, nu] = np.sum(expm_der * ucf_cell_grad)

        # Cauchy stress used for convergence testing
        convergence_crit_stress = -(virial / volume)
        if self.constant_volume:
            # apply constraint to force
            dglf_trace = deform_grad_log_force.trace()
            np.fill_diagonal(deform_grad_log_force,
                             np.diag(deform_grad_log_force) - dglf_trace / 3.0)
            # apply constraint to Cauchy stress used for convergence testing
            ccs_trace = convergence_crit_stress.trace()
            np.fill_diagonal(convergence_crit_stress,
                             np.diag(convergence_crit_stress) - ccs_trace / 3.0)

        atoms_forces = self.atoms.get_forces(**kwargs)
        atoms_forces = atoms_forces @ cur_deform_grad

        # pack gradients into vector
        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = deform_grad_log_force / self.exp_cell_factor
        self.stress = full_3x3_to_voigt_6_stress(convergence_crit_stress)
        return forces


class ExpCellFilter(UnitCellFilter):

    @deprecated(DeprecationWarning(
        'Use FrechetCellFilter for better convergence w.r.t. cell variables.'
    ))
    def __init__(self, atoms, mask=None,
                 cell_factor=None,
                 hydrostatic_strain=False,
                 constant_volume=False,
                 scalar_pressure=0.0):
        r"""Create a filter that returns the atomic forces and unit cell
            stresses together, so they can simultaneously be minimized.

            The first argument, atoms, is the atoms object. The optional second
            argument, mask, is a list of booleans, indicating which of the six
            independent components of the strain are relaxed.

            - True = relax to zero
            - False = fixed, ignore this component

            Degrees of freedom are the positions in the original undeformed
            cell, plus the log of the deformation tensor (extra 3 "atoms"). This
            gives forces consistent with numerical derivatives of the potential
            energy with respect to the cell degrees of freedom.

            For full details see:
                E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
                Phys. Rev. B 59, 235 (1999)

            You can still use constraints on the atoms, e.g. FixAtoms, to
            control the relaxation of the atoms.

            >>> # this should be equivalent to the StrainFilter
            >>> atoms = Atoms(...)
            >>> atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
            >>> ecf = ExpCellFilter(atoms)

            You should not attach this ExpCellFilter object to a
            trajectory. Instead, create a trajectory for the atoms, and
            attach it to an optimizer like this:

            >>> atoms = Atoms(...)
            >>> ecf = ExpCellFilter(atoms)
            >>> qn = QuasiNewton(ecf)
            >>> traj = Trajectory('TiO2.traj', 'w', atoms)
            >>> qn.attach(traj)
            >>> qn.run(fmax=0.05)

            Helpful conversion table:

            - 0.05 eV/A^3   = 8 GPA
            - 0.003 eV/A^3  = 0.48 GPa
            - 0.0006 eV/A^3 = 0.096 GPa
            - 0.0003 eV/A^3 = 0.048 GPa
            - 0.0001 eV/A^3 = 0.02 GPa

            Additional optional arguments:

            cell_factor: (DEPRECATED)
                Retained for backwards compatibility, but no longer used.

            hydrostatic_strain: bool (default False)
                Constrain the cell by only allowing hydrostatic deformation.
                The virial tensor is replaced by np.diag([np.trace(virial)]*3).

            constant_volume: bool (default False)
                Project out the diagonal elements of the virial tensor to allow
                relaxations at constant volume, e.g. for mapping out an
                energy-volume curve.

            scalar_pressure: float (default 0.0)
                Applied pressure to use for enthalpy pV term. As above, this
                breaks energy/force consistency.

        Implementation details:

        The implementation is based on that of Christoph Ortner in JuLIP.jl:
        https://github.com/libAtoms/JuLIP.jl/blob/expcell/src/Constraints.jl#L244

        We decompose the deformation gradient as

            F = exp(U) F0
            x =  F * F0^{-1} z  = exp(U) z

        If we write the energy as a function of U we can transform the
        stress associated with a perturbation V into a derivative using a
        linear map V -> L(U, V).

        \phi( exp(U+tV) (z+tv) ) ~ \phi'(x) . (exp(U) v) + \phi'(x) .
                                                    ( L(U, V) exp(-U) exp(U) z )

        where

                \nabla E(U) : V  =  [S exp(-U)'] : L(U,V)
                                =  L'(U, S exp(-U)') : V
                                =  L(U', S exp(-U)') : V
                                =  L(U, S exp(-U)) : V     (provided U = U')

        where the : operator represents double contraction,
        i.e. A:B = trace(A'B), and

            F = deformation tensor - 3x3 matrix
            F0 = reference deformation tensor - 3x3 matrix, np.eye(3) here
            U = cell degrees of freedom used here - 3x3 matrix
            V = perturbation to cell DoFs - 3x3 matrix
            v = perturbation to position DoFs
            x = atomic positions in deformed cell
            z = atomic positions in original cell
            \phi = potential energy
            S = stress tensor [3x3 matrix]
            L(U, V) = directional derivative of exp at U in direction V, i.e
            d/dt exp(U + t V)|_{t=0} = L(U, V)

        This means we can write

            d/dt E(U + t V)|_{t=0} = L(U, S exp (-U)) : V

        and therefore the contribution to the gradient of the energy is

            \nabla E(U) / \nabla U_ij =  [L(U, S exp(-U))]_ij

        .. deprecated:: 3.23.0
            Use :class:`~ase.filters.FrechetCellFilter` for better convergence
            w.r.t. cell variables.
        """
        Filter.__init__(self, atoms=atoms, indices=range(len(atoms)))
        UnitCellFilter.__init__(self, atoms=atoms, mask=mask,
                                cell_factor=cell_factor,
                                hydrostatic_strain=hydrostatic_strain,
                                constant_volume=constant_volume,
                                scalar_pressure=scalar_pressure)
        if cell_factor is not None:
            # cell_factor used in UnitCellFilter does not affect on gradients of
            # ExpCellFilter.
            warn("cell_factor is deprecated")
        self.cell_factor = 1.0

        # We defer the scipy import to avoid high immediate import overhead
        from scipy.linalg import expm, logm
        self.expm = expm
        self.logm = logm

    def get_forces(self, **kwargs):
        forces = UnitCellFilter.get_forces(self, **kwargs)

        # forces on atoms are same as UnitCellFilter, we just
        # need to modify the stress contribution
        stress = self.atoms.get_stress(**kwargs)
        volume = self.atoms.get_volume()
        virial = -volume * (voigt_6_to_full_3x3_stress(stress) +
                            np.diag([self.scalar_pressure] * 3))

        cur_deform_grad = self.deform_grad()
        cur_deform_grad_log = self.logm(cur_deform_grad)

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        deform_grad_log_force_naive = virial.copy()
        Y = np.zeros((6, 6))
        Y[0:3, 0:3] = cur_deform_grad_log
        Y[3:6, 3:6] = cur_deform_grad_log
        Y[0:3, 3:6] = - virial @ self.expm(-cur_deform_grad_log)
        deform_grad_log_force = -self.expm(Y)[0:3, 3:6]
        for (i1, i2) in [(0, 1), (0, 2), (1, 2)]:
            ff = 0.5 * (deform_grad_log_force[i1, i2] +
                        deform_grad_log_force[i2, i1])
            deform_grad_log_force[i1, i2] = ff
            deform_grad_log_force[i2, i1] = ff

        # check for reasonable alignment between naive and
        # exact search directions
        all_are_equal = np.all(np.isclose(deform_grad_log_force,
                                          deform_grad_log_force_naive))
        if all_are_equal or \
            (np.sum(deform_grad_log_force * deform_grad_log_force_naive) /
             np.sqrt(np.sum(deform_grad_log_force**2) *
                     np.sum(deform_grad_log_force_naive**2)) > 0.8):
            deform_grad_log_force = deform_grad_log_force_naive

        # Cauchy stress used for convergence testing
        convergence_crit_stress = -(virial / volume)
        if self.constant_volume:
            # apply constraint to force
            dglf_trace = deform_grad_log_force.trace()
            np.fill_diagonal(deform_grad_log_force,
                             np.diag(deform_grad_log_force) - dglf_trace / 3.0)
            # apply constraint to Cauchy stress used for convergence testing
            ccs_trace = convergence_crit_stress.trace()
            np.fill_diagonal(convergence_crit_stress,
                             np.diag(convergence_crit_stress) - ccs_trace / 3.0)

        # pack gradients into vector
        natoms = len(self.atoms)
        forces[natoms:] = deform_grad_log_force
        self.stress = full_3x3_to_voigt_6_stress(convergence_crit_stress)
        return forces
