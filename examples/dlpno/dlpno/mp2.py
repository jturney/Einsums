#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""DLPNO-MP2, ported from psi4's ``dlpno/mp2.cc``.

The physics of the local MP2 residual, and the ComputeGraph the port solves it
with, are in :mod:`dlpno.lmp2_solver`. What is left here is the DLPNO-MP2
*method*: the phase sequence, and the three additive pieces psi4 reports the
correlation energy as.

The split is not tidiness. psi4 runs the same iteration a second time inside
``DLPNOCCSD::pno_lmp2_iterations``, to converge the amplitudes its strong/weak
pair split and its CC-level PNOs are derived from, and duplicates ~800 lines of
the most engineered code in the module to do it. Here the solver takes an
explicit :class:`~dlpno.lmp2_solver.LMP2Problem`, so the CCSD layer builds a
second problem instead of a second solver.
"""

from .base import DLPNOBase
from .lmp2_solver import LMP2Problem, LMP2Solver

__all__ = ["DLPNOMP2"]


class DLPNOMP2(DLPNOBase):
    """DLPNO-MP2: PNO overlaps, then the local MP2 equations as a ComputeGraph."""

    def __init__(self, reference, thresholds=None, verbose=True, use_diis=True,
                 integral_source=None):
        super().__init__(reference, thresholds, verbose, integral_source=integral_source)
        self.use_diis = use_diis
        self.solver = None
        self.e_lmp2 = 0.0
        self.e_lmp2_os = 0.0
        self.e_lmp2_ss = 0.0
        self.n_iterations = 0

    # -- the solver, and the problem it is handed --------------------------

    def lmp2_problem(self):
        """This calculation's converged setup, as an :class:`LMP2Problem`.

        Everything the iteration reads, by value. ``pno_transform`` has to have
        run: the PNO set and the store layout are its outputs.
        """
        return LMP2Problem(
            naocc=self.ref.naocc,
            ij_to_i_j=self.ij_to_i_j,
            i_j_to_ij=self.i_j_to_ij,
            ij_to_ji=self.ij_to_ji,
            n_pno=self.n_pno,
            e_pno=self.e_pno,
            X_pno=self.X_pno,
            F_lmo=self.F_lmo,
            S_pao=self.S_pao,
            lmopair_to_paos=self.lmopair_to_paos,
            layout=self.layout,
            K_all=self.K_all,
            T_all=self.T_all,
            Tt_all=self.Tt_all,
            cut=self.cut,
        )

    def lmp2_solver(self):
        """The solver for this calculation, built once."""
        if self.solver is None:
            self.solver = LMP2Solver(self.lmp2_problem(), use_diis=self.use_diis,
                                     verbose=self.verbose)
        return self.solver

    # The solver's outputs, where the tools and the drivers have always read
    # them. Properties rather than copies so nothing can go stale.

    @property
    def plan(self):
        return None if self.solver is None else self.solver.plan

    @property
    def _S_cls(self):
        return None if self.solver is None else self.solver._S_cls

    @property
    def _S_T(self):
        return None if self.solver is None else self.solver._S_T

    # -- phases ------------------------------------------------------------

    def plan_pno_couplings(self):
        """Which pairs couple to which partners; see :meth:`LMP2Solver.plan_couplings`."""
        return self.lmp2_solver().plan_couplings()

    def compute_pno_overlaps(self):
        """PNO overlaps; see :meth:`LMP2Solver.compute_overlaps`."""
        self.lmp2_solver().compute_overlaps()
        return self

    def lmp2_iterations(self, optimize=True):
        """Solve the local MP2 equations; see :meth:`LMP2Solver.iterate`."""
        solver = self.lmp2_solver()
        solver.iterate(optimize=optimize)
        self.e_lmp2 = solver.e_lmp2
        self.e_lmp2_os = solver.e_lmp2_os
        self.e_lmp2_ss = solver.e_lmp2_ss
        self.n_iterations = solver.n_iterations
        self.t_capture = solver.t_capture
        self.t_iterate = solver.t_iterate
        return self

    # -- psi4 DLPNOMP2::compute_energy -------------------------------------

    def compute_energy(self, optimize=True, session=None):
        """Run the whole DLPNO-MP2 pipeline and return the total energy.

        Args:
            optimize: Apply the ComputeGraph optimization passes.
            session: An :class:`einsums.stages.Session` to run the phases in.
                The phases then dispatch through the stage registry, which is
                what makes ``session.report()`` able to say what each cost and
                what a C++ backend would be replacing. Default None keeps the
                plain call sequence, so nothing about the numbers depends on
                whether anyone is measuring.
        """
        if session is not None:
            from .stages import run_phases

            run_phases(self, session, lmp2_iterations={"optimize": optimize})
        else:
            self.setup_orbitals()
            self.compute_doi()
            self.prep_sparsity()
            self.compute_metric()
            self.compute_qia()
            self.precompute_fits()
            self.pno_transform()
            self.compute_pno_overlaps()
            self.lmp2_iterations(optimize=optimize)

        # Three additive pieces, as psi4 reports them: the converged LMP2
        # energy in the truncated PNO bases, the energy lost to that truncation,
        # and the dipole-estimated energy of the pairs never formed at all.
        self.e_corr = self.e_lmp2 + self.de_pno_total + self.de_dipole
        self._print(
            "\n  Total DLPNO-MP2 Correlation Energy: "
            f"{self.e_corr:16.12f}\n"
            f"    MP2 Correlation Energy:           {self.e_lmp2:16.12f}\n"
            f"    PNO Truncation Correction:        {self.de_pno_total:16.12f}\n"
            f"    Dipole Pair Correction:           {self.de_dipole:16.12f}"
        )
        return self.ref.e_scf + self.e_corr
