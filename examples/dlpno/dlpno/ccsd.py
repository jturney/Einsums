#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""DLPNO-CCSD, ported from psi4's ``dlpno/ccsd.cc``.

The T1-dressed local coupled-cluster formulation of Jiang et al. (JCP 161,
082502, 2024), which is what the psi4 tree this is ported from implements.

**What CC adds over MP2 is mostly the road to the iteration, not the iteration.**
DLPNO-MP2 screens pairs once, builds PNOs once and solves. DLPNO-CCSD runs a
three-stage cascade first, and each stage exists because the one after it is too
expensive to run on everything the one before admits:

1. *Crude prescreening.* Semicanonical MP2 pair energies over deliberately
   cheap domains. Pairs below ``t_cut_pairs_mp2`` are deleted outright and
   their energy is booked once, as ``de_lmp2_eliminated``.
2. *Refined prescreening.* Full domains, MP2-level PNOs, and a complete
   iterative local MP2. Pairs below ``t_cut_pairs`` become *weak*: they keep
   their converged LMP2 energy as ``de_weak`` and never enter the CC residual,
   but unlike eliminated pairs they stay in every domain, because the strong
   pairs' integrals are indexed against lists that include them.
3. *PNO rebuild.* The surviving pairs get new PNOs from the converged LMP2
   amplitudes rather than semicanonical ones, at the much looser CC cutoffs -
   looser being affordable precisely because the amplitudes are better.

The distinction between "eliminated" and "weak" is the one to keep straight
while reading: both are pairs whose energy is estimated rather than solved for,
but only the first leaves the pair list, and getting that wrong changes the
domains of every pair that coupled through it.

The energy is then the sum psi4 reports term by term:

    E(corr) = e_lccsd + de_weak + de_lmp2_eliminated + de_dipole + de_pno_total
"""

from dataclasses import replace

import numpy as np

from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import DLPNOBase, batched_eigh
from .cc_integrals import compute_pno_integrals
from .cc_overlaps import PnoOverlapStore
from .layout import PairLayout
from .lccsd import LCCSDSolver
from .lmp2_solver import LMP2Problem, LMP2Solver
from .pno_xform import pair_quantities

__all__ = ["DLPNOCCSD"]


class DLPNOCCSD(DLPNOBase):
    """DLPNO-CCSD: the prescreening cascade, then the T1-dressed CC equations.

    Not a subclass of :class:`~dlpno.mp2.DLPNOMP2`, matching psi4, whose
    ``DLPNOCCSD`` also derives straight from ``DLPNO``. What CC needs from the
    MP2 solver it gets by composition: the local MP2 it runs during prescreening
    is :class:`~dlpno.lmp2_solver.LMP2Solver` handed a second problem, not a
    second implementation.
    """

    def __init__(self, reference, thresholds=None, verbose=True, use_diis=True,
                 integral_source=None):
        super().__init__(reference, thresholds, verbose, integral_source=integral_source)
        self.use_diis = use_diis

        #: The prescreening LMP2, kept so its per-pair energies and timings stay
        #: readable after it has run.
        self.lmp2 = None
        self.e_lmp2 = 0.0
        self.e_lccsd = 0.0

        #: Correlation energy of the pairs crude prescreening deleted, psi4's
        #: ``de_lmp2_eliminated_``.
        self.de_lmp2_eliminated = 0.0
        #: What the MP2-level PNO truncation alone cost, snapshotted before
        #: ``recompute_pnos`` accumulates onto it. See :meth:`e_mp2_corr`.
        self.de_pno_mp2 = 0.0
        #: Converged LMP2 energy of the weak pairs, psi4's ``de_weak_``. Set by
        #: :meth:`recompute_pnos` at the MP2 level and overwritten by the CC
        #: iteration with its own singles-inclusive value.
        self.de_weak = 0.0
        #: The same quantity BEFORE the CC iteration overwrites it. psi4 prints
        #: this one on its way past and publishes the other as a psivar, and the
        #: two differ by the singles contribution, so a validation that compares
        #: only one of them cannot tell which it is looking at.
        self.de_weak_mp2 = 0.0

        # The strong/weak split. Both are pair lists over the SAME domains as
        # ij_to_i_j - filter_pairs classifies, it does not remove.
        self.is_strong = []
        self.ij_to_i_j_strong = []
        self.i_j_to_ij_strong = None
        self.ij_to_ji_strong = []
        self.ij_to_i_j_weak = []
        self.i_j_to_ij_weak = None
        self.ij_to_ji_weak = []

        #: Per pair, its converged LMP2 energy. What the strong/weak split is
        #: decided on, and worth keeping for the classification report.
        self.e_pair_lmp2 = None

        #: The three PNO overlap families; see :meth:`compute_pno_overlaps`.
        self.overlaps = None
        #: The pair-local integral blocks; see :meth:`compute_pno_integrals`.
        self.integral_blocks = None

    # -- PNO criteria ------------------------------------------------------

    def pno_criteria(self):
        """psi4's ``compute_pair_energies<false>`` truncation, at MP2-level cutoffs.

        Three differences from the MP2 branch, all of them psi4's:

        * the trace and energy criteria are live, at their ``_mp2`` thresholds;
        * the occupation cutoff is ``t_cut_pno_mp2``, which is TIGHTER than the
          MP2 method's own - these amplitudes decide the strong/weak split, and
          one truncated away here is never recovered;
        * diagonal pairs get psi4's two separate adjustments, folded into one
          factor. See :meth:`~dlpno.base.DLPNOBase.pno_criteria` for why folding
          them is exact.
        """
        return dict(
            t_cut_pno=float(self.cut.t_cut_pno_mp2),
            t_cut_trace=float(self.cut.t_cut_trace_mp2),
            t_cut_energy=float(self.cut.t_cut_energy_mp2),
            diag_factor=2.0 * float(self.cut.t_cut_pno_diag_scale),
        )

    # -- psi4 DLPNOCCSD::compute_pair_energies<true> ------------------------

    def semicanonical_pair_energies(self):
        """Semicanonical MP2 pair energies, without building any PNOs.

        The crude half of psi4's ``compute_pair_energies``, whose ``crude``
        template parameter guards exactly the PNO construction this skips. One
        non-iterative MP2 step per pair in its orthocanonical PAO domain, and
        the pair energy ``K_ij . Tt_ij`` that falls out of it.

        Returned per pair ordinal, with the lower triangle mirrored: the pair
        energy is symmetric, and psi4 fills ``e_ijs[ji]`` from ``e_ijs[ij]``
        rather than recomputing it.
        """
        args = self.plan_pno_transform()
        st = pair_quantities(
            args["q_ia"], args["fits"], args["fit_of"], args["fit_pos"],
            args["dom_X"], args["dom_e"], args["dom_F"], args["dom_of"],
            args["ribfs"], args["paos"], args["lmo_j"], args["shift"],
        )
        e_pair = np.zeros(self.n_lmo_pairs)
        for u, ij in enumerate(self._pno_upper):
            e_pair[ij] = e_pair[self.ij_to_ji[ij]] = float(
                ten.view(st[u]["e_ij_initial"])[0])
        return e_pair

    # -- psi4 DLPNOCCSD::filter_pairs<true> ---------------------------------

    def eliminate_pairs(self, e_pair):
        """Delete the pairs crude prescreening found negligible.

        Unlike the strong/weak split this REMOVES pairs: they leave
        ``ij_to_i_j`` and therefore every domain built from it, and their
        semicanonical energy is booked once into ``de_lmp2_eliminated``. psi4's
        comment on the branch is "Otherwise, it dies forever :)".

        Diagonal pairs are always kept, whatever their energy: a correlation
        method without ``ii`` pairs is not the method.

        The caller must rebuild the pair domains afterwards, which
        ``prep_sparsity(False, False)`` does - ``lmopair_to_lmos`` is defined by
        which pairs exist, so deleting one changes the domain of every pair that
        coupled through it.
        """
        naocc = self.ref.naocc
        keep = []
        dropped = 0.0
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if i == j or abs(e_pair[ij]) >= self.cut.t_cut_pairs_mp2:
                keep.append(ij)
            else:
                dropped += e_pair[ij]

        self.de_lmp2_eliminated = dropped
        n_before = self.n_lmo_pairs
        self.i_j_to_ij = np.full((naocc, naocc), -1, dtype=int)
        self.ij_to_i_j = [self.ij_to_i_j[ij] for ij in keep]
        for new, (i, j) in enumerate(self.ij_to_i_j):
            self.i_j_to_ij[i, j] = new
        self.ij_to_ji = [int(self.i_j_to_ij[j, i]) for (i, j) in self.ij_to_i_j]

        self._print(
            f"  crude:    {n_before - self.n_lmo_pairs} of {n_before} pairs "
            f"eliminated, {self.n_lmo_pairs} survive; "
            f"eliminated pairs contribute {self.de_lmp2_eliminated:.10f} Eh"
        )
        return self

    # -- psi4 DLPNOCCSD::filter_pairs<false> --------------------------------

    def filter_pairs(self, e_pair):
        """Split the surviving pairs into strong and weak.

        A classification, not a deletion. Weak pairs stay in ``ij_to_i_j`` and
        in every domain, because the strong pairs' integrals are indexed against
        lists - ``lmopair_to_lmos`` above all - that include them; what changes
        is that the CC residual skips them and they keep their converged LMP2
        energy instead. Diagonal pairs are always strong.

        psi4 discards the weak-pair energy this computes, because
        ``recompute_pnos`` immediately recomputes it in the rebuilt PNO basis
        and ``lccsd_iterations`` overwrites it again with the value that
        includes the singles contribution. This follows that, and reports the
        counts, which is what the validation compares.
        """
        naocc = self.ref.naocc
        self.e_pair_lmp2 = np.asarray(e_pair, dtype=float)
        self.is_strong = [
            (i == j) or abs(e_pair[ij]) >= self.cut.t_cut_pairs
            for ij, (i, j) in enumerate(self.ij_to_i_j)
        ]

        self.i_j_to_ij_strong = np.full((naocc, naocc), -1, dtype=int)
        self.i_j_to_ij_weak = np.full((naocc, naocc), -1, dtype=int)
        self.ij_to_i_j_strong, self.ij_to_i_j_weak = [], []
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if self.is_strong[ij]:
                self.i_j_to_ij_strong[i, j] = len(self.ij_to_i_j_strong)
                self.ij_to_i_j_strong.append((i, j))
            else:
                self.i_j_to_ij_weak[i, j] = len(self.ij_to_i_j_weak)
                self.ij_to_i_j_weak.append((i, j))
        self.ij_to_ji_strong = [int(self.i_j_to_ij_strong[j, i])
                                for (i, j) in self.ij_to_i_j_strong]
        self.ij_to_ji_weak = [int(self.i_j_to_ij_weak[j, i])
                              for (i, j) in self.ij_to_i_j_weak]

        n_strong = len(self.ij_to_i_j_strong)
        self._print(
            f"  pairs:    {n_strong} strong, {len(self.ij_to_i_j_weak)} weak "
            f"of {self.n_lmo_pairs} ({100.0 * n_strong / max(self.n_lmo_pairs, 1):.1f}% strong)"
        )
        return self

    # -- psi4 DLPNOCCSD::pno_lmp2_iterations --------------------------------

    def pno_lmp2_iterations(self):
        """Converge local MP2 in the MP2-level PNO basis; return the pair energies.

        The whole of psi4's ``DLPNOCCSD::pno_lmp2_iterations``, which is
        ``DLPNOMP2::lmp2_iterations`` copied with small differences. Here it is
        the same :class:`~dlpno.lmp2_solver.LMP2Solver` handed a second problem
        - see that module's docstring for why that is the shape.

        Convergence is tightened by psi4's hard-coded factor of 100 on both
        criteria. These amplitudes are not an answer: they are the input to the
        strong/weak split and to the PNO rebuild, and an under-converged
        amplitude moves a pair across a classification boundary, which no later
        stage can repair.
        """
        scale = self.cut.mp2_in_cc_convergence_scale
        cut = replace(self.cut,
                      e_convergence=self.cut.e_convergence * scale,
                      r_convergence=self.cut.r_convergence * scale)
        problem = LMP2Problem(
            naocc=self.ref.naocc,
            ij_to_i_j=self.ij_to_i_j, i_j_to_ij=self.i_j_to_ij,
            ij_to_ji=self.ij_to_ji,
            n_pno=self.n_pno, e_pno=self.e_pno, X_pno=self.X_pno,
            F_lmo=self.F_lmo, S_pao=self.S_pao,
            lmopair_to_paos=self.lmopair_to_paos,
            layout=self.layout,
            K_all=self.K_all, T_all=self.T_all, Tt_all=self.Tt_all,
            cut=cut,
        )
        self.lmp2 = LMP2Solver(problem, use_diis=self.use_diis, verbose=self.verbose)
        self.lmp2.compute_overlaps()
        self.lmp2.iterate()
        self.e_lmp2 = self.lmp2.e_lmp2
        return self.lmp2.pair_energies()

    # -- psi4 DLPNOCCSD::recompute_pnos -------------------------------------

    def recompute_pnos(self):
        """Rebuild every pair's PNO basis from the converged LMP2 amplitudes.

        The same construction as the semicanonical one - pair density,
        diagonalize, truncate, recanonicalize - with three differences that
        together are why it is worth doing twice:

        * the density comes from CONVERGED amplitudes rather than one
          semicanonical step, so the leading eigenvectors are the ones the
          coupled-cluster amplitudes will actually populate;
        * it therefore truncates at ``t_cut_pno``, two orders of magnitude
          looser than the ``t_cut_pno_mp2`` that produced the basis it starts
          from, and still recovers more;
        * it runs entirely inside the existing PNO basis. There is no PAO
          domain work: ``X_pno`` already maps the raw pair PAOs to PNOs, so the
          new transform is composed onto it rather than rebuilt.

        The truncation error ACCUMULATES onto ``de_pno``, which already holds
        what the first construction lost. Both are truncations of the same
        untruncated pair energy, and psi4 books their sum.

        A note for anyone comparing against psi4 line by line: its
        ``recompute_pnos`` computes ``orthocanonicalizer(S_pao_ij, F_pao_ij)``
        and a transformed ``S_pao_ij`` and then never reads either. That is dead
        code, and it is omitted here rather than reproduced.

        The PNO counts change, so this ends by building a NEW
        :class:`~dlpno.layout.PairLayout` and new stores. That is the reason the
        layout is a value rather than fields on this object.
        """
        upper = [ij for ij, (i, j) in enumerate(self.ij_to_i_j) if i <= j]
        n_pno_old = list(self.n_pno)

        # The pair's Fock matrix over its RAW PAO domain, which is what
        # X_pno maps from. Memoized by domain, like everything else
        # domain-shaped: pairs share domains heavily.
        F_dom, dom_of, doms = {}, [], []
        for ij in upper:
            key = id(self.lmopair_to_paos[ij])
            hit = F_dom.get(key)
            if hit is None:
                hit = len(doms)
                F_dom[key] = hit
                doms.append(sparse.submatrix_rows_and_cols(
                    self.F_pao, self.lmopair_to_paos[ij], self.lmopair_to_paos[ij],
                    name="F (PAO domain)"))
            dom_of.append(hit)

        # The blocks this pair currently holds, as views into the stores.
        K_view = [self.pair_view(self.K_all, ij) for ij in upper]
        T_view = [self.pair_view(self.T_all, ij) for ij in upper]
        Tt_view = [self.pair_view(self.Tt_all, ij) for ij in upper]

        # -- pair density and the Fock matrix in the current PNO basis ------
        # Both are padded-block operations, so the trailing rows and columns are
        # zero and the eigendecomposition sees a block-diagonal matrix whose
        # padding block is empty. The truncation loop then reads only the first
        # n_pno entries, which is what keeps the padding inert.
        D_all, F_init = [], []
        g1 = cg.Graph("PNO rebuild densities")
        with cg.capture(g1):
            for u, ij in enumerate(upper):
                n = n_pno_old[ij]
                X = self.X_pno[ij]
                F_init.append(ten.triplet(X, doms[dom_of[u]], X, trans_a=True,
                                          name="F (PNO)"))
                Tt, T = Tt_view[u][:n, :n], T_view[u][:n, :n]
                D = ten.doublet(Tt, T, trans_b=True, name="D (pair)")
                la.axpby(1.0, ten.doublet(Tt, T, trans_a=True), 1.0, D)
                D_all.append(D)
        self._run(g1)

        occs, vecs = batched_eigh(D_all, self._run, descending=True, label="PNO rebuild")

        # -- the pair integrals in the untruncated rebuilt basis ------------
        # Needed by the energy criterion, which is always live here: the CC
        # branch's t_cut_energy is 0.99 at NORMAL.
        K_init, Tt_init = [], []
        g2 = cg.Graph("PNO rebuild projection")
        with cg.capture(g2):
            for u, ij in enumerate(upper):
                n = n_pno_old[ij]
                X = vecs[u]
                K_init.append(ten.triplet(X, K_view[u][:n, :n], X, trans_a=True,
                                          name="K (untrunc PNO)"))
                Tt_init.append(ten.triplet(X, Tt_view[u][:n, :n], X, trans_a=True,
                                           name="Tt (untrunc PNO)"))
        self._run(g2)

        # -- the truncation decision ----------------------------------------
        keep = []
        for u, ij in enumerate(upper):
            i, j = self.ij_to_i_j[ij]
            scale = 1.0
            if i == j:
                scale *= 2.0 * self.cut.t_cut_pno_diag_scale
            if i < self.ref.n_core or j < self.ref.n_core:
                scale *= self.cut.t_cut_pno_core_scale
            e_total = float(np.vdot(self.pair_block(self.K_all, ij),
                                    self.pair_block(self.Tt_all, ij)).real)
            keep.append(_truncate_inclusive(
                ten.view(occs[u]), scale, int(self.cut.min_pnos),
                float(self.cut.t_cut_pno), float(self.cut.t_cut_trace),
                float(self.cut.t_cut_energy),
                ten.view(K_init[u]), ten.view(Tt_init[u]), e_total,
                n_pno_old[ij]))

        X_trunc = [vecs[u][:, :keep[u]] for u in range(len(upper))]

        # -- recanonicalize what survives -----------------------------------
        g3 = cg.Graph("PNO rebuild canonicalization")
        with cg.capture(g3):
            Fmo = [ten.triplet(X_trunc[u], F_init[u], X_trunc[u], trans_a=True,
                               name="C^T F C") for u in range(len(upper))]
        self._run(g3)
        e_new, canons = batched_eigh(Fmo, self._run, descending=True,
                                     label="PNO rebuild canon")

        # -- the rebuilt blocks ---------------------------------------------
        K_new, T_new, Tt_new, X_new = [], [], [], []
        g4 = cg.Graph("PNO rebuild projection (final)")
        with cg.capture(g4):
            for u, ij in enumerate(upper):
                n = n_pno_old[ij]
                R = ten.doublet(X_trunc[u], canons[u], name="X (rebuilt)")
                K_new.append(ten.triplet(R, K_view[u][:n, :n], R, trans_a=True, name="K"))
                T_new.append(ten.triplet(R, T_view[u][:n, :n], R, trans_a=True, name="T"))
                Tt_new.append(ten.triplet(R, Tt_view[u][:n, :n], R, trans_a=True, name="Tt"))
                # Composed onto the existing transform, so X_pno keeps meaning
                # "raw pair PAOs to canonical PNOs" rather than becoming a
                # rotation within a basis nothing else knows about.
                X_new.append(ten.doublet(self.X_pno[ij], R, name="X (PAO->PNO)"))
        self._run(g4)

        self._install_rebuilt_pnos(upper, keep, K_new, T_new, Tt_new, X_new, e_new)
        return self

    def _install_rebuilt_pnos(self, upper, keep, K_new, T_new, Tt_new, X_new, e_new):
        """Replace the PNO set, the layout and the stores with the rebuilt ones.

        Split out because it is bookkeeping rather than arithmetic and because
        it is where the truncation error is booked: ``de_pno`` accumulates onto
        what the first PNO construction already lost, exactly as psi4 does with
        its ``+=``.
        """
        # The additional truncation error, measured BEFORE the stores are
        # replaced, since it is the difference between the two bases.
        de_extra = {}
        for u, ij in enumerate(upper):
            before = float(np.vdot(self.pair_block(self.K_all, ij),
                                   self.pair_block(self.Tt_all, ij)).real)
            after = float(np.vdot(ten.view(K_new[u]), ten.view(Tt_new[u])).real)
            de_extra[ij] = before - after

        for u, ij in enumerate(upper):
            i, j = self.ij_to_i_j[ij]
            ji = self.ij_to_ji[ij]
            for target in ((ij, ji) if i < j else (ij,)):
                self.X_pno[target] = X_new[u]
                self.e_pno[target] = e_new[u]
                self.n_pno[target] = keep[u]
                self.de_pno[target] += de_extra[ij]

        # A new layout, because the PNO counts have changed. The old stores are
        # read for the last time here, one pair at a time, and then dropped.
        old_layout, old_K, old_T, old_Tt = (
            self.layout, self.K_all, self.T_all, self.Tt_all)
        self.npno_max = max(self.n_pno) if self.n_pno else 0
        self.layout = PairLayout.choose(self.n_pno, self.cut, report=self._print)
        self.K_all = self.new_pair_stores("K")
        self.T_all = self.new_pair_stores("T")
        self.Tt_all = self.new_pair_stores("Tt")
        del old_layout, old_K, old_T, old_Tt

        for u, ij in enumerate(upper):
            n = self.n_pno[ij]
            if n == 0:
                continue
            i, j = self.ij_to_i_j[ij]
            K, T, Tt = (ten.view(K_new[u]), ten.view(T_new[u]), ten.view(Tt_new[u]))
            self.pair_block(self.K_all, ij)[:n, :n] = K
            self.pair_block(self.T_all, ij)[:n, :n] = T
            self.pair_block(self.Tt_all, ij)[:n, :n] = Tt
            if i < j:
                ji = self.ij_to_ji[ij]
                self.pair_block(self.K_all, ji)[:n, :n] = K.T
                self.pair_block(self.T_all, ji)[:n, :n] = T.T
                self.pair_block(self.Tt_all, ji)[:n, :n] = Tt.T

        # The weak pairs' contribution, in the rebuilt basis. Overwritten again
        # by the CC iteration, which adds the singles part; this is the value
        # psi4 reports if the calculation stops at MP2.
        self.de_weak = self.de_weak_mp2 = sum(
            float(np.vdot(self.pair_block(self.K_all, ij),
                          self.pair_block(self.Tt_all, ij)).real)
            for ij in range(self.n_lmo_pairs)
            if not self.is_strong[ij] and self.n_pno[ij]
        )
        self.de_pno_total = sum(self.de_pno)

        counts = self.n_pno
        self._print(
            f"  PNOs:     avg {sum(counts) / max(len(counts), 1):.1f}, "
            f"min {min(counts)}, max {max(counts)} per LMO pair (rebuilt)"
        )
        self._print(f"  PNO truncation energy = {self.de_pno_total:.12f}")
        self._print(f"  weak pair energy      = {self.de_weak:.12f}")
        return self

    # -- psi4 DLPNOCCSD::compute_energy -------------------------------------

    def prescreen_pairs(self):
        """The whole three-stage cascade, in psi4's order.

        The threshold scaling around the crude pass is psi4's, and the direction
        is worth stating because the naming invites the opposite reading: both
        are thresholds a quantity must EXCEED to keep its atom or PAO, so
        multiplying them up discards more and the crude domains come out
        SMALLER. That is what makes the pass cheap enough to be worth running
        before the real one.
        """
        full = self.cut

        self.cut = replace(full, t_cut_mkn=full.t_cut_mkn * 100,
                           t_cut_do=full.t_cut_do * 2)
        self.prep_sparsity(initial=True, final=False)
        self.compute_qia()
        self.eliminate_pairs(self.semicanonical_pair_energies())

        self.cut = full
        self.prep_sparsity(initial=False, final=False)
        self.compute_qia()
        self.precompute_fits()
        self.pno_transform()
        # What the MP2-level truncation alone cost, before the rebuild adds to
        # it. psi4 reports its ``MP2 CORRELATION ENERGY`` psivar at this point
        # in the cascade, so this is the value that comparison needs; ``de_pno``
        # itself keeps accumulating.
        self.de_pno_mp2 = self.de_pno_total
        self.filter_pairs(self.pno_lmp2_iterations())

        self.prep_sparsity(initial=False, final=True)
        self.recompute_pnos()
        self.compute_pno_integrals()
        self.compute_pno_overlaps()
        return self

    # -- psi4 DLPNOCCSD::compute_pno_integrals ------------------------------

    def compute_pno_integrals(self):
        """Every pair-local integral block the residual reads.

        See :mod:`dlpno.cc_integrals`. Needs ``(Q|i j)`` and ``(Q|u v)``, so it
        declares them here rather than during the cascade: the cascade builds
        ``(Q|i u)`` twice at two different domain settings, and these two are
        wanted once, after the pair list has settled.
        """
        self.compute_qij()
        self.compute_qab()
        self.integral_blocks = compute_pno_integrals(self)
        self.integral_blocks.report(self._print)
        return self

    # -- psi4 DLPNOCCSD::compute_pno_overlaps -------------------------------

    def compute_pno_overlaps(self):
        """The three PNO overlap families the CC residual reads.

        See :class:`~dlpno.cc_overlaps.PnoOverlapStore` for what the three are
        and why psi4 keeps them apart. Built once, after the PNO rebuild has
        settled the bases they are overlaps between, and read through
        :meth:`S_PNO` everywhere afterwards.
        """
        self.overlaps = PnoOverlapStore(self).build()
        self.overlaps.report(self._print)
        return self

    def S_PNO(self, ij, mn):
        """``S(ij, mn)``, psi4's dispatcher over the three families."""
        return self.overlaps.get(ij, mn)

    def e_mp2_corr(self):
        """The MP2 correlation energy psi4 reports partway through its CC run.

        psi4 sets ``MP2 CORRELATION ENERGY`` between the refined prescreening
        and the PNO rebuild, so it carries the MP2-level truncation correction
        and NOT the rebuild's, and it does not carry ``de_weak``: at that point
        the weak pairs are still part of ``e_lmp2``. Both facts matter for the
        comparison, which is why this is a method rather than a printed line.
        """
        return (self.e_lmp2 + self.de_lmp2_eliminated + self.de_dipole
                + self.de_pno_mp2)

    def lccsd_iterations(self):
        """Solve the T1-dressed coupled-cluster equations; see :mod:`dlpno.lccsd`."""
        self.lccsd = LCCSDSolver(self, verbose=self.verbose)
        self.lccsd.iterate()
        self.e_lccsd = self.lccsd.e_lccsd
        # psi4 overwrites the MP2-level weak-pair energy here with the one that
        # includes the singles contribution, and publishes THAT as
        # ``DLPNO LMP2 WEAK PAIR ENERGY``.
        self.de_weak = self.lccsd.de_weak
        return self

    def compute_energy(self, optimize=True, session=None, method="ccsd"):
        """Run the DLPNO-CCSD pipeline and return the total energy.

        Args:
            method: ``"ccsd"`` for the full calculation, or ``"mp2"`` to stop
                after the prescreening cascade and report the MP2-level energy
                psi4 also publishes at that point. The second is what M2's
                validation compares against and is much cheaper, so it stays a
                supported mode rather than a historical one.
        """
        self.setup_orbitals()
        self.compute_doi()
        self.compute_metric()
        self.prescreen_pairs()

        if method == "ccsd":
            self.lccsd_iterations()
            self.e_corr = (self.e_lccsd + self.de_weak + self.de_lmp2_eliminated
                           + self.de_dipole + self.de_pno_total)
            self._print(
                "\n  Total DLPNO-CCSD Correlation Energy: "
                f"{self.e_corr:16.12f}\n"
                f"    CCSD Correlation Energy:          {self.e_lccsd:16.12f}\n"
                f"    Weak Pair Contribution:           {self.de_weak:16.12f}\n"
                f"    Eliminated Pair Correction:       {self.de_lmp2_eliminated:16.12f}\n"
                f"    Dipole Pair Correction:           {self.de_dipole:16.12f}\n"
                f"    PNO Truncation Correction:        {self.de_pno_total:16.12f}"
            )
            return self.ref.e_scf + self.e_corr

        self.e_corr = self.e_mp2_corr()
        self._print(
            "\n  MP2 Correlation Energy (psi4's psivar at this point): "
            f"{self.e_corr:16.12f}\n"
            f"    Local MP2 Energy:                 {self.e_lmp2:16.12f}\n"
            f"    Eliminated Pair Correction:       {self.de_lmp2_eliminated:16.12f}\n"
            f"    Dipole Pair Correction:           {self.de_dipole:16.12f}\n"
            f"    PNO Truncation (MP2 level):       {self.de_pno_mp2:16.12f}\n"
            "\n  Carried into the coupled-cluster stage:\n"
            f"    Weak Pair Energy:                 {self.de_weak:16.12f}\n"
            f"    PNO Truncation (after rebuild):   {self.de_pno_total:16.12f}"
        )
        return self.ref.e_scf + self.e_corr


def _truncate_inclusive(occ, scale, min_pnos, t_cut_pno, t_cut_trace,
                        t_cut_energy, K_init, Tt_init, e_total, nvir):
    """How many rebuilt PNOs survive, by psi4's ``recompute_pnos`` selection loop.

    The same four OR'd criteria as
    :func:`~dlpno.pno_xform._truncate`, with one difference that is not
    cosmetic: here the energy is evaluated on the set INCLUDING ``a``, where
    ``compute_pair_energies`` evaluates it on the set before ``a`` joins. psi4
    writes the two loops in the opposite order (ccsd.cc:650 against 1059), so
    the same thresholds keep a different number of PNOs, and reproducing each
    faithfully is the only way the counts match.
    """
    occ_sum = 0.0
    e_pno = 0.0
    occ_total = float(np.sum(occ[:nvir])) if nvir else 0.0
    cut = scale * t_cut_pno
    energy_floor = t_cut_energy * abs(e_total)
    kept = []

    for a in range(nvir):
        if not (abs(occ[a]) >= cut
                or (occ_total != 0.0 and occ_sum / occ_total < t_cut_trace)
                or abs(e_pno) < energy_floor
                or a < min_pnos):
            continue
        kept.append(a)
        ix = np.ix_(kept, kept)
        e_pno = float(np.vdot(K_init[ix], Tt_init[ix]).real)
        occ_sum += occ[a]
    return min(nvir, max(1, len(kept)))
