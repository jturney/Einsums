#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The T1-dressed local CCSD iteration, from psi4's ``DLPNOCCSD::lccsd_iterations``.

Jiang et al., JCP 161, 082502 (2024). Every method below names the equation it
implements, because the equations are the specification and the code is only a
transcription of them.

**Every contraction is written as an explicit einsum index string or a named
GEMM.** That is not a style choice. The residual is thirty-odd contractions over
blocks that live in several different PNO bases, bridged by overlap matrices,
and half of them are built by re-blocking a tensor whose two virtual indices are
interchangeable in the integral but not in the term. psi4 expresses these as
sequences of ``doublet``/``triplet`` calls with transpose flags and reshapes,
where a swapped pair reads as a plausible integral and surfaces only as a wrong
correlation energy several equations downstream. Naming the indices makes that
class of error impossible to write.

**The iteration is captured, not eager.** Every quantity here has a fixed shape
and a fixed dependency pattern once sparsity is frozen; only values change from
iteration to iteration. So the whole iteration is recorded once into a set of
ComputeGraphs before the solver loop starts and replayed to convergence, and
Python appears in the per-iteration path only in the convergence predicate and
the DIIS call. Nothing is allocated per iteration and no index is resolved
twice.

**Plan, then emit.** Each phase is split in two. ``_plan_*`` derives the work
list - which pairs, which neighbours, which overlap bridges, and what shape
class each record belongs to - as pure integer bookkeeping over the sparsity
maps, once. ``_emit_*`` walks that list and records operations into whatever
capture is ambient. The split is what makes the performance campaign an emitter
change: the first cut emits one operation per record, and grouping records by
shape class into batched calls (the lever that took LMP2 from 754 dispatches an
iteration to 13) touches only the emitter and none of the derivation. See
decision 3 of ``DESIGN-ccsd.md``.

**Phase graphs, not one loop graph.** Each phase is its own graph, replayed in
order by a host ``while``. Graph boundaries are the slice-versus-whole safety
line, and there are a lot of slices here, so the correctness cut is liberal with
them. Folding the phases into a single ``add_loop`` body with the convergence
test and DIIS in the predicate is the performance campaign's job, and the
boundaries get audited away there rather than here.

**Where scratch is shared and where it is not.** Everything of order
``n_pno^2`` is allocated per pair, so each pair's chain is independent and a
future executor can thread over them. The rank-3 scratch that carries the
auxiliary index - the dressed ``B~^Q_{ab}`` above all - is a single buffer sized
to the largest pair and shared by every pair, because one per pair is 15 MiB
times the pair count and the store would be larger than every integral in the
calculation put together. Sharing serializes exactly those chains, which is the
right trade for a correctness cut and is what the ``ScratchPrivatization`` pass
exists to undo.
"""

import time

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import tensors as ten

__all__ = ["LCCSDSolver"]


class LCCSDSolver:
    """Solve the T1-dressed local coupled-cluster singles and doubles equations.

    Constructed from a :class:`~dlpno.ccsd.DLPNOCCSD` whose cascade has run, so
    the PNO bases, the overlaps and every integral block already exist and are
    fixed for the life of the solve.

    Args:
        cc: the calculation, after ``prescreen_pairs``.
        verbose: print the iteration table.
        use_diis: extrapolate the amplitudes with :func:`einsums.graph.diis`.
        clamp_singles: hold ``T1`` at zero by discarding its residual. Not a
            physical model - it is the bisection ``check_ccsd_defect.py`` uses
            to separate the T1-dressing terms from the pure doubles ones.
    """

    def __init__(self, cc, verbose=True, use_diis=True, clamp_singles=False):
        self.cc = cc
        self.verbose = verbose
        self.use_diis = use_diis
        self.clamp_singles = clamp_singles

        self.B = cc.integral_blocks
        self.naocc = cc.ref.naocc
        self.F_lmo_np = np.asarray(ten.view(cc.F_lmo))

        self.e_lccsd = 0.0
        self.de_weak = 0.0
        self.n_iterations = 0
        self.t_plan = 0.0
        self.t_capture = 0.0
        self.t_iterate = 0.0

        #: Tensors and views created while emitting. A view handed to a captured
        #: op is held by the graph as a slot pointer, so one that dies at the end
        #: of the expression that made it leaves the replay reading freed memory.
        self._keep = []
        self._graphs = None
        self._plan = None
        self._diis = None

    # -- small accessors ---------------------------------------------------

    def _print(self, *args):
        if self.verbose:
            print(*args, flush=True)

    def diag(self, i):
        """The diagonal pair ``ii``, which always exists."""
        return int(self.cc.i_j_to_ij[i, i])

    def is_strong(self, ij):
        i, j = self.cc.ij_to_i_j[ij]
        return self.cc.i_j_to_ij_strong[i, j] != -1

    def _S(self, ij, mn):
        """``S(ij, mn)``, memoized.

        psi4's dispatcher answers most requests from a stored family and builds
        the rest on the fly. Memoizing here moves every on-the-fly build to
        capture time, so the replayed iteration never builds an overlap.
        """
        hit = self._S_cache.get((ij, mn))
        if hit is None:
            hit = self.cc.S_PNO(ij, mn)
            self._S_cache[ij, mn] = hit
        return hit

    def _pair(self, stores, ij):
        """The logical ``(n, n)`` einsums view of pair ``ij``'s block, memoized."""
        key = (id(stores), ij)
        hit = self._view_cache.get(key)
        if hit is None:
            n = self.cc.n_pno[ij]
            layout = self.cc.layout
            hit = stores[layout.bucket_of[ij]][:n, :n, layout.slot_of[ij]]
            self._view_cache[key] = hit
        return hit

    def T2(self, ij):
        """Pair ``ij``'s doubles amplitudes as a numpy view, unpadded."""
        n = self.cc.n_pno[ij]
        return self.cc.pair_block(self.cc.T_all, ij)[:n, :n]

    def Tt2(self, ij):
        """Pair ``ij``'s antisymmetrized amplitudes ``2T - T^T``, as numpy."""
        n = self.cc.n_pno[ij]
        return self.cc.pair_block(self.cc.Tt_all, ij)[:n, :n]

    def _T2(self, ij):
        """Pair ``ij``'s doubles amplitudes as a captured view."""
        return self._pair(self.cc.T_all, ij)

    def _Tt2(self, ij):
        """Pair ``ij``'s antisymmetrized amplitudes as a captured view."""
        return self._pair(self.cc.Tt_all, ij)

    def _K2(self, ij):
        """The exchange operator ``(i a | j b)`` in pair ``ij``'s PNO basis."""
        return self._pair(self.cc.K_all, ij)

    def _T1(self, i):
        """LMO ``i``'s singles amplitudes as a captured ``(n, 1)`` view."""
        hit = self._T1_view.get(i)
        if hit is None:
            n = self.cc.n_pno[self.diag(i)]
            hit = self.T1_all[:n, i:i + 1]
            self._T1_view[i] = hit
        return hit

    # -- Jiang Eq. 70, and every other projection of a singles amplitude ----

    def _proj(self, p, x):
        """``S(a_p, a_xx) t_x^{a_xx}``, as a ``(1, n_p)`` captured view.

        Every term that reads a singles amplitude reads it in some pair's PNO
        basis rather than in its own, so the projection is the single most
        common operation in the iteration. psi4 rebuilds it at each use; here it
        is built once per distinct ``(pair, LMO)`` request and shared.

        Most requests are for an LMO in the pair's own neighbour list, which is
        exactly Jiang Eq. 70's ``T_{n_ij}^{a_ij}``, and those land as rows of
        that pair's ``T_n_ij`` block so the block also exists whole for the terms
        that contract it as a matrix. The rest - a pair against an LMO outside
        its neighbour list, which Eq. 83c and Eq. 88's second term both ask for -
        get a standalone row. Both are the same GEMM against the same overlap,
        so the emitter does not distinguish them.

        Returned as ``(1, n_p)`` rather than as a vector because both forms are
        wanted: ``[0, :]`` is the rank-1 operand ``ger`` and the einsums take,
        and the ``(1, n_p)`` form is what a GEMM contracts through ``trans_b``.
        """
        return self._proj_row[p, x]

    def _plan_projections(self):
        """Which singles projections the iteration reads, and where each lives.

        Walks every term that reads one and records the ``(pair, LMO)`` request.
        Deduplicating them here is what turns psi4's repeated
        ``S_PNO(...) * T_ia`` products into one GEMM apiece.
        """
        cc = self._cc_maps()

        # Eq. 70 proper: every pair against every LMO in its neighbour list.
        # EVERY row is registered, including those whose LMO has no PNOs of its
        # own - the block has a row for each neighbour either way, and the terms
        # that contract it as a matrix read the whole thing. A row with nothing
        # to project stays at the zero it was allocated with, which is what psi4
        # leaves there and what those terms want.
        rows = []
        for ij in cc["live"]:
            for n_ij, n in enumerate(cc["lmos"][ij]):
                rows.append((ij, n_ij, n, self.diag(n)))
        in_rows = {(ij, n) for ij, _, n, _ in rows}

        # A pair's own two LMOs are always in its neighbour list, and several
        # terms index the row by position rather than searching for it, so a
        # missing one would be read as numpy's -1 - the last row - rather than
        # raise. Stated here once instead of guarded at four call sites.
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            for x in (i, j):
                assert (ij, x) in in_rows, (
                    f"LMO {x} is outside its own pair {ij}'s neighbour list")

        # Everything else that projects a singles amplitude into a pair: a pair
        # against an LMO that is NOT one of its neighbours. Eq. 83c, Eq. 84c and
        # Eq. 88's second term each ask for one.
        requests = set()

        def want(p, x):
            if p != -1 and cc["n_pno"][p] and (p, x) not in in_rows:
                requests.add((p, x))

        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            for l in cc["lmos"][ij]:
                want(int(cc["i_j_to_ij"][i, l]), j)   # Eq. 83c, labelled by ki
                want(int(cc["i_j_to_ij"][l, j]), i)   # Eq. 84c, labelled by ik
                want(self.diag(i), l)                 # Eq. 88's second term

        extras = sorted(requests)
        self._n_proj_rows = len(rows)
        self._n_proj_extra = len(extras)
        # What actually gets a GEMM: a projection of an amplitude that exists.
        emit = [(ij, n, nn) for ij, _, n, nn in rows if cc["n_pno"][nn]]
        emit += [(p, x, self.diag(x)) for p, x in extras
                 if cc["n_pno"][self.diag(x)]]
        return rows, emit

    def _emit_projections(self):
        """Record Eq. 70 and its relatives: one GEMM per distinct projection.

        ``dest`` is a ``(1, n_p)`` row, so the product is written transposed:
        ``t^T S^T`` rather than ``S t``. That is not a detour - the destination
        is a row of a column-major block either way, and writing it directly is
        cheaper than producing a column and permuting it.
        """
        for p, x, xx in self._plan.projections:
            la.gemm(1.0, self._T1(x), self._S(p, xx), 0.0, self._proj_row[p, x],
                    trans_a=True, trans_b=True)

    # -- Jiang Eq. 91-92 ---------------------------------------------------

    def _plan_t1_ints(self):
        """The pairs whose dressed three-index factors the residual reads.

        Strong pairs only: Eq. 75 and Eq. 82 are the only consumers and neither
        runs over a weak pair.
        """
        cc = self._cc_maps()
        out = []
        for ij in cc["live"]:
            if not self.is_strong(ij):
                continue
            i, _ = cc["ij_to_i_j"][ij]
            out.append((ij, int(cc["ij_to_ji"][ij]), i))
        return out

    def _emit_t1_ints(self):
        """The T1-dressed three-index factors.

        Eq. 91:  ``B~^Q_{ki} = B^Q_{ki} + B^Q_{ka} t_i^a``
        Eq. 92:  ``B~^Q_{ai} = B^Q_{ai} - t_k^a B~^Q_{ki} + B^Q_{ab} t_i^b``

        Dressing the integrals rather than the amplitudes is what makes this
        formulation worth having: the singles are folded into the factors once
        per iteration, and every term downstream then reads a plain integral.
        """
        cc = self.cc
        for ij, _ji, i in self._plan.t1_ints:
            T_i = self._proj(ij, i)[0, :]                       # (a,)
            Qma = self.B.qma(cc.ij_to_ji, ij)                   # (Q, k, a)
            Qab = self.B.qab(cc.ij_to_ji, ij)                   # (Q, a, b)

            i_Qk = self.i_Qk_t1[ij]
            la.axpby(1.0, self.B.i_Qk[ij], 0.0, i_Qk)
            einsums.einsum("qk <- qka ; a", i_Qk, Qma, T_i, c_pf=1.0)

            i_Qa = self.i_Qa_t1[ij]
            la.axpby(1.0, self.B.i_Qa[ij], 0.0, i_Qa)
            la.gemm(-1.0, i_Qk, self.T_n[ij], 1.0, i_Qa)        # (Q,k)(k,a)
            einsums.einsum("qa <- qab ; b", i_Qa, Qab, T_i, c_pf=1.0)

    # -- Jiang Eq. 94-101 --------------------------------------------------

    def _plan_t1_fock(self):
        """The three work lists the dressed Fock matrices need.

        psi4 dresses in two passes, the first over CONTRACTED indices (the
        ``bar`` quantities, Eq. 98-101) and the second over free ones (the
        ``tilde`` quantities, Eq. 94-97). The pair sets differ between them and
        between the terms within them, which is the whole content of this
        method: Eq. 98 runs over ALL pairs because the occupied block picks up a
        contribution from every one of them, weak pairs included, while Eq. 99
        to 101 run over strong upper-triangle pairs only.
        """
        cc = self._cc_maps()
        occ = [(ij, *cc["ij_to_i_j"][ij], int(cc["ij_to_ji"][ij]))
               for ij in cc["live"]]

        bar = []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            if i > j or not self.is_strong(ij):
                continue
            bar.append((ij, i, j))

        # Eq. 94 wants Fkc_bar on the diagonal pair jj and the row of it
        # belonging to i, so i has to be one of jj's neighbours. It always is -
        # pair ij exists, so i and j interact - but reading a missing slot as
        # numpy's -1 would silently take the last row rather than fail.
        kj = []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            jj = self.diag(j)
            if cc["n_pno"][jj] == 0 or not self.is_strong(jj):
                continue
            i_jj = int(cc["lmos_dense"][jj, i])
            assert i_jj != -1, (
                f"LMO {i} is outside diagonal pair {j}{j}'s neighbour list, so "
                f"Eq. 94 has no F~_kc row to read for pair ({i},{j})")
            kj.append((ij, i, j, jj, i_jj))

        # Eq. 95, over ALL pairs: Fkc_bar is not built over weak pairs, so psi4
        # builds this one separately and from L_iajb.
        kc = []
        for ij in cc["live"]:
            i, _ = cc["ij_to_i_j"][ij]
            for k in cc["lmos"][ij]:
                ik = int(cc["i_j_to_ij"][i, k])
                kk = self.diag(k)
                if ik == -1 or not cc["n_pno"][ik] or not cc["n_pno"][kk]:
                    continue
                kc.append((ij, ik, k))
        return occ, bar, kj, kc

    def _emit_t1_fock_bar(self):
        """Eq. 98-101, the quantities dressed over contracted indices.

        Two intermediates carry the whole dressing. ``gamma_Q = B^Q_{me} t_m^e``
        is the Coulomb-like half of all three, one scalar per auxiliary
        function. The exchange-like half is a three-factor contraction, and
        WHICH index is contracted first decides how big the intermediate is:
        folding the amplitude into the occupied factor gives ``(Q, k, m)`` for
        Eq. 99 and ``(Q, a, m)`` for Eq. 101, both of which carry an occupied
        axis, where the obvious order carries a second virtual one and is a
        factor of ``n_pno / n_lmo`` larger and slower.
        """
        cc = self.cc
        for ij, i, j in self._plan.fock_bar:
            nq = ten.shape(self.B.i_Qk[ij])[0]
            nk, na = ten.shape(self.T_n[ij])
            T_n = self.T_n[ij]                                  # (k, e)
            Qma = self.B.qma(cc.ij_to_ji, ij)                   # (Q, k, a)
            Qab = self.B.qab(cc.ij_to_ji, ij)                   # (Q, a, b)

            gamma = self.gamma_Q[ij]
            einsums.einsum("q <- qme ; me", gamma, Qma, T_n)

            # Eq. 99, Coulomb-like then exchange-like.
            V = self._shared("V", [nq, nk, nk])
            einsums.einsum("qkm <- qke ; me", V, Qma, T_n)
            Fkc = self.Fkc_bar[ij]
            einsums.einsum("kc <- q ; qkc", Fkc, gamma, Qma, ab_pf=2.0)
            einsums.einsum("kc <- qkm ; qmc", Fkc, V, Qma, c_pf=1.0, ab_pf=-1.0)

            # Eq. 101. The orbital energies are the undressed diagonal and are
            # constant, so they are written once at allocation and copied here.
            Fab = self.Fab_bar[ij]
            la.axpby(1.0, self.e_diag[ij], 0.0, Fab)
            einsums.einsum("ab <- q ; qab", Fab, gamma, Qab, c_pf=1.0, ab_pf=2.0)
            U = self._shared("U", [nq, na, nk])
            einsums.einsum("qam <- qae ; me", U, Qab, T_n)
            einsums.einsum("ab <- qam ; qmb", Fab, U, Qma, c_pf=1.0, ab_pf=-1.0)

            # Eq. 100, only on the diagonal pairs, where the singles live.
            if i == j:
                y = self._shared("y", [nq, na])
                einsums.einsum("qe <- me ; qm", y, T_n, self.B.i_Qk[ij])
                Fai = self.Fai_bar[i]
                einsums.einsum("a <- q ; qa", Fai[:, 0], gamma, self.B.i_Qa[ij],
                               ab_pf=2.0)
                einsums.einsum("a <- qae ; qe", Fai[:, 0], Qab, y,
                               c_pf=1.0, ab_pf=-1.0)

    def _emit_t1_fock_occupied(self):
        """Eq. 98: the dressed occupied Fock matrix, over ALL pairs.

        One element of ``F_bar`` per pair, so this is thousands of scalar
        reductions landing in thousands of distinct matrix elements. Each one is
        a dot product into a length-1 tensor and an accumulate into a ``(1, 1)``
        view of the matrix - the captured spelling of ``+=`` on a scalar, and
        the shape of every scalar accumulation below.
        """
        la.axpby(1.0, self.cc.F_lmo, 0.0, self.Fij_bar)
        for ij, i, j, ji in self._plan.fock_occupied:
            T_n = self.T_n[ij]
            dest = self._element(self.Fij_bar, i, j)
            s1, s2 = self.fock_occ_scalar[ij]
            la.dot(s1, T_n, self.B.J_ijmb[ij])
            la.axpby(2.0, s1.reshape_view([1, 1]), 1.0, dest)
            la.dot(s2, T_n, self.B.K_mibj[ji])
            la.axpby(-1.0, s2.reshape_view([1, 1]), 1.0, dest)

    def _emit_t1_fock_tilde(self):
        """Eq. 94-97, the quantities dressed over free indices.

        The one thing to be careful of is that Eq. 96's last term reads the
        *bar* occupied Fock matrix, not the fully dressed one.
        """
        cc = self.cc

        # Eq. 94.
        la.axpby(1.0, self.Fij_bar, 0.0, self.Fkj)
        for ij, i, j, jj, i_jj in self._plan.fock_kj:
            s = self.fock_kj_scalar[ij]
            la.dot(s, self.Fkc_bar[jj][i_jj, :], self._T1(j)[:, 0])
            la.axpby(1.0, s.reshape_view([1, 1]), 1.0,
                     self._element(self.Fkj, i, j))

        # Eq. 95.
        for ij in self._plan.live:
            la.scale(0.0, self.Fkc[ij])
        for ij, ik, k in self._plan.fock_kc:
            tmp = self.fock_kc_tmp[ij, ik]
            la.gemm(1.0, self.B.L_iajb[ik], self._proj(ik, k), 0.0, tmp,
                    trans_b=True)
            la.gemm(1.0, self._S(ij, ik), tmp, 1.0, self.Fkc[ij])

        # Eq. 97 and Eq. 96.
        for ij, i, j in self._plan.fock_bar:
            la.axpby(1.0, self.Fab_bar[ij], 0.0, self.Fab[ij])
            la.gemm(-1.0, self.T_n[ij], self.Fkc_bar[ij], 1.0, self.Fab[ij],
                    trans_a=True)
            if i != j:
                continue
            Fai = self.Fai[i]
            la.axpby(1.0, self.Fai_bar[i], 0.0, Fai)
            la.gemm(1.0, self.Fab_bar[ij], self._T1(i), 1.0, Fai)
            tmp = self.fock_ai_tmp[i]
            la.gemm(1.0, self.Fkc_bar[ij], self._T1(i), 0.0, tmp)
            la.gemm(-1.0, self.T_n[ij], tmp, 1.0, Fai, trans_a=True)
            # The last term reads the BAR occupied matrix, not Fkj.
            col = self.fock_ai_col[i]
            la.gather(col, self.Fij_bar, [list(cc.lmopair_to_lmos[ij]), [i]])
            la.gemm(-1.0, self.T_n[ij], col, 1.0, Fai, trans_a=True)

    # -- Jiang Eq. 82 -------------------------------------------------------

    def _plan_beta(self):
        """Eq. 82 runs over the strong pairs, which are the only ones Eq. 77 reads."""
        cc = self._cc_maps()
        return [(ij, int(cc["ij_to_ji"][ij])) for ij in cc["live"]
                if self.is_strong(ij)]

    def _emit_beta(self):
        """Eq. 82: ``beta_{ij}^{kl} = B~^Q_{ki} B~^Q_{lj} + t_{ij}^{cd} B^Q_{kc} B^Q_{ld}``."""
        cc = self.cc
        for ij, ji in self._plan.beta:
            Qma = self.B.qma(cc.ij_to_ji, ij)                   # (Q, k, c)
            nq, nk, na = ten.shape(Qma)
            la.gemm(1.0, self.i_Qk_t1[ij], self.i_Qk_t1[ji], 0.0, self.beta[ij],
                    trans_a=True)
            W = self._shared("bW", [nq, nk, na])
            einsums.einsum("qkd <- qkc ; cd", W, Qma, self._T2(ij))
            einsums.einsum("kl <- qkd ; qld", self.beta[ij], W, Qma, c_pf=1.0)

    # -- Jiang Eq. 83 -------------------------------------------------------

    def _plan_gamma(self):
        """Eq. 83, labelled by the pair ``ki``, over ALL pairs.

        Four terms. 83a and 83b are per pair; 83c and 83d walk the pair's
        neighbour list and each needs a different set of bridges, so the plan
        records the resolved partner pairs rather than resolving them again on
        every replay.
        """
        cc = self._cc_maps()
        head, tail = [], []
        for ki in cc["live"]:
            k, i = cc["ij_to_i_j"][ki]
            ii = self.diag(i)
            head.append((ki, k, i, ii,
                         bool(cc["n_pno"][ii] and self.B.K_ivvv[ki] is not None)))
            for l in cc["lmos"][ki]:
                kl = int(cc["i_j_to_ij"][k, l])
                ll = self.diag(l)
                li = int(cc["i_j_to_ij"][l, i])
                c = (kl != -1 and cc["n_pno"][kl] and cc["n_pno"][ll]
                     and cc["n_pno"][ii])
                d = (kl != -1 and li != -1 and cc["n_pno"][kl] and cc["n_pno"][li])
                if c or d:
                    tail.append((ki, l, kl, ll, li, bool(c), bool(d)))
        return head, tail

    def _emit_gamma(self):
        """Eq. 83, four terms.

        83b is where the rank-3 ``K_ivvv`` earns itself: it is ``t_i^b (k b | a
        c)``, so ``b`` pairs with ``k`` and ``(a, c)`` are the free indices.
        Written against a flattened block the two would have to be separated by
        a reshape whose order depends on the storage layout.

        83c is the term the milestone's defect was in. ``b`` pairs with ``k``,
        which is the FIRST index of pair ``kl``'s exchange operator, hence the
        transpose; the untransposed form is ``(k c | l b) t_i^b``, wrong by
        exactly a swap of which occupied each virtual attaches to, and equal to
        it only on the diagonal pairs.
        """
        for ki, k, i, ii, has_vvv in self._plan.gamma_head:
            # 83a: -t_l^a (k i | l c)
            la.gemm(-1.0, self.T_n[ki], self.B.J_ijmb[ki], 0.0, self.gamma[ki],
                    trans_a=True)
            # 83b: +t_i^b (k b | a c)
            if has_vvv:
                einsums.einsum("ac <- bac ; b", self.gamma[ki],
                               self.B.K_ivvv[ki], self._proj(ki, i)[0, :],
                               c_pf=1.0)

        for ki, l, kl, ll, li, do_c, do_d in self._plan.gamma_tail:
            t1, t2, t3 = self.gamma_tmp[ki]
            n_kl = self.cc.n_pno[kl]
            if do_c:
                # 83c: -t_i^b (k b | l c) t_l^a. The integral is pair kl's own
                # exchange operator, which the PNO transform already left in the
                # amplitude stores.
                _, i_of_ki = self.cc.ij_to_i_j[ki]
                a = t1[:n_kl, :1]
                la.gemm(1.0, self._K2(kl), self._proj(kl, i_of_ki), 0.0, a,
                        trans_a=True, trans_b=True)
                b = t2[:, :1]
                la.gemm(1.0, self._S(ki, kl), a, 0.0, b)
                la.ger(-1.0, self._proj(ki, l)[0, :], b[:, 0], self.gamma[ki])
            if do_d:
                # 83d: -0.5 t_{li}^{ad} (k d | l c)
                n_li = self.cc.n_pno[li]
                a = self.gamma_tmp_d[ki][0][:n_li, :n_kl]
                la.gemm(1.0, self._T2(li), self._S(li, kl), 0.0, a)
                b = self.gamma_tmp_d[ki][1][:n_li, :n_kl]
                la.gemm(1.0, a, self._K2(kl), 0.0, b)
                c = t3[:, :n_kl]
                la.gemm(1.0, self._S(ki, li), b, 0.0, c)
                la.gemm(-0.5, c, self._S(kl, ki), 1.0, self.gamma[ki])

    # -- Jiang Eq. 84 -------------------------------------------------------

    def _plan_delta(self):
        """Eq. 84, labelled by the pair ``ik``, over ALL pairs.

        psi4 calls this one "a doozy" and it is: four terms, two of them
        re-blockings of the three-external block in opposite directions.
        """
        cc = self._cc_maps()
        head, tail = [], []
        for ik in cc["live"]:
            i, k = cc["ij_to_i_j"][ik]
            ki = int(cc["ij_to_ji"][ik])
            ii = self.diag(i)
            head.append((ik, i, k, ki, ii,
                         bool(cc["n_pno"][ii] and self.B.K_ivvv[ki] is not None)))
            for l in cc["lmos"][ik]:
                ll = self.diag(l)
                lk = int(cc["i_j_to_ij"][l, k])
                il = int(cc["i_j_to_ij"][i, l])
                c = (lk != -1 and cc["n_pno"][lk] and cc["n_pno"][ll]
                     and cc["n_pno"][ii])
                d = (il != -1 and lk != -1 and cc["n_pno"][il] and cc["n_pno"][lk])
                if c or d:
                    tail.append((ik, l, ll, lk, il, bool(c), bool(d)))
        return head, tail

    def _emit_delta(self):
        """Eq. 84, four terms.

        84b's two halves differ only in which slot the singles amplitude
        contracts into, which is the whole reason ``K_ivvv`` is stored rank 3:
        against a flat block the two re-blockings run in opposite directions and
        neither is expressible without knowing the storage order.

        **84c's two products are taken transposed, and that is not cosmetic.**
        Written the way the equation reads, they multiply a ``(1, n)`` row into
        a matrix; Eq. 83c, which is the same term with the indices exchanged,
        multiplies a matrix into an ``(n, 1)`` column. A GEMM with one row is a
        degenerate case, and this one was worth a factor of two on the whole
        iteration: at a six-monomer chain on ten threads it took this phase from
        68 ms to 419 and the LCCSD row from 12.2 s to 24.0. Since the result is
        only ever consumed as the rank-1 operand of the outer product below,
        taking it as a column costs nothing.

        The interaction with the thread count is not reproduced by timing the
        two shapes in isolation, where the row form is slow but slow equally at
        one thread and at ten. Do not go looking for it in a microbenchmark; the
        phase-by-phase replay in the design notes is what localized it.
        """
        for ik, i, k, ki, ii, has_vvv in self._plan.delta_head:
            # 84a: -t_l^a [2 (i l | k c) - (i k | l c)], whose integral
            # combination is constant and was formed once at allocation.
            la.gemm(-1.0, self.T_n[ik], self.M_delta[ik], 0.0, self.delta[ik],
                    trans_a=True)
            # 84b: +t_i^b [2 (k c | a b) - (k b | c a)]
            if has_vvv:
                K3 = self.B.K_ivvv[ki]                 # (e, a, f) for pair ki
                T_i = self._proj(ik, i)[0, :]
                einsums.einsum("ac <- cab ; b", self.delta[ik], K3, T_i,
                               c_pf=1.0, ab_pf=2.0)
                einsums.einsum("ac <- bac ; b", self.delta[ik], K3, T_i,
                               c_pf=1.0, ab_pf=-1.0)

        for ik, l, ll, lk, il, do_c, do_d in self._plan.delta_tail:
            t1, t2, t3 = self.delta_tmp[ik]
            n_lk = self.cc.n_pno[lk]
            if do_c:
                # 84c: -t_l^a L_{lk}^{bc} t_i^b, transposed so both products are
                # column-shaped. Written the way the equation reads it is
                # (1,n) x (n,n) twice, and a GEMM with one row is a degenerate
                # case that costs 54.6 us against 0.95 for the transpose at
                # n = 18. Since the result is only ever consumed as the rank-1
                # operand of the outer product below, taking it as a column
                # costs nothing and is the same arithmetic.
                i_of_ik, _ = self.cc.ij_to_i_j[ik]
                a = t1[:n_lk, :1]
                la.gemm(1.0, self.B.L_iajb[lk], self._proj(lk, i_of_ik), 0.0, a,
                        trans_a=True, trans_b=True)
                b = t2[:, :1]
                la.gemm(1.0, self._S(lk, ik), a, 0.0, b, trans_a=True)
                la.ger(-1.0, self._proj(ik, l)[0, :], b[:, 0], self.delta[ik])
            if do_d:
                # 84d: +0.5 u_{il}^{ad} L_{lk}^{dc}
                n_il = self.cc.n_pno[il]
                a = self.delta_tmp_d[ik][0][:n_il, :n_lk]
                la.gemm(1.0, self._Tt2(il), self._S(il, lk), 0.0, a)
                b = self.delta_tmp_d[ik][1][:n_il, :n_lk]
                la.gemm(1.0, a, self.B.L_iajb[lk], 0.0, b)
                c = t3[:, :n_lk]
                la.gemm(1.0, self._S(ik, il), b, 0.0, c)
                la.gemm(0.5, c, self._S(lk, ik), 1.0, self.delta[ik])

    # -- Jiang Eq. 86 -------------------------------------------------------

    def _plan_Fkj_double_tilde(self):
        """Eq. 86 walks every pair against its own neighbour list."""
        cc = self._cc_maps()
        out = []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            for l in cc["lmos"][ij]:
                li = int(cc["i_j_to_ij"][l, i])
                lj = int(cc["i_j_to_ij"][l, j])
                if li == -1 or lj == -1 or not cc["n_pno"][li] or not cc["n_pno"][lj]:
                    continue
                out.append((ij, i, j, l, li, lj))
        return out

    def _emit_Fkj_double_tilde(self):
        """Eq. 86: ``F''_{ij} = F~_{ij} + u_{lj}^{cd} (l c | i d)``."""
        la.axpby(1.0, self.Fkj, 0.0, self.Fkj2)
        for ij, i, j, l, li, lj in self._plan.fkj2:
            a, b, s = self.fkj2_tmp[ij, l]
            n_lj = self.cc.n_pno[lj]
            la.gemm(1.0, self._S(li, lj), self._Tt2(lj), 0.0, a[:, :n_lj])
            la.gemm(1.0, a[:, :n_lj], self._S(lj, li), 0.0, b)
            la.dot(s, self._K2(li), b)
            la.axpby(1.0, s.reshape_view([1, 1]), 1.0,
                     self._element(self.Fkj2, i, j))

    # -- Jiang Eq. 87a, 88, 89, 90 -----------------------------------------

    def _plan_R_ia(self):
        """The singles residual's three work lists.

        Eq. 88's A and Eq. 90's C are driven by the pair ``ik``, Eq. 89's B by
        the pair ``kl``, which is why psi4 walks the pair list twice rather than
        nesting.
        """
        cc = self._cc_maps()
        ik_list, a2_list, b_list = [], [], []
        for ik in cc["live"]:
            i, k = cc["ij_to_i_j"][ik]
            ki = int(cc["ij_to_ji"][ik])
            ii = self.diag(i)
            if not cc["n_pno"][ii]:
                continue
            ik_list.append((ik, i, k, ki, ii,
                            self.B.K_ivvv[ki] is not None,
                            bool(cc["n_pno"][ki])))
            for l in cc["lmos"][ik]:
                kl = int(cc["i_j_to_ij"][k, l])
                ll = self.diag(l)
                if kl == -1 or not cc["n_pno"][kl] or not cc["n_pno"][ll]:
                    continue
                a2_list.append((ik, i, ki, ii, l, kl))
        for kl in cc["live"]:
            members = []
            for i_kl, i in enumerate(cc["lmos"][kl]):
                ii = self.diag(i)
                if cc["n_pno"][ii]:
                    members.append((i_kl, i, ii))
            if members:
                b_list.append((kl, members))
        return ik_list, a2_list, b_list

    def _emit_R_ia(self):
        """The singles residual.

        Eq. 87a is the initialization from the dressed ``F_ai``; the three named
        contributions are A (Eq. 88), B (Eq. 89) and C (Eq. 90).
        """
        la.scale(0.0, self.R1_all)
        if self.clamp_singles:
            return
        for i in range(self.naocc):
            if self.Fai[i] is not None:
                la.axpby(1.0, self.Fai[i], 1.0, self._R1(i))

        for ik, i, k, ki, ii, has_vvv, has_fkc in self._plan.r_ia_ik:
            # Eq. 88 A1: u_{ki}^{cd} (k c | d a). K_ivvv[ki] is (e, a, f) with e
            # the index paired with k, so (k c | d a) is K3[c, d, a].
            if has_vvv:
                A1 = self.r_ia_a1[ik]
                einsums.einsum("a <- cd ; cda", A1[:, 0], self._Tt2(ki),
                               self.B.K_ivvv[ki])
                la.gemm(1.0, self._S(ki, ii), A1, 1.0, self._R1(i), trans_a=True)
            # Eq. 90 C: F_kc u_{ik}^{ac}.
            if has_fkc:
                C = self.r_ia_c[ik]
                la.gemm(1.0, self._Tt2(ik), self.Fkc[ki], 0.0, C)
                la.gemm(1.0, self._S(ik, ii), C, 1.0, self._R1(i), trans_a=True)

        # Eq. 88 A2: -u_{ki}^{cd} (k c | l d) t_l^a.
        for ik, i, ki, ii, l, kl in self._plan.r_ia_a2:
            a, b, s = self.r_ia_a2_tmp[ik, l]
            n_ki = self.cc.n_pno[ki]
            la.gemm(1.0, self._S(kl, ki), self._Tt2(ki), 0.0, a[:, :n_ki])
            la.gemm(1.0, a[:, :n_ki], self._S(ki, kl), 0.0, b)
            la.dot(s, b, self._K2(kl))
            la.ger(-1.0, self._proj(ii, l)[0, :], s, self._R1(i))

        # Eq. 89 B: -u_{kl}^{ac} [(k i | l c) + t_i^b (k b | l c)].
        for kl, members in self._plan.r_ia_b:
            K_kilc, B_ai = self.r_ia_b_tmp[kl]
            la.axpby(1.0, self.B.K_mibj[kl], 0.0, K_kilc)
            la.gemm(1.0, self.T_n[kl], self._K2(kl), 1.0, K_kilc)
            # Held transposed as (a_kl, i_kl) so that one LMO's slab is a
            # contiguous column rather than a strided row.
            la.gemm(1.0, self._Tt2(kl), K_kilc, 0.0, B_ai, trans_b=True)
            for i_kl, i, ii in members:
                la.gemm(-1.0, self._S(kl, ii), B_ai[:, i_kl:i_kl + 1], 1.0,
                        self._R1(i), trans_a=True)

    # -- Jiang Eq. 19, 75-81 -----------------------------------------------

    def _plan_R_iajb(self):
        """The doubles residual's four work lists.

        psi4 assembles the residual in two containers, and which container a
        term goes in is a property of the term rather than of the pair: ``R``
        collects the terms that are ALREADY symmetric under ``(ij, ab) <-> (ji,
        ba)`` and so are computed once for ``i <= j`` and mirrored, while ``Rn``
        collects the rest, which are symmetrized at the end by ``R[ij] += Rn[ij]
        + Rn[ji]^T``. Getting a term into the wrong container double-counts it.
        """
        cc = self._cc_maps()
        sym, sym_kl, non, non_g = [], [], [], []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            if not self.is_strong(ij):
                continue
            if i <= j:
                sym.append((ij, i, j, int(cc["ij_to_ji"][ij])))
                lmos = cc["lmos"][ij]
                for k_ij, k in enumerate(lmos):
                    for l_ij, l in enumerate(lmos):
                        kl = int(cc["i_j_to_ij"][k, l])
                        if kl != -1 and cc["n_pno"][kl]:
                            sym_kl.append((ij, k_ij, l_ij, kl))
            # Eq. 78 (C) and Eq. 79 (D) share the neighbour walk but resolve
            # different partner pairs from it.
            cs, ds = [], []
            for k_ij, k in enumerate(cc["lmos"][ij]):
                ki = int(cc["i_j_to_ij"][k, i])
                kj = int(cc["i_j_to_ij"][k, j])
                if (ki != -1 and kj != -1 and cc["n_pno"][ki] and cc["n_pno"][kj]
                        and self.B.J_ikac[ij] is not None
                        and self.B.J_ikac[ij][k_ij] is not None):
                    cs.append((k_ij, ki, kj))
                ik = int(cc["i_j_to_ij"][i, k])
                jk = int(cc["i_j_to_ij"][j, k])
                if (ik != -1 and jk != -1 and cc["n_pno"][ik] and cc["n_pno"][jk]
                        and self.B.K_iakc[ij] is not None
                        and self.B.K_iakc[ij][k_ij] is not None
                        and self.B.J_ikac[ij][k_ij] is not None):
                    ds.append((k_ij, ik, jk))
            non.append((ij, i, j, cs, ds))
            gs = []
            for k in range(self.naocc):
                ik = int(cc["i_j_to_ij"][i, k])
                if ik != -1 and cc["n_pno"][ik]:
                    gs.append((k, ik))
            non_g.append((ij, i, j, gs))
        return sym, sym_kl, non, non_g

    def _emit_R_iajb_zero(self):
        """Clear both residual containers. Their own graph, so nothing races it."""
        for R, Rn in zip(self.R_all, self.Rn_all):
            la.scale(0.0, R)
            la.scale(0.0, Rn)

    def _emit_R_iajb_symmetric(self):
        """Eq. 75-77 and Eq. 80: the terms already symmetric in ``(ij, ab)``.

        All four are accumulated into one per-pair buffer and then added to
        ``R[ij]`` and, transposed, to ``R[ji]``, rather than added four times to
        each. Eq. 77 and Eq. 85 share the double walk over the pair's
        neighbours, and ``u_{kl} (k c | l d)^T`` in Eq. 85 depends only on
        ``kl``, so it is hoisted to one product per pair.
        """
        cc = self.cc
        # The Eq. 85 factor, shared by every pair whose neighbour walk reaches kl.
        for kl in self._plan.live:
            la.gemm(1.0, self._Tt2(kl), self._K2(kl), 0.0, self.E_kl[kl],
                    trans_b=True)

        for ij, i, j, ji in self._plan.r_sym:
            acc = self.r_sym_acc[ij]
            Qma = self.B.qma(cc.ij_to_ji, ij)
            Qab = self.B.qab(cc.ij_to_ji, ij)
            nq, nk, na = ten.shape(Qma)

            # Eq. 75: the dressed exchange operator.
            la.gemm(1.0, self.i_Qa_t1[ij], self.i_Qa_t1[ji], 0.0, acc,
                    trans_a=True)

            # Eq. 76, with the Eq. 93 dressing B~^Q_{ab} = B^Q_{ab} - t_k^a B^Q_{kb}.
            Qab_t1 = self._shared("Qab_t1", [nq, na, na])
            la.axpby(1.0, Qab, 0.0, Qab_t1)
            einsums.einsum("qab <- ka ; qkb", Qab_t1, self.T_n[ij], Qma,
                           c_pf=1.0, ab_pf=-1.0)
            W = self._shared("AW", [nq, na, na])
            einsums.einsum("qad <- qac ; cd", W, Qab_t1, self._T2(ij))
            einsums.einsum("ab <- qad ; qbd", acc, W, Qab_t1, c_pf=1.0)

            # Eq. 77's accumulator and Eq. 85's F''_bc, which the double walk
            # below fills and Eq. 80 then reads.
            la.scale(0.0, self.r_sym_B[ij])
            la.axpby(1.0, self.Fab[ij], 0.0, self.F_bc[ij])

        # Eq. 77 (B) and Eq. 85 (F_bc), the shared double walk.
        for ij, k_ij, l_ij, kl in self._plan.r_sym_kl:
            S_kl = self._S(kl, ij)
            n_ij = self.cc.n_pno[ij]
            n_kl = self.cc.n_pno[kl]
            h1, h2, tmp = self.r_sym_kl_tmp[ij]
            a = h1[:, :n_kl]
            la.gemm(1.0, S_kl, self._T2(kl), 0.0, a, trans_a=True)
            la.gemm(1.0, a, S_kl, 0.0, tmp)
            la.ger(1.0, tmp.reshape_view([n_ij * n_ij]),
                   self.beta[ij][k_ij:k_ij + 1, l_ij],
                   self.r_sym_B[ij].reshape_view([n_ij * n_ij, 1]))
            b = h2[:, :n_kl]
            la.gemm(1.0, S_kl, self.E_kl[kl], 0.0, b, trans_a=True)
            la.gemm(-1.0, b, S_kl, 1.0, self.F_bc[ij])

        for ij, i, j, ji in self._plan.r_sym:
            acc = self.r_sym_acc[ij]
            la.axpby(1.0, self.r_sym_B[ij], 1.0, acc)
            # Eq. 80, already carrying its own permutation.
            la.gemm(1.0, self._T2(ij), self.F_bc[ij], 1.0, acc, trans_b=True)
            la.gemm(1.0, self.F_bc[ij], self._T2(ij), 1.0, acc)
            la.axpby(1.0, acc, 1.0, self._pair(self.R_all, ij))
            if i != j:
                la.axpby(1.0, cg.permute_view(acc, [1, 0]), 1.0,
                         self._pair(self.R_all, ji))

    def _emit_R_iajb_nonsymmetric(self):
        """Eq. 78, 79 and 81: the terms that carry their own permutation."""
        for ij, i, j, cs, ds in self._plan.r_non:
            C, D = self.r_non_C[ij], self.r_non_D[ij]
            la.scale(0.0, C)
            la.scale(0.0, D)

            # Eq. 78 (C).
            for k_ij, ki, kj in cs:
                n_kj = self.cc.n_pno[kj]
                total = self.r_non_tot[ij][:, :n_kj]
                la.axpby(1.0, self.B.J_ikac[ij][k_ij], 0.0, total)
                bridge = self.r_non_bridge[ij][:, :self.cc.n_pno[ki]]
                la.gemm(1.0, self._S(ij, ki), self.gamma[ki], 0.0, bridge)
                la.gemm(1.0, bridge, self._S(ki, kj), 1.0, total)
                half = self.r_non_half[ij][:, :n_kj]
                la.gemm(1.0, total, self._T2(kj), 0.0, half, trans_b=True)
                la.gemm(-1.0, half, self._S(kj, ij), 1.0, C)

            # Eq. 79 (D).
            for k_ij, ik, jk in ds:
                n_jk = self.cc.n_pno[jk]
                total = self.r_non_tot[ij][:, :n_jk]
                la.axpby(1.0, self.M_D[ij][k_ij], 0.0, total)
                bridge = self.r_non_bridge[ij][:, :self.cc.n_pno[ik]]
                la.gemm(1.0, self._S(ij, ik), self.delta[ik], 0.0, bridge)
                la.gemm(1.0, bridge, self._S(ik, jk), 1.0, total)
                half = self.r_non_half[ij][:, :n_jk]
                la.gemm(1.0, total, self._Tt2(jk), 0.0, half, trans_b=True)
                la.gemm(1.0, half, self._S(jk, ij), 1.0, D)

            Rn = self._pair(self.Rn_all, ij)
            la.axpby(0.5, C, 1.0, Rn)
            la.axpby(1.0, cg.permute_view(C, [1, 0]), 1.0, Rn)
            la.axpby(0.5, D, 1.0, Rn)

        # Eq. 81 (G). Its prefactor is an element of F'' rather than a constant,
        # so the accumulation is a rank-1 update against a length-1 slice: the
        # captured spelling of "scalar times matrix".
        for ij, i, j, gs in self._plan.r_non_g:
            n_ij = self.cc.n_pno[ij]
            G = self.r_non_G[ij]
            la.scale(0.0, G)
            for k, ik in gs:
                a, tmp = self.r_non_g_tmp[ij]
                n_ik = self.cc.n_pno[ik]
                la.gemm(1.0, self._S(ij, ik), self._T2(ik), 0.0, a[:, :n_ik])
                la.gemm(1.0, a[:, :n_ik], self._S(ik, ij), 0.0, tmp)
                la.ger(-1.0, tmp.reshape_view([n_ij * n_ij]),
                       self.Fkj2[k:k + 1, j], G.reshape_view([n_ij * n_ij, 1]))
            la.axpby(1.0, G, 1.0, self._pair(self.Rn_all, ij))

    def _emit_R_iajb_combine(self):
        """``R[ij] += Rn[ij] + Rn[ji]^T``.

        The untransposed half is one operation per BUCKET rather than per pair:
        both containers share a layout, weak pairs were never written into
        ``Rn``, and the padding of both is identically zero, so adding the whole
        store is the same arithmetic as adding pair by pair.

        The transposed half is one accumulate per pair through a transposing
        view, so it needs no scratch. Pair ``ji`` always has the same PNO count
        as ``ij`` and therefore sits in the same bucket, so the whole family is
        a slot permutation composed with a transpose - one ``gather`` per bucket
        rather than one operation per pair. That is an emitter change and it
        belongs to the performance campaign; it is recorded in "Observed pass
        candidates".
        """
        for R, Rn in zip(self.R_all, self.Rn_all):
            la.axpby(1.0, Rn, 1.0, R)
        for ij, i, j, ji in self._plan.r_combine:
            la.axpby(1.0, cg.permute_view(self._pair(self.Rn_all, ji), [1, 0]),
                     1.0, self._pair(self.R_all, ij))

    # -- Jiang Eq. 45 -------------------------------------------------------

    def _emit_energy(self):
        """Eq. 45: ``E = (t_{ij}^{ab} + t_i^a t_j^b) L_{ij}^{ab}``.

        The weak pairs' contribution is accumulated separately rather than
        skipped, because psi4 computes it here and then subtracts it: their
        doubles are frozen at the converged LMP2 values, but their SINGLES
        contribution is live, which is why the published ``DLPNO LMP2 WEAK PAIR
        ENERGY`` is not the value ``recompute_pnos`` printed earlier.
        """
        la.scale(0.0, self.e_total)
        la.scale(0.0, self.e_weak)
        for ij, i, j, strong, has_singles in self._plan.energy:
            tau = self.tau[ij]
            la.axpby(1.0, self._T2(ij), 0.0, tau)
            if has_singles:
                la.ger(1.0, self._proj(ij, i)[0, :], self._proj(ij, j)[0, :], tau)
            s = self.energy_scalar[ij]
            la.dot(s, tau, self.B.L_iajb[ij])
            la.axpby(1.0, s, 1.0, self.e_total)
            if not strong:
                la.axpby(1.0, s, 1.0, self.e_weak)

    # -- Jiang Eq. 103-104 --------------------------------------------------

    def _emit_step(self):
        """The Jacobi update, both amplitude families, over whole stores.

        The step is materialized rather than fused into the update because it is
        DIIS's error vector and has to exist as its own tensor for
        :func:`einsums.graph.diis` to snapshot between replays. The padding
        stays exactly zero: the residual's padding is zero and the denominator's
        is one.
        """
        la.direct_division(-1.0, self.R1_all, self.D1_all, 0.0, self.step1_all)
        la.axpby(1.0, self.step1_all, 1.0, self.T1_all)
        for R, D, S, T in zip(self.R_all, self.D_all, self.step_all,
                              self.cc.T_all):
            la.direct_division(-1.0, R, D, 0.0, S)
            la.axpby(1.0, S, 1.0, T)

    def _emit_antisymmetrize(self):
        """``Tt = 2 T - T^T``, every pair in two operations per bucket."""
        for Tt, T in zip(self.cc.Tt_all, self.cc.T_all):
            einsums.permute("abp <- bap", Tt, T)
            la.axpby(2.0, T, -1.0, Tt)

    # -- planning ------------------------------------------------------------

    def _cc_maps(self):
        """The sparsity maps the plan reads, resolved once."""
        cc = self.cc
        return {
            "live": [ij for ij in range(cc.n_lmo_pairs) if cc.n_pno[ij]],
            "n_pno": cc.n_pno,
            "ij_to_i_j": cc.ij_to_i_j,
            "i_j_to_ij": cc.i_j_to_ij,
            "ij_to_ji": cc.ij_to_ji,
            "lmos": cc.lmopair_to_lmos,
            "lmos_dense": cc.lmopair_to_lmos_dense,
        }

    def plan(self):
        """Derive every work list, once, as pure integer bookkeeping.

        Not one floating-point array is touched here and nothing depends on the
        value of an integral: this is where the method's index logic lives, and
        separating it from the emission is what lets the emission be replaced
        wholesale by a batched one without re-deriving a single map.

        Each list is also classified by shape class, on the same key the LMP2
        planner uses - ``(bucket of ij, bucket of the partner)`` - extended by
        the neighbour count where a term's shape depends on it. Nothing reads
        the classes yet; they are the vocabulary the batched emitter will group
        on, and deriving them now is what M4 owes M8.
        """
        t0 = time.perf_counter()
        maps = self._cc_maps()
        p = _Plan()
        p.live = maps["live"]
        p.projection_rows, p.projections = self._plan_projections()
        p.t1_ints = self._plan_t1_ints()
        (p.fock_occupied, p.fock_bar, p.fock_kj, p.fock_kc) = self._plan_t1_fock()
        p.beta = self._plan_beta()
        p.gamma_head, p.gamma_tail = self._plan_gamma()
        p.delta_head, p.delta_tail = self._plan_delta()
        p.fkj2 = self._plan_Fkj_double_tilde()
        p.r_ia_ik, p.r_ia_a2, p.r_ia_b = self._plan_R_ia()
        p.r_sym, p.r_sym_kl, p.r_non, p.r_non_g = self._plan_R_iajb()
        p.r_combine = [(ij, *maps["ij_to_i_j"][ij], int(maps["ij_to_ji"][ij]))
                       for ij in maps["live"] if self.is_strong(ij)]
        p.energy = [(ij, i, j, self.is_strong(ij),
                     bool(maps["n_pno"][self.diag(i)]
                          and maps["n_pno"][self.diag(j)]))
                    for ij in maps["live"]
                    for i, j in [maps["ij_to_i_j"][ij]]]
        p.classify(self.cc.layout.bucket_of, self.cc.n_pno,
                   self.cc.lmopair_to_lmos)
        self._plan = p
        self.t_plan = time.perf_counter() - t0
        return p

    # -- allocation ----------------------------------------------------------

    def _shared(self, role, shape_):
        """A slice of the one buffer of its role, sized to the largest pair.

        The rank-3 intermediates carry the auxiliary index, so one per pair is
        of order 15 MiB times the pair count - larger than every integral in the
        calculation put together. One buffer per role serializes the chains that
        read it, which is the correctness cut's accepted trade and exactly what
        ``ScratchPrivatization`` exists to undo.
        """
        buf = self._shared_buf[role]
        flat = int(np.prod(shape_))
        return self._keepv(buf[:flat].reshape_view(list(shape_)))

    def _keepv(self, v):
        self._keep.append(v)
        return v

    def _element(self, M, i, j):
        """The ``(1, 1)`` view of one matrix element, memoized."""
        key = (id(M), i, j)
        hit = self._view_cache.get(key)
        if hit is None:
            hit = M[i:i + 1, j:j + 1]
            self._view_cache[key] = hit
        return hit

    def _R1(self, i):
        """LMO ``i``'s singles residual as a captured ``(n, 1)`` view."""
        hit = self._R1_view.get(i)
        if hit is None:
            n = self.cc.n_pno[self.diag(i)]
            hit = self.R1_all[:n, i:i + 1]
            self._R1_view[i] = hit
        return hit

    def _allocate(self):
        """Every tensor the iteration writes, allocated once.

        The emitters below create only views and record operations; nothing here
        is reallocated per iteration, which is the standing advantage over
        psi4's per-iteration ``make_shared`` traffic in ``lccsd_iterations``.
        """
        cc = self.cc
        p = self._plan
        naocc = self.naocc
        n_pno = cc.n_pno
        F = self.F_lmo_np

        self._S_cache = {}
        self._view_cache = {}
        self._T1_view = {}
        self._R1_view = {}

        def z(name, shape_):
            return ten.zeros(name, shape_)

        # -- the singles, on the diagonal pairs -----------------------------
        # One padded block store rather than a list of blocks, so DIIS
        # extrapolates the singles the same way it extrapolates the doubles and
        # the Jacobi step is one operation for the whole family.
        diag = [self.diag(i) for i in range(naocc)]
        M1 = max([n_pno[ii] for ii in diag] + [1])
        self.T1_all = z("T1", [M1, naocc])
        self.R1_all = z("R1", [M1, naocc])
        self.step1_all = z("step T1", [M1, naocc])
        self.D1_all = z("D1", [M1, naocc])
        d1 = ten.view(self.D1_all)
        d1[...] = 1.0
        for i, ii in enumerate(diag):
            n = n_pno[ii]
            if n:
                d1[:n, i] = ten.view(cc.e_pno[ii]) - F[i, i]
        #: The singles as numpy views, one per LMO, for callers that inject or
        #: read amplitudes (``check_ccsd_defect.py``'s bridge).
        t1 = ten.view(self.T1_all)
        self.T_ia = [t1[:n_pno[ii], i:i + 1] for i, ii in enumerate(diag)]

        # -- the doubles residual, and both Jacobi denominators --------------
        self.R_all = cc.new_pair_stores("R")
        self.Rn_all = cc.new_pair_stores("Rn")
        self.step_all = cc.new_pair_stores("step T2")
        self.D_all = cc.new_pair_stores("D")
        for D in self.D_all:
            ten.view(D)[...] = 1.0
        for ij, (i, j) in enumerate(cc.ij_to_i_j):
            n = n_pno[ij]
            if not n:
                continue
            e = ten.view(cc.e_pno[ij])
            cc.pair_block(self.D_all, ij)[:n, :n] = (
                e[:, None] + e[None, :] - F[i, i] - F[j, j])

        # -- Eq. 70 and the other projections -------------------------------
        self.T_n = {}
        for ij in p.live:
            nk = len(cc.lmopair_to_lmos[ij])
            self.T_n[ij] = z(f"T_n ({ij})", [max(nk, 1), n_pno[ij]])
        self._proj_row = {}
        for ij, n_ij, n, _nn in p.projection_rows:
            self._proj_row[ij, n] = self._keepv(self.T_n[ij][n_ij:n_ij + 1, :])
        for ij, x, _xx in p.projections:
            if (ij, x) not in self._proj_row:
                t = z(f"t({x}) in {ij}", [1, n_pno[ij]])
                self._keep.append(t)
                self._proj_row[ij, x] = t

        # -- Eq. 91-92 -------------------------------------------------------
        self.i_Qk_t1 = [None] * cc.n_lmo_pairs
        self.i_Qa_t1 = [None] * cc.n_lmo_pairs
        for ij, _ji, _i in p.t1_ints:
            self.i_Qk_t1[ij] = z(f"B~(Q|k i) {ij}", ten.shape(self.B.i_Qk[ij]))
            self.i_Qa_t1[ij] = z(f"B~(Q|a i) {ij}", ten.shape(self.B.i_Qa[ij]))

        # -- Eq. 94-101 -------------------------------------------------------
        self.Fij_bar = z("F_bar (occ)", [naocc, naocc])
        self.Fkj = z("F~ (occ)", [naocc, naocc])
        self.Fkj2 = z("F'' (occ)", [naocc, naocc])
        self.fock_occ_scalar = {ij: (ten.scalar(), ten.scalar())
                                for ij, _, _, _ in p.fock_occupied}
        self.fock_kj_scalar = {ij: ten.scalar() for ij, *_ in p.fock_kj}
        self.Fkc_bar = [None] * cc.n_lmo_pairs
        self.Fab_bar = [None] * cc.n_lmo_pairs
        self.Fab = [None] * cc.n_lmo_pairs
        self.e_diag = [None] * cc.n_lmo_pairs
        self.Fai_bar = [None] * naocc
        self.Fai = [None] * naocc
        self.gamma_Q = {}
        self.fock_ai_tmp = [None] * naocc
        self.fock_ai_col = [None] * naocc
        for ij, i, j in p.fock_bar:
            nq = ten.shape(self.B.i_Qk[ij])[0]
            nk, na = ten.shape(self.T_n[ij])
            self.Fkc_bar[ij] = z(f"F_bar (kc) {ij}", [nk, na])
            self.Fab_bar[ij] = z(f"F_bar (ab) {ij}", [na, na])
            self.Fab[ij] = z(f"F~ (ab) {ij}", [na, na])
            self.e_diag[ij] = z(f"e_pno diag {ij}", [na, na])
            ten.view(self.e_diag[ij])[...] = np.diag(ten.view(cc.e_pno[ij]))
            self.gamma_Q[ij] = z(f"gamma_Q {ij}", [nq])
            if i == j:
                self.Fai_bar[i] = z(f"F_bar (ai) {i}", [na, 1])
                self.Fai[i] = z(f"F~ (ai) {i}", [na, 1])
                self.fock_ai_tmp[i] = z(f"F~ (ai) tmp {i}", [nk, 1])
                self.fock_ai_col[i] = z(f"F_bar column {i}", [nk, 1])
        self.Fkc = [None] * cc.n_lmo_pairs
        for ij in p.live:
            self.Fkc[ij] = z(f"F~ (kc) {ij}", [n_pno[ij], 1])
        self.fock_kc_tmp = {(ij, ik): z(f"F~ (kc) tmp {ij},{ik}", [n_pno[ik], 1])
                            for ij, ik, _k in p.fock_kc}

        # -- Eq. 82-86 --------------------------------------------------------
        self.beta = [None] * cc.n_lmo_pairs
        for ij, _ji in p.beta:
            nk = ten.shape(self.T_n[ij])[0]
            self.beta[ij] = z(f"beta {ij}", [nk, nk])
        self.gamma = [None] * cc.n_lmo_pairs
        self.delta = [None] * cc.n_lmo_pairs
        self.M_delta = [None] * cc.n_lmo_pairs
        for ij in p.live:
            self.gamma[ij] = z(f"gamma {ij}", [n_pno[ij], n_pno[ij]])
            self.delta[ij] = z(f"delta {ij}", [n_pno[ij], n_pno[ij]])
            # Eq. 84a's integral combination is constant.
            self.M_delta[ij] = ten.from_numpy(
                f"2K-J {ij}",
                2.0 * ten.view(self.B.K_mibj[ij]) - ten.view(self.B.J_ijmb[ij]))

        # Per-pair scratch is sized to the WIDEST partner that pair's work list
        # reaches, and each use slices the part it needs. One buffer per pair
        # and role rather than one per record: the records of a pair accumulate
        # into that pair's intermediate and are therefore serialized by the
        # dependency anyway, so sharing costs no parallelism and saves an
        # allocation per neighbour.
        gw = _widest(p.gamma_tail, 0, [(2, n_pno), (4, n_pno)])
        self.gamma_tmp, self.gamma_tmp_d = {}, {}
        for ki, (wide, tall) in gw.items():
            n = n_pno[ki]
            self.gamma_tmp[ki] = (z(f"83c a {ki}", [wide, 1]),
                                  z(f"83c b {ki}", [n, 1]),
                                  z(f"83d c {ki}", [n, wide]))
            self.gamma_tmp_d[ki] = (z(f"83d a {ki}", [tall, wide]),
                                    z(f"83d b {ki}", [tall, wide]))
        dw = _widest(p.delta_tail, 0, [(3, n_pno), (4, n_pno)])
        self.delta_tmp, self.delta_tmp_d = {}, {}
        for ik, (wide, tall) in dw.items():
            n = n_pno[ik]
            # Column-shaped, mirroring 83c's; see _emit_delta on why.
            self.delta_tmp[ik] = (z(f"84c a {ik}", [wide, 1]),
                                  z(f"84c b {ik}", [n, 1]),
                                  z(f"84d c {ik}", [n, wide]))
            self.delta_tmp_d[ik] = (z(f"84d a {ik}", [tall, wide]),
                                    z(f"84d b {ik}", [tall, wide]))

        self.fkj2_tmp = {}
        for ij, _i, _j, l, li, lj in p.fkj2:
            self.fkj2_tmp[ij, l] = (z(f"86 a {ij},{l}", [n_pno[li], n_pno[lj]]),
                                    z(f"86 b {ij},{l}", [n_pno[li], n_pno[li]]),
                                    ten.scalar())

        # -- Eq. 87a-90 -------------------------------------------------------
        self.r_ia_a1 = {ik: z(f"88 A1 {ik}", [n_pno[ki], 1])
                        for ik, _i, _k, ki, _ii, has, _f in p.r_ia_ik if has}
        self.r_ia_c = {ik: z(f"90 C {ik}", [n_pno[ik], 1])
                       for ik, _i, _k, _ki, _ii, _h, has in p.r_ia_ik if has}
        self.r_ia_a2_tmp = {}
        for ik, _i, ki, _ii, l, kl in p.r_ia_a2:
            self.r_ia_a2_tmp[ik, l] = (z(f"88 A2 a {ik},{l}", [n_pno[kl], n_pno[ki]]),
                                       z(f"88 A2 b {ik},{l}", [n_pno[kl], n_pno[kl]]),
                                       ten.scalar())
        self.r_ia_b_tmp = {}
        for kl, _members in p.r_ia_b:
            nk, na = ten.shape(self.T_n[kl])
            self.r_ia_b_tmp[kl] = (z(f"89 K {kl}", [nk, na]),
                                   z(f"89 B {kl}", [na, nk]))

        # -- Eq. 19, 75-81 -----------------------------------------------------
        self.E_kl = [None] * cc.n_lmo_pairs
        for ij in p.live:
            self.E_kl[ij] = z(f"85 u K^T {ij}", [n_pno[ij], n_pno[ij]])
        self.r_sym_acc = {}
        self.r_sym_B = {}
        self.F_bc = {}
        self.r_sym_kl_tmp = {}
        for ij, i, j, _ji in p.r_sym:
            n = n_pno[ij]
            self.r_sym_acc[ij] = z(f"R sym {ij}", [n, n])
            self.r_sym_B[ij] = z(f"77 B {ij}", [n, n])
            self.F_bc[ij] = z(f"85 F'' bc {ij}", [n, n])
        for ij, (wide,) in _widest(p.r_sym_kl, 0, [(3, n_pno)]).items():
            n = n_pno[ij]
            self.r_sym_kl_tmp[ij] = (z(f"77 h1 {ij}", [n, wide]),
                                     z(f"85 h2 {ij}", [n, wide]),
                                     z(f"77 tmp {ij}", [n, n]))

        self.M_D = [None] * cc.n_lmo_pairs
        for ij in p.live:
            if self.B.K_iakc[ij] is None or self.B.J_ikac[ij] is None:
                continue
            self.M_D[ij] = [
                None if (K is None or J is None) else ten.from_numpy(
                    f"2K-J np {ij}", 2.0 * ten.view(K) - ten.view(J))
                for K, J in zip(self.B.K_iakc[ij], self.B.J_ikac[ij])]

        self.r_non_C, self.r_non_D, self.r_non_G = {}, {}, {}
        self.r_non_tot, self.r_non_bridge, self.r_non_half = {}, {}, {}
        self.r_non_g_tmp = {}
        for ij, _i, _j, cs, ds in p.r_non:
            n = n_pno[ij]
            self.r_non_C[ij] = z(f"78 C {ij}", [n, n])
            self.r_non_D[ij] = z(f"79 D {ij}", [n, n])
            wide = max([n_pno[r[2]] for r in cs] + [n_pno[r[2]] for r in ds]
                       + [n_pno[r[1]] for r in cs] + [n_pno[r[1]] for r in ds]
                       + [1])
            self.r_non_tot[ij] = z(f"78/79 total {ij}", [n, wide])
            self.r_non_bridge[ij] = z(f"78/79 bridge {ij}", [n, wide])
            self.r_non_half[ij] = z(f"78/79 half {ij}", [n, wide])
        for ij, _i, _j, gs in p.r_non_g:
            n = n_pno[ij]
            self.r_non_G[ij] = z(f"81 G {ij}", [n, n])
            wide = max([n_pno[ik] for _k, ik in gs] + [1])
            self.r_non_g_tmp[ij] = (z(f"81 a {ij}", [n, wide]),
                                    z(f"81 tmp {ij}", [n, n]))

        # -- Eq. 45 ------------------------------------------------------------
        self.tau = {ij: z(f"tau {ij}", [n_pno[ij], n_pno[ij]]) for ij in p.live}
        self.energy_scalar = {ij: ten.scalar() for ij in p.live}
        self.e_total = ten.scalar("E(iteration)")
        self.e_weak = ten.scalar("E(weak)")

        # -- the shared rank-3 scratch ------------------------------------------
        self._shared_buf = {}
        for role, dims in self._shared_sizes().items():
            self._shared_buf[role] = ten.empty(f"scratch {role}", [dims])

    def _shared_sizes(self):
        """How large each shared rank-3 buffer has to be, in elements."""
        cc = self.cc
        p = self._plan
        need = {"V": 1, "U": 1, "y": 1, "bW": 1, "Qab_t1": 1, "AW": 1}
        for ij, _i, _j in p.fock_bar:
            nq = ten.shape(self.B.i_Qk[ij])[0]
            nk, na = ten.shape(self.T_n[ij])
            need["V"] = max(need["V"], nq * nk * nk)
            need["U"] = max(need["U"], nq * na * nk)
            need["y"] = max(need["y"], nq * na)
        for ij, _ji in p.beta:
            nq, nk, na = ten.shape(self.B.qma(cc.ij_to_ji, ij))
            need["bW"] = max(need["bW"], nq * nk * na)
        for ij, _i, _j, _ji in p.r_sym:
            nq, na, _ = ten.shape(self.B.qab(cc.ij_to_ji, ij))
            need["Qab_t1"] = max(need["Qab_t1"], nq * na * na)
            need["AW"] = max(need["AW"], nq * na * na)
        return need

    # -- capture -------------------------------------------------------------

    def capture(self):
        """Record every phase into its own graph, once.

        The order below is the order they replay in, and it is psi4's order in
        ``lccsd_iterations``. Boundaries are liberal on purpose: each one is a
        point where a slice-versus-whole aliasing question cannot arise, and
        this iteration is made of slices.
        """
        t0 = time.perf_counter()
        phases = [
            ("Eq. 70 projections", self._emit_projections),
            ("Eq. 91-92 t1 ints", self._emit_t1_ints),
            ("Eq. 98 F_bar (occ)", self._emit_t1_fock_occupied),
            ("Eq. 99-101 F_bar", self._emit_t1_fock_bar),
            ("Eq. 94-97 F~", self._emit_t1_fock_tilde),
            ("Eq. 82 beta", self._emit_beta),
            ("Eq. 83 gamma", self._emit_gamma),
            ("Eq. 84 delta", self._emit_delta),
            ("Eq. 86 F''", self._emit_Fkj_double_tilde),
            ("Eq. 87-90 R_ia", self._emit_R_ia),
            ("residual containers", self._emit_R_iajb_zero),
            ("Eq. 75-77, 80, 85", self._emit_R_iajb_symmetric),
            ("Eq. 78-79, 81", self._emit_R_iajb_nonsymmetric),
            ("Eq. 19 combine", self._emit_R_iajb_combine),
        ]
        self._graphs = []
        for name, emit in phases:
            g = cg.Graph(f"lccsd: {name}")
            with cg.capture(g):
                emit()
            self._graphs.append(g)

        self._g_step = cg.Graph("lccsd: Eq. 103-104 step")
        with cg.capture(self._g_step):
            self._emit_step()
        self._g_antisym = cg.Graph("lccsd: antisymmetrize")
        with cg.capture(self._g_antisym):
            self._emit_antisymmetrize()
        self._g_energy = cg.Graph("lccsd: Eq. 45 energy")
        with cg.capture(self._g_energy):
            self._emit_energy()

        self.t_capture = time.perf_counter() - t0
        return self

    def num_nodes(self):
        """Nodes across every captured graph, which is what the replay costs."""
        return (sum(g.num_nodes() for g in self._graphs)
                + self._g_step.num_nodes() + self._g_antisym.num_nodes()
                + self._g_energy.num_nodes())

    # -- replay ---------------------------------------------------------------

    def evaluate_residuals(self):
        """Replay everything up to but not including the amplitude update.

        Every phase before the Jacobi step is a pure function of the amplitudes,
        so this evaluates the port's equations at whatever amplitudes are
        currently in the stores. That is what makes the residual-level
        comparison against the canonical oracle possible at probe amplitudes
        rather than only at the solution; see ``check_ccsd_defect.py``.
        """
        self.prepare()
        for g in self._graphs:
            g.execute()
        return self.residuals()

    def residuals(self):
        """The current residuals as numpy, laid out the way the oracle wants."""
        cc = self.cc
        R1 = ten.view(self.R1_all)
        R_ia = [R1[:cc.n_pno[self.diag(i)], i:i + 1].copy()
                for i in range(self.naocc)]
        R_iajb = {}
        for ij in self._plan.live:
            n = cc.n_pno[ij]
            R_iajb[ij] = cc.pair_block(self.R_all, ij)[:n, :n].copy()
        return R_ia, R_iajb

    def _residual_norms(self):
        """Max RMS over the singles and over the doubles, on logical blocks only.

        The padding is identically zero and would otherwise dilute the norm.
        """
        cc = self.cc
        R1 = ten.view(self.R1_all)
        rms_s = 0.0
        for i in range(self.naocc):
            n = cc.n_pno[self.diag(i)]
            if n:
                block = R1[:n, i]
                rms_s = max(rms_s, float(np.sqrt(np.mean(block ** 2))))
        rms_d = 0.0
        for ij in self._plan.live:
            n = cc.n_pno[ij]
            block = cc.pair_block(self.R_all, ij)[:n, :n]
            rms_d = max(rms_d, float(np.sqrt(np.mean(block ** 2))))
        return rms_s, rms_d

    # -- psi4 DLPNOCCSD::lccsd_iterations ------------------------------------

    def prepare(self):
        """Plan, allocate and capture. Idempotent."""
        if self._graphs is None:
            self.plan()
            self._allocate()
            self.capture()
            self._report()
        return self

    def _report(self):
        p = self._plan
        self._print("\n  ==> Local CCSD <==\n")
        self._print(
            f"    plan:     {len(p.projections)} projections "
            f"({self._n_proj_extra} outside the neighbour lists), "
            f"{len(p.r_sym_kl)} Eq. 77/85 couplings, "
            f"{p.n_classes()} shape classes over {len(p.class_report())} terms")
        self._print(f"    captured: {self.num_nodes()} nodes in "
                    f"{len(self._graphs) + 3} phase graphs "
                    f"({self.t_plan * 1e3:.0f} ms planning, "
                    f"{self.t_capture * 1e3:.0f} ms capture)")

    def iterate(self):
        """Drive the coupled-cluster equations to convergence.

        The doubles enter at the converged LMP2 values ``recompute_pnos`` left
        in the stores and the singles at zero, which is psi4's starting guess.

        Every graph is replayed under the DEFAULT executor. An OpenMP team here
        would nest the residual's GEMMs inside OpenBLAS's own threads, which is
        the hazard ``mp2.py`` documents and pays for in the last digits of the
        energy; scoping executors per phase is the answer, and it belongs to the
        campaign that measures which phases want one.
        """
        cc = self.cc
        self.prepare()

        if self.use_diis:
            self._diis = cg.diis(
                [(self.T1_all, self.step1_all)]
                + list(zip(cc.T_all, self.step_all)),
                k=cc.cut.diis_max_vecs)

        self._print(f"\n    {'iter':>4}  {'Corr. Energy':>18} {'Delta E':>12} "
                    f"{'Max R1':>10} {'Max R2':>10}")
        t0 = time.perf_counter()
        e_prev = 0.0
        for iteration in range(cc.cut.maxiter + 1):
            for g in self._graphs:
                g.execute()
            rms_s, rms_d = self._residual_norms()
            self._g_step.execute()
            if self._diis is not None:
                self._diis.step()
            self._g_antisym.execute()
            self._g_energy.execute()

            e_curr = float(ten.view(self.e_total)[0])
            e_weak = float(ten.view(self.e_weak)[0])
            self._print(f"    {iteration:>4}  {e_curr:>18.12f} "
                        f"{e_curr - e_prev:>12.3e} {rms_s:>10.3e} {rms_d:>10.3e}")
            converged = (iteration > 0
                         and abs(e_curr - e_prev) < cc.cut.e_convergence
                         and max(rms_s, rms_d) < cc.cut.r_convergence)
            e_prev = e_curr
            if converged:
                self.t_iterate = time.perf_counter() - t0
                self.n_iterations = iteration + 1
                self.e_lccsd = e_curr - e_weak
                self.de_weak = e_weak
                self._print(f"    replay:   {self.t_iterate:.3f} s over "
                            f"{self.n_iterations} iterations")
                return self
        raise RuntimeError("maximum DLPNO-CCSD iterations exceeded")


def _widest(records, owner, fields):
    """Per owner, the largest value of each field over its records.

    One pass rather than one scan of the whole list per owner. The obvious
    nesting is quadratic in the record count and at a six-monomer chain that is
    tens of millions of comparisons for an answer that is a running maximum.

    Args:
        records: the work list.
        owner: which field of a record identifies its owner.
        fields: ``(index, table)`` pairs; the width is ``table[record[index]]``,
            and a record whose index is ``-1`` does not contribute.
    """
    out = {}
    for r in records:
        row = out.get(r[owner])
        if row is None:
            row = out[r[owner]] = [1] * len(fields)
        for slot, (index, table) in enumerate(fields):
            key = r[index]
            if key != -1 and table[key] > row[slot]:
                row[slot] = table[key]
    return {k: tuple(v) for k, v in out.items()}


class _Plan:
    """The iteration's work lists, and the shape classes they fall into.

    A plain container rather than a dataclass because it is filled field by
    field as each term is planned, and because the only thing anything does with
    it is iterate over one of its lists.

    The shape classes are the part that is not just bookkeeping. Every hot
    emission here is currently one operation per record; the campaign's first
    lever is to group records whose operands have identical shapes into one
    batched call, and a class is exactly a group that can. The key is the LMP2
    planner's - the PNO bucket of the pair and of its partner - extended by the
    neighbour count for the terms whose shape depends on it. Deriving them now
    is what M4 owes M8; nothing reads them yet.
    """

    #: Per work list, the function from a record to its shape-class key.
    _KEYS = {
        "projections": lambda b, n, k, r: (b[r[0]], b[r[2]]),
        "t1_ints": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "fock_occupied": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "fock_bar": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "fock_kj": lambda b, n, k, r: (b[r[0]], b[r[3]]),
        "fock_kc": lambda b, n, k, r: (b[r[0]], b[r[1]]),
        "beta": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "gamma_head": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "gamma_tail": lambda b, n, k, r: (b[r[0]], b[r[2]]),
        "delta_head": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "delta_tail": lambda b, n, k, r: (b[r[0]], b[r[3]]),
        "fkj2": lambda b, n, k, r: (b[r[4]], b[r[5]]),
        "r_ia_ik": lambda b, n, k, r: (b[r[0]], b[r[4]]),
        "r_ia_a2": lambda b, n, k, r: (b[r[2]], b[r[5]]),
        "r_ia_b": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "r_sym": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "r_sym_kl": lambda b, n, k, r: (b[r[0]], b[r[3]]),
        "r_combine": lambda b, n, k, r: (b[r[0]],),
        "energy": lambda b, n, k, r: (b[r[0]],),
    }

    def __init__(self):
        self.live = []
        self.classes = {}
        self.cls = {}

    def classify(self, bucket_of, n_pno, lmos):
        """Intern each work list's shape classes and label every record."""
        for name, key in self._KEYS.items():
            records = getattr(self, name, None)
            if not records:
                continue
            seen, labels = {}, []
            for r in records:
                k = key(bucket_of, n_pno, lmos, r)
                labels.append(seen.setdefault(k, len(seen)))
            self.classes[name] = sorted(seen, key=seen.get)
            self.cls[name] = labels

    def n_classes(self):
        return sum(len(v) for v in self.classes.values())

    def class_report(self):
        """Per work list, how many records and how many shape classes."""
        return {name: (len(getattr(self, name)), len(cls))
                for name, cls in self.classes.items()}
