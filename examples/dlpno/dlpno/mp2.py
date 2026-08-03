#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Local MP2 in the PNO basis, ported from psi4's ``dlpno/mp2.cc``.

The local MP2 residual (Pinski et al., JCP 143, 034108, equation 13) is

    R_ij = K_ij + (e_a + e_b - F_ii - F_jj) T_ij
           - sum_k F_ik  S(ij,kj) T_kj S(ij,kj)^T
           - sum_k F_kj  S(ij,ik) T_ik S(ij,ik)^T

with one small dense block per LMO pair and a PNO overlap matrix bridging every
pair that couples through the occupied Fock matrix. That is thousands of tiny
GEMMs whose shapes and dependency pattern are fixed for the whole calculation
and change only in their *values* from iteration to iteration, which is exactly
what a ComputeGraph is for: capture once, replay per iteration.

Three graphs are captured, in the order psi4 runs them:

``residual``  R_ij from the current amplitudes, and the iteration energy
``step``      the Jacobi update T_ij -= R_ij / D_ij
``amplitude`` the antisymmetrized amplitudes Tt_ij = 2 T_ij - T_ij^T

DIIS sits between ``step`` and ``amplitude`` because psi4 extrapolates the
amplitudes before rebuilding Tt. It runs on the host: it is a solver detail over
flattened buffers rather than tensor algebra, and it writes back into the same
tensors the graphs already reference, so the replay picks the new values up. The
natural next step is to fold the whole loop into one graph with a loop node.
"""

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import DLPNOBase

__all__ = ["DLPNOMP2"]


class _DIIS:
    """Pulay extrapolation over flattened amplitude and residual vectors."""

    def __init__(self, max_vecs):
        self.max_vecs = max_vecs
        self.amplitudes = []
        self.errors = []

    def extrapolate(self, t_flat, r_flat):
        self.amplitudes.append(np.array(t_flat, copy=True))
        self.errors.append(np.array(r_flat, copy=True))
        if len(self.amplitudes) > self.max_vecs:
            # psi4 drops the largest-error vector; dropping the oldest is the
            # usual alternative and keeps this readable.
            self.amplitudes.pop(0)
            self.errors.pop(0)

        n = len(self.amplitudes)
        if n < 2:
            return t_flat

        B = np.zeros((n + 1, n + 1))
        B[:n, :n] = np.array([[e1 @ e2 for e2 in self.errors] for e1 in self.errors])
        B[n, :n] = B[:n, n] = -1.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return t_flat
        return sum(ci * ti for ci, ti in zip(c, self.amplitudes))


class DLPNOMP2(DLPNOBase):
    """DLPNO-MP2: PNO overlaps, then the local MP2 equations as a ComputeGraph."""

    def __init__(self, reference, thresholds=None, verbose=True, use_diis=True):
        super().__init__(reference, thresholds, verbose)
        self.use_diis = use_diis
        self.S_pno_ij_kj = {}
        self.S_pno_ij_ik = {}
        self.e_lmp2 = 0.0
        self.e_lmp2_os = 0.0
        self.e_lmp2_ss = 0.0
        self.n_iterations = 0

    # -- psi4 DLPNOMP2::compute_pno_overlaps -------------------------------

    def compute_pno_overlaps(self):
        """PNO overlap matrices ``S(ij, kj)`` and ``S(ij, ik)``, as one graph.

        Every pair has its own PNO basis, so coupling pair ``ij`` to pair ``kj``
        needs the overlap between the two bases. Built once and reused every
        iteration.

        **The overlaps are pre-scaled by the square root of their Fock
        prefactor.** ``GEMMBatching`` requires bit-identical ``alpha`` across a
        group, and the residual's natural form carries a different prefactor
        ``-F_ik`` on every coupling, which fragments the batches into one group
        per distinct Fock element. Since

            f * S T S^T = sign(f) * (sqrt(|f|) S) T (sqrt(|f|) S)^T

        and S is constant for the whole calculation, folding ``sqrt(|f|)`` in
        here leaves ``alpha`` at exactly +1 or -1 in the residual. The overlaps
        stop being pure overlaps, hence the name.

        Two things make this cheap. First, each pair's PNO transform is
        scattered into the FULL PAO axis, so ``X`` is zero outside its own
        domain and the domain restriction becomes implicit: the full-space
        triple product picks exactly the terms the restricted one did, and the
        per-coupling ``S_pao`` gather disappears. That gather was copying the
        whole of ``S_pao`` once per coupling, 42% of this phase.

        Second, a pair's partners are concatenated into one matrix, so the
        overlaps against ALL of them are a single GEMM:

            half[ij]  = X[ij]^T S_pao                      (one per pair)
            S_cat[ij] = half[ij] [ X[mn1] | X[mn2] | ... ] (one per pair)

        and each coupling's block is a column slice of the result rather than a
        separate allocation. That is 2 GEMMs per pair instead of 2 per coupling:
        338 graph nodes for ethanol instead of 8112, which matters because
        capture costs ~38 us a node and this phase runs once. Capturing the
        per-coupling form would have spent more on building the graph than the
        eager version spends computing.
        """
        naocc = self.ref.naocc
        F_lmo = ten.view(self.F_lmo)
        f_cut = self.cut.f_cut
        npao = ten.shape(self.C_pao)[1]

        self.S_pno_ij_kj = {}
        self.S_pno_ij_ik = {}
        self.k_couple_kj = [{} for _ in range(self.n_lmo_pairs)]
        self.k_couple_ik = [{} for _ in range(self.n_lmo_pairs)]

        # Which partners each pair couples to, in the order they are laid out.
        partners = [[] for _ in range(self.n_lmo_pairs)]  # (partner, k, is_ik, factor)
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if self.n_pno[ij] == 0:
                continue
            for k in range(naocc):
                kj = int(self.i_j_to_ij[k, j])
                if kj != -1 and i != k and abs(F_lmo[i, k]) > f_cut and self.n_pno[kj] > 0:
                    partners[ij].append((kj, k, False, -float(F_lmo[i, k])))
                ik = int(self.i_j_to_ij[i, k])
                if ik != -1 and j != k and abs(F_lmo[k, j]) > f_cut and self.n_pno[ik] > 0:
                    partners[ij].append((ik, k, True, -float(F_lmo[k, j])))

        # Each pair's PNO transform on the full PAO axis, padded to its bucket.
        X_pad = [None] * self.n_lmo_pairs
        for ij in range(self.n_lmo_pairs):
            n = self.n_pno[ij]
            if n == 0:
                continue
            Xp = ten.zeros(f"X (padded) {ij}", [npao, self.pair_dim(ij)])
            ten.view(Xp)[np.asarray(self.lmopair_to_paos[ij], dtype=int), :n] = ten.view(self.X_pno[ij])
            X_pad[ij] = Xp

        # Partner blocks side by side, and the per-column sqrt prefactor.
        self._S_cat = {}
        cat, scale = {}, {}
        for ij, plist in enumerate(partners):
            if not plist:
                continue
            widths = [self.pair_dim(p) for p, _, _, _ in plist]
            block = ten.zeros(f"X (partners) {ij}", [npao, sum(widths)])
            factors = np.empty(sum(widths))
            view = ten.view(block)
            off = 0
            for (p, _, _, factor), w in zip(plist, widths):
                view[:, off:off + w] = ten.view(X_pad[p])
                factors[off:off + w] = np.sqrt(abs(factor))
                off += w
            cat[ij] = block
            scale[ij] = factors
            self._S_cat[ij] = ten.zeros(f"S (cat) {ij}", [self.pair_dim(ij), sum(widths)])

        half = {ij: ten.zeros(f"half {ij}", [self.pair_dim(ij), npao]) for ij in cat}
        g = cg.Graph("pno overlaps")
        with cg.capture(g):
            for ij in cat:
                einsums.einsum("ab <- ca ; cb", half[ij], X_pad[ij], self.S_pao)
                einsums.einsum("ab <- ac ; cb", self._S_cat[ij], half[ij], cat[ij])
        pm = cg.PassManager()
        pm.populate_default()
        g.apply(pm)
        g.execute()

        # One vectorized scaling per pair: every coupling owns a column range.
        for ij, factors in scale.items():
            ten.view(self._S_cat[ij])[...] *= factors[np.newaxis, :]

        # Each coupling's overlap is a column slice, not its own allocation.
        for ij, plist in enumerate(partners):
            off = 0
            for p, k, is_ik, factor in plist:
                w = self.pair_dim(p)
                block = self._S_cat[ij][:, off:off + w]
                if is_ik:
                    self.S_pno_ij_ik[ij, k] = block
                    self.k_couple_ik[ij][k] = float(np.sign(factor))
                else:
                    self.S_pno_ij_kj[ij, k] = block
                    self.k_couple_kj[ij][k] = float(np.sign(factor))
                off += w

        self._print(
            f"  overlaps: {sum(len(p) for p in partners)} couplings via "
            f"{g.num_nodes()} nodes ({2 * len(cat)} captured)"
        )
        return self

    def _allocate_iteration_tensors(self):
        """Residuals, denominators, scratch, and the captured views."""
        F_lmo = ten.view(self.F_lmo)
        naocc = self.ref.naocc
        P = self.n_lmo_pairs

        self.R_all = self.new_pair_stores("R")
        # D[a,b] = e_a + e_b - F_ii - F_jj. Padded entries are left at 1 so the
        # Jacobi step divides by something finite there; every numerator that
        # reaches them is zero, so the padding stays zero for good.
        self.D_all = self.new_pair_stores("D")
        for D in self.D_all:
            ten.view(D)[...] = 1.0
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            n = self.n_pno[ij]
            if n == 0:
                continue
            D = ten.zeros("D (pair)", [n, n])
            la.outer_sum(D, [self.e_pno[ij], self.e_pno[ij]], [1.0, 1.0])
            la.shift(-float(F_lmo[i, i] + F_lmo[j, j]), D)
            self.pair_block(self.D_all, ij)[:n, :n] = ten.view(D)

        # Views handed to a captured op must outlive the graph: the graph holds
        # the tensor by slot pointer, so a view created inline in the capture
        # loop is freed the moment the expression ends and the replay then reads
        # released memory. It does not necessarily fault; it silently works
        # until the allocator reuses the block. So every view is built once,
        # here, and kept for the life of the object.
        self._R_view = [self.pair_view(self.R_all, ij) if self.n_pno[ij] else None
                        for ij in range(P)]
        self._T_view = [self.pair_view(self.T_all, ij) if self.n_pno[ij] else None
                        for ij in range(P)]
        # One scratch block per surviving (pair, k) coupling. Its shape is set
        # by BOTH pairs' buckets: S(ij,kj) is (M_ij, M_kj). Sharing scratch
        # across couplings would serialize contractions that are mutually
        # independent, and independence at one dependency level is exactly what
        # GEMMBatching groups on.
        self._tmp_kj, self._tmp_ik = {}, {}
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if self.n_pno[ij] == 0:
                continue
            M_ij = self.pair_dim(ij)
            for k in self.k_couple_kj[ij]:
                M_p = self.pair_dim(int(self.i_j_to_ij[k, j]))
                self._tmp_kj[ij, k] = ten.zeros(f"tmp_kj({ij},{k})", [M_ij, M_p])
            for k in self.k_couple_ik[ij]:
                M_p = self.pair_dim(int(self.i_j_to_ij[i, k]))
                self._tmp_ik[ij, k] = ten.zeros(f"tmp_ik({ij},{k})", [M_ij, M_p])

        # Partial accumulators. Coupling number c of a pair goes to accumulator
        # c % G at dependency level c // G, so couplings sharing a level are
        # independent even within one pair and the whole level batches together.
        # Shape (M, M, n_pairs_in_bucket, G) keeps A[:, :, :, g] a contiguous
        # rank-3 slice, which the final reduction needs (strided views cannot be
        # captured).
        self.n_acc = max(1, int(self.cut.n_accumulators))
        self.A_all = [
            ten.zeros(f"A (bucket {b}, dim {M})", [M, M, len(members), self.n_acc])
            for b, (M, members) in enumerate(zip(self.bucket_dims, self.bucket_members))
        ]
        self._A_view = [
            [self.A_all[self.bucket_of[ij]][:, :, self.slot_of[ij], g]
             for g in range(self.n_acc)] if self.n_pno[ij] else None
            for ij in range(P)
        ]
        self._A_slice = [[A[:, :, :, g] for g in range(self.n_acc)] for A in self.A_all]

        self.e_iter = ten.zeros("E(iteration)", [1])
        self._e_part = ten.zeros("E(part)", [1])

    def _capture_residual(self):
        """Graph: the local MP2 residual for every pair, plus the iteration energy.

        Everything elementwise over pairs is one rank-3 operation on the whole
        store, so the prologue, the update and the energy cost a handful of
        nodes regardless of how many pairs there are.

        The Fock coupling stays a plain loop over (pair, k) of two 2D einsums,
        which is what psi4 writes. Two details make that loop batchable, and
        both are measured in ``bench_batching.py``:

        * every block is padded to ``npno_max``, so all ``n_pairs * naocc``
          coupling GEMMs share one shape and ``GEMMBatching`` collapses them
          into a handful of ``gemm_batch`` calls;
        * they are written with ``einsum``, not ``linalg.gemm``. A 2D x 2D -> 2D
          einsum with one link index carries the ``gemm_hint`` the pass groups
          on; ``linalg.gemm`` captures as ``OpKind::Gemm`` and the pass skips it.

        Hand-stacking k into a batch axis here was measurably *worse* (2.1x
        against 4.2x on the water dimer): it fixes the batch at one pair's worth
        of k, where the pass batches across every pair and every k at once.

        This graph contains ONLY the view-level accumulations. The prologue
        (which writes ``R_all`` whole) and the energy (which reads it whole) are
        separate graphs on purpose: the graph does not know that a write through
        ``R_all[:, :, ij]`` touches ``R_all``, so mixing parent-level and
        view-level access to one store in a single graph lets the scheduler
        interleave them. It is not hypothetical -- with all three phases
        captured together the energy dot landed at node 201 of 405, summing a
        half-built residual and quietly returning a wrong correlation energy.
        Graphs execute in the order they are replayed, so keeping each
        granularity in its own graph is an explicit barrier.
        """
        work = self._coupling_work()
        g = cg.Graph("lmp2 residual")
        with cg.capture(g):
            # Phase 1: every S T product. Distinct scratch, depends only on T,
            # so the whole lot is one dependency level.
            #
            # Emitted as cg.batched_gemm rather than one einsum per coupling.
            # GEMMBatching would fuse the per-coupling form to the same thing,
            # but only after the graph has been built one node at a time, and
            # capture costs ~38 us a node: 8112 nodes was 0.31 s of pure
            # bookkeeping for a graph that ends up ~30 nodes wide.
            groups = {}
            for _, tmp, S, T, _, _ in work:
                groups.setdefault(ten.shape(tmp), []).append((S, T, tmp))
            for members in groups.values():
                cg.batched_gemm(1.0, [S for S, _, _ in members], [T for _, T, _ in members],
                                0.0, [tmp for _, _, tmp in members])

            # Phase 2: accumulate into the residuals, grouped by (level, sign)
            # so members of a group write different accumulators, and by shape
            # so one gemm_batch can take them. Level 0 assigns (beta = 0) and
            # later levels accumulate, which also means the accumulators never
            # need zeroing between iterations: every slot a pair uses is
            # overwritten at level 0, and slots it never uses stay at their
            # allocated zero.
            by_key = {}
            for level, tmp, S, _, target, sign in work:
                key = (level, sign, ten.shape(tmp), ten.shape(target))
                by_key.setdefault(key, []).append((tmp, S, target))
            for key in sorted(by_key):
                level, sign = key[0], key[1]
                members = by_key[key]
                cg.batched_gemm(sign, [tmp for tmp, _, _ in members], [S for _, S, _ in members],
                                0.0 if level == 0 else 1.0,
                                [t for _, _, t in members], trans_b=True)
        return g

    def _coupling_work(self):
        """Every coupling as ``(level, tmp, S, T, target, sign)``.

        ``level`` is ``c // G`` for the pair's ``c``-th coupling, and the target
        is accumulator ``c % G``. Two couplings share a target only when they
        share a pair *and* their levels differ, so everything at one level is
        mutually independent and batches as a unit. That is the whole point: with
        a single accumulator a pair's couplings serialize, capping the batch at
        one per pair per level.
        """
        work = []
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if self.n_pno[ij] == 0:
                continue
            c = 0
            for k, sign in self.k_couple_kj[ij].items():
                work.append((c // self.n_acc, self._tmp_kj[ij, k], self.S_pno_ij_kj[ij, k],
                             self._T_view[int(self.i_j_to_ij[k, j])],
                             self._A_view[ij][c % self.n_acc], sign))
                c += 1
            for k, sign in self.k_couple_ik[ij].items():
                work.append((c // self.n_acc, self._tmp_ik[ij, k], self.S_pno_ij_ik[ij, k],
                             self._T_view[int(self.i_j_to_ij[i, k])],
                             self._A_view[ij][c % self.n_acc], sign))
                c += 1
        return work

    def _capture_reduce(self):
        """Graph: fold the partial accumulators into the residual.

        ``G`` rank-3 axpby nodes per bucket, independent of how many pairs or
        couplings there are. Separate from the coupling graph because it reads
        ``A`` whole while the couplings write single blocks of it, and the graph
        does not order view writes against parent reads.
        """
        g = cg.Graph("lmp2 coupling reduction")
        with cg.capture(g):
            for slices, R in zip(self._A_slice, self.R_all):
                for A_g in slices:
                    la.axpby(1.0, A_g, 1.0, R)
        return g

    def _capture_prologue(self):
        """Graph: ``R = K + (e_a + e_b - F_ii - F_jj) T`` for every pair, in two nodes."""
        g = cg.Graph("lmp2 residual prologue")
        with cg.capture(g):
            for D, T, R, K in zip(self.D_all, self.T_all, self.R_all, self.K_all):
                la.direct_product(1.0, D, T, 0.0, R)
                la.axpby(1.0, K, 1.0, R)
        return g

    def _capture_energy(self):
        """Graph: ``E = sum_ij (K_ij + R_ij) . Tt_ij``, in three nodes.

        psi4's ``compute_iteration_energy`` over the whole store rather than
        4*n_pairs operations.
        """
        g = cg.Graph("lmp2 iteration energy")
        with cg.capture(g):
            la.scale(0.0, self.e_iter)
            for K, R, Tt in zip(self.K_all, self.R_all, self.Tt_all):
                la.dot(self._e_part, K, Tt)
                la.axpby(1.0, self._e_part, 1.0, self.e_iter)
                la.dot(self._e_part, R, Tt)
                la.axpby(1.0, self._e_part, 1.0, self.e_iter)
        return g

    def _capture_step(self):
        """Graph: the Jacobi amplitude update ``T -= R / D``, every pair in one node."""
        g = cg.Graph("lmp2 amplitude step")
        with cg.capture(g):
            for R, D, T in zip(self.R_all, self.D_all, self.T_all):
                la.direct_division(-1.0, R, D, 1.0, T)
        return g

    def _capture_antisymmetrize(self):
        """Graph: ``Tt = 2 T - T^T``, every pair in two nodes."""
        g = cg.Graph("lmp2 antisymmetrized amplitudes")
        with cg.capture(g):
            for Tt, T in zip(self.Tt_all, self.T_all):
                einsums.permute("abp <- bap", Tt, T)
                la.axpby(2.0, T, -1.0, Tt)
        return g

    def _flat_views(self, stores):
        """Flat numpy views of every bucket store, for DIIS."""
        views = [ten.view(s).ravel(order="F") for s in stores]
        for v in views:
            assert v.base is not None, "DIIS needs a view, not a copy"
        return views

    def _flat_view(self, store):
        """The flat numpy view of a whole pair store, for DIIS.

        One contiguous buffer covers every pair, so DIIS needs no gathering.
        Einsums tensors are column major, so ``np.asarray`` hands back an
        F-contiguous array and a plain ``reshape(-1)`` would silently *copy*;
        DIIS would then extrapolate into a throwaway buffer and the solve would
        quietly fall back to unaccelerated Jacobi. ``ravel(order="F")`` is a
        genuine view, which the assertion pins down.
        """
        flat = ten.view(store).ravel(order="F")
        assert flat.base is not None, "DIIS needs a view, not a copy"
        return flat

    def lmp2_iterations(self, optimize=True):
        """Solve the local MP2 equations by replaying the captured graphs.

        Records ``t_capture`` (building and optimizing the graphs, paid once)
        and ``t_iterate`` (replaying them) separately. The split matters when
        comparing against an eager implementation: capture is fixed while the
        iteration count grows, and at ten iterations it is the larger of the two.
        """
        import time as _time
        _t0 = _time.perf_counter()
        self._allocate_iteration_tensors()

        g_prologue = self._capture_prologue()
        g_residual = self._capture_residual()
        g_reduce = self._capture_reduce()
        g_energy = self._capture_energy()
        g_step = self._capture_step()
        g_tt = self._capture_antisymmetrize()
        graphs = [g_prologue, g_residual, g_reduce, g_energy, g_step, g_tt]

        self._print(
            "\n  ==> Local MP2 <==\n"
            f"    captured {sum(g.num_nodes() for g in graphs)} nodes in "
            f"{len(graphs)} graphs (prologue {g_prologue.num_nodes()}, "
            f"coupling {g_residual.num_nodes()}, reduce {g_reduce.num_nodes()}, "
            f"energy {g_energy.num_nodes()}, "
            f"step {g_step.num_nodes()}, Tt {g_tt.num_nodes()})"
        )
        if optimize:
            pm = cg.PassManager()
            pm.populate_default()
            before = sum(g.num_nodes() for g in graphs)
            for g in graphs:
                g.apply(pm)
            self._print(
                f"    optimization: {pm.size} passes, nodes "
                f"{before} -> {sum(g.num_nodes() for g in graphs)}"
            )

        self.t_capture = _time.perf_counter() - _t0
        self._print(f"    capture + optimize: {self.t_capture * 1e3:.1f} ms (one-time)")
        self._print(f"\n    {'iter':>4}  {'Corr. Energy':>18} {'Delta E':>12} {'Max R':>12}")
        _t0 = _time.perf_counter()

        diis = _DIIS(self.cut.diis_max_vecs) if self.use_diis else None
        t_views = self._flat_views(self.T_all)
        r_views = self._flat_views(self.R_all)
        offsets = np.cumsum([0] + [v.size for v in t_views])

        e_curr = e_prev = 0.0
        for iteration in range(self.cut.maxiter + 1):
            g_prologue.execute()
            g_residual.execute()
            g_reduce.execute()
            g_energy.execute()
            e_prev, e_curr = e_curr, float(ten.view(self.e_iter)[0])
            # Max RMS over the pairs, on each pair's logical block: the padding
            # is identically zero and would otherwise dilute the norm.
            r_curr = 0.0
            for ij, n in enumerate(self.n_pno):
                if n:
                    block = self.pair_block(self.R_all, ij)[:n, :n]
                    r_curr = max(r_curr, float(np.sqrt(np.vdot(block, block).real / block.size)))

            g_step.execute()

            if diis is not None:
                t_new = diis.extrapolate(np.concatenate(t_views), np.concatenate(r_views))
                for v, lo, hi in zip(t_views, offsets[:-1], offsets[1:]):
                    v[...] = t_new[lo:hi]

            g_tt.execute()

            self._print(
                f"    {iteration:>4}  {e_curr:>18.12f} {e_curr - e_prev:>12.3e} {r_curr:>12.3e}"
            )
            self.n_iterations = iteration + 1

            if iteration > 0 and abs(e_curr - e_prev) < self.cut.e_convergence \
                    and abs(r_curr) < self.cut.r_convergence:
                break
        else:
            raise RuntimeError("maximum DLPNO iterations exceeded")

        self.t_iterate = _time.perf_counter() - _t0
        self.e_lmp2 = e_curr
        # Padded components are zero, so one dot over the whole store is the
        # sum over pairs.
        self.e_lmp2_os = sum(ten.vector_dot(K, T) for K, T in zip(self.K_all, self.T_all))
        self.e_lmp2_ss = e_curr - self.e_lmp2_os

        # The residual's contiguous-batch trick rests on T_ji == T_ij^T. That is
        # an identity of the LMP2 equations rather than something imposed here,
        # so it is worth confirming rather than assuming.
        skew = max(
            (np.abs(self.pair_block(self.T_all, ij)[:n, :n]
                    - self.pair_block(self.T_all, self.ij_to_ji[ij])[:n, :n].T).max()
             for ij, n in enumerate(self.n_pno) if n),
            default=0.0,
        )
        assert skew < 1e-10, f"pair-transpose symmetry broken by {skew:.3e}"
        return self

    # -- psi4 DLPNOMP2::compute_energy -------------------------------------

    def compute_energy(self, optimize=True):
        """Run the whole DLPNO-MP2 pipeline and return the total energy."""
        self.setup_orbitals()
        self.prep_sparsity()
        self.compute_metric()
        self.compute_qia()
        self.pno_transform()
        self.compute_pno_overlaps()
        self.lmp2_iterations(optimize=optimize)

        # de_dipole_ is the correction for pairs dropped by dipole screening;
        # nothing is dropped yet, so it is zero.
        self.e_corr = self.e_lmp2 + self.de_pno_total
        self._print(
            "\n  Total DLPNO-MP2 Correlation Energy: "
            f"{self.e_corr:16.12f}\n"
            f"    MP2 Correlation Energy:           {self.e_lmp2:16.12f}\n"
            f"    PNO Truncation Correction:        {self.de_pno_total:16.12f}"
        )
        return self.ref.e_scf + self.e_corr
