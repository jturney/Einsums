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
        # The scatters are captured with the GEMMs that consume them rather than
        # run here, so they parallelize too and the graph orders them itself.
        X_pad = [None] * self.n_lmo_pairs
        for ij in range(self.n_lmo_pairs):
            if self.n_pno[ij]:
                X_pad[ij] = ten.zeros(f"X (padded) {ij}", [npao, self.pair_dim(ij)])

        # Where each coupling's block sits in its pair's output, and the
        # per-column sqrt prefactor. No partner blocks are concatenated: see
        # below.
        self._couplings, raw_width = {}, {}
        for ij, plist in enumerate(partners):
            if not plist:
                continue
            entries, off = [], 0
            for (p, k, is_ik, factor) in plist:
                w = self.pair_dim(p)
                entries.append((p, k, is_ik, off, w, float(np.sign(factor))))
                off += w
            self._couplings[ij] = entries
            raw_width[ij] = off

        # The concatenation is padded to a width its whole group shares, so the
        # residual's second half is one batched GEMM per group rather than one
        # per coupling. See _choose_width_groups.
        self._cat_groups = self._choose_width_groups(raw_width)
        self._cat_width = {ij: width for width, members in self._cat_groups for ij in members}

        self._S_cat, scale, offsets = {}, {}, {}
        for ij, entries in self._couplings.items():
            # Padding columns are allocated zero and never written, and they are
            # what keeps the padded product exact: a zero column of S_cat meets a
            # zero column of tmp_cat and contributes nothing. Their prefactor is
            # therefore arbitrary; 1.0 keeps the scaling below a pure multiply.
            factors = np.ones(self._cat_width[ij])
            for (p, _, _, off, w, _), (_, _, _, factor) in zip(entries, partners[ij]):
                factors[off:off + w] = np.sqrt(abs(factor))
            scale[ij] = factors
            offsets[ij] = [off for _, _, _, off, _, _ in entries]
            self._S_cat[ij] = ten.zeros(f"S (cat) {ij}",
                                        [self.pair_dim(ij), self._cat_width[ij]])

        half = {ij: ten.zeros(f"half {ij}", [self.pair_dim(ij), npao]) for ij in self._S_cat}

        # Both GEMMs are emitted as batches over pairs of buckets rather than
        # one node per pair, which is what lets the partner concatenation go.
        #
        # The concatenation was the dominant cost of this phase and none of it
        # was arithmetic: laying a pair's partners side by side so its overlaps
        # are one GEMM re-copies every partner block once per pair that couples
        # to it, which at ethanol/cc-pVTZ is 335 MiB of allocate-and-memcpy
        # over blocks that already exist in 14 MiB of X_pad. It was also the
        # one part of the phase that could not be parallelized, being numpy
        # slice assignment on the calling thread.
        #
        # gemm_batch takes a single m/n/k and a single leading dimension for
        # the whole batch, which is exactly what bucketing already guarantees:
        # group the couplings by (bucket of ij, bucket of partner) and every
        # member of a group agrees on all of them. Each output is a column
        # slice of its pair's S_cat, so the blocks land where the concatenated
        # form put them and the rest of the phase is unchanged. Sixteen groups
        # at four buckets, against 4056 couplings.
        # Each coupling's output block. Sliced once and reused, by the batch
        # below, by the coupling map at the end, and every iteration afterwards
        # by the residual: a slice is a pybind round trip building a fresh view
        # object, and at 4056 couplings doing it twice costs more than the batch
        # emission it feeds.
        blocks = self._S_block = {}
        by_shape = {}
        for ij, plist in enumerate(partners):
            for idx, ((p, _, _, _), off) in enumerate(zip(plist, offsets.get(ij, []))):
                block = self._S_cat[ij][:, off:off + self.pair_dim(p)]
                blocks[ij, idx] = block
                by_shape.setdefault((self.bucket_of[ij], self.bucket_of[p]), []).append(
                    (half[ij], X_pad[p], block))

        by_bucket = {}
        for ij in self._S_cat:
            by_bucket.setdefault(self.bucket_of[ij], []).append(ij)

        g = cg.Graph("pno overlaps")
        with cg.capture(g):
            for ij in range(self.n_lmo_pairs):
                if self.n_pno[ij]:
                    sparse.scatter_into(X_pad[ij], self.X_pno[ij],
                                        [self.lmopair_to_paos[ij], range(self.n_pno[ij])])
            for members in by_bucket.values():
                cg.batched_gemm(1.0, [X_pad[ij] for ij in members],
                                [self.S_pao] * len(members), 0.0,
                                [half[ij] for ij in members], trans_a=True)
            for members in by_shape.values():
                cg.batched_gemm(1.0, [h for h, _, _ in members], [x for _, x, _ in members],
                                0.0, [s for _, _, s in members])
        # No pass pipeline. This graph is emitted in the form the passes would
        # produce - the batches are already fused, and there is nothing to share
        # between them - so applying them is pure cost, and it is not small:
        # 14 ms to apply, plus the 27 ms of populate_default it forces.
        #
        # Parallelism across the batch, each member serial underneath. The
        # OpenMP executor rather than Dataflow because Dataflow runs nodes on
        # std::thread workers, which is the pattern the OpenMP-built OpenBLAS
        # miscomputes; see DLPNOBase.precompute_fits.
        self._run(g)

        # One vectorized scaling per pair: every coupling owns a column range.
        for ij, factors in scale.items():
            ten.view(self._S_cat[ij])[...] *= factors[np.newaxis, :]

        # Each coupling's overlap is a column slice, not its own allocation.
        for ij, plist in enumerate(partners):
            for idx, (p, k, is_ik, factor) in enumerate(plist):
                if is_ik:
                    self.S_pno_ij_ik[ij, k] = blocks[ij, idx]
                    self.k_couple_ik[ij][k] = float(np.sign(factor))
                else:
                    self.S_pno_ij_kj[ij, k] = blocks[ij, idx]
                    self.k_couple_kj[ij][k] = float(np.sign(factor))

        self._print(
            f"  overlaps: {sum(len(p) for p in partners)} couplings via "
            f"{g.num_nodes()} nodes ({len(by_bucket) + len(by_shape)} captured), "
            f"{len(self._cat_groups)} width groups"
        )
        return self

    def _choose_width_groups(self, raw_width):
        """Partition the pairs into groups sharing one padded concatenation width.

        The residual's second half is ``R_ij = sum_c sign_c tmp_c S_c^T``, and
        summing over the couplings of a pair is a contraction, so the whole sum
        is a single GEMM against the concatenated operands - ``(M, W) x (M, W)^T``
        with ``W`` the pair's total coupling width. That turns 32948 GEMMs into
        776 at a six-monomer water chain, which is the point: the residual issues
        ~11000 flops per GEMM call, so it is bound by the per-call cost rather
        than by arithmetic, and the only thing that helps is fewer calls.

        Batching those 776 needs them to agree on ``m``, ``n`` and ``k``. The
        first two come from the PNO bucket; ``k`` is ``W``, which no two pairs
        need share, so it is padded. Padding every pair in a bucket to that
        bucket's widest costs 1.42x the flops, because coupling counts inside one
        bucket spread widely (258..983 columns in the smallest). Sorting by width
        first and cutting each bucket into ``n_width_groups`` chunks costs 1.10x
        at four, which buys back most of the padding for four times the calls -
        16 in total, against 32948.

        The padding is inert rather than merely cheap: a padded column of
        ``S_cat`` is zero and so is its partner in ``tmp_cat``, so it contributes
        nothing to any residual.
        """
        n_groups = max(1, int(self.cut.n_width_groups))
        by_bucket = {}
        for ij in raw_width:
            by_bucket.setdefault(self.bucket_of[ij], []).append(ij)

        groups = []
        for members in by_bucket.values():
            members.sort(key=lambda ij: raw_width[ij])
            chunk = -(-len(members) // n_groups)
            for start in range(0, len(members), chunk):
                part = members[start:start + chunk]
                groups.append((max(raw_width[ij] for ij in part), part))
        return groups

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
        # Scratch for the couplings, concatenated per pair in exactly the layout
        # S_cat uses, so the second half of the residual can read it whole. Each
        # coupling still gets its own block - they are written independently and
        # sharing one would serialize contractions that are not related - but the
        # blocks are column ranges of one buffer rather than separate tensors.
        #
        # The padding columns are allocated zero and nothing ever writes them,
        # which is what makes the padded GEMM exact rather than approximately so.
        self._tmp_cat = {
            ij: ten.zeros(f"tmp (cat) {ij}", [self.pair_dim(ij), self._cat_width[ij]])
            for ij in self._couplings
        }
        # Sliced once and kept: a slice is a pybind round trip building a fresh
        # view, and a view handed to a captured op has to outlive the graph.
        self._tmp_block = {
            (ij, idx): self._tmp_cat[ij][:, off:off + w]
            for ij, entries in self._couplings.items()
            for idx, (_, _, _, off, w, _) in enumerate(entries)
        }

        self.e_iter = ten.zeros("E(iteration)", [1])
        self._e_part = ten.zeros("E(part)", [1])

    def _capture_couplings(self):
        """Graph: ``tmp_c = sign_c S_c T_c`` for every coupling, into ``tmp_cat``.

        The first half of ``R_ij = sum_c sign_c S_c T_c S_c^T``. Every coupling
        has its own ``T``, so this half stays one GEMM per coupling; what it does
        not have to be is one graph *node* per coupling. Emitted as
        ``cg.batched_gemm`` rather than one einsum each: ``GEMMBatching`` would
        fuse the per-coupling form to the same thing, but only after the graph
        has been built a node at a time, and capture costs ~38 us a node - 8112
        nodes was 0.31 s of pure bookkeeping for a graph that ends up ~30 nodes
        wide.

        Each product lands in its coupling's column range of the pair's
        ``tmp_cat``, which is what lets :meth:`_capture_residual` read the whole
        concatenation back as one operand. The sign rides on the batch's
        ``alpha`` here rather than in the second half, because ``S`` appears
        twice there and a sign folded into ``S`` would square away; grouping by
        it splits each shape group in two, which costs nothing.
        """
        g = cg.Graph("lmp2 couplings")
        with cg.capture(g):
            groups = {}
            for tmp, S, T, sign in self._coupling_work():
                groups.setdefault((ten.shape(tmp), sign), []).append((S, T, tmp))
            for (_, sign), members in groups.items():
                cg.batched_gemm(sign, [S for S, _, _ in members], [T for _, T, _ in members],
                                0.0, [tmp for _, _, tmp in members])
        return g

    def _capture_residual(self):
        """Graph: fold every coupling into its pair's residual, one GEMM per pair.

        The second half of ``R_ij = sum_c sign_c S_c T_c S_c^T``. The sum over a
        pair's couplings is a contraction, so with the ``tmp_c`` and the ``S_c``
        both concatenated along it the whole sum is a single ``(M, W) x (M, W)^T``
        GEMM - not a loop of GEMMs and an accumulator tree.

        That is the difference between being bound by arithmetic and being bound
        by dispatch. The residual issues ~11000 flops per GEMM call, and a call
        costs ~0.2 us however small it is, so the per-coupling form spent most of
        its time entering and leaving the BLAS. This collapses 32948 of those
        calls into 776 at a six-monomer water chain, and
        :meth:`_choose_width_groups` batches the 776 into 16.

        It also removes the partial accumulators the per-coupling form needed.
        Those existed because a pair's couplings all wrote one residual and so
        serialized; spreading them over ``G`` accumulators cut the dependency
        depth by ``G`` at the cost of a reduction graph. A contraction has no
        dependency depth, so both are gone.

        Separate from :meth:`_capture_couplings` because it reads ``tmp_cat``
        whole while that one writes single column ranges of it, and the graph
        does not know a write through a view touches its parent. Separate from
        the prologue (which writes ``R_all`` whole) and the energy (which reads
        it whole) for the same reason, in the other direction. It is not
        hypothetical -- with the phases captured together the energy dot landed
        at node 201 of 405, summing a half-built residual and quietly returning a
        wrong correlation energy. Graphs execute in the order they are replayed,
        so keeping each granularity in its own graph is an explicit barrier.
        """
        g = cg.Graph("lmp2 residual")
        with cg.capture(g):
            # beta = 1: the prologue has already put K + D T in R.
            for _, members in self._cat_groups:
                cg.batched_gemm(1.0, [self._tmp_cat[ij] for ij in members],
                                [self._S_cat[ij] for ij in members],
                                1.0, [self._R_view[ij] for ij in members], trans_b=True)
        return g

    def _coupling_work(self):
        """Every coupling as ``(tmp block, S block, T view, sign)``.

        The tmp and S blocks are the same column range of their pair's two
        concatenations, which is what lets :meth:`_capture_residual` contract the
        pair of them whole.
        """
        return [(self._tmp_block[ij, idx], self._S_block[ij, idx],
                 self._T_view[partner], sign)
                for ij, entries in self._couplings.items()
                for idx, (partner, _, _, _, _, sign) in enumerate(entries)]

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
        g_couple = self._capture_couplings()
        g_residual = self._capture_residual()
        g_energy = self._capture_energy()
        g_step = self._capture_step()
        g_tt = self._capture_antisymmetrize()
        graphs = [g_prologue, g_couple, g_residual, g_energy, g_step, g_tt]

        self._print(
            "\n  ==> Local MP2 <==\n"
            f"    captured {sum(g.num_nodes() for g in graphs)} nodes in "
            f"{len(graphs)} graphs (prologue {g_prologue.num_nodes()}, "
            f"coupling {g_couple.num_nodes()}, residual {g_residual.num_nodes()}, "
            f"energy {g_energy.num_nodes()}, "
            f"step {g_step.num_nodes()}, Tt {g_tt.num_nodes()})"
        )
        if optimize:
            # Not the two batched graphs. They are emitted in the form the
            # passes would produce - the batches are already fused and there is
            # nothing to share between them - so applying the pipeline is pure
            # cost, and the cost scales with the batch rather than the node
            # count: 445 ms on a 32-node coupling graph whose nodes carry ~1000
            # operands each, against a replay of 324 ms for the whole solve.
            # It buys nothing measurable either, the two graphs being 48 of the
            # 85 nodes and the whole pipeline removing exactly one of them.
            # Same argument, and the same decision, as compute_pno_overlaps.
            batched = {id(g_couple), id(g_residual)}
            targets = [g for g in graphs if id(g) not in batched]
            pm = self.pass_manager()
            before = sum(g.num_nodes() for g in targets)
            for g in targets:
                g.apply(pm)
            self._print(
                f"    optimization: {pm.size} passes on {len(targets)} graphs, "
                f"nodes {before} -> {sum(g.num_nodes() for g in targets)}"
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
            g_couple.execute()
            g_residual.execute()
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
