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

The whole iteration - the antisymmetrized amplitudes ``Tt_ij = 2 T_ij -
T_ij^T``, the residual, the iteration energy, and the Jacobi update ``T_ij -=
R_ij / D_ij`` - is captured once as a single graph with a loop node and
replayed to convergence (see :meth:`DLPNOMP2.lmp2_iterations`).

DIIS is ``einsums.graph.diis``, installed as the loop's predicate: it
extrapolates the amplitude stores in place between replays, so the body picks
the new values up, and it sits between the step and the rebuild of ``Tt``
because psi4 extrapolates the amplitudes before rebuilding. Its error vector
is the materialized Jacobi step (the standard coupled-cluster choice), where
the earlier host-side implementation used the raw residual, so trajectories
agree with it to convergence tolerance rather than bit for bit.
"""

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import DLPNOBase

from .contracts import CouplingPlan

__all__ = ["DLPNOMP2"]


class DLPNOMP2(DLPNOBase):
    """DLPNO-MP2: PNO overlaps, then the local MP2 equations as a ComputeGraph."""

    def __init__(self, reference, thresholds=None, verbose=True, use_diis=True):
        super().__init__(reference, thresholds, verbose)
        self.use_diis = use_diis
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
        # Through the stage registry, so a selected cpp backend runs from
        # every entry point, not only run_phases.
        from .stages import compute_pno_overlaps as overlap_stage

        plan = self.plan_pno_couplings()
        overlaps = overlap_stage(
            self.X_pno, self.S_pao, self.lmopair_to_paos, self.n_pno,
            self.bucket_of, self.bucket_dims, plan,
        )
        self._S_cls, self._S_T = overlaps.S_cls, overlaps.S_T
        self._print(
            f"  overlaps: {len(plan.pair)} couplings, "
            f"{len(plan.classes)} shape classes"
        )
        return self

    def plan_pno_couplings(self):
        """Which pairs couple to which partners, and where every block goes.

        The planning half of the overlap phase, and the half that stays Python.
        Pure integer bookkeeping: not one floating-point array is touched, and
        nothing here depends on the values of the integrals. It is where the
        method's logic lives, it is index arithmetic rather than arithmetic, and
        nothing measured says it is hot.

        Splitting it out is what makes the other half's contract narrow enough
        to state. As one phase this was 27 fields of ``self``; the numerics
        take seven parameters.
        """
        naocc = self.ref.naocc
        F_lmo = ten.view(self.F_lmo)
        f_cut = self.cut.f_cut

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

        self._n_partners = sum(len(p) for p in partners)
        return self._plan_couplings(partners)

    def _plan_couplings(self, partners):
        """Lay the couplings out twice, and record the map between the two.

        The residual is ``R_ij = sum_c sign_c S_c T_c S_c^T``, and its two halves
        want opposite groupings. The second half sums over a pair's couplings, so
        it wants them consecutive **by pair**. The first half applies each
        partner's ``T``, and many pairs couple through the same partner, so it
        wants them consecutive **by partner**. The couplings are a bipartite
        graph, so no single order is consecutive both ways: the intermediate is
        produced in one order and read in the other, and something has to move it
        between. That something is one permuting gather per shape class - see
        :meth:`_capture_repack`.

        Everything is per *shape class*, a pair of PNO buckets ``(bucket of ij,
        bucket of the partner)``. Within a class every coupling block is the same
        ``M_b x M_p``, which is what lets a class be one tensor and its GEMMs be
        one batch.

        Sets, per class:

        * ``_cls_partner_groups`` - the phase-1 batches. Partners sorted by how
          many couplings they carry and cut into ``n_width_groups`` chunks, each
          padded to its chunk's widest, so a chunk is one batched GEMM.
        * ``_cls_pair_groups`` - the same for phase 2, over pairs.
        * ``_cls_perm`` - for each destination slot in pair order, the source slot
          in partner order. Padding slots point at a reserved zero slot, so the
          gather writes a full destination and the padded columns it leaves are
          genuinely zero rather than stale.

        Returns a :class:`~dlpno.contracts.CouplingPlan`: the couplings as flat
        per-field arrays indexed by coupling ordinal, plus the per-class layout
        they refer to. That is the form the promoted C++ stage consumes, and it
        is the only form it CAN consume - a dict keyed by a shape-class tuple is
        not expressible across the boundary. The per-class structures the
        iteration phase needs (``_cls_perm``, ``_cls_pair_groups``,
        ``_cls_partner_groups``, ``_pair_slot``) stay on the object: they are
        read by ``lmp2_iterations``, which is not being promoted.
        """
        n_groups = max(1, int(self.cut.n_width_groups))

        # -- couplings by (pair, class), which is the order S_cat is built in --
        per_pair = {}
        for ij, plist in enumerate(partners):
            if not plist:
                continue
            by_cls = {}
            for entry in plist:
                by_cls.setdefault(self.bucket_of[entry[0]], []).append(entry)
            per_pair[ij] = by_cls

        counts = {}  # class -> {pair: how many couplings}
        for ij, by_cls in per_pair.items():
            for b_p, lst in by_cls.items():
                counts[self.bucket_of[ij], b_p] = counts.get((self.bucket_of[ij], b_p), {})
                counts[self.bucket_of[ij], b_p][ij] = len(lst)

        def chunk_by_size(keys, size_of):
            """Sort by size and cut into n_groups chunks, each padded to its widest."""
            keys = sorted(keys, key=size_of)
            step = -(-len(keys) // n_groups)
            return [(max(size_of(k) for k in keys[s:s + step]), keys[s:s + step])
                    for s in range(0, len(keys), step)]

        # -- phase 2: pair order, padded per group ---------------------------
        #
        # Everything per-class is a LIST indexed by class ordinal, not a dict
        # keyed by the (bucket, bucket) tuple. The tuple is still the identity
        # of a class and _classes still names them in order; what changed is
        # that nothing downstream looks a class up by it.
        #
        # That is a requirement rather than a preference: a stage signature can
        # only carry the closed cross-boundary type list, and a dict keyed by a
        # tuple is not on it and will not be. See the M3 section of the design
        # notes. It is also faster and plainer C++ - a vector indexed by class
        # against a map lookup in the inner loop.
        self._classes = sorted(counts)
        self._cls_index = {cls: ci for ci, cls in enumerate(self._classes)}
        n_cls = len(self._classes)

        self._cls_pair_groups = [None] * n_cls
        self._cls_slots_pair = [0] * n_cls
        pair_slot = {}
        for ci, cls in enumerate(self._classes):
            groups = chunk_by_size(list(counts[cls]), lambda ij: counts[cls][ij])
            slot = 0
            laid = []
            for n_pad, members in groups:
                for ij in members:
                    pair_slot[ci, ij] = slot
                    slot += n_pad
                laid.append((n_pad, members))
            self._cls_pair_groups[ci] = laid
            self._cls_slots_pair[ci] = slot

        # -- the couplings themselves, in that same order --------------------
        # Structure of arrays: one flat list per field, all the same length,
        # indexed by coupling ordinal. This was a dict from pair to a list of
        # six-tuples, which is the shape that cannot cross into C++ and the
        # shape that costs a pointer chase per coupling once there. See
        # dlpno/contracts.py.
        #
        # A coupling carries its class ORDINAL, not the class tuple; the two
        # consumers that want block dimensions read them back through
        # ``self._classes[ci]``.
        c_pair, c_partner, c_cls = [], [], []
        c_dest, c_sign, c_factor = [], [], []
        self._pair_slot = pair_slot
        self._coupled_pairs = sorted(per_pair)
        for ij in self._coupled_pairs:
            by_cls = per_pair[ij]
            for b_p in sorted(by_cls):
                ci = self._cls_index[self.bucket_of[ij], b_p]
                for c, (p, k, is_ik, factor) in enumerate(by_cls[b_p]):
                    c_pair.append(ij)
                    c_partner.append(p)
                    c_cls.append(ci)
                    c_dest.append(pair_slot[ci, ij] + c)
                    c_sign.append(float(np.sign(factor)))
                    c_factor.append(factor)

        # -- phase 1: partner order, padded per group ------------------------
        by_partner = [{} for _ in range(n_cls)]  # per class: {partner: [coupling ordinals]}
        for n, (p, ci) in enumerate(zip(c_partner, c_cls)):
            by_partner[ci].setdefault(p, []).append(n)

        self._cls_partner_groups = [None] * n_cls
        self._cls_slots_partner = [0] * n_cls
        c_src = [0] * len(c_pair)
        for ci in range(n_cls):
            groups = chunk_by_size(list(by_partner[ci]),
                                   lambda q: len(by_partner[ci][q]))
            slot = 0
            laid = []
            for n_pad, members in groups:
                for q in members:
                    for c, n in enumerate(by_partner[ci][q]):
                        c_src[n] = slot + c
                    laid.append((q, slot, n_pad))
                    slot += n_pad
            # Reserved zero slot: every padded destination reads from it, so the
            # gather writes its whole destination and the padding stays zero.
            self._cls_slots_partner[ci] = slot + 1
            self._cls_partner_groups[ci] = self._regroup_partners(laid)

        # -- the map from partner order to pair order ------------------------
        # One pass over the couplings, not one per class: there are 32948 of
        # them and 16 classes, and the obvious nesting is a third of a second of
        # pure Python.
        self._cls_perm = [[self._cls_slots_partner[ci] - 1] * self._cls_slots_pair[ci]
                          for ci in range(n_cls)]
        # ...and its inverse, for lifting the transposed copy of S out of the
        # pair-major one at setup. Its padding reads the reserved zero slot at
        # the end of the pair-major side.
        cls_inv_perm = [[self._cls_slots_pair[ci]] * self._cls_slots_partner[ci]
                        for ci in range(n_cls)]
        for ci, dst, src in zip(c_cls, c_dest, c_src):
            self._cls_perm[ci][dst] = src
            cls_inv_perm[ci][src] = dst
        self._cls_inv_perm = cls_inv_perm

        self.plan = CouplingPlan(
            classes=list(self._classes),
            slots_pair=list(self._cls_slots_pair),
            slots_partner=list(self._cls_slots_partner),
            inv_perm=cls_inv_perm,
            pair=c_pair,
            partner=c_partner,
            cls=c_cls,
            dest_slot=c_dest,
            src_slot=c_src,
            sign=c_sign,
            factor=c_factor,
            coupled_pairs=self._coupled_pairs,
        )
        return self.plan

    @staticmethod
    def _regroup_partners(laid):
        """``[(n_pad, [(partner, slot), ...])]`` - one entry per batchable group."""
        groups = {}
        for q, slot, n_pad in laid:
            groups.setdefault(n_pad, []).append((q, slot))
        return sorted(groups.items())

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

        # The Jacobi step, materialized per bucket rather than fused into the
        # amplitude update: it is DIIS's error vector, and cg.diis snapshots
        # (amplitude, step) pairs between replays (see _emit_step).
        self.step_all = self.new_pair_stores("step")

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
        # The intermediate, held twice: V in partner order, which is what the
        # first half produces, and W in pair order, which is what the second half
        # reads. Both are (M_p, slots, M_b) per shape class and hold TRANSPOSED
        # blocks, because column major only gives a GEMM contiguous output blocks
        # when the shared operand is on the left, and here the shared operand is
        # the partner's T.
        #
        # Padding slots are allocated zero. V's are computed (from zero columns
        # of S^T, so they come out zero) and never read; W's are written by the
        # repack from V's reserved zero slot. Either way the padded columns a
        # GEMM sees are genuinely zero rather than stale.
        self._V = [
            ten.zeros(f"V (class {cls})",
                      [self.bucket_dims[cls[1]], self.bucket_dims[cls[0]],
                       self._cls_slots_partner[ci]])
            for ci, cls in enumerate(self._classes)
        ]
        self._W = [
            ten.zeros(f"W (class {cls})",
                      [self.bucket_dims[cls[1]], self._cls_slots_pair[ci],
                       self.bucket_dims[cls[0]]])
            for ci, cls in enumerate(self._classes)
        ]

        # Every operand is a slice reinterpreted as a matrix. Sliced and reshaped
        # once and kept: both are pybind round trips, and a view handed to a
        # captured op has to outlive the graph.
        #
        # reshape_view is what makes this layout usable at all. A phase-1 slice
        # is (M_p, M_b, n) and the GEMM wants (M_p, M_b * n); a phase-2 slice is
        # (M_p, n, M_b) and the GEMM wants (M_p * n, M_b). Both merges are free -
        # the axes already abut - but linalg.reshape would have copied the whole
        # 66 MiB intermediate twice per iteration to express them.
        self._V_group, self._ST_group = {}, {}
        for ci, cls in enumerate(self._classes):
            M_p, M_b = self.bucket_dims[cls[1]], self.bucket_dims[cls[0]]
            for n_pad, members in self._cls_partner_groups[ci]:
                for q, slot in members:
                    self._V_group[ci, q] = (
                        self._V[ci][:, :, slot:slot + n_pad].reshape_view([M_p, M_b * n_pad]))
                    self._ST_group[ci, q] = (
                        self._S_T[ci][:, :, slot:slot + n_pad].reshape_view([M_p, M_b * n_pad]))

        self._W_pair, self._S_pair = {}, {}
        for ci, cls in enumerate(self._classes):
            M_p, M_b = self.bucket_dims[cls[1]], self.bucket_dims[cls[0]]
            for n_pad, members in self._cls_pair_groups[ci]:
                for ij in members:
                    slot = self._pair_slot[ci, ij]
                    self._W_pair[ci, ij] = (
                        self._W[ci][:, slot:slot + n_pad, :].reshape_view([M_p * n_pad, M_b]))
                    self._S_pair[ci, ij] = self._S_cls[ci][:, slot * M_p:(slot + n_pad) * M_p]

        self.e_iter = ten.zeros("E(iteration)", [1])
        self._e_part = ten.zeros("E(part)", [1])

    def _capture_couplings(self):
        """Graph: ``V_c = (sign_c S_c T_c)^T``, one GEMM per PARTNER.

        The first half of ``R_ij = sum_c sign_c S_c T_c S_c^T``, and the mirror
        image of the second. The second half collapses because a pair's
        couplings share that pair; this one collapses because many pairs couple
        through the same partner, so they share that partner's ``T``. Stack
        those couplings' overlaps and the whole set is one GEMM: 32948 down to
        2822 at a six-monomer water chain, in 64 batches.

        Two things fall out of column major and are worth stating rather than
        rediscovering. A GEMM's output blocks are contiguous only when the
        SHARED operand is on the left, and the shared operand here is ``T``, so
        the products come out transposed - hence ``V_c`` rather than ``tmp_c``.
        And the sign rides on ``S_T`` rather than on the batch's ``alpha``,
        because one batch now spans many pairs and the sign varies between them;
        ``S`` appears twice in the congruence, so it has to ride on exactly one
        of the two copies, and this is the one the first half reads.
        """
        g = cg.Graph("lmp2 couplings")
        with cg.capture(g):
            self._emit_couplings()
        return g

    def _emit_couplings(self):
        """Record this phase's ops into the ambient capture."""
        for ci in range(len(self._classes)):
            for _, members in self._cls_partner_groups[ci]:
                cg.batched_gemm(1.0, [self._T_view[q] for q, _ in members],
                                [self._ST_group[ci, q] for q, _ in members], 0.0,
                                [self._V_group[ci, q] for q, _ in members], trans_a=True)

    def _capture_repack(self):
        """Graph: the intermediate from partner order into pair order.

        The price of having both halves be big GEMMs. They want the couplings
        grouped opposite ways and the couplings are a bipartite graph, so no one
        order serves both and the intermediate has to move once per iteration.

        One permuting gather per shape class, which is the whole of it: the
        reordering along the coupling axis and the swap of the other two are the
        same pass. Written as two passes - gather then permute - it is 12.71 ms
        serial and 5.72 threaded over the 66 MiB this moves at n=6, against 6.93
        and 3.29 for the one pass.

        Under the OpenMP executor because the classes are independent and a
        gather does not thread inside itself.

        Padding slots read from the reserved zero slot at the end of each class,
        so the destination is written whole and its padded columns are zero
        rather than stale - which is what lets the second half's GEMM ignore
        them.
        """
        g = cg.Graph("lmp2 repack")
        with cg.capture(g):
            self._emit_repack()
        g.set_executor(cg.OpenMPExecutor())
        return g

    def _emit_repack(self):
        """Record this phase's ops into the ambient capture."""
        for ci, cls in enumerate(self._classes):
            M_p, M_b = self.bucket_dims[cls[1]], self.bucket_dims[cls[0]]
            la.gather(self._W[ci], self._V[ci],
                      [list(range(M_p)), list(range(M_b)), self._cls_perm[ci]],
                      [0, 2, 1])

    def _capture_residual(self):
        """Graph: fold every coupling into its pair's residual, one GEMM per pair.

        The second half of ``R_ij = sum_c sign_c S_c T_c S_c^T``. The sum over a
        pair's couplings is a contraction, so with the intermediate and the
        ``S_c`` both concatenated along it the whole sum is a single GEMM - not a
        loop of GEMMs and an accumulator tree.

        That is the difference between being bound by arithmetic and being bound
        by dispatch. The residual issues ~11000 flops per GEMM call, and a call
        costs ~0.2 us however small it is, so the per-coupling form spent most of
        its time entering and leaving the BLAS.

        One GEMM per (pair, shape class) rather than per pair, because the
        intermediate is per class: a pair's couplings to differently sized
        partners cannot share one operand. That is at most four per pair and they
        accumulate into the same residual with ``beta = 1``, which is safe
        because they are separate batches replayed in order.

        It also removes the partial accumulators the per-coupling form needed.
        Those existed because a pair's couplings all wrote one residual and so
        serialized; spreading them over ``G`` accumulators cut the dependency
        depth by ``G`` at the cost of a reduction graph. A contraction has no
        dependency depth, so both are gone.

        This standalone form exists for tests and reproducers; the solver
        captures the same ops into one loop body through :meth:`_emit_residual`.
        The separation used to be load-bearing: the graph could not link a view
        covering its parent EXACTLY (a single-member class's ``W`` slice is one),
        so it saw no repack -> residual dependency, and the sort turned that
        missing edge into a reorder - captured together, the energy dot once
        landed at node 201 of 405, summing a half-built residual and quietly
        returning a wrong correlation energy. Both defects were fixed in
        ``8e4174d48``, which is what made the fold in :meth:`lmp2_iterations`
        possible.
        """
        g = cg.Graph("lmp2 residual")
        with cg.capture(g):
            self._emit_residual()
        return g

    def _emit_residual(self):
        """Record this phase's ops into the ambient capture."""
        # beta = 1: the prologue has already put K + D T in R, and earlier
        # classes have already added theirs.
        for ci in range(len(self._classes)):
            for _, members in self._cls_pair_groups[ci]:
                cg.batched_gemm(1.0, [self._W_pair[ci, ij] for ij in members],
                                [self._S_pair[ci, ij] for ij in members], 1.0,
                                [self._R_view[ij] for ij in members],
                                trans_a=True, trans_b=True)

    def _capture_prologue(self):
        """Graph: ``R = K + (e_a + e_b - F_ii - F_jj) T`` for every pair, in two nodes."""
        g = cg.Graph("lmp2 residual prologue")
        with cg.capture(g):
            self._emit_prologue()
        return g

    def _emit_prologue(self):
        """Record this phase's ops into the ambient capture."""
        for D, T, R, K in zip(self.D_all, self.T_all, self.R_all, self.K_all):
            la.direct_product(1.0, D, T, 0.0, R)
            la.axpby(1.0, K, 1.0, R)

    def _capture_energy(self):
        """Graph: ``E = sum_ij (K_ij + R_ij) . Tt_ij``, in three nodes.

        psi4's ``compute_iteration_energy`` over the whole store rather than
        4*n_pairs operations.
        """
        g = cg.Graph("lmp2 iteration energy")
        with cg.capture(g):
            self._emit_energy()
        return g

    def _emit_energy(self):
        """Record this phase's ops into the ambient capture."""
        la.scale(0.0, self.e_iter)
        for K, R, Tt in zip(self.K_all, self.R_all, self.Tt_all):
            la.dot(self._e_part, K, Tt)
            la.axpby(1.0, self._e_part, 1.0, self.e_iter)
            la.dot(self._e_part, R, Tt)
            la.axpby(1.0, self._e_part, 1.0, self.e_iter)

    def _capture_step(self):
        """Graph: the Jacobi amplitude update ``T -= R / D``, via the step store."""
        g = cg.Graph("lmp2 amplitude step")
        with cg.capture(g):
            self._emit_step()
        return g

    def _emit_step(self):
        """Record this phase's ops into the ambient capture.

        Two nodes per store rather than one fused ``T -= R / D``, because the
        step is DIIS's error vector and has to exist as its own tensor for
        ``cg.diis`` to snapshot between replays. The padding stays exactly
        zero (R's padding is 0 and D's is 1), so padded entries contribute
        nothing to the DIIS B matrix.
        """
        for R, D, S, T in zip(self.R_all, self.D_all, self.step_all, self.T_all):
            la.direct_division(-1.0, R, D, 0.0, S)
            la.axpby(1.0, S, 1.0, T)

    def _capture_antisymmetrize(self):
        """Graph: ``Tt = 2 T - T^T``, every pair in two nodes."""
        g = cg.Graph("lmp2 antisymmetrized amplitudes")
        with cg.capture(g):
            self._emit_antisymmetrize()
        return g

    def _emit_antisymmetrize(self):
        """Record this phase's ops into the ambient capture."""
        for Tt, T in zip(self.Tt_all, self.T_all):
            einsums.permute("abp <- bap", Tt, T)
            la.axpby(2.0, T, -1.0, Tt)

    def lmp2_iterations(self, optimize=True):
        """Solve the local MP2 equations as ONE graph with a loop node.

        The body is the whole iteration - antisymmetrize, prologue, couplings,
        repack, residual, energy, step - captured once and replayed by a
        ``Graph::add_loop`` node. The predicate is ``cg.diis(...).wrap(...)``
        around the convergence test: the one place per iteration that has to
        be host code, because both read values the replay just produced. The
        accelerator extrapolates the (T, step) bucket stores in place with
        einsums operations; its error vector is the materialized Jacobi step,
        where the deleted host-side ``_DIIS`` used the raw residual, so the
        two agree to convergence tolerance rather than bit for bit. Once the
        convergence test reports done, ``wrap`` skips the extrapolation - the
        old code extrapolated one final time; that difference is also below
        convergence tolerance.

        ``Tt`` is emitted FIRST, not last. A loop predicate runs after the whole
        body, and DIIS has to land between the step and the rebuild of ``Tt``
        (psi4 extrapolates the amplitudes before rebuilding). Computing ``Tt``
        at the top of the next iteration puts it after the extrapolation with no
        change in meaning: at iteration k the body sees the same ``T`` the old
        seven-graph order did, and ``pno_transform`` has already left ``Tt``
        consistent with ``T`` for iteration 0, which the body then recomputes.

        Records ``t_capture`` (building and optimizing, paid once) and
        ``t_iterate`` (replaying) separately, as before.

        On why this is one graph now. The seven were separate because merging
        them changed results, and that was a real ComputeGraph defect, not a
        semantic constraint: a view covering its parent exactly never
        alias-linked, and the sort turned the missing edge into a reorder (see
        :meth:`_capture_residual`; fixed in ``8e4174d48``). With the fix the
        merge is bit-identical - what is STILL not safe is giving the merged
        graph the OpenMP executor that repack alone wanted: with the residual's
        batched GEMMs in the same graph, an OpenMP team nests them inside
        OpenBLAS's own threads and the energy moves in the last few digits. So
        the body runs under the default executor. Repack's parallelism is
        bought back INSIDE the node instead: ``cg::gather`` runs its outer walk
        on an OpenMP team when it is not already inside one (its writes are
        disjoint by construction), which threads the repack under the default
        executor without nesting anything. Scoping an executor to part of a
        graph remains the general answer for node-level parallelism here.
        """
        import time as _time
        _t0 = _time.perf_counter()
        self._allocate_iteration_tensors()

        state = {"curr": 0.0, "prev": 0.0, "iters": 0, "converged": False}

        def advance(iteration):
            """Convergence test: read the iteration out and decide.

            Returns True to keep going. Everything here is host code by
            necessity - the energy and the residual norm are values the replay
            has only just produced. ``cg.diis`` wraps this, so the DIIS step
            runs right after a True return.
            """
            state["prev"], state["curr"] = state["curr"], float(ten.view(self.e_iter)[0])
            # Max RMS over the pairs, on each pair's logical block: the padding
            # is identically zero and would otherwise dilute the norm.
            r_curr = 0.0
            for ij, n in enumerate(self.n_pno):
                if n:
                    block = self.pair_block(self.R_all, ij)[:n, :n]
                    r_curr = max(r_curr, float(np.sqrt(np.vdot(block, block).real / block.size)))

            self._print(
                f"    {iteration:>4}  {state['curr']:>18.12f} "
                f"{state['curr'] - state['prev']:>12.3e} {r_curr:>12.3e}"
            )
            state["iters"] = iteration + 1
            state["converged"] = (
                iteration > 0
                and abs(state["curr"] - state["prev"]) < self.cut.e_convergence
                and abs(r_curr) < self.cut.r_convergence
            )
            return not state["converged"]

        if self.use_diis:
            acc = cg.diis(list(zip(self.T_all, self.step_all)), k=self.cut.diis_max_vecs)
            predicate = acc.wrap(advance)
        else:
            predicate = advance

        g = cg.Graph("lmp2")
        body = g.add_loop("lmp2 iteration", self.cut.maxiter + 1, predicate)
        with cg.capture(body):
            self._emit_antisymmetrize()
            self._emit_prologue()
            self._emit_couplings()
            self._emit_repack()
            self._emit_residual()
            self._emit_energy()
            self._emit_step()

        self._print(
            "\n  ==> Local MP2 <==\n"
            f"    captured {body.num_nodes()} nodes in one loop body"
        )
        if optimize:
            # Deliberately not applying the pass pipeline to the body. The
            # batched phases dominate it (the couplings, repack and residual are
            # most of the nodes and carry ~1000 operands each), and applying the
            # pipeline to them was measured at 445 ms against a 324 ms replay
            # for the whole solve while removing exactly one node. The four
            # small phases used to get passes when they were their own graphs;
            # folding trades that for the boundaries. Worth re-measuring rather
            # than assuming, which is why the flag still exists.
            self._print("    optimization: skipped (batched body, see docstring)")

        self.t_capture = _time.perf_counter() - _t0
        self._print(f"    capture: {self.t_capture * 1e3:.1f} ms (one-time)")
        self._print(f"\n    {'iter':>4}  {'Corr. Energy':>18} {'Delta E':>12} {'Max R':>12}")
        _t0 = _time.perf_counter()

        g.execute()

        if not state["converged"]:
            raise RuntimeError("maximum DLPNO iterations exceeded")
        e_curr = state["curr"]
        self.n_iterations = state["iters"]

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
