#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The PNO overlap numerics: the first DLPNO stage with a stated contract.

A free function taking only cross-boundary types, which is what makes it
promotable. It was a method reading 27 fields off ``DLPNOMP2`` until the
planning half was split away into ``plan_pno_couplings``; what is left takes
seven parameters and returns one contract.

Its own module, and not for tidiness: this file is the thing the C++ port
replaces, so its boundary is the boundary the port has to reproduce. Anything
that creeps back in here as a reference to the solver object is a field that
would have to join the contract.
"""

import numpy as np

from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import DLPNOBase
from .contracts import PnoOverlaps

__all__ = ["compute_pno_overlaps"]

#: Replay a setup graph as an OpenMP team over its independent nodes. A
#: staticmethod on DLPNOBase; bound here so this module needs no instance.
#: Never a ThreadPoolExecutor - see the note on DLPNOBase._run.
_run_setup_graph = DLPNOBase._run


def compute_pno_overlaps(
    X_pno,
    S_pao,
    lmopair_to_paos,
    n_pno,
    bucket_of,
    bucket_dims,
    plan,
):
    """The arithmetic the plan describes. This is the half that gets promoted.

    Everything here is driven by ``plan``: the operand lists for both
    batched GEMMs, the scaling, and the permuting gather are all read off
    its flat arrays. Which is the point - roughly half this phase's time is
    building the 32,948 tensor views those operand lists are made of, every
    one a pybind round trip, and a C++ backend builds them natively and
    never materializes a view.
    """
    npao = ten.shape(S_pao)[0]
    n_lmo_pairs = len(n_pno)

    # Each pair's PNO transform on the full PAO axis, padded to its bucket.
    # The scatters are captured with the GEMMs that consume them rather than
    # run here, so they parallelize too and the graph orders them itself.
    X_pad = [None] * n_lmo_pairs
    for ij in range(n_lmo_pairs):
        if n_pno[ij]:
            X_pad[ij] = ten.zeros(f"X (padded) {ij}", [npao, bucket_dims[bucket_of[ij]]])

    # One overlap tensor per shape class, in pair order, rank 2 so that a
    # coupling, a (pair, class) run, and the whole class are all column
    # ranges of it. The extra slot at the end is a reserved zero, which the
    # transposed copy's padding reads from.
    S_cls = []
    for ci, cls in enumerate(plan.classes):
        M_b, M_p = bucket_dims[cls[0]], bucket_dims[cls[1]]
        S_cls.append(ten.zeros(f"S (class {cls})",
                                     [M_b, M_p * (plan.slots_pair[ci] + 1)]))

    half = {ij: ten.zeros(f"half {ij}", [bucket_dims[bucket_of[ij]], npao])
            for ij in plan.coupled_pairs}

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
    # Each coupling's output block, a column range of its class's tensor.
    # Sliced once and kept: a slice is a pybind round trip building a fresh
    # view object, and at 32948 couplings that is 0.24 s, so doing it twice
    # would cost more than everything else in this phase.
    by_shape = {}
    for ij, p, ci, dst in zip(plan.pair, plan.partner, plan.cls, plan.dest_slot):
        cls = plan.classes[ci]
        M_b, M_p = bucket_dims[cls[0]], bucket_dims[cls[1]]
        # The destination is described, not constructed. Column block
        # `slot` of a (M_b, M_p * N) column-major tensor starts at
        # element slot * M_p * M_b and inherits the parent's leading
        # dimension, which is exactly what batched_gemm_blocked wants.
        by_shape.setdefault(ci, []).append(
            (half[ij], X_pad[p], dst * M_p * M_b))

    by_bucket = {}
    for ij in plan.coupled_pairs:
        by_bucket.setdefault(bucket_of[ij], []).append(ij)

    g = cg.Graph("pno overlaps")
    with cg.capture(g):
        for ij in range(n_lmo_pairs):
            if n_pno[ij]:
                sparse.scatter_into(X_pad[ij], X_pno[ij],
                                    [lmopair_to_paos[ij], range(n_pno[ij])])
        for members in by_bucket.values():
            cg.batched_gemm(1.0, [X_pad[ij] for ij in members],
                            [S_pao] * len(members), 0.0,
                            [half[ij] for ij in members], trans_a=True)
        for ci, members in by_shape.items():
            cls = plan.classes[ci]
            M_b, M_p = bucket_dims[cls[0]], bucket_dims[cls[1]]
            cg.batched_gemm_blocked(1.0, [h for h, _, _ in members],
                                    [x for _, x, _ in members], 0.0,
                                    S_cls[ci], [o for _, _, o in members],
                                    M_b, M_p)
    # No pass pipeline. This graph is emitted in the form the passes would
    # produce - the batches are already fused, and there is nothing to share
    # between them - so applying them is pure cost, and it is not small:
    # 14 ms to apply, plus the 27 ms of populate_default it forces.
    #
    # Parallelism across the batch, each member serial underneath. The
    # OpenMP executor rather than Dataflow because Dataflow runs nodes on
    # std::thread workers, which is the pattern the OpenMP-built OpenBLAS
    # miscomputes; see DLPNOBase.precompute_fits.
    _run_setup_graph(g)

    # One vectorized scaling per class: every coupling owns a column range.
    # Built in one pass over the couplings rather than one per class - there
    # are 32948 of them against 16 classes, and the obvious nesting is a
    # third of a second of pure Python.
    factors = [np.ones(bucket_dims[cls[1]] * (plan.slots_pair[ci] + 1))
               for ci, cls in enumerate(plan.classes)]
    for ci, dst, factor in zip(plan.cls, plan.dest_slot, plan.factor):
        M_p = bucket_dims[plan.classes[ci][1]]
        factors[ci][dst * M_p:(dst + 1) * M_p] = np.sqrt(abs(factor))
    for ci in range(len(plan.classes)):
        ten.view(S_cls[ci])[...] *= factors[ci][np.newaxis, :]

    # The partner-major transposed copy the first half consumes, taken from
    # the pair-major one by a single permuting gather per class rather than
    # a second GEMM per coupling. Transposing and reordering are the same
    # pass, and 32948 more output slices would have cost 0.65 s.
    #
    # It reads S AFTER the scaling above, so it inherits sqrt|f| and only
    # the sign is left to apply. The sign has to ride on exactly one of the
    # two copies - S appears on both sides of the congruence, so a sign in
    # both would square away - and this is the copy the first half reads.
    S_T = []
    # Held past the capture: the graph keeps tensors by slot pointer, so a
    # view built inline in the loop is freed at the end of the expression
    # and the replay reads released memory.
    src_views = []
    for ci, cls in enumerate(plan.classes):
        M_b, M_p = bucket_dims[cls[0]], bucket_dims[cls[1]]
        S_T.append(ten.zeros(
            f"S^T (class {cls})", [M_p, M_b, plan.slots_partner[ci]]))
        src_views.append(S_cls[ci].reshape_view(
            [M_b, M_p, plan.slots_pair[ci] + 1]))

    g2 = cg.Graph("pno overlaps transposed")
    with cg.capture(g2):
        for ci, cls in enumerate(plan.classes):
            M_b, M_p = bucket_dims[cls[0]], bucket_dims[cls[1]]
            la.gather(S_T[ci], src_views[ci],
                      [list(range(M_b)), list(range(M_p)), plan.inv_perm[ci]],
                      [1, 0, 2])
    g2.set_executor(cg.OpenMPExecutor())
    g2.execute()

    signed = [np.ones(n) for n in plan.slots_partner]
    for ci, src, sign in zip(plan.cls, plan.src_slot, plan.sign):
        signed[ci][src] = sign
    for ci in range(len(plan.classes)):
        ten.view(S_T[ci])[...] *= signed[ci][np.newaxis, np.newaxis, :]

    return PnoOverlaps(S_cls=S_cls, S_T=S_T)
