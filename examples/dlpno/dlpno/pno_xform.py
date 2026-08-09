#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The PNO transform numerics: the second DLPNO stage with a stated contract.

A free function taking only cross-boundary types, which is what makes it
promotable. It was the numerics half of ``DLPNOMP2.pno_transform``, a method
reading 32 fields off the solver until the planning half was split away into
``plan_pno_transform``; what is left takes fifteen parameters and returns one
contract.

Its own module for the same reason ``pno_overlaps.py`` is: this file is the
thing the C++ port replaces, so its boundary is the boundary the port has to
reproduce. Anything that creeps back in as a reference to the solver object is
a field that would have to join the contract.

Why this stage was promoted, so the reason survives the port: re-profiled
2026-08-08, capture EMISSION is 45-56% of the phase - ~1,630 nodes across the
three graphs at ~8 us of Python per node - while the truncation scan everyone
suspected is 1.1-1.3%. Emission cost is per node and linear in the pair count,
which is exactly the cost a language change removes and a batching DSL cannot.
"""

import numpy as np

from einsums import linalg as la
import einsums.graph as cg

from . import tensors as ten
from .base import DLPNOBase, batched_eigh
from .contracts import PnoTransform

__all__ = ["transform_pnos"]

#: Replay a setup graph as an OpenMP team over its independent nodes. A
#: staticmethod on DLPNOBase; bound here so this module needs no instance.
#: Never a ThreadPoolExecutor - see the note on DLPNOBase._run.
_run_setup_graph = DLPNOBase._run


def transform_pnos(
    q_ia,
    fits,
    fit_of,
    fit_pos,
    dom_X,
    dom_e,
    dom_F,
    dom_of,
    ribfs,
    paos,
    lmo_j,
    shift,
    pno_scale,
    min_pnos,
    t_cut_pno,
):
    """Build each upper pair's truncated, canonical PNO basis.

    For every upper pair: form the exchange operator over the PAO domain,
    take one semicanonical MP2 step for the amplitudes, diagonalize the
    resulting pair density to get the PNOs, keep the ones whose occupation
    number clears the (per-pair scaled) ``T_CUT_PNO``, and recanonicalize
    what survives. The energy lost to the truncation comes back per pair as
    ``de_pno``; the caller accumulates psi4's PNO truncation correction.

    Three stages rather than one loop, so the two eigendecompositions per
    pair can be batched, and each stage captured as one graph so its pairs
    run as an OpenMP team rather than one at a time on the calling thread.
    Each stage is the same arithmetic in the same order as the per-pair loop
    it replaces; only the issue order changes.

    Everything domain-shaped arrives deduplicated: ``fits`` and the three
    ``dom_*`` lists hold one entry per DISTINCT domain, and ``fit_of`` /
    ``dom_of`` map each upper pair onto them. Pairs sharing a domain share
    the tensors, exactly as the memo caches shared them when this was a
    method.

    Args:
        q_ia: Three-index integrals ``(Q | i a)``, the full tensor.
        fits: Per distinct (auxiliary domain, LMO-set) block, the solved fit
            coefficients ``J^-1 (Q | i u)``, rank 3 ``(nq, nk, nu)``.
        fit_of: Per upper pair, its index into ``fits``.
        fit_pos: Per upper pair, LMO ``i``'s slot on ``fits``'s middle axis.
        dom_X: Per distinct PAO domain, the orthocanonicalizer ``X_pao``.
        dom_e: Per distinct PAO domain, the canonical orbital energies.
        dom_F: Per distinct PAO domain, the canonical-basis Fock matrix.
        dom_of: Per upper pair, its index into the ``dom_*`` lists.
        ribfs: Per upper pair, the auxiliary indices of its fit domain.
        paos: Per upper pair, the PAO indices its domain covers.
        lmo_j: Per upper pair, LMO ``j`` for the ``(Q | j a)`` gather.
        shift: Per upper pair, ``F_ii + F_jj``.
        pno_scale: Per upper pair, the core-pair scaling on ``T_CUT_PNO``.
        min_pnos: Pairs keep at least this many PNOs, domain size permitting.
        t_cut_pno: The occupation-number cutoff.
    """
    n_upper = len(fit_of)

    # Stage 1: everything up to the pair density, which depends on nothing
    # outside its own pair.
    st = [dict() for _ in range(n_upper)]
    g1 = cg.Graph("PNO stage 1")
    with cg.capture(g1):
        for u in range(n_upper):
            # The exchange operator K_ij[a,b] = (i a | j b) over the pair's
            # PAO domain, density-fitted within its auxiliary domain: one
            # gather straight into a rank-3 block plus one GEMM through its
            # rank-2 view. Selecting a single LMO leaves a length-1 middle
            # axis; dropping it in numpy would be a host copy, and a host
            # copy in the middle of this pins the whole pair loop to the
            # calling thread. The view costs nothing: (nq, 1, nu) column
            # major has exactly the layout of (nq, nu).
            fit_i = fits[fit_of[u]][:, fit_pos[u], :]
            blk = ten.zeros("(Q|j a)", [len(ribfs[u]), 1, len(paos[u])])
            la.gather(blk, q_ia,
                      [[int(p) for p in ribfs[u]], [int(lmo_j[u])],
                       [int(p) for p in paos[u]]])
            K_pao = ten.doublet(fit_i, blk[:, 0, :], trans_a=True, name="K (ia|jb)")

            # Rotate into the domain's canonical PAO basis, removing linear
            # dependencies. Shared by every pair on the same domain.
            X_pao, e_pao, F_pao_ij = dom_X[dom_of[u]], dom_e[dom_of[u]], dom_F[dom_of[u]]
            K_pao = ten.triplet(X_pao, K_pao, X_pao, trans_a=True, name="K (can PAO)")

            # One semicanonical MP2 step gives the amplitudes the PNOs come from.
            D_pao = ten.pair_denominator(e_pao, shift[u])
            T_pao = ten.zeros("T (can PAO)", ten.shape(K_pao))
            la.direct_division(1.0, K_pao, D_pao, 0.0, T_pao)
            Tt_pao = ten.antisymmetrize(T_pao)

            # Pair density; its eigenvectors are the PNOs, eigenvalues the
            # PNO occupation numbers.
            D_ij = ten.doublet(Tt_pao, T_pao, trans_b=True, name="D (pair)")
            la.axpby(1.0, ten.doublet(Tt_pao, T_pao, trans_a=True), 1.0, D_ij)

            st[u] = dict(
                K_pao=K_pao, T_pao=T_pao, Tt_pao=Tt_pao, D_ij=D_ij,
                X_pao=X_pao, F_pao_ij=F_pao_ij,
                # Pointer-writer dots: the returning form has no value to
                # return until the graph runs, and throws under capture.
                e_ij_initial=ten.dot_into(ten.scalar("e_ij"), K_pao, Tt_pao),
                e_ij_os_initial=ten.dot_into(ten.scalar("e_ij os"), K_pao, T_pao),
            )
    _run_setup_graph(g1)

    occs, vecs = batched_eigh([s["D_ij"] for s in st], _run_setup_graph,
                              descending=True, label="PNO")

    # Stage 2: the truncation decision, which is data dependent and so cannot
    # be captured, then the Fock matrix in each surviving subspace, which can.
    for u, (pno_occ, X_pno) in enumerate(zip(occs, vecs)):
        nvir_ij = ten.shape(st[u]["K_pao"])[0]
        keep = min(min_pnos, nvir_ij)
        occ = ten.view(pno_occ)
        for a in range(keep, nvir_ij):
            if abs(occ[a]) >= pno_scale[u] * t_cut_pno:
                keep += 1
        st[u]["keep"] = keep
        # The survivors are the LEADING columns, because the eigenvectors
        # came back sorted by descending occupation number, so truncating is
        # a view rather than a gather: no node, no allocation, and no copy
        # of a matrix we already have. Taken out here rather than under the
        # capture below, where slicing would record a view node of its own
        # and put back the node it just saved.
        st[u]["X_pno"] = X_pno[:, :keep]

    g2 = cg.Graph("PNO stage 2")
    with cg.capture(g2):
        for u in range(n_upper):
            # Orthonormal but not canonical yet; rotate so F is diagonal.
            st[u]["Fmo"] = ten.triplet(st[u]["X_pno"], st[u]["F_pao_ij"],
                                       st[u]["X_pno"], trans_a=True,
                                       name="C^T F C")
    _run_setup_graph(g2)

    e_pnos, canons = batched_eigh([s["Fmo"] for s in st], _run_setup_graph,
                                  descending=True, label="PNO canon")

    # Stage 3: rotate into the canonical PNO basis and record the pair. The
    # lower-triangle mirror (K_ji = K_ij^T) is the caller's: it is a host
    # transpose at scatter time, not arithmetic, so it stays outside the
    # contract.
    K_out = [None] * n_upper
    T_out = [None] * n_upper
    X_out = [None] * n_upper
    g3 = cg.Graph("PNO stage 3")
    with cg.capture(g3):
        for u, pno_canon in enumerate(canons):
            K_pao, T_pao, Tt_pao = st[u]["K_pao"], st[u]["T_pao"], st[u]["Tt_pao"]
            X_pao = st[u]["X_pao"]

            X_pno = ten.doublet(st[u]["X_pno"], pno_canon, name="X (PNO)")

            K_pno = ten.triplet(X_pno, K_pao, X_pno, trans_a=True, name="K (PNO)")
            T_pno = ten.triplet(X_pno, T_pao, X_pno, trans_a=True, name="T (PNO)")
            Tt_pno = ten.triplet(X_pno, Tt_pao, X_pno, trans_a=True, name="Tt (PNO)")

            st[u]["e_ij_trunc"] = ten.dot_into(ten.scalar("e trunc"), K_pno, Tt_pno)
            st[u]["e_ij_os_trunc"] = ten.dot_into(ten.scalar("e os trunc"), K_pno, T_pno)

            # PAO domain -> canonical PNO, in one transform.
            K_out[u] = K_pno
            T_out[u] = T_pno
            X_out[u] = ten.doublet(X_pao, X_pno, name="X (PAO->PNO)")
    _run_setup_graph(g3)

    # The truncation-energy dots come back as the scalar tensors the graph
    # wrote; the caller subtracts them after run(). Floats here would violate
    # the contract vocabulary: a captured stage returns before its graph
    # executes, so a computed float would be read too early.
    return PnoTransform(
        K_pno=K_out, T_pno=T_out, X_pno=X_out,
        e_pno=e_pnos, n_pno=[s["keep"] for s in st],
        e_initial=[s["e_ij_initial"] for s in st],
        e_os_initial=[s["e_ij_os_initial"] for s in st],
        e_trunc=[s["e_ij_trunc"] for s in st],
        e_os_trunc=[s["e_ij_os_trunc"] for s in st],
    )
