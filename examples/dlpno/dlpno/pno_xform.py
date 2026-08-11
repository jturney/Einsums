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

__all__ = ["transform_pnos", "pair_quantities"]


def pair_quantities(q_ia, fits, fit_of, fit_pos, dom_X, dom_e, dom_F, dom_of,
                    ribfs, paos, lmo_j, shift):
    """Each upper pair's exchange operator, semicanonical amplitudes and density.

    Everything that depends on nothing outside its own pair, as one graph so the
    pairs run as an OpenMP team rather than one at a time on the calling thread.

    Split out because psi4 runs it twice for different reasons. It is the first
    half of ``DLPNOCCSD::compute_pair_energies``, whose ``crude`` template
    parameter guards the PNO construction that follows: the crude prescreening
    pass wants only ``e_ij_initial`` from here and throws the rest away, while
    the refined pass and :func:`transform_pnos` go on to build the PNOs. Making
    the crude pass call :func:`transform_pnos` and ignore its output would
    quadruple the phase - psi4 measures 0.83 s against 3.93 s for the two passes
    on ethanol/cc-pVTZ, and nearly all of that gap is the eigendecompositions.

    Returns one dict per upper pair, in the planner's order.
    """
    n_upper = len(fit_of)
    st = [dict() for _ in range(n_upper)]
    g = cg.Graph("PNO pair quantities")
    with cg.capture(g):
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
    _run_setup_graph(g)
    return st


def _truncate(occ, scale, min_pnos, t_cut_pno, t_cut_trace, t_cut_energy,
              K_init, Tt_init, e_initial, nvir):
    """How many PNOs survive, by psi4's three OR'd criteria plus the floor.

    A faithful port of the selection loop in ``DLPNOCCSD::compute_pair_energies``
    (ccsd.cc:649-662), which is the MP2 branch's loop with two disjuncts added.
    Both extra criteria are *running* quantities, so unlike the occupation test
    they are not monotone in ``a`` and the loop cannot be rewritten as a
    threshold scan.

    What survives is the leading ``keep`` COLUMNS, which is not the same set as
    the indices that passed. The two can differ: the occupation criterion is
    monotone only while the eigenvalues stay positive, and a pair density that
    has picked up a numerically negative tail can have ``|occ|`` rise again, so
    a later index passes after an earlier one failed. psi4 truncates on the
    count and accumulates the energy on the holed index list, so this tracks
    both.

    One asymmetry worth preserving rather than tidying: the energy is evaluated
    on the set BEFORE ``a`` joins it while the occupation sum includes ``a``.
    psi4's ``recompute_pnos`` has it the other way round, and the two are not
    interchangeable - see :meth:`~dlpno.ccsd.DLPNOCCSD.recompute_pnos`.
    """
    occ_sum = 0.0
    e_pno = 0.0
    occ_total = float(np.sum(occ)) if nvir else 0.0
    cut = scale * t_cut_pno
    energy_floor = t_cut_energy * abs(e_initial)
    kept = []
    # Zero-safe on ``occ_total``: an empty domain has nothing to keep and no
    # total to divide by. psi4 never reaches the divide because it never
    # reaches a pair with no virtuals.
    K = None if K_init is None else ten.view(K_init)
    Tt = None if Tt_init is None else ten.view(Tt_init)

    for a in range(nvir):
        if not (abs(occ[a]) >= cut
                or (occ_total != 0.0 and occ_sum / occ_total < t_cut_trace)
                or abs(e_pno) < energy_floor
                or a < min_pnos):
            continue
        if K is not None:
            # The energy the PNOs kept SO FAR recover, before a joins them.
            ix = np.ix_(kept, kept)
            e_pno = float(np.vdot(K[ix], Tt[ix]).real) if kept else 0.0
        occ_sum += occ[a]
        kept.append(a)
    # psi4's floor of one, which only bites when every criterion is off and the
    # occupations are all exactly zero.
    return min(nvir, max(1, len(kept)))

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
    t_cut_trace=0.0,
    t_cut_energy=0.0,
):
    """Build each upper pair's truncated, canonical PNO basis.

    For every upper pair: form the exchange operator over the PAO domain,
    take one semicanonical MP2 step for the amplitudes, diagonalize the
    resulting pair density to get the PNOs, keep the ones that clear the
    truncation criteria, and recanonicalize what survives. The energy lost to
    the truncation comes back per pair as ``de_pno``; the caller accumulates
    psi4's PNO truncation correction.

    **Three truncation criteria, OR'd, not applied in sequence.** psi4's MP2
    branch (``DLPNO::pno_transform``) has only the first; its coupled-cluster
    branch (``DLPNOCCSD::compute_pair_energies``) adds the other two, and this
    serves both because zero switches each of the extra two off:

    * *occupation* - keep ``a`` while ``|occ[a]| >= pno_scale[u] * t_cut_pno``.
    * *trace* - keep while the occupation recovered so far is below
      ``t_cut_trace`` of the total. Zero disables it, since a fraction is never
      below zero.
    * *energy* - keep while the pair energy recovered so far is below
      ``t_cut_energy`` of the untruncated one. Zero disables it the same way.

    A PNO survives if it passes ANY of them, so the count is the largest of the
    four cut points (``min_pnos`` being the fourth), not the smallest.

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
    st = pair_quantities(q_ia, fits, fit_of, fit_pos, dom_X, dom_e, dom_F,
                         dom_of, ribfs, paos, lmo_j, shift)
    n_upper = len(st)

    occs, vecs = batched_eigh([s["D_ij"] for s in st], _run_setup_graph,
                              descending=True, label="PNO")

    # The two extra criteria need the pair's integrals in the UNTRUNCATED PNO
    # basis, which is a pair of triple products per pair that the occupation
    # criterion alone does not want. Built only when the energy criterion is
    # live, so an MP2 run pays nothing for a coupled-cluster feature.
    if t_cut_energy > 0.0:
        g_init = cg.Graph("PNO initial projection")
        with cg.capture(g_init):
            for u, X_pno in enumerate(vecs):
                st[u]["K_init"] = ten.triplet(X_pno, st[u]["K_pao"], X_pno,
                                              trans_a=True, name="K (untrunc PNO)")
                st[u]["Tt_init"] = ten.triplet(X_pno, st[u]["Tt_pao"], X_pno,
                                               trans_a=True, name="Tt (untrunc PNO)")
        _run_setup_graph(g_init)

    # Stage 2: the truncation decision, which is data dependent and so cannot
    # be captured, then the Fock matrix in each surviving subspace, which can.
    for u, (pno_occ, X_pno) in enumerate(zip(occs, vecs)):
        nvir_ij = ten.shape(st[u]["K_pao"])[0]
        occ = ten.view(pno_occ)
        st[u]["keep"] = _truncate(
            occ, pno_scale[u], min_pnos, t_cut_pno, t_cut_trace, t_cut_energy,
            st[u].get("K_init"), st[u].get("Tt_init"),
            float(ten.view(st[u]["e_ij_initial"])[0]), nvir_ij,
        )
        keep = st[u]["keep"]
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
