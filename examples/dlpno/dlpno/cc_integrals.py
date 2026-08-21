#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Every pair-local integral block the coupled-cluster residual reads.

A port of psi4's ``DLPNOCCSD::compute_pno_integrals``, and one of the two
biggest phases in psi4's own profile (10.7 s of ethanol/cc-pVTZ's 42.1 s at one
thread, second only to the iteration itself). So it is a performance surface,
not bookkeeping, and it is structured accordingly: everything is grouped by pair
and emitted into captured graphs so the independent per-pair chains thread,
rather than run one at a time on the calling thread.

**The naming is Jiang Eq. 71-74, which psi4 states once and then relies on.**

    J_{pq}^{rs} = (pq|rs)          K_{pq}^{rs} = (pr|qs)
    L_{pq}^{rs} = 2(pr|qs) - (ps|qr)

so ``K_mibj`` is ``(m i | b j)`` and ``J_ijmb`` is ``(i j | m b)`` - the letters
say which indices are adjacent in the charge-density notation, not which are
occupied.

**Two metric powers, which is the part that is easy to get wrong.** Most blocks
are fitted symmetrically, ``B^Q_{pq} = sum_P (P|Q)^{-1/2} (P|pq)``, so any
``B^T B`` reconstructs an integral and each factor is reusable. The two
non-projected families are the exception: they contract a fitted factor against
an UNFITTED one, so their fitted side carries the full inverse instead. Applying
the same power to both sides of those would double-count the metric.
"""

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import DLPNOBase
from .contracts import PnoIntegralBlocks

__all__ = ["PnoIntegrals", "compute_pno_integrals", "pno_integral_blocks"]

_run_setup_graph = DLPNOBase._run


class PnoIntegrals:
    """The blocks, indexed by pair ordinal.

    One object rather than a dozen parallel lists, because every consumer wants
    several of them for the same pair and psi4's own member-per-block layout is
    the thing that makes its residual hard to read.

    Blocks absent for a pair - the density-fitted ones on a weak pair, which
    psi4 also skips - are ``None`` rather than zero, so a consumer reading one
    it should not touches an obvious failure instead of quietly adding nothing.
    """

    def __init__(self, n_pairs):
        # 1-external
        self.K_mibj = [None] * n_pairs
        self.J_ijmb = [None] * n_pairs
        self.L_mibj = [None] * n_pairs
        # 2-external
        self.L_iajb = [None] * n_pairs
        # 3-external. Rank 3 ``(e, a, f)``, NOT psi4's flattened ``(e, a*f)``.
        #
        # psi4 stores it flat and then re-blocks it two different ways: Jiang
        # Eq. 83b reads it as ``(b, a*c)`` and Eq. 84b as ``(c*a, b)``. Both
        # re-blockings depend on the storage order, psi4's Matrix is row major
        # and einsums tensors are column major, so a flat port would have to
        # invert every one of them by hand.
        #
        # Worse, the obvious test cannot catch a mistake. ``(i e | a f)`` is
        # symmetric under ``a <-> f``, so storing the transpose reproduces the
        # right integral and passes an oracle check; it only goes wrong several
        # equations later, where a re-blocking separates the two indices. Rank 3
        # makes every consumer name its indices instead of counting strides.
        self.K_ivvv = [None] * n_pairs
        # density-fitted factors, strong pairs only
        self.i_Qk = [None] * n_pairs
        self.i_Qa = [None] * n_pairs
        self.Qma = [None] * n_pairs
        self.Qab = [None] * n_pairs
        # 2-external non-projected, per pair per neighbour
        self.J_ikac = [None] * n_pairs
        self.K_iakc = [None] * n_pairs

    def qma(self, ij_to_ji, ij):
        """``B^Q_{m a}`` for pair ``ij``, psi4's ``QIA_PNO``.

        Stored only for ``i <= j``, because ``ij`` and ``ji`` share a PNO basis
        and a neighbour list, so the two blocks are identical rather than
        transposes. psi4 maps the read the same way; doing it here rather than
        at each call site is what keeps a caller from silently reading a
        ``None`` for a lower-triangle pair.
        """
        return self.Qma[ij] if self.Qma[ij] is not None else self.Qma[ij_to_ji[ij]]

    def qab(self, ij_to_ji, ij):
        """``B^Q_{a b}`` for pair ``ij``, psi4's ``QAB_PNO``. See :meth:`qma`."""
        return self.Qab[ij] if self.Qab[ij] is not None else self.Qab[ij_to_ji[ij]]

    def nbytes(self):
        """Total stored bytes, for the in-core decision's memory report."""
        total = 0
        for name in ("K_mibj", "J_ijmb", "L_mibj", "L_iajb", "K_ivvv",
                     "i_Qk", "i_Qa", "Qma", "Qab"):
            for block in getattr(self, name):
                if block is not None:
                    total += int(np.prod(ten.shape(block))) * 8
        for name in ("J_ikac", "K_iakc"):
            for per_pair in getattr(self, name):
                for block in (per_pair or []):
                    if block is not None:
                        total += int(np.prod(ten.shape(block))) * 8
        return total

    def report(self, printer):
        by_kind = {}
        for name in ("K_mibj", "J_ijmb", "L_iajb", "K_ivvv", "Qma", "Qab"):
            blocks = [b for b in getattr(self, name) if b is not None]
            if blocks:
                by_kind[name] = sum(int(np.prod(ten.shape(b))) for b in blocks) * 8
        parts = ", ".join(f"{k} {v / 2**20:.1f}" for k, v in sorted(by_kind.items()))
        printer(f"  CC ints:  {parts} MiB; total {self.nbytes() / 2**20:.1f} MiB")


def compute_pno_integrals(cc):
    """Build every block, for every pair that needs one.

    Follows psi4's own split: the contracted integrals are built for ALL pairs
    (weak pairs still contribute their singles term to the energy and their
    ``L_iajb`` to the intermediates), while the density-fitted factors and the
    two non-projected families are built for strong pairs only, because only the
    strong pairs' residual reads them.

    Only ``i <= j`` is looped; the ``ji`` blocks that are not transposes are
    built alongside their partner, exactly as psi4 does.

    The pairs go through in chunks that fit the memory budget rather than all
    at once. Nothing here reads another pair's raw blocks, so the split changes
    no arithmetic and no summation order - it only releases the gathers sooner.
    See :func:`_chunks` for why that matters.

    Three pieces since the promotion: :func:`_plan_chunk` is the index
    bookkeeping and stays Python, :func:`pno_integral_blocks` is the contracted
    numerics that has a C++ backend, and :func:`_scatter` puts the blocks where
    the consumers look for them. The numerics goes through the stage registry,
    so whichever backend has been selected runs from every entry point.
    """
    # Imported here rather than at module scope because ``stages`` imports the
    # solvers, which import this module.
    from .stages import compute_pno_integrals as integral_stage

    out = PnoIntegrals(cc.n_lmo_pairs)
    upper = [ij for ij, (i, j) in enumerate(cc.ij_to_i_j)
             if i <= j and cc.n_pno[ij]]

    chunks = _chunks(cc, upper)
    if len(chunks) > 1:
        # Worth a line, because the chunk count is the difference between this
        # phase's peak and its budget, and a benchmark whose peak depends on an
        # unreported number is not reproducible.
        cc._print(f"  CC ints:  {len(upper)} pairs in {len(chunks)} chunks "
                  f"against a {cc.cut.in_core_memory / 2**30:.2f} GiB budget")
    for chunk in chunks:
        plan = _plan_chunk(cc, chunk)
        _scatter(cc, chunk, plan, integral_stage(**plan), out)
    return out


def _plan_chunk(cc, chunk):
    """One chunk's stage arguments: every domain list, flattened.

    Pure index bookkeeping against the sparsity, which is what makes it the
    seam. Nothing here depends on the value of an integral, and the only
    floating-point objects that cross are the whole three-index tensors and the
    pairs' PNO transforms, which are read, never written.

    The neighbour bookkeeping is the part worth reading twice. Each of the two
    non-projected families carries a virtual index belonging to a NEIGHBOUR pair
    ``kj``, so the numerics needs that neighbour's PNO transform and the
    position of its PAO domain inside this pair's extended domain. Both are
    described here as an index per neighbour slot into two deduplicated lists,
    rather than as a nested structure per pair: a slot is ``-1`` when the
    neighbour pair does not exist or carries no PNOs, which is the same test
    psi4 makes, and the numerics skips it on both sides of the boundary.
    """
    args = dict(
        q_ij=cc.q_ij, q_ia=cc.q_ia, q_ab=cc.q_ab, metric=cc.metric,
        X_pno=[], i_lmo=[], j_lmo=[], n_pno=[], strong=[],
        ribfs=[], paos=[], lmos=[], extended=[],
        rot_X=[], rot_paos=[], nb_ij=[], nb_ji=[],
    )
    rot_of = {}

    def slots(lmos, partner):
        """Per neighbour ``k``, where to find pair ``(k, partner)``'s basis."""
        out = []
        for k in lmos:
            kj = int(cc.i_j_to_ij[k, partner])
            if kj == -1 or not cc.n_pno[kj]:
                out.append(-1)
                continue
            if kj not in rot_of:
                rot_of[kj] = len(args["rot_X"])
                args["rot_X"].append(cc.X_pno[kj])
                args["rot_paos"].append([int(u) for u in cc.lmopair_to_paos[kj]])
            out.append(rot_of[kj])
        return out

    for ij in chunk:
        i, j = cc.ij_to_i_j[ij]
        ribfs, paos, lmos, extended = _domains(cc, ij)
        is_strong = cc.i_j_to_ij_strong[i, j] != -1
        args["X_pno"].append(cc.X_pno[ij])
        args["i_lmo"].append(int(i))
        args["j_lmo"].append(int(j))
        args["n_pno"].append(int(cc.n_pno[ij]))
        args["strong"].append(bool(is_strong))
        args["ribfs"].append([int(q) for q in ribfs])
        args["paos"].append([int(u) for u in paos])
        args["lmos"].append([int(m) for m in lmos])
        args["extended"].append([int(u) for u in extended])
        # Only a strong pair reads a neighbour block, so only a strong pair
        # contributes slots - and with them the neighbour transforms that cross
        # the boundary. The weak pairs are the majority.
        args["nb_ij"].append(slots(lmos, j) if is_strong else [])
        args["nb_ji"].append(slots(lmos, i) if is_strong and i != j else [])
    return args


def pno_integral_blocks(q_ij, q_ia, q_ab, metric, X_pno, i_lmo, j_lmo, n_pno,
                        strong, ribfs, paos, lmos, extended, rot_X, rot_paos,
                        nb_ij, nb_ji):
    """Every block one chunk of pairs needs, in the chunk's order.

    The contracted numerics behind :func:`compute_pno_integrals`, and the half
    with a C++ backend. Everything indexed by pair is indexed by the pair's
    position in the chunk; nothing here knows a pair ordinal, which is what lets
    the same code serve any chunking.

    See :class:`dlpno.contracts.PnoIntegralBlocks` for the output layout and
    :func:`_plan_chunk` for where the arguments come from.
    """
    raw = _raw_blocks(q_ij, q_ia, q_ab, X_pno, i_lmo, j_lmo, n_pno, ribfs,
                      paos, lmos)
    _fit(metric, ribfs, raw)
    blocks = _contract(raw, i_lmo, j_lmo, strong)
    _non_projected(q_ia, q_ab, X_pno, raw, i_lmo, j_lmo, strong, ribfs, paos,
                   lmos, extended, rot_X, rot_paos, nb_ij, nb_ji, blocks)
    return PnoIntegralBlocks(**blocks)


def _absent():
    """The placeholder for a block a pair legitimately does not have.

    Zero extent rather than ``None``, because the contract's lists are parallel
    and a C++ ``std::vector<RuntimeTensor<double>>`` has no third state. The
    caller turns it back into the ``None`` its consumers expect.
    """
    return ten.zeros("absent", [0, 0])


def _scatter(cc, chunk, plan, blocks, out):
    """File one chunk's blocks under their pair ordinals.

    Host-side bookkeeping and two host-side combinations, which is everything
    the numerics deliberately left out: the ``L`` families are elementwise
    sums of blocks that already exist, and the ``ji`` mirroring is a question
    about pair ordinals, which the numerics does not know.

    Every contract field is read into a local ONCE. On the C++ backend each
    attribute access converts a ``std::vector`` into a fresh Python list, so
    indexing ``blocks.K_mibj[p]`` inside the loop would be quadratic in the
    chunk length - the same trap ``_finish_pno_transform`` records.
    """
    K_mibj, K_mjai = blocks.K_mibj, blocks.K_mjai
    J_ijmb = blocks.J_ijmb
    K_ivvv, K_jvvv = blocks.K_ivvv, blocks.K_jvvv
    i_Qk, i_Qa, j_Qk, j_Qa = blocks.i_Qk, blocks.i_Qa, blocks.j_Qk, blocks.j_Qa
    Qma, Qab = blocks.Qma, blocks.Qab
    J_ikac, K_iakc = blocks.J_ikac, blocks.K_iakc
    J_jkac, K_jakc = blocks.J_jkac, blocks.K_jakc
    strong, nb_ij, nb_ji = plan["strong"], plan["nb_ij"], plan["nb_ji"]

    # Where each pair's neighbour slabs start in the flat lists: the same
    # prefix sum the numerics walks, so slot s of pair p is at offset(p) + s.
    base_ij, base_ji, at_ij, at_ji = [], [], 0, 0
    for p in range(len(chunk)):
        base_ij.append(at_ij)
        base_ji.append(at_ji)
        at_ij += len(nb_ij[p])
        at_ji += len(nb_ji[p])

    def neighbours(flat, base, slots):
        """One pair's per-neighbour list, with a ``None`` for every dead slot."""
        return [None if s < 0 else flat[base + k] for k, s in enumerate(slots)]

    for p, ij in enumerate(chunk):
        i, j = cc.ij_to_i_j[ij]
        ji = cc.ij_to_ji[ij]
        out.K_mibj[ij] = K_mibj[p]
        out.J_ijmb[ij] = J_ijmb[p]
        out.K_ivvv[ij] = K_ivvv[p]
        if i != j:
            out.K_mibj[ji] = K_mjai[p]
            out.K_ivvv[ji] = K_jvvv[p]
            # Symmetric in the pair, so ji SHARES the object rather than
            # holding a copy, exactly as psi4 does.
            out.J_ijmb[ji] = out.J_ijmb[ij]

        # L_mibj and L_iajb, both pure host-side combinations of blocks that
        # already exist.
        #
        # psi4 computes L_mibj_[ji] as 3 K_ji - 2 K_ij, because it subtracts the
        # already-overwritten L_mibj_[ij] instead of K_mibj_[ij]. That is a bug,
        # and a harmless one there because nothing ever reads L_mibj_ - it is
        # resized, written and cleared. Written correctly here.
        K_ij = ten.view(out.K_mibj[ij])
        if i == j:
            out.L_mibj[ij] = ten.from_numpy("L (m i|b j)", K_ij)
        else:
            K_ji = ten.view(out.K_mibj[ji])
            out.L_mibj[ij] = ten.from_numpy("L (m i|b j)", 2.0 * K_ij - K_ji)
            out.L_mibj[ji] = ten.from_numpy("L (m j|a i)", 2.0 * K_ji - K_ij)

        # L_iajb = 2 (i a|j b) - (i b|j a), from the exchange operator the PNO
        # transform already left in the amplitude stores.
        n = cc.n_pno[ij]
        K = cc.pair_block(cc.K_all, ij)[:n, :n]
        out.L_iajb[ij] = ten.from_numpy("L (i a|j b)", 2.0 * K - K.T)
        if i != j:
            out.L_iajb[ji] = ten.from_numpy("L (j a|i b)",
                                            ten.view(out.L_iajb[ij]).T)

        # The density-fitted factors and the non-projected families, strong
        # pairs only: nothing else reads them.
        if not strong[p]:
            continue
        out.i_Qk[ij] = i_Qk[p]
        out.i_Qa[ij] = i_Qa[p]
        out.Qma[ij] = Qma[p]
        out.Qab[ij] = Qab[p]
        if i != j:
            out.i_Qk[ji] = j_Qk[p]
            out.i_Qa[ji] = j_Qa[p]
        out.J_ikac[ij] = neighbours(J_ikac, base_ij[p], nb_ij[p])
        out.K_iakc[ij] = neighbours(K_iakc, base_ij[p], nb_ij[p])
        if i != j:
            out.J_ikac[ji] = neighbours(J_jkac, base_ji[p], nb_ji[p])
            out.K_iakc[ji] = neighbours(K_jakc, base_ji[p], nb_ji[p])


def _pair_bytes(cc, ij):
    """What one pair holds for as long as its chunk is in flight."""
    ribfs, paos, lmos, _extended = _domains(cc, ij)
    nq, nu, nk, na = len(ribfs), len(paos), len(lmos), cc.n_pno[ij]
    kept = (nq                      # (Q|i j)
            + 2 * nq * nk           # (Q|i m), (Q|j m)
            + 2 * nq * na           # (Q|i a), (Q|j a)
            + nq * nk * na          # (Q|m a)
            + nq * na * na          # (Q|a b)
            + 2 * nq * (nk + na)    # the four full-inverse copies from _fit
            + nq * nq)              # _fit's domain metric, which it builds for
                                    # every pair in the chunk before solving any
    # The (Q|u v) gather and its half transform used to dominate this and are
    # absent because they are absent from the phase: both backends stream that
    # block through cache rather than allocating it.
    gathered = (nq                  # (Q|i j) raw
                + 2 * nq * nk       # (Q|i m), (Q|j m) raw
                + 2 * nq * nu       # (Q|i u), (Q|j u) raw
                + nq * nk * nu)     # (Q|m u) raw, the largest block left
    return 8 * (kept + gathered)


def _pair_transient_bytes(cc, ij):
    """What building ONE pair needs on top, and does not keep.

    ``_fit`` stacks every block into one right-hand side and copies the metric;
    ``_non_projected`` gathers over the EXTENDED PAO domain, which is the wider
    of the two and so usually the one that sets this. Both are alive for one
    pair at a time, so a chunk pays the largest of them once rather than the
    sum over its members.
    """
    ribfs, paos, lmos, extended = _domains(cc, ij)
    nq, nu, nk, na = len(ribfs), len(paos), len(lmos), cc.n_pno[ij]
    ne = len(extended)
    fit = (nq * (1 + 2 * nk + 2 * na + nk * na + na * na)  # the stacked rhs
           + 2 * nq * nq)                    # the metric copy and its root,
                                             # both per pair and both dropped
    ext = (nq * nu * ne         # (Q|u v_ext) raw
           + nq * na * ne       # (Q|a v_ext)
           + nq * nk * ne       # (Q|m u_ext) raw
           + 2 * nk * na * ne)  # the two contracted results
    return 8 * max(fit, ext)


def _chunks(cc, upper):
    """Split the pairs into runs whose working set fits the memory budget.

    The blocks a pair is BUILT from are larger than the blocks it produces:
    every raw gather carries a PAO index where the fitted block carries a PNO
    one, and the PAO domain is several times the PNO count. Building every pair
    before fitting any of them therefore peaks at something no store reflects -
    10.9 GiB at ethanol/cc-pVTZ against the 0.6 GiB of integrals the phase
    returns, which is most of what stood between that configuration and a
    completed run. That measurement had ``(Q | u v)`` materialized per pair at
    ``nq * nu^2``; both backends stream it now, so the widest single block left
    is the extended-domain gather :func:`_non_projected` builds one pair at a
    time.

    In-core with a measured failure, per design decision 10: a pair too large
    to build alone reports its own requirement and its domain sizes, because
    there is no disk path to fall back to.
    """
    budget = int(cc.cut.in_core_memory)
    chunks, current, total, head = [], [], 0, 0
    for ij in upper:
        need = _pair_bytes(cc, ij)
        mine = _pair_transient_bytes(cc, ij)
        if need + mine > budget:
            ribfs, paos, lmos, extended = _domains(cc, ij)
            i, j = cc.ij_to_i_j[ij]
            raise MemoryError(
                f"pair {ij} = ({i}, {j}) needs "
                f"{(need + mine) / 2**20:.0f} MiB on its own ({cc.n_pno[ij]} "
                f"PNOs, {len(ribfs)} auxiliary functions, {len(paos)} PAOs "
                f"widened to {len(extended)}, {len(lmos)} neighbours) against "
                f"a budget of {budget / 2**20:.0f} MiB. The largest single "
                f"block is the extended-domain (Q|u v_ext) gather at "
                f"{8 * len(ribfs) * len(paos) * len(extended) / 2**20:.0f} "
                "MiB. There is "
                "no disk path (design decision 10), so the options are: raise "
                "Thresholds.in_core_memory if the machine has the memory, or "
                "tighten t_cut_do to shrink the PAO domains.")
        if current and total + need + max(head, mine) > budget:
            chunks.append(current)
            current, total, head = [], 0, 0
        current.append(ij)
        total += need
        head = max(head, mine)
    if current:
        chunks.append(current)
    return chunks


def _domains(cc, ij):
    """The four domain lists and the extended PAO domain for one pair.

    ``extended_pao_domain`` is psi4's: the pair's own PAO domain widened by the
    PAO domain of every LMO it interacts with. The non-projected integrals need
    it because they carry an index belonging to a NEIGHBOUR pair ``kj``, whose
    PAOs are not a subset of ``ij``'s.
    """
    ribfs = cc.lmopair_to_ribfs[ij]
    paos = cc.lmopair_to_paos[ij]
    lmos = cc.lmopair_to_lmos[ij]
    extended = set(paos)
    for k in lmos:
        extended.update(cc.lmo_to_paos[k])
    return ribfs, paos, lmos, sorted(extended)


def _raw_blocks(q_ij, q_ia, q_ab, X_pno, i_lmo, j_lmo, n_pno, ribfs, paos, lmos):
    """The seven three-index blocks, gathered and half-transformed per pair.

    Six are slices of the three full tensors, so they are captured into one
    graph and replayed as an OpenMP team over the pairs. The seventh,
    ``(Q | a b)``, is gathered and rotated by a single grouped node for the
    whole chunk and never passes through a stored ``(Q | u v)`` block at all.
    """
    raw = []
    for p in range(len(i_lmo)):
        nq, nu, nk = len(ribfs[p]), len(paos[p]), len(lmos[p])
        na = n_pno[p]
        q_vv = ten.zeros("(Q|a b)", [nq, na * na])
        raw.append(dict(
            nq=nq, nu=nu, nk=nk, na=na,
            q_pair=ten.zeros("(Q|i j)", [nq, 1]),
            q_io=ten.zeros("(Q|i m)", [nq, nk]),
            q_jo=ten.zeros("(Q|j m)", [nq, nk]),
            q_iv=ten.zeros("(Q|i a)", [nq, na]),
            q_jv=ten.zeros("(Q|j a)", [nq, na]),
            q_ov=ten.zeros("(Q|m a)", [nq, nk * na]),
            q_vv=q_vv,
            # The rotation writes (Q, a, b) and the fit reads (Q, a*b), so the
            # store is the flat one - a numpy reshape of a rank-3 column-major
            # view would transpose the pair of virtual axes - and the rank-3
            # spelling is a view of it, built once because both the rotation
            # and the ``Qab`` output hold it.
            q_vv_block=q_vv.reshape_view([nq, na, na]),
        ))

    # The gathers, into rank-3 blocks whose views the contractions read. A
    # single-LMO selection leaves a length-1 axis, which costs nothing in column
    # major and saves the host copy that dropping it in numpy would be.
    scratch = []
    rot_c, rot_q, rot_u, rot_x = [], [], [], []
    g = cg.Graph("CC integrals: gather")
    with cg.capture(g):
        for p, r in enumerate(raw):
            nq, nu, nk = r["nq"], r["nu"], r["nk"]
            qs, us, ms = ribfs[p], paos[p], lmos[p]
            i, j = i_lmo[p], j_lmo[p]

            s = {}
            scratch.append(s)
            s["ij_blk"] = ten.zeros("(Q|i j) raw", [nq, 1, 1])
            la.gather(s["ij_blk"], q_ij, [qs, [i], [j]])
            s["io_blk"] = ten.zeros("(Q|i m) raw", [nq, 1, nk])
            la.gather(s["io_blk"], q_ij, [qs, [i], ms])
            s["jo_blk"] = ten.zeros("(Q|j m) raw", [nq, 1, nk])
            la.gather(s["jo_blk"], q_ij, [qs, [j], ms])
            s["iu_blk"] = ten.zeros("(Q|i u) raw", [nq, 1, nu])
            la.gather(s["iu_blk"], q_ia, [qs, [i], us])
            s["ju_blk"] = ten.zeros("(Q|j u) raw", [nq, 1, nu])
            la.gather(s["ju_blk"], q_ia, [qs, [j], us])
            s["mu_blk"] = ten.zeros("(Q|m u) raw", [nq, nk, nu])
            la.gather(s["mu_blk"], q_ia, [qs, ms, us])

            rot_c.append(r["q_vv_block"])
            rot_q.append(qs)
            rot_u.append(us)
            rot_x.append(X_pno[p])

        # X^T (Q | u v) X for every pair in the chunk, as ONE node. Both
        # transformed indices are interior axes of a rank-3 block, so neither is
        # a merged-axis GEMM, and the obvious spellings are both traps: a GEMM
        # pair per auxiliary function is 2 * naux operations per pair - around
        # 100,000 at ethanol/cc-pVTZ - each building a view, which exhausts
        # einsums' small-buffer pool long before it finishes; a gather of the
        # whole domain block followed by two contractions instead materializes
        # the two widest tensors the phase holds and streams them four times
        # over to produce a result an order smaller. The grouped node does the
        # same arithmetic a cache-resident block of auxiliary functions at a
        # time, reading ``(Q|u v)`` once out of the parent, so neither the
        # gathered block nor the half transform is ever allocated.
        #
        # Symmetric by contract: one index list and one transform serve both
        # rotated axes, which is what this rotation is.
        if rot_c:
            la.grouped_gather_rotate(rot_c, q_ab, rot_q, rot_u, rot_x)
    _run_setup_graph(g)

    # The PAO -> PNO half transforms. Separate graph because they consume the
    # gathers above; separate from the fit below because the metric solve is
    # not capturable in the same pass.
    g2 = cg.Graph("CC integrals: PAO to PNO")
    with cg.capture(g2):
        for p, r in enumerate(raw):
            s = scratch[p]
            nq, nu, nk, na = r["nq"], r["nu"], r["nk"], r["na"]
            X = X_pno[p]
            # (Q|i j) and (Q|i m) need no transform, only a reshape of the
            # length-1 axis away, which a view already is.
            la.axpby(1.0, s["ij_blk"].reshape_view([nq, 1]), 0.0, r["q_pair"])
            la.axpby(1.0, s["io_blk"].reshape_view([nq, nk]), 0.0, r["q_io"])
            la.axpby(1.0, s["jo_blk"].reshape_view([nq, nk]), 0.0, r["q_jo"])
            # (Q|i u) X -> (Q|i a), and the same for j.
            la.gemm(1.0, s["iu_blk"].reshape_view([nq, nu]), X, 0.0, r["q_iv"])
            la.gemm(1.0, s["ju_blk"].reshape_view([nq, nu]), X, 0.0, r["q_jv"])
            # (Q m | u) X -> (Q m | a). The (Q, m) axes are adjacent and both
            # lead, so the whole block is one GEMM through a merged view.
            la.gemm(1.0, s["mu_blk"].reshape_view([nq * nk, nu]), X, 0.0,
                    r["q_ov"].reshape_view([nq * nk, na]))
    _run_setup_graph(g2)

    # And now the gathers are dead. They carry a PAO index where the blocks
    # they become carry a PNO one, and the PAO domain is several times the PNO
    # count, so they exist only to feed the two graphs above and the scratch is
    # dropped here rather than travelling with ``raw``. It used to, which cost
    # 10 GiB of resident memory at ethanol/cc-pVTZ against the 0.6 GiB of
    # integrals the phase actually returns; nothing ever read it.
    scratch.clear()
    return raw


def _fit(metric, ribfs, raw):
    """Apply the metric, at the two different powers psi4 uses.

    The full-inverse copies come first and are kept: they are the fitted side of
    the two non-projected families, whose other side is unfitted. Then the
    metric is replaced by its square root and every primary block is solved
    against it, which leaves them as symmetric DF factors.

    ``power(0.5)`` is a matrix square root through an eigendecomposition, which
    is what ``einsums.linalg.pow`` does; the ``1e-14`` psi4 passes is its
    eigenvalue floor and matters only for a near-singular auxiliary basis.
    """
    metrics = [sparse.submatrix_rows_and_cols(metric, qs, qs,
                                              name="(P|Q) domain")
               for qs in ribfs]

    # The full-inverse copies, one solve per pair against a fresh copy of the
    # metric (gesv overwrites its left-hand side).
    for p, r in enumerate(raw):
        A = metrics[p]
        rhs = ten.zeros("full-inverse rhs",
                        [r["nq"], 2 * r["nk"] + 2 * r["na"]])
        view = ten.view(rhs)
        nk, na = r["nk"], r["na"]
        view[:, :nk] = ten.view(r["q_io"])
        view[:, nk:2 * nk] = ten.view(r["q_jo"])
        view[:, 2 * nk:2 * nk + na] = ten.view(r["q_iv"])
        view[:, 2 * nk + na:] = ten.view(r["q_jv"])
        Acopy = ten.from_numpy("(P|Q) copy", ten.view(A))
        la.gesv(Acopy, rhs)
        view = ten.view(rhs)
        r["q_io_inv"] = ten.from_numpy("J^-1 (Q|i m)", view[:, :nk])
        r["q_jo_inv"] = ten.from_numpy("J^-1 (Q|j m)", view[:, nk:2 * nk])
        r["q_iv_inv"] = ten.from_numpy("J^-1 (Q|i a)", view[:, 2 * nk:2 * nk + na])
        r["q_jv_inv"] = ten.from_numpy("J^-1 (Q|j a)", view[:, 2 * nk + na:])

    # Then the symmetric fit. One solve per pair over every primary block at
    # once: they share the metric, and gesv factorizes once for all columns.
    for p, r in enumerate(raw):
        A_half = la.pow(metrics[p], 0.5)
        nk, na = r["nk"], r["na"]
        widths = [("q_pair", 1), ("q_io", nk), ("q_jo", nk), ("q_iv", na),
                  ("q_jv", na), ("q_ov", nk * na), ("q_vv", na * na)]
        total = sum(w for _, w in widths)
        rhs = ten.zeros("symmetric fit rhs", [r["nq"], total])
        view = ten.view(rhs)
        at = 0
        for name, w in widths:
            view[:, at:at + w] = ten.view(r[name]).reshape(r["nq"], w)
            at += w
        la.gesv(A_half, rhs)
        view = ten.view(rhs)
        at = 0
        for name, w in widths:
            ten.view(r[name])[...] = view[:, at:at + w].reshape(ten.shape(r[name]))
            at += w


def _contract(raw, i_lmo, j_lmo, strong):
    """The contracted integrals, ``B^T B`` over the pair's auxiliary domain.

    Returns the contract's fields as lists in the chunk's order, with a
    zero-extent placeholder wherever a pair does not have the block: on the
    diagonal for the ``ji`` partners, and on a weak pair for the density-fitted
    factors. See :class:`dlpno.contracts.PnoIntegralBlocks`.
    """
    blocks = {name: [] for name in
              ("K_mibj", "J_ijmb", "K_ivvv", "K_mjai", "K_jvvv",
               "i_Qk", "i_Qa", "j_Qk", "j_Qa", "Qma", "Qab")}
    K_mibj, K_mjai = blocks["K_mibj"], blocks["K_mjai"]
    J_ijmb = blocks["J_ijmb"]
    K_ivvv, K_jvvv = blocks["K_ivvv"], blocks["K_jvvv"]

    g = cg.Graph("CC integrals: contract")
    with cg.capture(g):
        for p, r in enumerate(raw):
            i, j = i_lmo[p], j_lmo[p]
            nk, na = r["nk"], r["na"]

            # (m i | b j) and its ji partner (m j | a i).
            K_mibj.append(ten.zeros("K (m i|b j)", [nk, na]))
            la.gemm(1.0, r["q_io"], r["q_jv"], 0.0, K_mibj[p], trans_a=True)
            if i != j:
                K_mjai.append(ten.zeros("K (m j|a i)", [nk, na]))
                la.gemm(1.0, r["q_jo"], r["q_iv"], 0.0, K_mjai[p], trans_a=True)
            else:
                K_mjai.append(_absent())

            # (i j | m b). Symmetric in the pair, so ji SHARES the object
            # rather than holding a copy, exactly as psi4 does.
            #
            # Built as a COLUMN and reshaped, not as a row. The pair factor is
            # (Q, 1), so contracting it on the left would make this a GEMM with
            # one row, which is a degenerate case worth avoiding on sight - it
            # cost the coupled-cluster iteration a factor of two at ten threads
            # in Eq. 84c before that one was turned round. There are only about
            # ninety of these per run so nothing here is measurably faster; it
            # is written this way so the shape is not copied.
            #
            # The reshape is exact rather than a reinterpretation: a
            # column-major (nk*na, 1) and a (nk, na) put element (k, a) at the
            # same offset k + nk*a.
            J = ten.zeros("J (i j|m b)", [nk * na, 1])
            la.gemm(1.0, r["q_ov"], r["q_pair"], 0.0, J, trans_a=True)
            J_ijmb.append(J.reshape_view([nk, na]))

            # (i e | a f), as (e, a, f).
            K_ivvv.append(ten.zeros("K (i e|a f)", [na, na, na]))
            la.gemm(1.0, r["q_iv"], r["q_vv"], 0.0,
                    K_ivvv[p].reshape_view([na, na * na]), trans_a=True)
            if i != j:
                K_jvvv.append(ten.zeros("K (j e|a f)", [na, na, na]))
                la.gemm(1.0, r["q_jv"], r["q_vv"], 0.0,
                        K_jvvv[p].reshape_view([na, na * na]), trans_a=True)
            else:
                K_jvvv.append(_absent())
    _run_setup_graph(g)

    # The density-fitted factors, strong pairs only: nothing else reads them.
    # They are the raw blocks themselves, not copies of them, which is what
    # keeps this phase's footprint the size of its output.
    for p, r in enumerate(raw):
        i, j = i_lmo[p], j_lmo[p]
        nq, nk, na = r["nq"], r["nk"], r["na"]
        off_diagonal = strong[p] and i != j
        blocks["i_Qk"].append(r["q_io"] if strong[p] else _absent())
        blocks["i_Qa"].append(r["q_iv"] if strong[p] else _absent())
        blocks["j_Qk"].append(r["q_jo"] if off_diagonal else _absent())
        blocks["j_Qa"].append(r["q_jv"] if off_diagonal else _absent())
        # Per auxiliary function, which is the axis every consumer loops over.
        blocks["Qma"].append(r["q_ov"].reshape_view([nq, nk, na])
                             if strong[p] else _absent())
        blocks["Qab"].append(r["q_vv_block"] if strong[p] else _absent())
    return blocks


def _non_projected(q_ia, q_ab, X_pno, raw, i_lmo, j_lmo, strong, ribfs, paos,
                   lmos, extended, rot_X, rot_paos, nb_ij, nb_ji, blocks):
    """``(i k | a_ij c_kj)`` and ``(i a_ij | k c_kj)``, per pair per neighbour.

    The two families whose second virtual index belongs to a NEIGHBOUR pair, so
    they cannot be expressed in ``ij``'s own PNO basis and go through the
    extended PAO domain instead. Strong pairs only.

    This is where the two metric powers matter. Each of these contracts a
    fully-inverse-fitted factor against an unfitted one, which reconstructs the
    integral exactly once; using the symmetric factors on both sides would
    apply the metric one and a half times.

    One pair at a time, deliberately. The extended-domain gathers are the widest
    thing this phase touches, and ``_pair_transient_bytes`` charges the chunk for
    the largest of them ONCE rather than for the sum over its members; hoisting
    them into a graph over every pair would spend the whole budget on scratch.
    """
    for name in ("J_ikac", "K_iakc", "J_jkac", "K_jakc"):
        blocks[name] = []
    for p, r in enumerate(raw):
        if not strong[p]:
            continue
        i, j = i_lmo[p], j_lmo[p]
        nq, nk, na = r["nq"], r["nk"], r["na"]
        es = extended[p]
        ne = len(es)

        # (Q | a_ij v_ext): half in the pair's PNOs, half in the extended PAOs.
        uv_ext = ten.zeros("(Q|u v_ext) raw", [nq, r["nu"], ne])
        la.gather(uv_ext, q_ab, [ribfs[p], paos[p], es])
        q_av = ten.zeros("(Q|a v_ext)", [nq, na, ne])
        mu_ext = ten.zeros("(Q|m u_ext) raw", [nq, nk, ne])
        g = cg.Graph("CC integrals: extended half transform")
        with cg.capture(g):
            einsums.einsum("Qae <- Que ; ua", q_av, uv_ext, X_pno[p])
            # (Q | m u_ext), raw and unfitted: the K_iakc side.
            la.gather(mu_ext, q_ia, [ribfs[p], lmos[p], es])
        _run_setup_graph(g)

        # J: (P|Q)^-1 (Q|i m) contracted with (Q | a v_ext).
        K_iovv = ten.zeros("(i m|a v_ext)", [nk, na * ne])
        la.gemm(1.0, r["q_io_inv"], q_av.reshape_view([nq, na * ne]), 0.0,
                K_iovv, trans_a=True)
        # K: (Q | m u_ext) contracted with (P|Q)^-1 (Q|i a).
        K_oviv = ten.zeros("(m u_ext|i a)", [nk, ne * na])
        la.gemm(1.0, mu_ext.reshape_view([nq, nk * ne]), r["q_iv_inv"], 0.0,
                K_oviv.reshape_view([nk * ne, na]), trans_a=True)

        # K_iovv came from a (nq, na, ne) block flattened in place, so the PNO
        # index runs fastest; K_oviv was built as (nk*ne, na) and reinterpreted,
        # so the extended PAO index does. See _slice_neighbours.
        blocks["J_ikac"] += _slice_neighbours(nb_ij[p], rot_X, rot_paos, es,
                                              K_iovv, na, ne, True, "(i k|a c)")
        blocks["K_iakc"] += _slice_neighbours(nb_ij[p], rot_X, rot_paos, es,
                                              K_oviv, na, ne, False, "(i a|k c)")
        if i != j:
            K_jovv = ten.zeros("(j m|a v_ext)", [nk, na * ne])
            la.gemm(1.0, r["q_jo_inv"], q_av.reshape_view([nq, na * ne]), 0.0,
                    K_jovv, trans_a=True)
            K_ovjv = ten.zeros("(m u_ext|j a)", [nk, ne * na])
            la.gemm(1.0, mu_ext.reshape_view([nq, nk * ne]), r["q_jv_inv"], 0.0,
                    K_ovjv.reshape_view([nk * ne, na]), trans_a=True)
            blocks["J_jkac"] += _slice_neighbours(nb_ji[p], rot_X, rot_paos, es,
                                                  K_jovv, na, ne, True,
                                                  "(j k|a c)")
            blocks["K_jakc"] += _slice_neighbours(nb_ji[p], rot_X, rot_paos, es,
                                                  K_ovjv, na, ne, False,
                                                  "(j a|k c)")


def _slice_neighbours(slots, rot_X, rot_paos, extended, block, na, ne,
                      pno_fastest, name):
    """Cut each neighbour's slab out and rotate it into that neighbour's PNOs.

    ``block`` is ``(nlmo, na * ne)``: one row per neighbour, holding a
    ``(PNO, extended PAO)`` slab flattened into the second axis. The
    neighbour's own PAO domain is a subset of the extended one, so the
    rotation is a selection along the extended axis followed by that
    neighbour's ``X_pno``.

    **Which index runs fastest in that flattened axis is not a convention to
    pick, it is a consequence of column major, and the two families disagree.**
    Merging two adjacent axes of a column-major tensor leaves the FIRST of them
    varying fastest, so a ``(nq, na, ne)`` block flattened to ``(nq, na*ne)``
    has ``W = a + na*e`` while a ``(nk*ne, na)`` result reinterpreted as
    ``(nk, ne*na)`` has ``W = e + ne*a``. Reading either with the other's
    stride silently transposes each slab, which is why this takes the layout as
    an argument rather than inferring it from the caller.

    Args:
        slots: Per neighbour, the index into ``rot_X``/``rot_paos`` of the
            neighbour pair's basis, or ``-1`` when it has none. A dead slot
            yields a zero-extent placeholder, which the caller reads back as
            ``None``.
        pno_fastest: True when the PNO index varies fastest in ``block``'s
            second axis (``W = a + na*e``), False when the extended PAO index
            does (``W = e + ne*a``).
    """
    view = ten.view(block)
    where = {u: pos for pos, u in enumerate(extended)}
    per_neighbour = []
    for k_ij, slot in enumerate(slots):
        if slot < 0:
            per_neighbour.append(_absent())
            continue
        keep = [where[u] for u in rot_paos[slot]]
        X_kj = np.asarray(rot_X[slot])
        if pno_fastest:
            # W = a + na*e, so C-order (ne, na) reads it correctly.
            slab = view[k_ij].reshape(ne, na)[keep, :]          # (u_kj, a_ij)
            per_neighbour.append(ten.from_numpy(name, (X_kj.T @ slab).T))
        else:
            # W = e + ne*a, so C-order (na, ne) reads it correctly.
            slab = view[k_ij].reshape(na, ne)[:, keep]          # (a_ij, u_kj)
            per_neighbour.append(ten.from_numpy(name, slab @ X_kj))
    return per_neighbour
