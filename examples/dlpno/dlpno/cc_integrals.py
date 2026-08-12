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

__all__ = ["PnoIntegrals", "compute_pno_integrals"]

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
    """
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
        raw = _raw_blocks(cc, chunk)
        _fit(cc, chunk, raw)
        _contract(cc, chunk, raw, out)
        _non_projected(cc, chunk, raw, out)
        raw.clear()
    return out


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
    gathered = (nq                  # (Q|i j) raw
                + 2 * nq * nk       # (Q|i m), (Q|j m) raw
                + 2 * nq * nu       # (Q|i u), (Q|j u) raw
                + nq * nk * nu      # (Q|m u) raw
                + nq * nu * nu      # (Q|u v) raw, the largest block by far
                + nq * na * nu)     # the (Q|a v) half transform
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

    The blocks a pair is BUILT from are an order larger than the blocks it
    produces: ``(Q | u v)`` over the pair's own domains is ``nq * nu^2``
    against the ``nq * na^2`` it becomes, and the PAO domain is several times
    the PNO count. Building every pair before fitting any of them therefore
    peaks at something no store reflects - 10.9 GiB at ethanol/cc-pVTZ against
    the 0.6 GiB of integrals the phase returns, which is most of what stood
    between that configuration and a completed run.

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
                f"block is the (Q|u v) gather at "
                f"{8 * len(ribfs) * len(paos) ** 2 / 2**20:.0f} MiB. There is "
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


def _raw_blocks(cc, upper):
    """The seven three-index blocks, gathered and half-transformed per pair.

    All seven are slices of the same three full tensors, so they are captured
    into one graph and replayed as an OpenMP team over the pairs.
    """
    raw = {}
    jobs = []
    for ij in upper:
        i, j = cc.ij_to_i_j[ij]
        ribfs, paos, lmos, extended = _domains(cc, ij)
        nq, nu, nk, na = len(ribfs), len(paos), len(lmos), cc.n_pno[ij]
        raw[ij] = dict(
            ribfs=ribfs, paos=paos, lmos=lmos, extended=extended,
            nq=nq, nu=nu, nk=nk, na=na, i=i, j=j,
            q_pair=ten.zeros("(Q|i j)", [nq, 1]),
            q_io=ten.zeros("(Q|i m)", [nq, nk]),
            q_jo=ten.zeros("(Q|j m)", [nq, nk]),
            q_iv=ten.zeros("(Q|i a)", [nq, na]),
            q_jv=ten.zeros("(Q|j a)", [nq, na]),
            q_ov=ten.zeros("(Q|m a)", [nq, nk * na]),
            q_vv=ten.zeros("(Q|a b)", [nq, na * na]),
        )
        jobs.append(ij)

    # The gathers, into rank-3 blocks whose views the contractions read. A
    # single-LMO selection leaves a length-1 axis, which costs nothing in column
    # major and saves the host copy that dropping it in numpy would be.
    scratch = {}
    g = cg.Graph("CC integrals: gather")
    with cg.capture(g):
        for ij in jobs:
            r = raw[ij]
            ribfs, paos, lmos = r["ribfs"], r["paos"], r["lmos"]
            nq, nu, nk, na = r["nq"], r["nu"], r["nk"], r["na"]
            qs = [int(q) for q in ribfs]
            us = [int(u) for u in paos]
            ms = [int(m) for m in lmos]

            s = scratch[ij] = {}
            s["ij_blk"] = ten.zeros("(Q|i j) raw", [nq, 1, 1])
            la.gather(s["ij_blk"], cc.q_ij, [qs, [r["i"]], [r["j"]]])
            s["io_blk"] = ten.zeros("(Q|i m) raw", [nq, 1, nk])
            la.gather(s["io_blk"], cc.q_ij, [qs, [r["i"]], ms])
            s["jo_blk"] = ten.zeros("(Q|j m) raw", [nq, 1, nk])
            la.gather(s["jo_blk"], cc.q_ij, [qs, [r["j"]], ms])
            s["iu_blk"] = ten.zeros("(Q|i u) raw", [nq, 1, nu])
            la.gather(s["iu_blk"], cc.q_ia, [qs, [r["i"]], us])
            s["ju_blk"] = ten.zeros("(Q|j u) raw", [nq, 1, nu])
            la.gather(s["ju_blk"], cc.q_ia, [qs, [r["j"]], us])
            s["mu_blk"] = ten.zeros("(Q|m u) raw", [nq, nk, nu])
            la.gather(s["mu_blk"], cc.q_ia, [qs, ms, us])
            s["uv_blk"] = ten.zeros("(Q|u v) raw", [nq, nu, nu])
            la.gather(s["uv_blk"], cc.q_ab, [qs, us, us])
    _run_setup_graph(g)

    # The PAO -> PNO half transforms. Separate graph because they consume the
    # gathers above; separate from the fit below because the metric solve is
    # not capturable in the same pass.
    g2 = cg.Graph("CC integrals: PAO to PNO")
    with cg.capture(g2):
        for ij in jobs:
            r, s = raw[ij], scratch[ij]
            nq, nu, nk, na = r["nq"], r["nu"], r["nk"], r["na"]
            X = cc.X_pno[ij]
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
            # X^T (Q | u v) X. Both transformed indices are interior axes of a
            # rank-3 block, so neither is a merged-axis GEMM; the obvious form
            # is a GEMM pair per auxiliary function, and that is a trap. It is
            # 2 * naux operations per pair - around 100,000 at ethanol/cc-pVTZ
            # against 254 this way - and each one builds a view, which exhausts
            # einsums' small-buffer pool long before it finishes.
            s["half"] = ten.zeros("(Q|a v) half", [nq, na, nu])
            einsums.einsum("Qav <- Quv ; ua", s["half"], s["uv_blk"], X)
            einsums.einsum("Qab <- Qav ; vb", r["q_vv"].reshape_view([nq, na, na]),
                           s["half"], X)
    _run_setup_graph(g2)

    # And now the gathers are dead. They are the largest thing this phase ever
    # holds - ``uv_blk`` alone is ``nq * nu^2`` per pair, an order above any
    # block it produces - and they exist only to feed the two graphs above, so
    # the scratch is dropped here rather than travelling with ``raw``. It used
    # to, which cost 10 GiB of resident memory at ethanol/cc-pVTZ against the
    # 0.6 GiB of integrals the phase actually returns; nothing ever read it.
    scratch.clear()
    return raw


def _fit(cc, upper, raw):
    """Apply the metric, at the two different powers psi4 uses.

    The full-inverse copies come first and are kept: they are the fitted side of
    the two non-projected families, whose other side is unfitted. Then the
    metric is replaced by its square root and every primary block is solved
    against it, which leaves them as symmetric DF factors.

    ``power(0.5)`` is a matrix square root through an eigendecomposition, which
    is what ``einsums.linalg.pow`` does; the ``1e-14`` psi4 passes is its
    eigenvalue floor and matters only for a near-singular auxiliary basis.
    """
    metrics = {}
    for ij in upper:
        r = raw[ij]
        ribfs = r["ribfs"]
        A = sparse.submatrix_rows_and_cols(cc.metric, ribfs, ribfs,
                                           name="(P|Q) domain")
        metrics[ij] = A

    # The full-inverse copies, one solve per pair against a fresh copy of the
    # metric (gesv overwrites its left-hand side).
    for ij in upper:
        r = raw[ij]
        A = metrics[ij]
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
    for ij in upper:
        r = raw[ij]
        A_half = la.pow(metrics[ij], 0.5)
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


def _contract(cc, upper, raw, out):
    """The contracted integrals, ``B^T B`` over the pair's auxiliary domain."""
    g = cg.Graph("CC integrals: contract")
    with cg.capture(g):
        for ij in upper:
            r = raw[ij]
            i, j = r["i"], r["j"]
            ji = cc.ij_to_ji[ij]
            nk, na = r["nk"], r["na"]

            # (m i | b j) and its ji partner (m j | a i).
            out.K_mibj[ij] = ten.zeros("K (m i|b j)", [nk, na])
            la.gemm(1.0, r["q_io"], r["q_jv"], 0.0, out.K_mibj[ij], trans_a=True)
            if i != j:
                out.K_mibj[ji] = ten.zeros("K (m j|a i)", [nk, na])
                la.gemm(1.0, r["q_jo"], r["q_iv"], 0.0, out.K_mibj[ji], trans_a=True)

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
            out.J_ijmb[ij] = J.reshape_view([nk, na])
            if i != j:
                out.J_ijmb[ji] = out.J_ijmb[ij]

            # (i e | a f), as (e, a, f).
            out.K_ivvv[ij] = ten.zeros("K (i e|a f)", [na, na, na])
            la.gemm(1.0, r["q_iv"], r["q_vv"], 0.0,
                    out.K_ivvv[ij].reshape_view([na, na * na]), trans_a=True)
            if i != j:
                out.K_ivvv[ji] = ten.zeros("K (j e|a f)", [na, na, na])
                la.gemm(1.0, r["q_jv"], r["q_vv"], 0.0,
                        out.K_ivvv[ji].reshape_view([na, na * na]), trans_a=True)
    _run_setup_graph(g)

    # L_mibj and L_iajb, both pure host-side combinations of blocks that
    # already exist.
    #
    # psi4 computes L_mibj_[ji] as 3 K_ji - 2 K_ij, because it subtracts the
    # already-overwritten L_mibj_[ij] instead of K_mibj_[ij]. That is a bug, and
    # a harmless one there because nothing ever reads L_mibj_ - it is resized,
    # written and cleared. Written correctly here.
    for ij in upper:
        i, j = raw[ij]["i"], raw[ij]["j"]
        ji = cc.ij_to_ji[ij]
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

    # The density-fitted factors, strong pairs only: nothing else reads them.
    for ij in upper:
        r = raw[ij]
        i, j = r["i"], r["j"]
        ji = cc.ij_to_ji[ij]
        nq, nk, na = r["nq"], r["nk"], r["na"]
        if cc.i_j_to_ij_strong[i, j] == -1:
            continue
        out.i_Qk[ij] = r["q_io"]
        out.i_Qa[ij] = r["q_iv"]
        if i != j:
            out.i_Qk[ji] = r["q_jo"]
            out.i_Qa[ji] = r["q_jv"]
        # Per auxiliary function, which is the axis every consumer loops over.
        out.Qma[ij] = r["q_ov"].reshape_view([nq, nk, na])
        out.Qab[ij] = r["q_vv"].reshape_view([nq, na, na])


def _non_projected(cc, upper, raw, out):
    """``(i k | a_ij c_kj)`` and ``(i a_ij | k c_kj)``, per pair per neighbour.

    The two families whose second virtual index belongs to a NEIGHBOUR pair, so
    they cannot be expressed in ``ij``'s own PNO basis and go through the
    extended PAO domain instead. Strong pairs only.

    This is where the two metric powers matter. Each of these contracts a
    fully-inverse-fitted factor against an unfitted one, which reconstructs the
    integral exactly once; using the symmetric factors on both sides would
    apply the metric one and a half times.
    """
    for ij in upper:
        r = raw[ij]
        i, j = r["i"], r["j"]
        ji = cc.ij_to_ji[ij]
        if cc.i_j_to_ij_strong[i, j] == -1:
            continue
        ribfs, paos, lmos, extended = (r["ribfs"], r["paos"], r["lmos"],
                                       r["extended"])
        nq, nk, na, ne = r["nq"], r["nk"], r["na"], len(extended)
        qs = [int(q) for q in ribfs]
        es = [int(u) for u in extended]

        # (Q | a_ij v_ext): half in the pair's PNOs, half in the extended PAOs.
        uv_ext = ten.zeros("(Q|u v_ext) raw", [nq, len(paos), ne])
        la.gather(uv_ext, cc.q_ab, [qs, [int(u) for u in paos], es])
        q_av = ten.zeros("(Q|a v_ext)", [nq, na, ne])
        mu_ext = ten.zeros("(Q|m u_ext) raw", [nq, nk, ne])
        g = cg.Graph("CC integrals: extended half transform")
        with cg.capture(g):
            einsums.einsum("Qae <- Que ; ua", q_av, uv_ext, cc.X_pno[ij])
            # (Q | m u_ext), raw and unfitted: the K_iakc side.
            la.gather(mu_ext, cc.q_ia, [qs, [int(m) for m in lmos], es])
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
        out.J_ikac[ij] = _slice_neighbours(cc, ij, lmos, extended, K_iovv,
                                           na, ne, True, "(i k|a c)")
        out.K_iakc[ij] = _slice_neighbours(cc, ij, lmos, extended, K_oviv,
                                           na, ne, False, "(i a|k c)")
        if i != j:
            K_jovv = ten.zeros("(j m|a v_ext)", [nk, na * ne])
            la.gemm(1.0, r["q_jo_inv"], q_av.reshape_view([nq, na * ne]), 0.0,
                    K_jovv, trans_a=True)
            K_ovjv = ten.zeros("(m u_ext|j a)", [nk, ne * na])
            la.gemm(1.0, mu_ext.reshape_view([nq, nk * ne]), r["q_jv_inv"], 0.0,
                    K_ovjv.reshape_view([nk * ne, na]), trans_a=True)
            out.J_ikac[ji] = _slice_neighbours(cc, ji, lmos, extended, K_jovv,
                                               na, ne, True, "(j k|a c)")
            out.K_iakc[ji] = _slice_neighbours(cc, ji, lmos, extended, K_ovjv,
                                               na, ne, False, "(j a|k c)")


def _slice_neighbours(cc, ij, lmos, extended, block, na, ne, pno_fastest, name):
    """Cut one neighbour's slab out and rotate it into that neighbour's PNOs.

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
        pno_fastest: True when the PNO index varies fastest in ``block``'s
            second axis (``W = a + na*e``), False when the extended PAO index
            does (``W = e + ne*a``).
    """
    _, j = cc.ij_to_i_j[ij]
    view = ten.view(block)
    where = {u: pos for pos, u in enumerate(extended)}
    per_neighbour = []
    for k_ij, k in enumerate(lmos):
        kj = int(cc.i_j_to_ij[k, j])
        if kj == -1 or not cc.n_pno[kj]:
            per_neighbour.append(None)
            continue
        keep = [where[u] for u in cc.lmopair_to_paos[kj]]
        X_kj = np.asarray(cc.X_pno[kj])
        if pno_fastest:
            # W = a + na*e, so C-order (ne, na) reads it correctly.
            slab = view[k_ij].reshape(ne, na)[keep, :]          # (u_kj, a_ij)
            per_neighbour.append(ten.from_numpy(name, (X_kj.T @ slab).T))
        else:
            # W = e + ne*a, so C-order (na, ne) reads it correctly.
            slab = view[k_ij].reshape(na, ne)[:, keep]          # (a_ij, u_kj)
            per_neighbour.append(ten.from_numpy(name, slab @ X_kj))
    return per_neighbour
