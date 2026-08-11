#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Overlaps between pairs of PNO bases, for the coupled-cluster residual.

Every pair has its own PNO basis, so any term coupling pair ``ij`` to pair ``mn``
needs the overlap between the two:

    S(ij, mn) = X_pno[ij]^T  S_pao[paos_ij, paos_mn]  X_pno[mn]

DLPNO-MP2 needs one family of these and gets it from :mod:`dlpno.pno_overlaps`,
which is built around the LMP2 coupling plan: bucketed shape classes, blocks
concatenated into per-class tensors, the Fock prefactor folded in. None of that
generalizes, because the CC residual reads overlaps in patterns the LMP2 plan
has no vocabulary for - psi4 keeps three separate families and a dispatcher over
them (``DLPNOCCSD::S_PNO``).

So this is the same *arithmetic* expressed as a request service rather than a
plan: hand it a list of ``(ij, mn)`` pairs and get back one matrix each. Two
things carry over from the LMP2 engine because they are what make it fast, and
neither depends on the plan:

* **Each pair's transform is scattered onto the FULL PAO axis.** ``X`` is then
  zero outside its own domain, the domain restriction becomes implicit, and the
  per-request ``S_pao`` gather disappears. That gather was 42% of the MP2 phase.
* **The left factor is computed once per distinct ``ij``**, not once per
  request. ``S_ij_kj`` alone asks for every pair against up to ``naocc``
  partners, so the sharing is worth an order of magnitude.
"""

import numpy as np

from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import DLPNOBase

__all__ = ["build_overlaps", "PnoOverlapStore"]

#: Distinguishes "no block stored" from a stored ``None``, which is what a
#: request naming a dead pair yields.
_MISSING = object()

#: Replay a setup graph as an OpenMP team over its independent nodes. A
#: staticmethod on DLPNOBase; bound here so this module needs no instance.
#: Never a ThreadPoolExecutor - see the note on DLPNOBase._run.
_run_setup_graph = DLPNOBase._run


def build_overlaps(X_pno, S_pao, lmopair_to_paos, n_pno, requests):
    """``S(ij, mn)`` for each requested pair of pairs.

    Args:
        X_pno: Per pair, its PAO-domain-to-PNO transform.
        S_pao: The full PAO overlap matrix.
        lmopair_to_paos: Per pair, the PAO indices its domain covers.
        n_pno: Per pair, its PNO count.
        requests: Iterable of ``(ij, mn)``.

    Returns a list of ``(n_pno[ij], n_pno[mn])`` tensors, one per request, in
    the order the requests were given. A request naming a pair with no PNOs
    yields ``None`` rather than a zero-sized tensor, because every consumer
    already has to branch on the dead-pair case and a shaped zero would only
    move the branch.
    """
    requests = [(int(ij), int(mn)) for ij, mn in requests]
    live = [r for r in requests if n_pno[r[0]] and n_pno[r[1]]]
    if not live:
        return [None] * len(requests)

    npao = ten.shape(S_pao)[0]
    needed = sorted({ij for r in live for ij in r})

    # Each pair's transform on the full PAO axis, so the domain restriction is
    # implicit and no request has to gather S_pao.
    X_full = {}
    g_pad = cg.Graph("PNO transforms on the full PAO axis")
    with cg.capture(g_pad):
        for ij in needed:
            X_full[ij] = ten.zeros(f"X (full PAO) {ij}", [npao, n_pno[ij]])
            sparse.scatter_into(X_full[ij], X_pno[ij],
                                [lmopair_to_paos[ij], range(n_pno[ij])])
    _run_setup_graph(g_pad)

    # The left factor, once per distinct pair rather than once per request.
    lefts = {ij: ten.zeros(f"half {ij}", [n_pno[ij], npao]) for ij in needed}
    g_half = cg.Graph("PNO overlap left factors")
    with cg.capture(g_half):
        cg.grouped_batched_gemm(
            1.0, [X_full[ij] for ij in needed], [S_pao] * len(needed),
            0.0, [lefts[ij] for ij in needed], trans_a=True)
    _run_setup_graph(g_half)

    out = {}
    g = cg.Graph("PNO overlaps")
    with cg.capture(g):
        a, b, c = [], [], []
        for ij, mn in live:
            if (ij, mn) in out:
                continue
            out[ij, mn] = ten.zeros(f"S ({ij},{mn})", [n_pno[ij], n_pno[mn]])
            a.append(lefts[ij])
            b.append(X_full[mn])
            c.append(out[ij, mn])
        cg.grouped_batched_gemm(1.0, a, b, 0.0, c)
    _run_setup_graph(g)

    return [out.get((ij, mn)) for ij, mn in requests]


class PnoOverlapStore:
    """psi4's three PNO overlap families, and the dispatcher over them.

    ``DLPNOCCSD::compute_pno_overlaps`` builds three lists and
    ``DLPNOCCSD::S_PNO`` decides which one answers a given ``(ij, mn)``. The
    split is not arbitrary: the three cover disjoint cases and together they
    cover every overlap the residual reads, with the sparsity of
    ``lmopair_to_lmos`` bounding all of them well short of pairs-squared.

    * ``kj`` - pair ``ij`` against pair ``kj`` for every LMO ``k``: the pairs
      sharing an index. Stored for EVERY ``ij``, not just the upper triangle,
      because ``S(ij, kj)`` and ``S(ji, ki)`` are different matrices.
    * ``nn`` - pair ``ij`` against the DIAGONAL pair ``kk``, for the ``k`` in
      its own neighbour list. This is what projects a singles amplitude, which
      lives on ``kk``, into a pair's basis.
    * ``mn`` - pair ``ij`` against a neighbour pair ``mn`` sharing NEITHER index
      with it. The largest family and the only one that could break the in-core
      decision, which is why :meth:`report` prints its size.

    The dispatcher exists because those cases overlap in a way that would be
    wasteful to store twice: ``S(ij, mn)`` with ``m == i`` is already in the
    ``kj`` family under a different name, and psi4 skips it rather than storing
    it again. Reproducing the skip list exactly is what keeps the memory claim
    true.
    """

    def __init__(self, cc):
        self.cc = cc
        self.kj = {}
        self.nn = {}
        self.mn = {}
        self._on_the_fly = 0

    # -- psi4 DLPNOCCSD::compute_pno_overlaps -------------------------------

    def build(self):
        """Build all three families, each as one batch of requests."""
        cc = self.cc
        naocc = cc.ref.naocc
        n_pno = cc.n_pno

        kj_keys, nn_keys, mn_keys = [], [], []
        for ij, (i, j) in enumerate(cc.ij_to_i_j):
            if not n_pno[ij]:
                continue
            # -- family kj: every ij, every k ------------------------------
            for k in range(naocc):
                kj = int(cc.i_j_to_ij[k, j])
                if kj != -1 and n_pno[kj]:
                    kj_keys.append((ij, k, kj))
            if i > j:
                continue
            # -- family nn: upper pairs, k in the pair's own neighbour list -
            for k in cc.lmopair_to_lmos[ij]:
                kk = int(cc.i_j_to_ij[k, k])
                if kk != -1 and n_pno[kk]:
                    nn_keys.append((ij, k, kk))
            # -- family mn: upper pairs, neighbour pairs sharing no index ---
            lmos = cc.lmopair_to_lmos[ij]
            for m_ij, m in enumerate(lmos):
                for n_ij in range(m_ij + 1, len(lmos)):
                    n = lmos[n_ij]
                    if i in (m, n) or j in (m, n):
                        continue
                    mn = int(cc.i_j_to_ij[m, n])
                    if mn != -1 and n_pno[mn]:
                        mn_keys.append((ij, m_ij, n_ij, mn))

        args = (cc.X_pno, cc.S_pao, cc.lmopair_to_paos, n_pno)
        for keys, store, slot in (
            (kj_keys, self.kj, lambda key: (key[0], key[1])),
            (nn_keys, self.nn, lambda key: (key[0], key[1])),
            (mn_keys, self.mn, lambda key: (key[0], key[1], key[2])),
        ):
            blocks = build_overlaps(*args, [(key[0], key[-1]) for key in keys])
            for key, block in zip(keys, blocks):
                store[slot(key)] = block
        return self

    # -- psi4 DLPNOCCSD::S_PNO ----------------------------------------------

    def get(self, ij, mn):
        """The overlap between pair ``ij``'s PNO basis and pair ``mn``'s.

        psi4's dispatcher, branch for branch. The first four cases exploit that
        ``ij`` and ``ji`` share a PNO basis, so an overlap against a pair
        sharing one index is already in the ``kj`` family under whichever of the
        two names makes the shared index the second one.

        The final case falls through to an on-the-fly build when either LMO is
        outside ``ij``'s neighbour list, which is exactly when the ``mn`` family
        did not store it. That is psi4's ``low_memory_overlap_`` branch taken
        selectively rather than globally, and :attr:`_on_the_fly` counts it so
        the cost is visible rather than assumed.
        """
        cc = self.cc
        i, j = cc.ij_to_i_j[ij]
        m, n = cc.ij_to_i_j[mn]
        ji = cc.ij_to_ji[ij]

        hit = _MISSING
        if i == m:
            hit = self.kj.get((ji, n), _MISSING)
        elif i == n:
            hit = self.kj.get((ji, m), _MISSING)
        elif j == m:
            hit = self.kj.get((ij, n), _MISSING)
        elif j == n:
            hit = self.kj.get((ij, m), _MISSING)
        elif m == n:
            hit = self.nn.get(((ji if i > j else ij), m), _MISSING)
        else:
            key_ij = ji if i > j else ij
            m_ij = int(cc.lmopair_to_lmos_dense[key_ij, m])
            n_ij = int(cc.lmopair_to_lmos_dense[key_ij, n])
            if m_ij != -1 and n_ij != -1:
                lo, hi = (m_ij, n_ij) if m_ij < n_ij else (n_ij, m_ij)
                hit = self.mn.get((key_ij, lo, hi), _MISSING)

        if hit is not _MISSING:
            return hit

        # Nothing stored. psi4 would return a null matrix here and rely on its
        # call sites never asking - every one of them loops over
        # ``lmopair_to_lmos``, which is exactly the set the families cover. That
        # holds, but it makes the lookup silently unsafe for any caller that has
        # not proved it, so this builds the block instead and counts it. A
        # nonzero count is a report that a residual term is reading outside the
        # sparsity its family was sized for, which is worth seeing rather than
        # crashing on or, worse, silently multiplying by a null.
        self._on_the_fly += 1
        return build_overlaps(cc.X_pno, cc.S_pao, cc.lmopair_to_paos,
                              cc.n_pno, [(ij, mn)])[0]

    # -- reporting -----------------------------------------------------------

    def sizes(self):
        """Stored elements per family, and their total in bytes."""
        counts = {}
        total = 0
        for name, store in (("kj", self.kj), ("nn", self.nn), ("mn", self.mn)):
            elements = sum(int(np.prod(ten.shape(b))) for b in store.values()
                           if b is not None)
            counts[name] = (len(store), elements)
            total += elements
        return counts, total * 8

    def report(self, printer):
        """One line per family, because risk 1 of the design is measured here.

        ``S_ij_mn`` is one of the two stores that could break the in-core
        decision, and the only way to know is to print it on the configurations
        that matter rather than to bound it on paper.
        """
        counts, nbytes = self.sizes()
        parts = ", ".join(f"{name} {n} blocks / {el} elements"
                          for name, (n, el) in counts.items())
        printer(f"  overlaps: {parts} ({nbytes / 2**20:.1f} MiB)")
        if self._on_the_fly:
            printer(f"            {self._on_the_fly} built on the fly")
