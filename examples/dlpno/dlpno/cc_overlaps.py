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

Both of those are per CALL, and a caller that asks in installments pays for them
once per installment. The (T0) driver is exactly that caller: it asks per chunk
of triplets, and a pair read by triplets in several chunks has its scatter and
its left factor rebuilt in each of them. :class:`OverlapHalfCache` closes that,
by keeping the two pair-side halves across calls.
"""

import numpy as np

from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import DLPNOBase

__all__ = ["build_overlaps", "OverlapHalfCache", "PnoOverlapStore"]

#: Distinguishes "no block stored" from a stored ``None``, which is what a
#: request naming a dead pair yields.
_MISSING = object()

#: Replay a setup graph as an OpenMP team over its independent nodes. A
#: staticmethod on DLPNOBase; bound here so this module needs no instance.
#: Never a ThreadPoolExecutor - see the note on DLPNOBase._run.
_run_setup_graph = DLPNOBase._run


class OverlapHalfCache:
    """The two pair-side halves of :func:`build_overlaps`, kept across calls.

    ``build_overlaps`` shares work within one call and nothing between calls,
    which is right for a caller that asks once and wrong for the (T0) driver,
    which asks once per chunk of triplets and once more per pass over them. The
    two things it shares within a call are also the two that depend on ONE
    entity rather than on a request: the full-PAO-axis transform ``X_full`` and
    the left factor ``X_full^T S_pao``. So they are exactly what survives a call
    boundary, and this holds them.

    **Only the entities below :attr:`n_cacheable` are eligible, and that is not
    a tuning knob.** The (T0) driver hands ``build_overlaps`` a CONCATENATED
    entity list - the pairs, then the chunk's triplets - so an entity index
    means "pair ``ij``" below the pair count and "triplet" above it. A triplet
    appears in exactly one chunk, so caching one would hold memory nothing will
    ask for again; worse, the triplet bases are rebuilt between passes, so an
    index above the pair count is not even a stable name. Restricting the cache
    to the pair range is what keeps the key stable and the two entity kinds
    apart.

    **Validity is checked, not assumed.** A pair's halves are a function of its
    transform, its PAO domain and its PNO count, and the CCSD PNO rebuild
    (``DLPNOCCSD::recompute_pnos``) replaces all three. Rather than reason about
    where the rebuild sits relative to the cache's lifetime, each entry keeps a
    reference to the three objects it was built from and is discarded if any of
    them is not the same object next time. Keeping the reference, rather than an
    ``id()``, is deliberate: a dropped tensor's address can be reissued to its
    replacement, and this port has already been bitten by exactly that.

    Lifetime is the caller's: whoever constructs one decides how long the halves
    live, and holding it for one calculation is what makes the reuse safe. There
    is no module-level instance on purpose.
    """

    def __init__(self, n_cacheable):
        #: Entity indices below this are pairs, and only pairs are cached.
        self.n_cacheable = int(n_cacheable)
        #: Entity to ``(X_full, left, provenance)``.
        self._halves = {}
        #: Halves asked for that were eligible, and how many had to be built.
        #: Their difference is the rebuild this cache removed, which is the
        #: only way to see that it is doing anything.
        self.requests = 0
        self.builds = 0

    def cacheable(self, entity):
        """Whether ``entity`` names a pair, and so may be cached."""
        return 0 <= entity < self.n_cacheable

    def holds(self, entity):
        """Whether an entry occupies memory for ``entity``, stale or not."""
        return entity in self._halves

    @staticmethod
    def provenance(X, paos, S_pao, n):
        """Everything an entity's halves were built from, and nothing else.

        The transform, the PAO domain and the overlap matrix are compared by
        IDENTITY, since a rebuild replaces the objects rather than editing them
        and an unrelated array that happens to hold equal values is not the same
        basis. The PNO count is compared by value, because it arrives as a fresh
        scalar object on every access and identity on it would never match.
        """
        return ((X, paos, S_pao), int(n))

    def get(self, entity, provenance):
        """The stored halves for ``entity``, or ``None`` if absent or stale."""
        hit = self._halves.get(entity)
        if hit is None:
            return None
        X_full, left, was = hit
        if (was[1] != provenance[1]
                or any(a is not b for a, b in zip(was[0], provenance[0]))):
            del self._halves[entity]
            return None
        return X_full, left

    def put(self, entity, X_full, left, provenance):
        self._halves[entity] = (X_full, left, provenance)

    def nbytes(self):
        """Bytes the stored halves hold.

        ``X_full`` is ``(npao, n_pno)`` and the left factor is its transpose's
        shape, so a cached pair costs ``2 npao n_pno`` doubles and the whole
        cache costs ``2 npao`` times the PNO count of every pair in it.
        """
        return 8 * sum(int(np.prod(ten.shape(X))) + int(np.prod(ten.shape(left)))
                       for X, left, _was in self._halves.values())

    def report(self):
        """One line, or ``None`` when the cache never answered anything."""
        saved = self.requests - self.builds
        if not saved:
            return None
        return (f"{saved} of {self.requests} pair overlap halves reused "
                f"({self.nbytes() / 2**20:.1f} MiB held)")


def build_overlaps(X_pno, S_pao, lmopair_to_paos, n_pno, requests, cache=None):
    """``S(ij, mn)`` for each requested pair of pairs.

    Args:
        X_pno: Per pair, its PAO-domain-to-PNO transform.
        S_pao: The full PAO overlap matrix.
        lmopair_to_paos: Per pair, the PAO indices its domain covers.
        n_pno: Per pair, its PNO count.
        requests: Iterable of ``(ij, mn)``.
        cache: Optional :class:`OverlapHalfCache`, which keeps the pair-side
            halves across calls. Opt in, because it only pays for a caller that
            asks in installments and only that caller knows how long the halves
            should live.

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

    # What the cache can answer, and what is left to build. An entity's halves
    # are a function of these three objects, so they are also what says whether
    # a stored entry still describes it.
    X_full, lefts, fresh = {}, {}, []
    for ij in needed:
        if cache is not None and cache.cacheable(ij):
            cache.requests += 1
            hit = cache.get(ij, cache.provenance(X_pno[ij],
                                                 lmopair_to_paos[ij],
                                                 S_pao, n_pno[ij]))
            if hit is not None:
                X_full[ij], lefts[ij] = hit
                continue
            cache.builds += 1
        fresh.append(ij)

    if fresh:
        # Each pair's transform on the full PAO axis, so the domain restriction
        # is implicit and no request has to gather S_pao.
        g_pad = cg.Graph("PNO transforms on the full PAO axis")
        with cg.capture(g_pad):
            for ij in fresh:
                X_full[ij] = ten.zeros(f"X (full PAO) {ij}", [npao, n_pno[ij]])
                sparse.scatter_into(X_full[ij], X_pno[ij],
                                    [lmopair_to_paos[ij], range(n_pno[ij])])
        _run_setup_graph(g_pad)

        # The left factor, once per distinct pair rather than once per request.
        for ij in fresh:
            lefts[ij] = ten.zeros(f"half {ij}", [n_pno[ij], npao])
        g_half = cg.Graph("PNO overlap left factors")
        with cg.capture(g_half):
            cg.grouped_batched_gemm(
                1.0, [X_full[ij] for ij in fresh], [S_pao] * len(fresh),
                0.0, [lefts[ij] for ij in fresh], trans_a=True)
        _run_setup_graph(g_half)

        # Stored only after the graph that fills them has run, so nothing can
        # read an entry whose scatter has not happened yet.
        if cache is not None:
            for ij in fresh:
                if cache.cacheable(ij):
                    cache.put(ij, X_full[ij], lefts[ij],
                              cache.provenance(X_pno[ij],
                                               lmopair_to_paos[ij],
                                               S_pao, n_pno[ij]))

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
