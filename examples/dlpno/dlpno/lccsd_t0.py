#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The semicanonical triples correction, from psi4's ``DLPNOCCSD_T::compute_lccsd_t0``.

Jiang et al., JCP 161, 082502 (2024), Eq. 109, 110 and 53. Per unique triplet
``i <= j <= k``::

    W_ijk^abc = P_ijk^abc [ (i a | b d) t_kj^cd  -  t_il^ab (j l | k c) ]
    V_ijk^abc = W_ijk^abc + t_i^a (jb|kc) + t_j^b (ia|kc) + t_k^c (ia|jb)
    T_ijk^abc = -W_ijk^abc / (e_a + e_b + e_c - F_ii - F_jj - F_kk)
    e_ijk     = p T_ijk . (8 V^abc - 4 V^cba - 4 V^acb - 4 V^bac
                                  + 2 V^cab + 2 V^bca)

``P_ijk^abc`` is applied explicitly, as psi4 does: the bracket is evaluated six
times with ``(i, j, k)`` permuted, and each result is scattered into ``W`` under
the matching permutation of ``(a, b, c)``. There is no shortcut, because unlike
the spin-orbital form the closed-shell ``W`` is not antisymmetric in ``abc``.

**Every contraction is an explicit einsum index string.** Same reason as
:mod:`dlpno.lccsd`: psi4 spells these as ``doublet`` calls against reshaped
matrices whose flattened axes mean different things in different terms, and a
swapped pair of virtual indices reads as a plausible integral. ``(i a | b d)``
is the worst of them - it is symmetric under exchanging its two trailing indices
in the INTEGRAL and not in the term - which is why
:mod:`dlpno.cc_integrals` already stores its pair-level analogue as rank 3 and
why this one is rank 3 too.

**The permutations are index permutations, not LMO labels.** psi4 shadows
``i, j, k`` inside its permutation loop and then looks quantities up by the
shadowed value, which works only because the six blocks it indexes agree
pairwise whenever two of the three LMOs coincide. Here a permutation carries
the POSITIONS ``(0, 1, 2)`` it permutes, so ``iij`` and ``ijj`` need no such
argument.

**Plan, emit, and chunk.** The phase is one-shot rather than iterative, so
capture-once-replay-many does not apply and the graphs exist for a different
reason: the per-triplet chains are independent, and a captured graph replayed
under the OpenMP executor threads across them where a Python loop cannot. What
capture costs instead is memory, because every triplet's intermediates have to
be alive at once. Seven tensors of ``n_tno^3`` per triplet is more than a large
molecule has, so the triplets are processed in CHUNKS sized against a memory
budget: within a chunk the chains are independent and thread, and between chunks
the only things retained are the energies and the pair-side overlap halves.

Those halves are the one deliberate exception, and they are an exception because
they do not depend on the chunk. ``S(pair, ijk)`` factors into a pair half and a
triplet half, the pair half is the same matrix for every chunk whose triplets
read that pair, and rebuilding it per chunk is work the chunking creates rather
than work the phase needs. So it is cached, in an
:class:`~dlpno.cc_overlaps.OverlapHalfCache` the driver owns for the calculation,
and the cache is charged to the memory budget alongside the retained stores
because it is alive while every chunk runs. Everything else here still dies with
its chunk.

**Nothing is shared between triplets.** The two buffers carrying an auxiliary
index alongside the raw PAO domain - the gathered ``(Q|u v)`` block and its half
transform - are by a wide margin the largest things here and are dead as soon as
``(Q|a b)`` exists, so one of each per CHUNK is the obvious economy, and it is
what this module did first. One buffer per triplet costs memory instead, which
is what the chunk budget is for: both are counted in
:meth:`LCCSDT0._triplet_bytes`, so a molecule large enough for them to matter
simply gets more chunks.

That is now a performance trade rather than a correctness one, and it was the
other way round for a while. Shared, this phase returned wrong answers that
changed from run to run - the water dimer's (T0) landed anywhere between -0.038
and -0.082 against a true -0.006037088212 - because ``Graph::resolve_alias``
gave up after a fixed 32 hops and ``link_alias_storage`` chained same-span
handles one to the next, so past the 32nd triplet two accesses to one buffer
resolved to different owners, the hazard scan saw no conflict, and the OpenMP
executor ran them together. Both are fixed (the linker path-compresses to the
root and the resolver's bound is now a cycle detector that throws), and
``libs/Einsums/ComputeGraph/tests/unit/AliasOrderSharedScratch.cpp`` holds the
schedule-level regression test.

Per triplet is still the right choice, for the reason it always was: a shared
buffer is correctly SERIALIZED, and serializing the chains is the one thing a
captured wide-and-shallow phase exists to avoid.
"""

import time

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import plan_widths
from .cc_overlaps import OverlapHalfCache, build_overlaps

__all__ = ["LCCSDT0"]

#: psi4's six permutations of ``(i, j, k)``, as permutations of the POSITIONS
#: ``0, 1, 2``, paired with the permutation of ``(a, b, c)`` each one scatters
#: into. The scatter string reads as an einsums permute right-hand side:
#: ``W[a, b, c] += Wperm[<spec>]``.
_PERMUTATIONS = [((0, 1, 2), "abc"), ((0, 2, 1), "acb"), ((1, 0, 2), "bac"),
                 ((1, 2, 0), "bca"), ((2, 0, 1), "cab"), ((2, 1, 0), "cba")]

#: The six terms of the energy expression (Jiang Eq. 53), as
#: ``(coefficient, permutation of V)``. psi4 spells these as six calls to
#: ``triples_permuter`` with permuted LMO labels; working out what each one
#: actually does to the virtual indices is done once, here.
_ENERGY_TERMS = [(8.0, "abc"), (-4.0, "cba"), (-4.0, "acb"),
                 (-4.0, "bac"), (2.0, "cab"), (2.0, "bca")]


class LCCSDT0:
    """Evaluate the (T0) correction over a prepared triplet list.

    Constructed from a :class:`~dlpno.triples.DLPNOCCSDT` whose
    ``triples_sparsity`` and ``tno_transform`` have run, so the triplet list,
    the TNO bases and the converged CCSD amplitudes are all fixed.
    """

    def __init__(self, cc, verbose=True, halves=None):
        self.cc = cc
        self.verbose = verbose
        self.F_lmo_np = np.asarray(ten.view(cc.F_lmo))
        self.e_ijk = np.zeros(cc.n_lmo_triplets)
        #: The pair-side overlap halves, shared across this phase's chunks.
        #:
        #: :meth:`_overlaps` runs once per chunk and the pairs it reads overlap
        #: heavily between chunks, so without this every chunk rescatters the
        #: same transforms onto the PAO axis and reforms the same left factors.
        #: The driver may hand one in to share it across the prescreening, the
        #: production and the crude passes as well, since the pair bases are
        #: fixed once the CCSD solve is done; one made here lasts for this
        #: phase, which is the conservative choice and still removes the
        #: per-chunk rebuild. Held on the instance rather than per call because
        #: the halves have to outlive the chunk that first built them.
        self.halves = (OverlapHalfCache(cc.n_lmo_pairs) if halves is None
                       else halves)

        self._plan = None
        self._retain = False
        self.W = self.V = self.T = None
        self._metric_cache = {}
        self.t_plan = 0.0
        self.n_nodes = 0
        #: Nodes captured into the batched-GEMM graphs, which is one per graph
        #: per chunk when the emitters are doing their job. Counted separately
        #: because it is the only signal that says so: a revert to one dispatch
        #: per plan record reproduces every energy in this port to the last bit.
        self.n_batched_nodes = 0
        self.n_chunks = 0

    def _print(self, *args):
        if self.verbose:
            print(*args, flush=True)

    # -- planning ------------------------------------------------------------

    def plan(self):
        """Derive every triplet's work list, once, as integer bookkeeping.

        Nothing here touches a floating-point array. Each record carries the
        three LMOs, the pair each permutation's Eq. 109a amplitude comes from,
        each LMO's pair with every neighbour, the diagonal pairs the singles
        live on, the four domain sizes and the energy prefactor - plus a shape
        class, on the key the pair-level planner uses extended by the neighbour
        count.

        It also carries ``bridges``, the DEDUPLICATED list of pairs whose
        amplitudes this triplet bridges into its TNO basis, and ``bridge_of``,
        the map from a pair to its slot in that list. Both halves of the
        congruence transform read them: the left half ``S^T t`` depends on the
        pair alone, so one per distinct pair is all the arithmetic there is,
        while the right half is per destination. Every permutation's pair is
        also some LMO's pair with some neighbour, so the deduplication removes
        the six permutations outright and whatever a repeated LMO adds on top.
        """
        t0 = time.perf_counter()
        cc = self.cc
        records = []
        for ijk, (i, j, k) in enumerate(cc.ijk_to_i_j_k):
            if not cc.n_tno[ijk]:
                continue
            lmos = cc.lmotriplet_to_lmos[ijk]
            labels = (i, j, k)
            # Per permutation, the pair its doubles amplitude comes from: psi4
            # reads ``T_iajb_[kj]``, the reverse of the permutation's last two.
            perms = [(positions, scatter,
                      int(cc.i_j_to_ij[labels[positions[2]], labels[positions[1]]]))
                     for positions, scatter in _PERMUTATIONS]
            neighbours = [[int(cc.i_j_to_ij[x, l]) for l in lmos]
                          for x in labels]
            bridges, bridge_of = [], {}
            for pair in ([pair for _, _, pair in perms]
                         + [pair for per_x in neighbours for pair in per_x]):
                # A dead pair is bridged by nobody: the terms reading it want
                # the zero their destination was allocated with, which is what
                # psi4's null matrix multiplies out to.
                if pair == -1 or not cc.n_pno[pair] or pair in bridge_of:
                    continue
                bridge_of[pair] = len(bridges)
                bridges.append(pair)
            equal = (i == j) + (j == k) + (i == k)
            records.append(dict(
                ijk=ijk, labels=labels, lmos=lmos, perms=perms,
                neighbours=neighbours,
                bridges=bridges, bridge_of=bridge_of,
                diag=[cc.diag(x) for x in labels],
                nq=len(cc.lmotriplet_to_ribfs[ijk]),
                nl=len(lmos),
                nu=len(cc.lmotriplet_to_paos[ijk]),
                nt=cc.n_tno[ijk],
                bridge_elements=cc.n_tno[ijk] * sum(int(cc.n_pno[p])
                                                    for p in bridges),
                shift=float(sum(self.F_lmo_np[x, x] for x in labels)),
                # One half when exactly two of i, j, k coincide. The
                # i == j == k case never reaches here: triples_sparsity does
                # not form it, and it would contribute exactly zero if it did.
                prefactor=0.5 if equal else 1.0,
                cls=(cc.n_tno[ijk], len(lmos)),
            ))
        self._plan = records
        self.t_plan = time.perf_counter() - t0
        return records

    def class_report(self):
        """How many triplets and how many shape classes, for the plan's report."""
        classes = {r["cls"] for r in self._plan}
        return len(self._plan), len(classes)

    # -- memory --------------------------------------------------------------

    @staticmethod
    def _retained_bytes(r):
        """Bytes one triplet keeps BEYOND its chunk under ``retain``.

        ``W``, ``V`` and the amplitudes. Everything else in the chunk's
        working set dies with the chunk; these three are what the iterative
        (T) reads, from every triplet, so they accumulate across chunks and
        are alive together at the end.
        """
        return 8 * 3 * r["nt"] ** 3

    @staticmethod
    def _triplet_bytes(r):
        """Bytes of transient storage one triplet needs while its chunk runs.

        The dominant term is the seven ``n_tno^3`` tensors - the three
        ``(x a | b d)`` integrals, ``W``, ``V``, the denominator and one scratch
        that becomes the amplitudes - until the PAO domain grows, at which point
        the gathered ``(Q | u v)`` block overtakes them. Both are counted,
        because which one dominates depends on the basis set and the chunk
        budget has to hold either way.

        ``bridge_elements`` is the one term the plan has to supply rather than
        derive from the four domain sizes: the half-transformed ``S^T t`` blocks
        are sized by the PNO counts of the pairs this triplet bridges, one per
        DISTINCT pair, and that set is what :meth:`plan` deduplicated.
        """
        nq, nl, nu, nt = r["nq"], r["nl"], r["nu"], r["nt"]
        return 8 * (8 * nt ** 3                    # K_xvvv x3, W, V, D, scratch, bracket
                    + nq * nu * nu + nq * nt * nu  # (Q|u v) and its half transform
                    + nq * nt * nt                 # (Q | a b)
                    + 3 * nt * nt * max(nl, 1)     # the t_xl stacks
                    + 6 * nt * nt                  # the six t_kj
                    + r["bridge_elements"]         # the half-transformed S^T t
                    + 2 * nq * (3 * nt + 3 * nl)   # the two fitted right-hand sides
                    + 3 * nq * (nl + nu)           # the gathered three-index blocks
                    + 6 * nl * nt + 3 * nt * nt    # (x l | y c) and (x a | y b)
                    + 2 * nq * nq)                 # the metric and its square root

    def _overlap_halves_bytes(self):
        """Bytes the cached pair-side overlap halves will hold for this phase.

        Not per triplet, and not per chunk: the whole point of
        :class:`~dlpno.cc_overlaps.OverlapHalfCache` is that a pair's halves
        outlive the chunk that built them, so they are alive alongside every
        chunk and belong on the same side of the budget as the retained stores.

        Two doubles-sized matrices per pair, each ``npao`` by the pair's PNO
        count, over the DISTINCT pairs this phase's triplets read - which is a
        strict bound rather than an estimate, since :meth:`_overlaps` asks for
        nothing else and the cache stores nothing else. Whatever a cache handed
        in already holds is added, because entries from an earlier pass occupy
        memory until something replaces them.
        """
        cc = self.cc
        npao = ten.shape(cc.S_pao)[0]
        pairs = {self._upper(pair) for r in self._plan
                 for pair in self._pairs_read(r)}
        pending = sum(int(cc.n_pno[pair]) for pair in pairs
                      if not self.halves.holds(pair))
        return self.halves.nbytes() + 8 * 2 * npao * pending

    def _chunks(self):
        """Split the triplets into runs that fit the memory budget.

        In-core with a measured failure, per design decision 10: a triplet too
        large to process even alone reports its own requirement, its domain
        sizes and what to change, because there is no disk path to fall back to.
        """
        budget = int(self.cc.cut.in_core_memory)
        # Everything retained is alive at the END, together with whatever
        # chunk is running then, so the retained total comes off the budget
        # before any chunk is sized against it. Bounding only the chunk is
        # what let this phase page instead of refusing.
        retained = sum(self._retained_bytes(r) for r in self._plan) if self._retain else 0
        # The cached overlap halves are alive alongside every chunk for the same
        # reason, so they come off the budget the same way.
        halves = self._overlap_halves_bytes()
        room = budget - retained - halves
        if room <= 0:
            raise MemoryError(
                f"the iterative (T) would keep {retained / 2**30:.2f} GiB of W, V "
                f"and amplitudes over {len(self._plan)} triplets, plus "
                f"{halves / 2**30:.2f} GiB of pair overlap halves, against a budget "
                f"of {budget / 2**30:.2f} GiB. Raise Thresholds.in_core_memory, "
                "loosen t_cut_tno, or set t0_approximation to stop at (T0), which "
                "retains nothing.")

        chunks, current, total = [], [], 0
        for r in self._plan:
            need = self._triplet_bytes(r)
            if need > room:
                raise MemoryError(
                    f"triplet {r['ijk']} = {r['labels']} needs "
                    f"{need / 2**20:.0f} MiB on its own ({r['nt']} TNOs, "
                    f"{r['nq']} auxiliary functions, {r['nu']} PAOs, "
                    f"{r['nl']} neighbours) against the "
                    f"{room / 2**20:.0f} MiB left of a "
                    f"{budget / 2**20:.0f} MiB budget after the "
                    f"{retained / 2**20:.0f} MiB of retained stores and the "
                    f"{halves / 2**20:.0f} MiB of pair overlap halves. Raise "
                    "Thresholds.in_core_memory or loosen t_cut_tno.")
            if current and total + need > room:
                chunks.append(current)
                current, total = [], 0
            current.append(r)
            total += need
        if current:
            chunks.append(current)
        return chunks

    # -- the run -------------------------------------------------------------

    def run(self, retain=False):
        """Compute every triplet's energy and return their sum.

        Args:
            retain: keep ``W``, ``V`` and the semicanonical amplitudes per
                triplet, in :attr:`W`, :attr:`V` and :attr:`T`. The iterative
                (T) of :mod:`dlpno.lccsd_t` reads all three and reads them
                across triplets, so it needs every one of them resident at
                once - which is psi4's ``save_memory`` argument and the reason
                psi4 grew a disk path here. Costs three ``n_tno^3`` blocks per
                triplet on top of what the phase already holds, so the chunk
                budget covers it (see :meth:`_triplet_bytes`) and a molecule
                that cannot afford it fails with its measured requirement
                rather than thrashing.
        """
        cc = self.cc
        if self._plan is None:
            self.plan()
        self.e_ijk = np.zeros(cc.n_lmo_triplets)
        self._retain = retain
        if retain:
            self.W = [None] * cc.n_lmo_triplets
            self.V = [None] * cc.n_lmo_triplets
            self.T = [None] * cc.n_lmo_triplets
        if not self._plan:
            return 0.0

        chunks = self._chunks()
        self.n_chunks = len(chunks)
        t0 = time.perf_counter()
        for chunk in chunks:
            self._run_chunk(chunk)
        elapsed = time.perf_counter() - t0

        total = float(np.sum(self.e_ijk))
        triplets, classes = self.class_report()
        self._print(
            f"    (T0):     {triplets} triplets in {len(chunks)} chunk(s), "
            f"{classes} shape classes, {self.n_nodes} captured nodes "
            f"({self.n_batched_nodes} batched GEMM), "
            f"{elapsed:.3f} s ({self.t_plan * 1e3:.0f} ms planning)")
        # Printed rather than assumed. The rebuild this removes was invisible to
        # every signal the port has - the energies were already right and the
        # replays already bit-identical - which is the shape of finding the
        # design document records going unseen twice for want of a report.
        reuse = self.halves.report()
        if reuse:
            self._print(f"              {reuse}")
        self._print(f"    (T0) energy = {total:.12f}")
        return total

    def _run_chunk(self, chunk):
        """One chunk: allocate, emit seven graphs, replay, read the energies.

        Several graphs rather than one for the reason ``lccsd.py`` gives: a
        boundary is a point where a slice-versus-whole aliasing question cannot
        arise, and there are a lot of slices here. The metric decomposition
        also has to happen between two of them, since it is not a captured
        operation.

        **Four of the seven hold exactly one node, and that is deliberate.**
        Those four are the batched GEMMs, and a batch is only threaded if
        something gives it the threads. Under the OpenMP executor that was the
        execution LEVEL: a level holding a single node ran unwrapped and the
        batch got the team, where two nodes on one level each ran inside a
        ``parallel for`` and the batch's own region nested and collapsed to one
        thread. A graph holding one node puts that node alone on level 0 by
        construction, which was a cheaper guarantee than reasoning about what
        else landed on a level. The cost is a graph boundary, and a boundary
        between two batches that were going to be separate calls anyway costs
        nothing but the replay.

        These seven are planned like everything else, but they are the one place
        in the port where the planner has the least to work with, and it is
        worth saying why. A phase graph here is replayed exactly ONCE per chunk,
        so there are no recorded timings for a second plan to improve on - the
        cold cost model is the whole of it. When it declines, ``plan_widths``
        leaves the OpenMP executor in place and the one-node rule above is still
        what threads the batch, which is why these graphs are still built one
        node at a time.
        """
        state = self._allocate(chunk)
        overlaps = self._overlaps(chunk)

        for name, batched, emit in (("gather", False, self._emit_gather),
                                    ("(Q|x a)", True, self._emit_rotate),
                                    ("fit", False, self._emit_fit),
                                    ("integrals", True, self._emit_integrals),
                                    ("S^T t", True, self._emit_projections),
                                    ("amplitudes", True, self._emit_amplitudes),
                                    ("Eq. 109-110, 53", False, self._emit_triples)):
            g = cg.Graph(f"lccsd(t0): {name}")
            with cg.capture(g):
                emit(chunk, state, overlaps)
            self.n_nodes += g.num_nodes()
            if batched:
                self.n_batched_nodes += g.num_nodes()
            plan_widths(g)
            g.execute()

        energies = ten.view(state["e_chunk"])
        for slot, r in enumerate(chunk):
            self.e_ijk[r["ijk"]] = float(energies[slot])

        if self._retain:
            # ``scratch`` holds the amplitudes by this point; see
            # :meth:`_emit_triples` for the buffer's two lives. Kept by
            # reference rather than copied - the chunk's state dies here and
            # these are the only three of its tensors anyone reads again.
            for r, per in zip(chunk, state["per"]):
                self.W[r["ijk"]] = per["W"]
                self.V[r["ijk"]] = per["V"]
                self.T[r["ijk"]] = per["scratch"]

    # -- overlaps ------------------------------------------------------------

    def _upper(self, pair):
        """The ``i <= j`` name of a pair.

        ``ij`` and ``ji`` share a PNO basis and a PAO domain - the transform is
        literally the same object - so their overlaps against a triplet are the
        same matrix. Canonicalizing the request halves the family.
        """
        i, j = self.cc.ij_to_i_j[pair]
        return pair if i <= j else int(self.cc.ij_to_ji[pair])

    def _overlaps(self, chunk):
        """``S(pair, ijk) = X_pno[pair]^T S_pao X_tno[ijk]`` for the chunk.

        Every amplitude the triples read lives in some pair's PNO basis and is
        contracted in a triplet's TNO basis, so this is the bridge, and it is
        wanted for three families: the pair of each permutation's last two
        LMOs, each of ``i, j, k`` against every neighbour, and the three
        diagonal pairs the singles live on.

        Built through the same :func:`~dlpno.cc_overlaps.build_overlaps` as the
        pair families, with the triplets appended to the pair list as extra
        entities: that function only cares that an entity has a transform, a
        domain and a column count. psi4 instead restricts ``S_pao`` to a merged
        "extended" domain and indexes into it; scattering each transform onto
        the full PAO axis makes the restriction implicit and gives the
        identical matrix.

        The concatenation is also what makes :attr:`halves` safe: pairs occupy
        the entity indices below ``n_lmo_pairs`` and the chunk's triplets sit
        above them, and only the pair range is eligible for the cache. A pair is
        read by triplets in many chunks and its halves are the same matrices
        every time; a triplet appears in one chunk and nowhere else.
        """
        cc = self.cc
        npairs = cc.n_lmo_pairs
        X_all = list(cc.X_pno) + list(cc.X_tno)
        paos_all = list(cc.lmopair_to_paos) + list(cc.lmotriplet_to_paos)
        n_all = list(cc.n_pno) + list(cc.n_tno)

        requests = set()
        for r in chunk:
            entity = npairs + r["ijk"]
            for pair in self._pairs_read(r):
                requests.add((self._upper(pair), entity))
        keys = sorted(requests)
        blocks = build_overlaps(X_all, cc.S_pao, paos_all, n_all, keys,
                                cache=self.halves)
        return {(pair, entity - npairs): block
                for (pair, entity), block in zip(keys, blocks)}

    def _pairs_read(self, r):
        """Every pair whose amplitudes one triplet reads."""
        pairs = {pair for _, _, pair in r["perms"]}
        for per_x in r["neighbours"]:
            pairs.update(per_x)
        pairs.update(r["diag"])
        return [p for p in pairs if p != -1 and self.cc.n_pno[p]]

    def _S(self, overlaps, pair, ijk):
        """The bridge for a pair, or ``None`` where there is nothing to bridge."""
        if pair == -1 or not self.cc.n_pno[pair]:
            return None
        return overlaps.get((self._upper(pair), ijk))

    # -- allocation ----------------------------------------------------------

    def _allocate(self, chunk):
        """Every tensor one chunk writes, plus the metric factorizations.

        The metric decomposition is here rather than in an emitter because
        ``pow(0.5)`` is an eigendecomposition and not a captured operation:
        calling it inside a capture would record the gather that feeds it and
        then read an unwritten tensor.

        Views are collected into ``keep`` for the reason ``lccsd.py`` gives: a
        view handed to a captured operation is held by the graph as a slot
        pointer, so one that dies with the expression that made it leaves the
        replay reading freed memory.
        """
        cc = self.cc
        state = {"keep": [], "per": []}
        state["e_chunk"] = ten.zeros("e (triplet)", [len(chunk)])

        for slot, r in enumerate(chunk):
            nq, nl, nu, nt = r["nq"], r["nl"], r["nu"], r["nt"]
            per = {}
            # Per triplet, not per chunk. Sharing these is a measured data race
            # under the OpenMP executor; see the module docstring.
            per["uv"] = ten.empty("(Q|u v) raw", [nq, nu, nu])
            per["half"] = ten.empty("(Q|a v) half", [nq, nt, nu])
            per["q_vv"] = ten.zeros("(Q|a b)", [nq, nt, nt])
            per["q_vv_flat"] = self._keep(
                state, per["q_vv"].reshape_view([nq, nt * nt]))
            per["raw_o"] = [ten.zeros("(Q|x m) raw", [nq, 1, nl]) for _ in range(3)]
            per["raw_v"] = [ten.zeros("(Q|x u) raw", [nq, 1, nu]) for _ in range(3)]
            per["raw_o_flat"] = [self._keep(state, b.reshape_view([nq, nl]))
                                 for b in per["raw_o"]]
            per["raw_v_flat"] = [self._keep(state, b.reshape_view([nq, nu]))
                                 for b in per["raw_v"]]

            # One right-hand side per metric power, laid out so every block a
            # consumer wants is a contiguous column slice. See _emit_fit for
            # why there are two.
            per["rhs_half"] = ten.zeros("J^-1/2 rhs", [nq, 3 * nt + 3 * nl])
            per["rhs_full"] = ten.zeros("J^-1 rhs", [nq, 3 * nt])
            per["rhs_virtual"] = self._keep(state, per["rhs_half"][:, :3 * nt])
            per["q_xv"] = [self._keep(state, per["rhs_half"][:, x * nt:(x + 1) * nt])
                           for x in range(3)]
            per["q_xo"] = [self._keep(state, per["rhs_half"][
                :, 3 * nt + x * nl:3 * nt + (x + 1) * nl]) for x in range(3)]
            per["q_xv_inv"] = [self._keep(state, per["rhs_full"][:, x * nt:(x + 1) * nt])
                               for x in range(3)]
            per["metric"] = self._metric_copies(r["ijk"])

            # (x a | b d) for x in i, j, k. Allocated FLAT, with the rank-3
            # naming as the view, because that is the orientation the batched
            # GEMM writes and one grouped call needs its whole destination list
            # to be of one kind; see :meth:`_emit_integrals` for the merge order
            # and the module docstring for why the rank-3 name exists at all.
            per["K_xvvv_flat"] = [ten.zeros("K (x a|b d)", [nt, nt * nt])
                                  for _ in range(3)]
            per["K_xvvv"] = [self._keep(state, K.reshape_view([nt, nt, nt]))
                             for K in per["K_xvvv_flat"]]
            # (y l | z c), one per permutation, and (x a | y b), one per pair of
            # the triplet's own LMOs.
            per["K_ooov"] = [ten.zeros("K (y l|z c)", [max(nl, 1), nt])
                             for _ in range(6)]
            per["K_ovov"] = [ten.zeros("K (x a|y b)", [nt, nt]) for _ in range(3)]

            # One block per DISTINCT pair this triplet bridges, holding the left
            # half ``S^T t`` of the congruence transform. The plan deduplicated
            # that set; see :meth:`_emit_projections`.
            per["bridge"] = [ten.zeros("S^T t", [nt, int(cc.n_pno[pair])])
                             for pair in r["bridges"]]
            # The amplitudes those read, as views made HERE rather than in the
            # emitter. Building a view inside a capture records a node for it,
            # so a per-request ``logical_view`` costs a third of the phase's
            # node count all by itself - which is the same reason every other
            # view in this method is made before the graph exists. Logical
            # rather than padded: ``S`` is sized to the pair's PNO count, so a
            # bucket's padding would not fit the product.
            per["T_pair"] = [self._keep(state,
                                        cc.layout.logical_view(cc.T_all, pair))
                             for pair in r["bridges"]]
            per["t1_diag"] = [self._keep(state, cc.T_ia[lmo][:, 0:1])
                              if pair != -1 and cc.n_pno[pair] else None
                              for lmo, pair in zip(r["labels"], r["diag"])]

            # The six t_kj as column slices of one store, so the batch that
            # writes them has a destination list of one kind. Slices of a
            # column-major matrix, so each is the contiguous block an owning
            # tensor would have been.
            per["t_kj_store"] = ten.zeros("t_kj (TNO) store", [nt, 6 * nt])
            per["t_kj"] = [self._keep(state,
                                      per["t_kj_store"][:, idx * nt:(idx + 1) * nt])
                           for idx in range(6)]
            per["t_xl"] = [ten.zeros("t_xl (TNO)", [nt, nt, max(nl, 1)])
                           for _ in range(3)]
            per["t_xl_slot"] = [
                [self._keep(state, stack[:, :, s]) for s in range(nl)]
                for stack in per["t_xl"]]
            per["t_x"] = [ten.zeros("t_x (TNO)", [nt, 1]) for _ in range(3)]
            per["t_x_vec"] = [self._keep(state, t.reshape_view([nt]))
                              for t in per["t_x"]]

            per["W"] = ten.zeros("W (triplet)", [nt, nt, nt])
            #: Where Eq. 53's bracket is accumulated. Normally ``W`` itself,
            #: which is dead by then; a separate buffer when the caller has
            #: asked to keep ``W`` for the iterative (T), because that is the
            #: one reader for which "dead by then" is false.
            per["bracket"] = (ten.zeros("Eq. 53 bracket", [nt, nt, nt])
                              if self._retain else per["W"])
            per["V"] = ten.zeros("V (triplet)", [nt, nt, nt])
            per["D"] = ten.zeros("denominator", [nt, nt, nt])
            #: Scratch: holds each permutation's contribution while ``W`` is
            #: assembled, then the amplitudes once ``W`` is complete.
            per["scratch"] = ten.zeros("W (permutation)", [nt, nt, nt])
            per["e"] = self._keep(state, state["e_chunk"][slot:slot + 1])
            state["per"].append(per)
        return state

    @staticmethod
    def _keep(state, view):
        state["keep"].append(view)
        return view

    def _metric_copies(self, ijk):
        """Fresh copies of one triplet domain's metric and its square root.

        Fresh because ``gesv`` overwrites its left-hand side, and memoized by
        DOMAIN because the eigendecomposition behind ``pow(0.5)`` is the
        expensive half and triplets sharing an auxiliary domain share it.
        """
        cc = self.cc
        domain = cc.lmotriplet_to_ribfs[ijk]
        hit = self._metric_cache.get(id(domain))
        if hit is None:
            A = sparse.submatrix_rows_and_cols(cc.metric, domain, domain,
                                               name="(P|Q) domain")
            hit = (ten.view(A).copy(), ten.view(la.pow(A, 0.5)).copy())
            self._metric_cache[id(domain)] = hit
        return (ten.from_numpy("(P|Q) copy", hit[0]),
                ten.from_numpy("(P|Q)^1/2 copy", hit[1]))

    # -- emission ------------------------------------------------------------

    def _emit_gather(self, chunk, state, overlaps):
        """Gather the three-index blocks, and rotate ``(Q|u v)`` to TNOs.

        ``(Q | a b)`` is finished here and never fitted at all - see
        :meth:`_emit_fit` for why - which is why the two-step rotation writes it
        in this graph and nothing solves against it in the next.
        """
        cc = self.cc
        for r, per in zip(chunk, state["per"]):
            ijk, nl = r["ijk"], r["nl"]
            qs = [int(q) for q in cc.lmotriplet_to_ribfs[ijk]]
            ms = [int(m) for m in r["lmos"]]
            us = [int(u) for u in cc.lmotriplet_to_paos[ijk]]
            X = cc.X_tno[ijk]

            for x, lmo in enumerate(r["labels"]):
                la.gather(per["raw_v"][x], cc.q_ia, [qs, [int(lmo)], us])
                if nl:
                    la.gather(per["raw_o"][x], cc.q_ij, [qs, [int(lmo)], ms])
                    la.axpby(1.0, per["raw_o_flat"][x], 0.0, per["q_xo"][x])

            # X^T (Q | u v) X, in two passes through the shared buffers. Both
            # transformed indices are interior axes of a rank-3 block, so
            # neither is a merged-axis GEMM; the per-auxiliary-function form is
            # the trap cc_integrals records - 2 naux operations per triplet, and
            # each one builds a view.
            la.gather(per["uv"], cc.q_ab, [qs, us, us])
            einsums.einsum("Qav <- Quv ; ua", per["half"], per["uv"], X)
            einsums.einsum("Qab <- Qav ; vb", per["q_vv"], per["half"], X)

    def _emit_rotate(self, chunk, state, overlaps):
        """``(Q | x u) X -> (Q | x a)``, one batch for the whole chunk.

        Three GEMMs per triplet, each writing straight into its column slice of
        the right-hand side the fit will solve against, and every one of them
        independent of every other - so they are one
        :func:`~einsums.graph.grouped_batched_gemm` rather than
        ``3 * len(chunk)`` dispatches. The shapes differ between triplets and
        need not agree: the grouped form sorts its members into uniform
        ``(m, n, k, lda, ldb, ldc)`` groups at capture, which is why nothing here
        pads or reads the plan's shape class.
        """
        cc = self.cc
        a, b, c = [], [], []
        for r, per in zip(chunk, state["per"]):
            for x in range(3):
                a.append(per["raw_v_flat"][x])
                b.append(cc.X_tno[r["ijk"]])
                c.append(per["q_xv"][x])
        cg.grouped_batched_gemm(1.0, a, b, 0.0, c)

    def _emit_fit(self, chunk, state, overlaps):
        """Apply both metric powers to the right-hand sides.

        **Two metric powers, and the same trap :mod:`dlpno.cc_integrals`
        documents.** Most blocks are fitted symmetrically, ``J^-1/2 (Q|pq)``, so
        a product of two of them reconstructs an integral. The three-external
        family is the exception: ``(i a | b d)`` contracts a fitted ``(Q|i a)``
        against a completely UNFITTED ``(Q|b d)``, so its left factor carries
        the full inverse instead. Applying one power throughout would apply the
        metric one and a half times, and ``(Q | a b)`` is never fitted at all.

        The full-inverse copies are taken BEFORE the symmetric fit overwrites
        the blocks they come from, which is psi4's order. Both steps stay in one
        graph, because what enforces that order is the write-after-read edge the
        hazard scan draws between overlapping slices of ``rhs_half``, not the
        graph boundary.
        """
        for per in state["per"]:
            la.axpby(1.0, per["rhs_virtual"], 0.0, per["rhs_full"])
            A_full, A_half = per["metric"]
            la.gesv(A_full, per["rhs_full"])
            la.gesv(A_half, per["rhs_half"])

    def _emit_integrals(self, chunk, state, overlaps):
        """The three integral families the triples contract, in ONE batch.

        ``(x a | b d)`` from the full-inverse factor against the unfitted
        ``(Q|a b)``; ``(y l | z c)`` and ``(x a | y b)`` from two symmetric
        factors each.

        All three are ``alpha op(A)^T op(B)`` with no accumulation, every
        destination is its own tensor, and nothing here reads what anything else
        here writes - so all twelve per triplet go out as one grouped batch over
        the whole chunk. The families differ only in shape, which is precisely
        what the grouped form absorbs.

        ``K_xvvv`` is written through the MERGED form of a rank-3 tensor, and the
        merge order is not a convention to pick: joining two adjacent axes of a
        column-major tensor leaves the FIRST varying fastest, and ``(Q|a b)``
        was flattened the same way, so ``K[a, b, d]`` comes out meaning
        ``(x a | b d)`` on both sides of the GEMM.
        """
        a, b, c = [], [], []
        for r, per in zip(chunk, state["per"]):
            for x in range(3):
                a.append(per["q_xv_inv"][x])
                b.append(per["q_vv_flat"])
                c.append(per["K_xvvv_flat"][x])

            # (y l | z c) per permutation: the occupied factor of the
            # permutation's middle position against the virtual factor of its
            # last.
            if r["nl"]:
                for idx, (positions, _scatter, _pair) in enumerate(r["perms"]):
                    a.append(per["q_xo"][positions[1]])
                    b.append(per["q_xv"][positions[2]])
                    c.append(per["K_ooov"][idx])

            # (j b | k c), (i a | k c) and (i a | j b), in the order Eq. 110
            # reads them.
            for slot, (x, y) in enumerate(((1, 2), (0, 2), (0, 1))):
                a.append(per["q_xv"][x])
                b.append(per["q_xv"][y])
                c.append(per["K_ovov"][slot])
        cg.grouped_batched_gemm(1.0, a, b, 0.0, c, trans_a=True)

    def _emit_projections(self, chunk, state, overlaps):
        """The left half ``S^T t`` of every congruence transform, in ONE batch.

        Every amplitude the triples read lives in some pair's PNO basis and is
        contracted in a triplet's TNO basis, so all of them arrive through the
        same congruence transform ``S^T t S`` against a pair-to-triplet overlap.
        Split in two, because the two halves batch differently: the left half
        depends on the PAIR alone, so it is computed once per distinct pair
        (:meth:`plan`'s ``bridges``), while the right half is once per
        destination.

        That sharing is worth having rather than incidental. A triplet reaches
        the same pair through all six permutations and through each of three
        LMOs' neighbour lists, so the requests outnumber the distinct pairs by
        the six permutations plus whatever a repeated LMO adds - and psi4, which
        rebuilds the overlap inside its permutation loop, does the whole
        transform once per request.

        The singles ride in the same batch, because they are the same operation:
        a diagonal pair's bridge against a one-column amplitude. A projection
        written as a COLUMN, not a row - a GEMM with one row is a degenerate case
        and Eq. 84c cost the CCSD iteration a factor of two at ten threads before
        it was turned round.
        """
        a, b, c = [], [], []
        for r, per in zip(chunk, state["per"]):
            ijk = r["ijk"]
            for slot, pair in enumerate(r["bridges"]):
                a.append(self._S(overlaps, pair, ijk))
                b.append(per["T_pair"][slot])
                c.append(per["bridge"][slot])

            for x in range(3):
                S = self._S(overlaps, r["diag"][x], ijk)
                if S is None or per["t1_diag"][x] is None:
                    continue
                a.append(S)
                b.append(per["t1_diag"][x])
                c.append(per["t_x"][x])
        if a:
            cg.grouped_batched_gemm(1.0, a, b, 0.0, c, trans_a=True)

    def _emit_amplitudes(self, chunk, state, overlaps):
        """The right half ``(S^T t) S``, into every destination, in ONE batch.

        Two families read it:

        * ``t_{k'j'}`` for each of the six permutations, Eq. 109a's doubles,
          into the six column slices of one store;
        * ``t_{xl}`` for ``x`` in ``i, j, k`` and ``l`` over the triplet's
          neighbours, Eq. 109b's, into slot ``l`` of a stack over ``l`` so that
          term becomes one einsum instead of one per neighbour.

        A dead pair contributes no member and leaves its destination at the zero
        it was allocated with, which is what psi4's null matrix multiplies out to
        and what the terms reading it want. Every destination is distinct - the
        six ``t_kj`` are disjoint column slices and the stack slots are disjoint
        planes - which is what makes one batch legal, since it orders nothing
        between its members.
        """
        a, b, c = [], [], []
        for r, per in zip(chunk, state["per"]):
            ijk, bridge_of = r["ijk"], r["bridge_of"]
            for idx, (_positions, _scatter, pair) in enumerate(r["perms"]):
                if pair not in bridge_of:
                    continue
                a.append(per["bridge"][bridge_of[pair]])
                b.append(self._S(overlaps, pair, ijk))
                c.append(per["t_kj"][idx])

            for x in range(3):
                for s, pair in enumerate(r["neighbours"][x]):
                    if pair not in bridge_of:
                        continue
                    a.append(per["bridge"][bridge_of[pair]])
                    b.append(self._S(overlaps, pair, ijk))
                    c.append(per["t_xl_slot"][x][s])
        if a:
            cg.grouped_batched_gemm(1.0, a, b, 0.0, c)

    def _emit_triples(self, chunk, state, overlaps):
        """Eq. 109, Eq. 110 and Eq. 53: ``W``, ``V``, the amplitudes, the energy.

        The four ``n_tno^3`` buffers are reused in a deliberate order, and the
        graph's hazard analysis is what makes that safe: ``scratch`` carries
        each permutation's block while ``W`` accumulates, then becomes the
        amplitudes once ``W`` is complete; ``W`` itself becomes the energy
        bracket once the amplitudes have read it. Nothing else here is of that
        order, and at production TNO counts each one is tens of megabytes.
        """
        for r, per in zip(chunk, state["per"]):
            scratch, W, V, D = per["scratch"], per["W"], per["V"], per["D"]

            # -- Eq. 109 ----------------------------------------------------
            for idx, (positions, scatter, _pair) in enumerate(r["perms"]):
                # + (x a | b d) t_{zy}^{cd}, then - t_{xl}^{ab} (y l | z c).
                einsums.einsum("abc <- abd ; cd", scratch,
                               per["K_xvvv"][positions[0]], per["t_kj"][idx])
                if r["nl"]:
                    einsums.einsum("abc <- abl ; lc", scratch,
                                   per["t_xl"][positions[0]], per["K_ooov"][idx],
                                   c_pf=1.0, ab_pf=-1.0)
                einsums.permute(f"abc <- {scatter}", W, scratch,
                                c_pf=0.0 if idx == 0 else 1.0)

            # -- Eq. 110 ----------------------------------------------------
            la.axpby(1.0, W, 0.0, V)
            einsums.einsum("abc <- a ; bc", V, per["t_x_vec"][0],
                           per["K_ovov"][0], c_pf=1.0)
            einsums.einsum("abc <- b ; ac", V, per["t_x_vec"][1],
                           per["K_ovov"][1], c_pf=1.0)
            einsums.einsum("abc <- c ; ab", V, per["t_x_vec"][2],
                           per["K_ovov"][2], c_pf=1.0)

            # -- Eq. 53 -----------------------------------------------------
            e_tno = self.cc.e_tno[r["ijk"]]
            la.outer_sum(D, [e_tno, e_tno, e_tno], [1.0, 1.0, 1.0])
            la.shift(-r["shift"], D)
            la.direct_division(-1.0, W, D, 0.0, scratch)

            # The bracket normally overwrites W, which the amplitudes have
            # just finished reading; under ``retain`` it gets its own buffer,
            # because then W has a reader left. The prefactor rides on the six
            # coefficients rather than scaling the result, which saves a pass
            # over n_tno^3.
            bracket = per["bracket"]
            for term, (coefficient, order) in enumerate(_ENERGY_TERMS):
                einsums.permute(f"abc <- {order}", bracket, V,
                                c_pf=0.0 if term == 0 else 1.0,
                                a_pf=coefficient * r["prefactor"])
            la.dot(per["e"], bracket, scratch)
