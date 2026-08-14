#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The iterative triples, from psi4's ``DLPNOCCSD_T::lccsd_t_iterations``.

Jiang et al., JCP 161, 082502 (2024), Eq. 111 and 112::

    R_ijk^abc = W_ijk^abc + T_ijk^abc (e_a + e_b + e_c - F_ii - F_jj - F_kk)
                - sum_l f_il T_ljk^abc - sum_l f_jl T_ilk^abc - sum_l f_kl T_ijl^abc
    T_ijk^abc -= R_ijk^abc / (e_a + e_b + e_c - F_ii - F_jj - F_kk)

**What this milestone actually buys is the last line of Eq. 111**, and it is
worth stating plainly because the rest of the file is machinery for it. (T0)
solves the same equation with those three sums dropped, which is exact only
where the occupied Fock matrix is diagonal. In the localized basis this port
works in it is not, and the difference is 1.5e-4 Eh on water/cc-pVDZ - five
percent of the correction. Restoring the coupling is the whole of (T).

That also hands the milestone its gate for free. (T0) needed two untruncated
comparisons because it is not invariant to a rotation of the occupied space;
(T) IS, so untruncated in the localized basis it must reproduce canonical
DF-CCSD(T) exactly - the number ``canonical_triples`` already pins and that
``test_lccsd_t0.py`` records (T0) missing.

**Jacobi, not psi4's in-place update.** psi4 updates ``T_ijk`` inside the same
OpenMP loop that reads its neighbours' amplitudes, so which iterate a given
term sees depends on thread scheduling. Both converge to the same fixed point -
it is the fixed point that is the answer - but only one of them gives the same
trajectory twice. Every residual here is computed against the previous
iterate and the amplitudes are updated afterwards, which costs one more
``n_tno^3`` block per triplet and makes a replay reproducible.

**The coupling is what makes this phase different from every other one in the
port.** Every term reads a DIFFERENT triplet's amplitudes, in that triplet's
own TNO basis and its own canonical index order, so each contribution needs
both a basis rotation (three GEMMs, one per index) and an index permutation.
Which permutation is not a free choice: an amplitude block is stored once, in
the ``i <= j <= k`` ordering of its own triplet, and read here in whatever
ordering the requesting triplet's ``a, b, c`` correspond to.

**Every one of those rotations is batched, and the permutation is what it cost
to get there.** A coupling is three GEMMs, all three of them independent of
every other coupling's until the last one accumulates, so the phase wants to
issue them as a handful of :func:`~einsums.graph.grouped_batched_gemm` calls
rather than a few thousand dispatches. Two things stood in the way and both are
resolved in :data:`_CHAIN`: the middle rotation of the chain as written is not a
GEMM at all (it contracts an interior axis), and the permutation cannot ride
along in a GEMM. Rotating in the order the storage already runs in makes all
three stages plain GEMMs, and because all three rotations use the SAME overlap
matrix the permutation commutes with them, so it can be absorbed either into
which layout the last GEMM writes or into one transposed copy of the partner's
amplitudes.

**What that leaves is three calls per coupling SLOT rather than six nodes per
coupling**, and the two are separated by however many triplets a call spans. A
call cannot span a triplet whose work is behind its own conditional, so the
phase is captured twice and each pass picks: gated per :attr:`LCCSDT.block`
where something has settled, and ungated over every triplet where nothing has,
which is every pass until the first triplet settles. The second is a
substitution rather than an approximation, and :meth:`LCCSDT.iterate` says why.
At methanol/cc-pVDZ that is 16482 nodes a pass becoming 360 on the passes that
cost anything and 8499 on the rest, with the energies and the pass count
unmoved to the last bit.
"""

import time

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import tensors as ten
from .base import DLPNOBase
from .cc_overlaps import build_overlaps

__all__ = ["LCCSDT", "permuter_spec", "chain_of", "rotation_stages",
          "prefer_wide_replay"]

_run_setup_graph = DLPNOBase._run


def _omp_max_threads():
    """The OpenMP runtime's current ceiling on a parallel region's team size.

    Delegates to :func:`einsums.hardware.get_max_threads`, which reads the
    same process-wide OpenMP runtime every Einsums C++ site consults -
    :class:`~einsums.graph.OpenMPExecutor` among them. It is deliberately
    not ``os.environ.get("OMP_NUM_THREADS")``: importing psi4 takes the
    process-wide count over, setting it to that variable if it was
    exported and to 1 if it was not, and ``psi4.set_num_threads`` (see
    ``bench_vs_psi4.py``) then overrides both, so only a runtime call sees
    the number actually in effect when this solver runs. Kept as a
    module-level function rather than called inline so tests can
    monkeypatch the thread count the gate observes.
    """
    return int(einsums.hardware.get_max_threads())


#: ``perm_idx`` from psi4's ``triples_permuter`` to the einsums permute
#: right-hand side that realizes it: ``Xperm[a, b, c] = X[<spec>]``.
_PERMUTER_SPEC = {0: "abc", 1: "acb", 2: "bac", 3: "bca", 4: "cab", 5: "cba"}


def permuter_spec(i, j, k):
    """psi4's ``triples_permuter`` index selection, as a permute spec.

    An amplitude block lives in the ``i <= j <= k`` ordering of its own
    triplet; a term that reads it needs it in the ordering the READING
    triplet's ``a, b, c`` correspond to. Passing that ordering here gives the
    spec that rearranges the block into it.

    The comparison chain is psi4's, character for character, including its
    behaviour on repeated labels - ``iij`` and ``ijj`` triplets exist and the
    ties decide which branch they take. The two cyclic orderings are the ones
    psi4's ``reverse`` flag exists to swap: asking for ``(1, 2, 0)`` selects
    ``perm_idx`` 4 and ``(2, 0, 1)`` selects 3, because permuting a tensor and
    permuting the request run in opposite directions for a 3-cycle.
    """
    if i <= j and j <= k and i <= k:
        idx = 0
    elif i <= k and k <= j and i <= j:
        idx = 1
    elif j <= i and i <= k and j <= k:
        idx = 2
    elif j <= k and k <= i and j <= i:
        idx = 3
    elif k <= i and i <= j and k <= j:
        idx = 4
    else:
        idx = 5
    return _PERMUTER_SPEC[idx]


#: How each permute spec is realized as three plain GEMMs: ``(odd, family)``.
#:
#: Write the rotated block as ``U[p1, p2, p3] = sum S[p1, a1] S[p2, a2]
#: S[p3, a3] A[a1, a2, a3]``, with ``A`` the partner's amplitudes and ``S`` the
#: triplet-to-triplet overlap. Reading ``A`` through a spec permutes which
#: output slot each of ``A``'s axes lands in, so a coupling wants
#: ``R[x, y, z] += w * U[...]`` with the three arguments in a spec-dependent
#: order: ``abc`` wants ``U[x, y, z]``, ``bca`` wants ``U[y, z, x]``, and so on.
#:
#: A chain of three GEMMs can produce three of the six orders and no more, and
#: which three is decided by geometry rather than by choice. Each stage has to
#: contract an axis that is at one END of the merged matrix view, so the chain
#: peels axes off in storage order, and the only freedom left is whether the
#: last GEMM writes its new index as the FASTEST axis or the slowest. That gives
#: the three cyclic orders; the three odd ones are unreachable, because no
#: single leading dimension can describe a transposed destination.
#:
#: So the odd half is bought with one transposed copy of the partner's block,
#: ``A'[a1, a2, a3] = A[a1, a3, a2]``, which turns ``U'`` into ``U`` with its
#: last two arguments swapped and every odd spec into a cyclic read of ``A'``.
#: One extra rank-3 block per triplet pays for it, and it is the SAME block the
#: unbatched form needed for its third rotation's output, so the phase's memory
#: does not move.
#:
#: ``f1`` and ``f2`` differ only in the last GEMM and share the first two; the
#: reverse chain differs in all three. The emitter still groups by family rather
#: than by direction, because a batch's transpose flags are shared by the whole
#: call and a slot holding two families would split every stage in two.
_CHAIN = {"abc": (False, "f1"), "bca": (False, "f2"), "cab": (False, "rev"),
          "acb": (True, "f1"), "bac": (True, "f2"), "cba": (True, "rev")}

#: Which of the two chain directions a family walks the axes in.
_DIRECTION = {"f1": "fwd", "f2": "fwd", "rev": "rev"}

#: ``(trans_a, trans_b)`` for the first two stages, by direction. The forward
#: chain peels ``a1, a2, a3`` off the front of the block and the reverse chain
#: peels them off the back, which is the only way to reach the third cyclic
#: order.
_ROTATE_FLAGS = {"fwd": (True, True), "rev": (False, True)}

#: ``(trans_a, trans_b)`` for the accumulating stage, by family. ``f1`` writes
#: the new index as the slowest axis of ``R`` and ``f2`` as the fastest; that
#: single choice is the whole difference between them.
_ACCUMULATE_FLAGS = {"f1": (True, True), "f2": (False, False), "rev": (False, True)}

#: The order couplings are sorted into within a triplet. The emitter walks the
#: families in this order too, so the sort is what fixes the order the couplings
#: accumulate into ``R`` in, and with it the arithmetic.
_FAMILY_ORDER = {"f1": 0, "f2": 1, "rev": 2}


def chain_of(spec):
    """``(odd, family)`` for a permute spec: how to realize it as GEMMs."""
    return _CHAIN[spec]


def rotation_stages(family, src, Sv, Sw, b1, b2, slow, fast, nt, npr):
    """One coupling's three GEMMs, as ``(a, b, c)`` operand triples.

    ``src`` is the partner's amplitude block, or its transposed copy for an odd
    spec; ``Sv`` the triplet-to-triplet overlap and ``Sw`` the same overlap
    scaled by the coupling's weight; ``b1`` and ``b2`` the triplet's two flat
    scratch buffers; ``slow`` and ``fast`` the two matrix views of the
    destination ``R``. All of them are rank-2 views, because a grouped call
    wants one operand kind per slot, and every one of them is built by the
    caller once rather than per pass.

    Each stage contracts an axis that sits at one END of a merged matrix view,
    which is what makes it a plain GEMM: the forward chain peels ``a1, a2, a3``
    off the front of the block and writes each new index as the trailing axis;
    the reverse chain peels them off the back and writes each new index as the
    leading one. The flags belong to :data:`_ROTATE_FLAGS` and
    :data:`_ACCUMULATE_FLAGS`, keyed by direction and family, and the last stage
    accumulates straight into ``R``.
    """
    if family == "rev":
        return ((Sv, src.reshape_view([npr * npr, npr]),
                 b1[:nt * npr * npr].reshape_view([nt, npr * npr])),
                (Sv, b1[:nt * npr * npr].reshape_view([nt * npr, npr]),
                 b2[:nt * nt * npr].reshape_view([nt, nt * npr])),
                (b2[:nt * nt * npr].reshape_view([nt * nt, npr]), Sw, slow))
    merged = b2[:nt * nt * npr].reshape_view([npr, nt * nt])
    return ((src.reshape_view([npr, npr * npr]), Sv,
             b1[:nt * npr * npr].reshape_view([npr * npr, nt])),
            (b1[:nt * npr * npr].reshape_view([npr, npr * nt]), Sv,
             b2[:nt * nt * npr].reshape_view([npr * nt, nt])),
            (merged, Sw, slow) if family == "f1" else (Sw, merged, fast))


#: Cutoff on the plan's mean TNO count above which the ungated wide replay
#: loses to the per-block gated one once more than one thread is in play.
#:
#: Calibrated from exactly two measured geometries and nothing finer than
#: that: water chain n=6, about 22 TNOs per triplet, where the wide graph
#: wins at both 1 and 10 threads; and ethanol/cc-pVTZ, about 52 TNOs, where
#: the wide graph wins serially (109 s -> 90.5 s) but loses to the gated
#: graph at 10 threads on that row (33.5 s -> 41.4 s). 32 is the midpoint of
#: the two, not a measured knee, and it is pending re-calibration once a
#: benchmark run brackets the actual crossover.
_WIDE_MAX_TNO = 32.0


def prefer_wide_replay(mean_tno, max_tno, threads, cutoff=None):
    """Whether the ungated wide-batch residual should replay over the gated one.

    The two graphs are bit-identical whenever the wide one is eligible at all
    (every triplet's gate open), so this is purely a choice about what a pass
    COSTS, made once per solve rather than per pass.

    Unconditionally wide at one thread: the wide graph's fewer dispatches are
    a clear win on both geometries measured so far, and there is no
    executor-level parallelism across triplets for the gated graph to win on
    instead - the two replays cost the same shape of work either way.

    Above one thread, wide only when the triplet blocks are small enough that
    the grouped batch's own internal threading still beats the executor
    spreading whole triplets across cores; the plan's mean TNO count is the
    proxy for that, compared against ``cutoff``.

    ``max_tno`` is accepted alongside the mean for a future re-calibration
    that wants the more conservative bound instead of, or in addition to, the
    mean; the rule below reads only the mean today.

    ``cutoff`` overrides :data:`_WIDE_MAX_TNO`. Passing something at or above
    the widest plan this solver will ever see forces wide whenever more than
    one thread is in play; passing something below the narrowest forces
    gated. Neither override reaches the one-thread branch, since both graphs
    give the same answer there and the substitution is only worth measuring
    where the thread count makes it move.
    """
    del max_tno
    if threads <= 1:
        return True
    return mean_tno <= (_WIDE_MAX_TNO if cutoff is None else cutoff)


class LCCSDT:
    """Iterate Eq. 111/112 to convergence over a prepared triplet list.

    Constructed from a :class:`~dlpno.triples.DLPNOCCSDT` and the
    :class:`~dlpno.lccsd_t0.LCCSDT0` that produced the semicanonical starting
    point with ``retain=True``, so ``W``, ``V`` and the initial amplitudes are
    already resident.
    """

    def __init__(self, cc, t0, verbose=True, use_diis=False, use_executor=True,
                 block=1, wide_max_tno=None):
        self.cc = cc
        self.t0 = t0
        self.verbose = verbose
        #: Triplets per skip gate, and so per batch in a gated pass.
        #:
        #: Batching wants one call spanning many triplets and skipping wants a
        #: conditional per triplet, and a batch cannot sit inside a per-triplet
        #: conditional. This is the dial between them: triplets are gated in
        #: blocks of this many, a block runs if ANY of its members is still
        #: moving, and a batch spans the block.
        #:
        #: **One, by default, and the reason is that widening it buys nothing on
        #: the passes that cost anything.** Until the first triplet settles every
        #: gate is open, and a pass with every gate open is replayed through the
        #: ungated one-batch graph instead - the same arithmetic, a fortieth of
        #: the dispatches - so the block width only ever decides what a pass
        #: costs AFTER something has settled, which is where the skip has already
        #: made the pass cheap. At a block of one the phase reproduces the
        #: per-triplet gate exactly: the same triplets skipped, the same pass
        #: count, and the same energy to the last bit.
        #:
        #: Wider blocks are a real lever and the cost of pulling it is measured,
        #: on methanol/cc-pVDZ at ``NORMAL``, where the gated residual holds
        #: 8499 nodes at a block of one:
        #:
        #: =====  ============  ======  =======================
        #: block  gated nodes   passes  (T) energy drift
        #: =====  ============  ======  =======================
        #: 1      8499          9       0 (bit-identical)
        #: 4      2581          9       1.4e-9
        #: 8      1534          10      2.4e-9
        #: 16     980           10      2.9e-9
        #: =====  ============  ======  =======================
        #:
        #: The drift is not error introduced here. A settled triplet sharing a
        #: block with a moving one is recomputed rather than frozen, so a wider
        #: block converges each triplet further, and the whole column is bounded
        #: by 3.2e-9, which is where the same code lands with the skip switched
        #: off entirely. It is movement within the band ``t_cut_iter`` already
        #: spans, and it is why the recorded fixture tolerance is 1e-8. The extra
        #: pass has the same cause: a frozen triplet contributes no energy change,
        #: so freezing more of them meets a 1e-10 test one pass sooner.
        self.block = block
        #: Extrapolate the amplitudes with :func:`einsums.graph.diis`.
        #:
        #: Off by default, unlike the coupled-cluster solver, because this is
        #: the lever M8 step 3 exists to measure rather than a settled choice:
        #: the port takes 18 Jacobi passes where psi4 takes 6 Gauss-Seidel ones,
        #: and at ethanol/cc-pVTZ that pass count is nearly the whole remaining
        #: gap on this phase. Leaving it off keeps every recorded gate
        #: bit-identical while the comparison is taken; DIIS moves where the
        #: iteration STOPS, not where it converges to, so turning it on shifts
        #: the energies within convergence tolerance.
        self.use_diis = use_diis
        self._diis = None
        #: Replay the residual and step under an OpenMP executor over triplets.
        self.use_executor = use_executor
        #: Override for :func:`prefer_wide_replay`'s cutoff on the plan's
        #: mean TNO count.
        #:
        #: ``None``, by default, meaning "use :data:`_WIDE_MAX_TNO`". That
        #: default rule is provisional - it is calibrated from exactly two
        #: measured geometries - and this is the escape hatch for measuring a
        #: third without editing the constant: pass a specific cutoff, or
        #: force the choice outright for a benchmark that wants one side held
        #: fixed while the other is varied. Passing something at or above the
        #: widest plan this solver will ever see forces the wide replay
        #: whenever more than one thread is in play; passing something below
        #: the narrowest forces the gated one. Neither reaches the one-thread
        #: case, where the wide replay is unconditional because the two
        #: graphs are bit-identical and there is no executor parallelism
        #: across triplets for the gated graph to win on instead.
        self.wide_max_tno = wide_max_tno
        self.F_lmo_np = np.asarray(ten.view(cc.F_lmo))
        self.e_ijk = np.array(t0.e_ijk, dtype=float)
        self.e_t = 0.0
        self.n_iterations = 0
        self.n_wide_passes = 0
        self.t_plan = 0.0
        self.t_iterate = 0.0
        self.n_nodes = 0
        #: Of :attr:`n_nodes`, the ones that are grouped batched GEMMs.
        self.n_batched_nodes = 0
        #: Captured nodes per replayed graph, which is what a pass costs in
        #: dispatches. The gated residual counts the work inside its
        #: conditionals, since that is what a pass with every block active
        #: executes.
        self.nodes_by_phase = {}
        self._plan = None
        self._keep = []

    def _print(self, *args):
        if self.verbose:
            print(*args, flush=True)

    # -- planning ------------------------------------------------------------

    def plan(self):
        """Which triplets couple to which, as integer bookkeeping.

        Three families per triplet, one per position the summed LMO replaces:
        ``ijl`` couples through ``F[l, k]``, ``ilk`` through ``F[l, j]`` and
        ``ljk`` through ``F[l, i]``. A coupling is kept only when the partner
        triplet exists and the Fock element clears ``f_cut_t``, which is
        psi4's screen and the reason this is not quadratic in the triplet
        count.

        Each record carries the permute spec that brings the partner's block
        into this triplet's index order, derived once here rather than at every
        use, and with it the ``(odd, family)`` :data:`_CHAIN` resolves that spec
        to. The couplings are then sorted by family, which is what lets a batch
        drawn across triplets at one slot be of one family and so one call.
        """
        t0 = time.perf_counter()
        cc = self.cc
        naocc = cc.ref.naocc
        f_cut = float(cc.cut.f_cut_t)
        F = self.F_lmo_np

        records = []
        for ijk, (i, j, k) in enumerate(cc.ijk_to_i_j_k):
            if not cc.n_tno[ijk]:
                continue
            couplings = []
            for l in range(naocc):
                # (i, j, l) replacing k, weighted by F[l, k]; and so on. The
                # ordering passed to the permuter is the one this triplet's
                # a, b, c mean, NOT the partner's stored order.
                for skip, labels, weight_at in (
                        (k, (i, j, l), (l, k)),
                        (j, (i, l, k), (l, j)),
                        (i, (l, j, k), (l, i))):
                    if l == skip:
                        continue
                    partner = int(cc.i_j_k_to_ijk[labels])
                    if partner == -1 or not cc.n_tno[partner]:
                        continue
                    f = float(F[weight_at])
                    if abs(f) < f_cut:
                        continue
                    spec = permuter_spec(*labels)
                    odd, family = chain_of(spec)
                    couplings.append((partner, -f, spec, odd, family))
            couplings.sort(key=lambda c: (_FAMILY_ORDER[c[4]], c[0]))
            records.append(dict(
                ijk=ijk, labels=(i, j, k), nt=cc.n_tno[ijk],
                shift=float(F[i, i] + F[j, j] + F[k, k]),
                couplings=couplings,
                prefactor=0.5 if (i == j or j == k or i == k) else 1.0,
            ))
        self._plan = records
        self.t_plan = time.perf_counter() - t0
        return records

    def _overlaps(self):
        """``S(ijk, partner)`` for every coupling the plan records.

        Triplet against triplet this time, where (T0) needed pair against
        triplet. Same builder: it only wants a transform, a domain and a
        column count per entity.
        """
        cc = self.cc
        requests = {(r["ijk"], coupling[0])
                    for r in self._plan for coupling in r["couplings"]}
        keys = sorted(requests)
        blocks = build_overlaps(list(cc.X_tno), cc.S_pao,
                               list(cc.lmotriplet_to_paos), list(cc.n_tno), keys)
        return dict(zip(keys, blocks))

    # -- allocation ----------------------------------------------------------

    def _allocate(self, overlaps):
        """The residual, the denominator, and everything a batch needs ready.

        ``W``, ``V`` and the amplitudes come from the (T0) pass and are read
        in place. What is new is one residual block per triplet - the price of
        a Jacobi update - and one denominator, which is constant across
        iterations and so is built once rather than per pass.

        The rest of this is the batch's price of admission, and all of it is
        paid once here rather than per pass:

        * the transposed amplitude copies the odd specs read, one per triplet
          that some coupling reaches with an odd spec. It replaces the buffer the
          third rotation used to write into, which the batched chain does not
          need because its last GEMM accumulates straight into ``R``, so the
          rank-3 store per triplet is what it was;
        * the two rotation scratch buffers per triplet, sized as before, but
          hoisted out of the emitter because their VIEWS are now built here too;
        * a weight-scaled copy of each coupling's overlap. The weight has to be
          applied somewhere, a grouped call shares one prefactor across every
          member, and the accumulating GEMM's own ``alpha`` is therefore not
          available. Scaling one of the three overlap factors instead puts the
          weight inside the batch for free. It doubles the overlap store, which
          is a few percent of a phase whose memory is rank-3 blocks;
        * every rank-2 view the calls read or write. A view built under capture
          is recorded as a node, which is two per coupling per pass, and a view
          built here is a Python object and no node at all.
        """
        cc = self.cc
        self.R = {}
        self.D = {}
        self.Tprime = {}
        self._scaled = []
        g = cg.Graph("lccsd(t): denominators")
        with cg.capture(g):
            for r in self._plan:
                ijk, nt = r["ijk"], r["nt"]
                self.R[ijk] = ten.zeros(f"R ({ijk})", [nt, nt, nt])
                D = ten.zeros(f"D ({ijk})", [nt, nt, nt])
                e = cc.e_tno[ijk]
                la.outer_sum(D, [e, e, e], [1.0, 1.0, 1.0])
                la.shift(-r["shift"], D)
                self.D[ijk] = D
        _run_setup_graph(g)

        # Allocated before any entry is built, because an entry that reads an
        # odd source holds a view of it.
        for ijk in sorted({c[0] for r in self._plan for c in r["couplings"]
                           if c[3]}):
            nt = int(cc.n_tno[ijk])
            self.Tprime[ijk] = ten.empty(f"T' ({ijk})", [nt, nt, nt])
        for r in self._plan:
            self._entries(r, overlaps)

    def _entries(self, r, overlaps):
        """One triplet's couplings, as ready-to-batch ``(a, b, c)`` view triples.

        Three stages per coupling, each of them a plain GEMM over merged axes:
        peel one axis off the partner's block, then off the intermediate, then
        off the last intermediate straight into ``R``. Nothing here is captured;
        what is built is the operand lists a later
        :func:`~einsums.graph.grouped_batched_gemm` will name.

        The scratch is one set per TRIPLET, which is what keeps a batch legal
        rather than merely small. Entries inside one grouped call may run
        concurrently, so no two of them may share a buffer or a destination; a
        batch drawn across triplets at the same coupling SLOT has one entry per
        triplet by construction, so both hold. It also keeps the store where the
        unbatched form put it: sized to this triplet's widest partner and sliced
        per coupling, taking a contiguous prefix rather than a strided view,
        because the merged-axis reshapes the rotations rely on are only valid on
        contiguous storage.
        """
        cc = self.cc
        ijk, nt = r["ijk"], r["nt"]
        live = [c for c in r["couplings"]
                if overlaps.get((ijk, c[0])) is not None]
        r["entries"] = []
        if not live:
            return

        widest = max(int(cc.n_tno[c[0]]) for c in live)
        b1 = ten.empty(f"S t (1) [{ijk}]", [nt * widest * widest])
        b2 = ten.empty(f"S t (2) [{ijk}]", [nt * nt * widest])
        R = self.R[ijk]
        # R as a matrix, twice: rows (x, y) for a family whose last GEMM writes
        # its new index slowest, rows x for the one that writes it fastest.
        slow = R.reshape_view([nt * nt, nt])
        fast = R.reshape_view([nt, nt * nt])
        self._keep += [b1, b2, slow, fast]

        for partner, weight, _spec, odd, family in live:
            npr = int(cc.n_tno[partner])
            S = overlaps[(ijk, partner)]
            S_w = ten.empty(f"w S [{ijk} {partner}]", [nt, npr])
            # Eager, and constant for the whole solve: the overlaps do not
            # change with the amplitudes.
            la.axpby(weight, S, 0.0, S_w)
            self._scaled.append(S_w)
            Sv = S.reshape_view([nt, npr])
            Sw = S_w.reshape_view([nt, npr])
            src = self.Tprime[partner] if odd else self.t0.T[partner]
            stages = rotation_stages(family, src, Sv, Sw, b1, b2, slow, fast,
                                     nt, npr)
            r["entries"].append((family, stages))
            self._keep += [Sv, Sw]
            self._keep += [v for stage in stages for v in stage]

    # -- emission ------------------------------------------------------------

    def _emit_prime(self):
        """The transposed amplitude copies the odd specs read.

        Rebuilt every pass, for every triplet that is read oddly, and not gated
        by the skip: a copy is only correct while it matches the amplitudes it
        was taken from, and gating it on which triplets moved LAST pass is not
        the same question as which triplets moved. An ``n_tno^3`` permute per
        triplet against the ``n_tno^4`` rotations it feeds is not a cost worth
        being clever about.

        Its own graph, so the copies are complete before any rotation reads one.
        A partner is read by triplets in every block, so a copy written inside
        one block's conditional and read inside another's would want an edge
        between two conditionals in both directions.
        """
        for ijk, out in self.Tprime.items():
            einsums.permute("abc <- acb", out, self.t0.T[ijk])

    def _emit_prep(self, records):
        """``R = W + T D``, the part of Eq. 111 that is not a coupling."""
        for r in records:
            ijk = r["ijk"]
            R = self.R[ijk]
            la.axpby(1.0, self.t0.W[ijk], 0.0, R)
            einsums.einsum("abc <- abc ; abc", R, self.t0.T[ijk], self.D[ijk],
                           c_pf=1.0)

    def _emit_couplings(self, records):
        """Eq. 111's coupling sums, as three grouped GEMMs per coupling slot.

        Slot-major within a family, over the whole block. Slot-major is the one
        grouping that batches without needing more scratch: a batch holds at most
        one coupling per triplet, so the ``n_tno^3``-scale buffers stay one set
        per triplet. Within a family, so that a slot's three calls are three
        calls and not up to seven: the flags belong to the family, so a slot
        holding two of them would split every stage in two.

        Exactly three calls per slot, then, and they chain: the second reads what
        the first wrote and the third what the second did, so most of them land on
        a level of their own. That matters as much as the count, because a level
        holding one node runs unwrapped and its batch gets the whole thread team,
        where a level of many runs the nodes in parallel with each batch on one
        thread.

        Both orderings are the sorted order of :meth:`plan`, which is what makes
        this reproducible: the couplings of one triplet accumulate into its ``R``
        in exactly that order, and the emission order is what the hazard scan
        turns into the write-after-write chain that enforces it.
        """
        for family, (trans_a, trans_b) in _ACCUMULATE_FLAGS.items():
            lanes = [[stages for member, stages in r["entries"]
                      if member == family] for r in records]
            rotate = _ROTATE_FLAGS[_DIRECTION[family]]
            for slot in range(max((len(lane) for lane in lanes), default=0)):
                at_slot = [lane[slot] for lane in lanes if slot < len(lane)]
                self._batch([stages[0] for stages in at_slot], 0.0, *rotate)
                self._batch([stages[1] for stages in at_slot], 0.0, *rotate)
                self._batch([stages[2] for stages in at_slot], 1.0,
                            trans_a, trans_b)

    @staticmethod
    def _batch(entries, beta, trans_a, trans_b):
        """One grouped GEMM over an already-checked-independent set of entries."""
        if not entries:
            return
        cg.grouped_batched_gemm(1.0, [a for a, _b, _c in entries],
                                [b for _a, b, _c in entries], beta,
                                [c for _a, _b, c in entries],
                                trans_a=trans_a, trans_b=trans_b)

    def _emit_residual(self, records):
        """Eq. 111 over one block of triplets."""
        self._emit_prep(records)
        self._emit_couplings(records)

    def _emit_update(self, records=None):
        """Eq. 112: the Jacobi step, applied after every residual is known.

        Written as two operations - ``R <- -R/D``, then ``T += R`` - rather
        than the one fused ``T <- T - R/D`` this used to be, because
        :func:`einsums.graph.diis` extrapolates over (amplitude, STEP) pairs
        and cannot see a step that was never materialized.

        It costs nothing. The division is in place, and ``R`` has already been
        read for the convergence norm by the time this graph replays, so no
        tensor is added and no value is needed after it is overwritten. The
        arithmetic is unchanged and elementwise, so the energies do not move.
        """
        for r in (records if records is not None else self._plan):
            ijk = r["ijk"]
            la.direct_division(-1.0, self.R[ijk], self.D[ijk], 0.0,
                               self.R[ijk])
            la.axpby(1.0, self.R[ijk], 1.0, self.t0.T[ijk])

    def _emit_energy(self):
        """Eq. 53 again, at the current amplitudes.

        The same bracket the semicanonical pass evaluates, against amplitudes
        that now solve the coupled equation rather than the diagonal one. psi4
        re-derives it in ``compute_t_iteration_energy``; here the coefficients
        are the ones ``lccsd_t0`` already resolved.
        """
        from .lccsd_t0 import _ENERGY_TERMS

        for r in self._plan:
            ijk, nt = r["ijk"], r["nt"]
            bracket = self._bracket[ijk]
            for term, (coefficient, order) in enumerate(_ENERGY_TERMS):
                einsums.permute(f"abc <- {order}", bracket, self.t0.V[ijk],
                                c_pf=0.0 if term == 0 else 1.0,
                                a_pf=coefficient * r["prefactor"])
            la.dot(self._e_view[ijk], bracket, self.t0.T[ijk])

    # -- the solve -----------------------------------------------------------

    def _make_blocks(self, skipping):
        """The triplets, partitioned into the units a gate switches on and off.

        One block with everything in it when nothing is being skipped, which is
        the widest batch and the fewest calls, and blocks of :attr:`block`
        otherwise.

        Sorted by coupling count, longest first, so that the triplets in a block
        run out of couplings at about the same slot. A block emits as many slots
        as its LONGEST member has couplings of a family, so mixing a 24-coupling
        triplet with a 4-coupling one would emit twenty slots carrying a single
        entry; sorting makes the number of calls track the couplings rather than
        the widest member. The tie-break on ``ijk`` keeps the partition a
        function of the plan alone.
        """
        if not skipping or self.block is None or self.block >= len(self._plan):
            return [list(self._plan)]
        order = sorted(self._plan, key=lambda r: (-len(r["entries"]), r["ijk"]))
        return [order[at:at + self.block]
                for at in range(0, len(order), self.block)]

    def iterate(self):
        """Drive Eq. 111/112 to convergence and return the (T) energy."""
        cc = self.cc
        if self._plan is None:
            self.plan()
        if not self._plan:
            return 0.0

        overlaps = self._overlaps()
        self._allocate(overlaps)

        # The energy bracket and its per-triplet scalar, allocated once.
        self._bracket, self._e_view = {}, {}
        e_all = ten.zeros("e (triplet)", [len(self._plan)])
        for slot, r in enumerate(self._plan):
            nt = r["nt"]
            self._bracket[r["ijk"]] = ten.zeros("bracket", [nt, nt, nt])
            self._e_view[r["ijk"]] = e_all[slot:slot + 1]
            self._keep.append(self._e_view[r["ijk"]])

        # The energy is captured whole either way. psi4 skips only the residual
        # and the amplitude update - `compute_t_iteration_energy` runs over
        # every triplet each pass - and it is right to: the energy is an n^3
        # contraction against the n^4 rotations the skip is there to avoid.
        g_energy = cg.Graph("lccsd(t): Eq. 53 energy")
        with cg.capture(g_energy):
            self._emit_energy()
        self.n_nodes += g_energy.num_nodes()
        self.nodes_by_phase["energy"] = g_energy.num_nodes()

        skipping = bool(cc.cut.t_skip_converged) and float(cc.cut.t_cut_iter) > 0.0
        self._active = {r["ijk"]: True for r in self._plan}
        blocks = self._make_blocks(skipping)

        # Whether a pass with every gate open should actually replay the wide
        # graph, decided once for the whole solve rather than per pass: see
        # :func:`prefer_wide_replay`. Threads below the executor rather than
        # from the runtime alone, because the wide graph's advantage over the
        # gated one at many threads is specifically the gated graph's
        # executor-level parallelism ACROSS triplets, and that does not exist
        # when `use_executor` is off - nothing here threads either way, so
        # there is nothing for the gated graph to win on and wide stays the
        # cheaper replay regardless of the runtime's thread ceiling.
        mean_tno = sum(r["nt"] for r in self._plan) / len(self._plan)
        max_tno = max(r["nt"] for r in self._plan)
        threads = _omp_max_threads() if self.use_executor else 1
        self._prefer_wide = prefer_wide_replay(mean_tno, max_tno, threads,
                                               self.wide_max_tno)

        # The transposed copies the odd specs read, one graph, always replayed.
        g_prime = None
        if self.Tprime:
            g_prime = cg.Graph("lccsd(t): T' for the odd couplings")
            with cg.capture(g_prime):
                self._emit_prime()
            self.n_nodes += g_prime.num_nodes()
            self.nodes_by_phase["T'"] = g_prime.num_nodes()

        # Both phases are ONE graph each, with the work of a BLOCK of triplets
        # gated behind a conditional when skipping is on. The obvious
        # alternative - a graph per triplet, driven from a Python loop - also
        # skips, and was what this did first, but it forfeits the thing that
        # matters more: a Python loop cannot be threaded (the OpenMP-built
        # OpenBLAS returns silently wrong numbers from caller-created threads),
        # so the phase could skip or thread but not both. One graph keeps the
        # executor's view of every triplet and lets it spread them over cores.
        #
        # A block rather than a triplet because a batch cannot sit inside a
        # per-triplet conditional; see :attr:`block`. The gate is the same
        # object for the residual and the step, so a block that recomputed a
        # settled member also steps it, and `self._active` is promoted to whole
        # blocks below before anything reads it.
        #
        # Residual and step stay SEPARATE graphs, and that separation is what
        # makes the threading legal rather than merely convenient. Inside the
        # residual every triplet reads its neighbours' amplitudes and writes
        # only its own R, so the triplets are independent; a step in the same
        # graph would write T while another triplet's residual was reading it.
        # Jacobi is what buys the parallelism, and only across a phase
        # boundary.
        def _gate(block):
            names = [r["ijk"] for r in block]
            return lambda: any(self._active[ijk] for ijk in names)

        g_residual = cg.Graph("lccsd(t): Eq. 111 residual")
        g_step = cg.Graph("lccsd(t): Eq. 112 step")
        g_every = None
        self.n_batched_nodes = 0
        gated, stepped = 0, 0
        if skipping:
            for index, block in enumerate(blocks):
                gate = _gate(block)
                branch, _ = g_residual.add_conditional(f"residual [{index}]", gate)
                with cg.capture(branch):
                    self._emit_residual(block)
                gated += branch.num_nodes()
                self.n_batched_nodes += branch.num_nodes() - 2 * len(block)
                branch, _ = g_step.add_conditional(f"step [{index}]", gate)
                with cg.capture(branch):
                    self._emit_update(block)
                stepped += branch.num_nodes()
            self.n_nodes += gated + stepped

            # The same residual a second time, ungated and batched over every
            # triplet at once, for the passes where every block is active.
            #
            # This is an equivalence rather than a heuristic, which is the only
            # reason it is worth a second capture. A pass in which every block
            # holds a moving triplet computes every triplet either way, in the
            # same order, with the same operands and prefactors; all that differs
            # is how many members each grouped call carries and therefore how
            # many calls there are. So the wide replay is bit-identical to the
            # gated one whenever it is chosen, and it is chosen exactly when the
            # gates would have let everything through - which is every pass until
            # the first triplet settles, and those are the expensive ones.
            #
            # It is also the only shape in which the batches thread. A level
            # holding one node runs unwrapped, so a grouped call gets the whole
            # team; inside a conditional under a level of many, it is one thread
            # of a parallel loop over blocks. The two replays cover both, and the
            # pass count picks between them.
            if len(blocks) > 1:
                g_every = cg.Graph("lccsd(t): Eq. 111 residual (every triplet)")
                with cg.capture(g_every):
                    self._emit_residual(self._plan)
                self.n_nodes += g_every.num_nodes()
                self.nodes_by_phase["residual (wide)"] = g_every.num_nodes()
        else:
            with cg.capture(g_residual):
                self._emit_residual(self._plan)
            with cg.capture(g_step):
                self._emit_update()
            self.n_batched_nodes = g_residual.num_nodes() - 2 * len(self._plan)
            self.nodes_by_phase["residual (wide)"] = g_residual.num_nodes()
        self.n_nodes += g_residual.num_nodes() + g_step.num_nodes()
        # What a pass costs in dispatches. A gated phase is charged with the work
        # inside its conditionals plus the gates themselves, since that is what a
        # pass with every gate open executes.
        if skipping:
            self.nodes_by_phase["residual (gated)"] = gated + g_residual.num_nodes()
        self.nodes_by_phase["step"] = stepped + g_step.num_nodes()

        # Spread the triplets over cores. This is the one solver phase in the
        # port that can take an executor today, and the reason is the scratch:
        # every triplet owns its rotation buffers, so nothing is shared for the
        # hazard scan to serialize. The coupled-cluster residual cannot, and
        # `lccsd.py::iterate` says why - its rank-3 scratch is one buffer
        # shared by every pair, so an executor there would order the chains it
        # was meant to overlap.
        #
        # Legality is Jacobi's, per the phase split above: within the residual
        # a triplet reads its neighbours' amplitudes and writes only its own R.
        if self.use_executor:
            g_residual.set_executor(cg.OpenMPExecutor())
            g_step.set_executor(cg.OpenMPExecutor())
            if g_prime is not None:
                g_prime.set_executor(cg.OpenMPExecutor())
            if g_every is not None:
                g_every.set_executor(cg.OpenMPExecutor())

        self._print(f"\n  ==> Local CCSD(T) <==\n")
        self._print(f"    plan:     {len(self._plan)} triplets, "
                    f"{sum(len(r['couplings']) for r in self._plan)} couplings, "
                    f"{self.n_nodes} captured nodes "
                    f"({self.n_batched_nodes} batched GEMM)"
                    + (f", {len(blocks)} gated block(s) of "
                       f"{min(self.block or len(self._plan), len(self._plan))}"
                       f" at t_cut_iter {cc.cut.t_cut_iter:.1e}"
                       if skipping else ""))
        self._print("    per pass: "
                    + ", ".join(f"{count} {phase}"
                                for phase, count in self.nodes_by_phase.items()))
        self._print(f"\n    {'iter':>4}  {'(T) Energy':>18} {'Delta E':>12} {'Max R':>10}")

        if self.use_diis:
            # One (amplitude, step) pair per triplet. The coupled-cluster
            # solver hands DIIS a handful of bucketed stores instead; there is
            # no padded triplet store to do that with here, so this is several
            # hundred small pairs and the extrapolation's own cost is part of
            # what the measurement has to justify.
            self._diis = cg.diis(
                [(self.t0.T[r["ijk"]], self.R[r["ijk"]]) for r in self._plan],
                k=cc.cut.diis_max_vecs)

        t0 = time.perf_counter()
        e_prev = float(np.sum(self.e_ijk))
        e_ijk_old = np.zeros_like(self.e_ijk)
        self.n_skipped = 0
        #: Passes replayed through the ungated one-batch residual.
        self.n_wide_passes = 0
        for iteration in range(cc.cut.maxiter + 1):
            # psi4's T_CUT_ITER screen, evaluated where psi4 evaluates it: on
            # how much a triplet's energy moved over the PREVIOUS pass. A
            # triplet whose contribution has stopped changing is not updated
            # again; its neighbours go on reading the amplitude it settled at.
            #
            # Unlike psi4 this costs no reproducibility. psi4 pairs the skip
            # with an in-place Gauss-Seidel update, so which iterate a term
            # sees depends on thread scheduling; here the skip is a
            # deterministic function of the previous pass's energies, so the
            # same input still gives the same replay twice.
            if skipping:
                moving = {r["ijk"] for r in self._plan
                          if abs(self.e_ijk[r["ijk"]] - e_ijk_old[r["ijk"]])
                          >= abs(e_ijk_old[r["ijk"]] * cc.cut.t_cut_iter)}
                # Promoted to whole blocks, because a block is what the gate
                # can switch off. A settled triplet whose block still holds a
                # moving one is recomputed rather than frozen, which is the
                # unskipped iteration's answer for it, and it has to be counted
                # as active everywhere below: it takes a step, its residual
                # enters the norm, and DIIS sees a real step vector for it.
                live = {r["ijk"] for block in blocks
                        if any(r["ijk"] in moving for r in block)
                        for r in block}
                active = [r for r in self._plan if r["ijk"] in live]
                self.n_skipped += len(self._plan) - len(active)
                # The gates the conditionals read. Set before the replay, so
                # one graph execution does exactly the active blocks.
                for ijk in self._active:
                    self._active[ijk] = ijk in live
                if g_prime is not None:
                    g_prime.execute()
                if (g_every is not None and self._prefer_wide
                        and len(live) == len(self._plan)):
                    self.n_wide_passes += 1
                    g_every.execute()
                else:
                    g_residual.execute()
                # Skipped triplets contribute nothing to the norm, exactly as
                # psi4's zero-initialized R_iajbkc_rms does.
                rms = max((ten.rms(self.R[r["ijk"]]) for r in active), default=0.0)
                g_step.execute()
                if self._diis is not None:
                    # A skipped triplet took no step, and DIIS has to be told
                    # so. Its R still holds the step from whichever pass last
                    # updated it, and extrapolating over that stale vector is
                    # not a small error: left in, the dimer goes from 11 passes
                    # to 29 and methanol to 48, both worse than taking neither
                    # lever. Zero is the true step for a triplet that did not
                    # move, and an n^3 fill against the n^4 pass it replaces is
                    # not a cost worth avoiding.
                    for ijk, on in self._active.items():
                        if not on:
                            ten.view(self.R[ijk])[...] = 0.0
            else:
                active = self._plan
                if g_prime is not None:
                    g_prime.execute()
                # One block, no gate: this graph IS the wide one.
                self.n_wide_passes += 1
                g_residual.execute()
                rms = max((ten.rms(self.R[r["ijk"]]) for r in self._plan),
                          default=0.0)
                g_step.execute()
            if self._diis is not None:
                self._diis.step()
            g_energy.execute()

            energies = ten.view(e_all)
            e_ijk_old = self.e_ijk.copy()
            for slot, r in enumerate(self._plan):
                self.e_ijk[r["ijk"]] = float(energies[slot])
            e_curr = float(np.sum(self.e_ijk))

            self._print(f"    {iteration:>4}  {e_curr:>18.12f} "
                        f"{e_curr - e_prev:>12.3e} {rms:>10.3e}")
            converged = (iteration > 0
                         and abs(e_curr - e_prev) < cc.cut.e_convergence
                         and rms < cc.cut.r_convergence)
            e_prev = e_curr
            if converged:
                self.t_iterate = time.perf_counter() - t0
                self.n_iterations = iteration + 1
                self.e_t = e_curr
                self._print(f"    replay:   {self.t_iterate:.3f} s over "
                            f"{self.n_iterations} iterations, "
                            f"{self.n_wide_passes} of them over every triplet")
                return e_curr
        raise RuntimeError("maximum DLPNO-(T) iterations exceeded")
