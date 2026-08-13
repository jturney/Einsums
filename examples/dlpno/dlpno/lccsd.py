#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The T1-dressed local CCSD iteration, from psi4's ``DLPNOCCSD::lccsd_iterations``.

Jiang et al., JCP 161, 082502 (2024). Every method below names the equation it
implements, because the equations are the specification and the code is only a
transcription of them.

**Every contraction is written as an explicit einsum index string or a named
GEMM.** That is not a style choice. The residual is thirty-odd contractions over
blocks that live in several different PNO bases, bridged by overlap matrices,
and half of them are built by re-blocking a tensor whose two virtual indices are
interchangeable in the integral but not in the term. psi4 expresses these as
sequences of ``doublet``/``triplet`` calls with transpose flags and reshapes,
where a swapped pair reads as a plausible integral and surfaces only as a wrong
correlation energy several equations downstream. Naming the indices makes that
class of error impossible to write.

**The iteration is captured, not eager.** Every quantity here has a fixed shape
and a fixed dependency pattern once sparsity is frozen; only values change from
iteration to iteration. So the whole iteration is recorded once into a set of
ComputeGraphs before the solver loop starts and replayed to convergence, and
Python appears in the per-iteration path only in the convergence predicate and
the DIIS call. Nothing is allocated per iteration and no index is resolved
twice.

**Plan, then emit - and the emission is batched.** Each phase is split in two.
``_plan_*`` derives the work list - which pairs, which neighbours, which overlap
bridges - as pure integer bookkeeping over the sparsity maps, once. ``_emit_*``
walks that list and records operations into whatever capture is ambient. The
split is what made the performance campaign an emitter change, and the campaign
has now happened: the first cut emitted one operation per record, and every GEMM
family is now a :func:`~einsums.graph.grouped_batched_gemm` - the lever that took
LMP2 from 754 dispatches an iteration to 13. See decision 3 of
``DESIGN-ccsd.md``, :class:`_Batch` for what may share a batch, and
:func:`_by_step` for how the neighbour walks are transposed so that an
accumulation chain still runs in its original order.

The grouping is NOT derived from the plan's shape classes, and does not need to
be: the grouped form sorts its members into uniform ``(m, n, k, lda, ldb, ldc)``
groups when it records the node, so one call per family stage covers every shape
the family has. The classes remain as the plan's own description of the work.

**What the GEMM batching left behind was the scalar half, and it was the larger
half.** With every GEMM family batched, the iteration was still dominated by
about seventeen hundred ``dot`` reductions into length-1 tensors and as many
scaled accumulations of those scalars into single matrix elements - a node each,
and no arithmetic in any of them to spread. Neither could join a batch: a dot
spelled as an ``m = n = 1`` GEMM reassociates the sum, and this port's gates are
bit-identity. :func:`~einsums.linalg.grouped_dot` and
:func:`~einsums.linalg.grouped_axpby` are the answer, and they are a different
kind of grouping from a batch: they run the entries in SEQUENCE inside one node,
each through the same kernel the single call used, so a family merges without
any numerical trade at all. See :class:`_DotRun` and :class:`_AxpbyRun`.

**Views are built before capture opens.** A view handed to a captured operation
is itself a node - the graph re-binds its dims, strides and pointer from the
parent at each replay - and the neighbour walks used to build five per record,
which was more nodes than the arithmetic they fed. :meth:`LCCSDSolver._allocate`
builds every one of them instead, where a view is eager and free.

**One graph an iteration.** The fourteen phases the correctness cut gave a graph
apiece are recorded back to back into a single graph, replayed once per
iteration by a host ``while``. The boundaries were the slice-versus-whole safety
line while the emitters were in flux; they were audited away once those settled,
and what replaced each of them is a hazard edge on the quantity the two phases
actually share. :meth:`LCCSDSolver.capture` states what the merge rests on and
:meth:`LCCSDSolver.phase_graph` is what a phase at a time still costs.

The amplitude step, the DIIS extrapolation, the antisymmetrization and the
energy stay in graphs of their own after it, and the first of those boundaries
is load-bearing rather than conservative: threading the residual is legal
because a pair reads its neighbours' amplitudes and writes only its own output,
which is Jacobi's property and holds only while nothing in the same graph
updates an amplitude. Going the rest of the way to an ``add_loop`` body with the
convergence test in the predicate would have to give that up, so it is not the
destination this stopped short of - it is a different trade.

**Where scratch is shared and where it is not.** Everything of order
``n_pno^2`` is allocated per pair, so each pair's chain is independent. The
rank-3 scratch that carries the auxiliary index - the dressed ``B~^Q_{ab}``
above all - cannot be: one buffer per pair is 15 MiB times the pair count and
the store would be larger than every integral in the calculation put together.
It is a POOL of a few buffers per role instead, sized to the largest pair and
handed out round-robin over the plan records of the phase emitting them, so
consecutive records land on different buffers. That is what makes those records
threadable: one buffer per role is correctly serialized by the hazard scan
rather than raced, and the serialization is exactly the parallelism the phase
wants. The pool is deliberately an EMITTER-level fix rather than a job for the
``ScratchPrivatization`` pass, which skips view-accessed tensors by design and
cannot see this idiom. See "M8 step 6" in ``DESIGN-ccsd.md``.

Each of the six roles belongs to exactly ONE phase, which is why the merge into
a single graph left the pool alone: a role's chains are the same chains they were
inside a phase graph, and no two phases contend for a buffer.

The pool changes no arithmetic at all and is measured not to: every hand-out is
overwritten in full before it is read, so at any pool depth and any thread count
the energies reproduce the single-buffer ones to the last bit. What the executor
then does to the last bit is a separate question, and ``use_executor`` below
answers it.

**The executor, and what one graph means for it.** One graph has one executor,
so the merge turned fourteen executor decisions into one:
:class:`einsums.graph.OpenMPExecutor` over the whole iteration, which spreads the
independent chains over cores with serial BLAS underneath - psi4's own
parallelism, which is over the pair list rather than inside a GEMM. The phases
that accumulate thousands of scalars into one shared occupied matrix through
``(1, 1)`` element views (Eq. 98, Eq. 94, Eq. 86) used to keep the serial
executor, and folding them in costs nothing and moves no bit: records writing
different elements are independent, and records writing the same element are
ordered by a write-after-write on it, so each element's sum keeps its term
order. Never drive any of this from Python threads: against the OpenMP-built
OpenBLAS conda resolves, a caller-created thread returns silently wrong numbers
(see ``base.py::_run``).
"""

import os
import time

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import tensors as ten

__all__ = ["LCCSDSolver"]

#: How many buffers each shared rank-3 role gets. About one per core, so a
#: phase's records can be in flight on every core at once, and capped because
#: the buffers are sized to the largest pair and the pool is what the store
#: costs: a handful of tens of MiB per buffer at production scale. Consecutive
#: records take consecutive buffers, so the pool only has to be as deep as the
#: width the executor can actually use.
_SCRATCH_POOL = min(os.cpu_count() or 1, 16)


class LCCSDSolver:
    """Solve the T1-dressed local coupled-cluster singles and doubles equations.

    Constructed from a :class:`~dlpno.ccsd.DLPNOCCSD` whose cascade has run, so
    the PNO bases, the overlaps and every integral block already exist and are
    fixed for the life of the solve.

    Args:
        cc: the calculation, after ``prescreen_pairs``.
        verbose: print the iteration table.
        use_diis: extrapolate the amplitudes with :func:`einsums.graph.diis`.
        clamp_singles: hold ``T1`` at zero by discarding its residual. Not a
            physical model - it is the bisection ``check_ccsd_defect.py`` uses
            to separate the T1-dressing terms from the pure doubles ones.
        use_executor: replay the iteration graph under an OpenMP executor. Off
            is the serial replay every phase used to get, and it is how the
            executor is bisected out of a result that moves.

            The two agree BIT FOR BIT at one, two and four threads and differ in
            the last bit of the correlation energy at ten, and the reason is not
            a race: it reproduces exactly across repeated runs. A node that runs
            inside an OpenMP team cannot thread its own contraction, and the two
            phases carrying the rank-3 scratch (Eq. 99-101 and Eq. 75-77) have
            einsums large enough that einsums threads them when they run on the
            calling thread, so the team changes their internal summation order.
            Bisecting the executor a phase at a time moved the bit for exactly
            those two and for no other, back when a phase was a graph;
            :meth:`phase_graph` is what that bisection costs now. The serial
            replay is no more canonical: by itself it moves over the same last
            bits between one thread and ten, which it did before this pool and
            this executor existed.

            The fold changed neither of those answers. Merging the fourteen
            phases is bit-identical to replaying them in sequence at one thread
            and, measured on the water dimer, at ten as well - the same
            amplitudes, residuals and energy to the last bit, executor on.
    """

    def __init__(self, cc, verbose=True, use_diis=True, clamp_singles=False,
                 use_executor=True):
        self.cc = cc
        self.verbose = verbose
        self.use_diis = use_diis
        self.clamp_singles = clamp_singles
        self.use_executor = use_executor

        self.B = cc.integral_blocks
        self.naocc = cc.ref.naocc
        self.F_lmo_np = np.asarray(ten.view(cc.F_lmo))

        self.e_lccsd = 0.0
        self.de_weak = 0.0
        self.n_iterations = 0
        self.t_plan = 0.0
        self.t_capture = 0.0
        self.t_iterate = 0.0

        #: Tensors and views created while emitting. A view handed to a captured
        #: op is held by the graph as a slot pointer, so one that dies at the end
        #: of the expression that made it leaves the replay reading freed memory.
        self._keep = []
        self._graphs = None
        self._plan = None
        self._diis = None

    # -- small accessors ---------------------------------------------------

    def _print(self, *args):
        if self.verbose:
            print(*args, flush=True)

    def diag(self, i):
        """The diagonal pair ``ii``, which always exists."""
        return int(self.cc.i_j_to_ij[i, i])

    def is_strong(self, ij):
        i, j = self.cc.ij_to_i_j[ij]
        return self.cc.i_j_to_ij_strong[i, j] != -1

    def _S(self, ij, mn):
        """``S(ij, mn)``, memoized.

        psi4's dispatcher answers most requests from a stored family and builds
        the rest on the fly. Memoizing here moves every on-the-fly build to
        capture time, so the replayed iteration never builds an overlap.
        """
        hit = self._S_cache.get((ij, mn))
        if hit is None:
            hit = self.cc.S_PNO(ij, mn)
            self._S_cache[ij, mn] = hit
        return hit

    def _pair(self, stores, ij):
        """The logical ``(n, n)`` einsums view of pair ``ij``'s block, memoized."""
        key = (id(stores), ij)
        hit = self._view_cache.get(key)
        if hit is None:
            n = self.cc.n_pno[ij]
            layout = self.cc.layout
            hit = stores[layout.bucket_of[ij]][:n, :n, layout.slot_of[ij]]
            self._view_cache[key] = hit
        return hit

    # -- views, all of them made before capture opens ------------------------
    #
    # A view handed to a captured operation is itself RECORDED: the graph holds
    # a ``view_rt`` node that re-binds the view's dims, strides and data pointer
    # from its parent at each replay. So a view built inside an emitter costs a
    # node exactly as a GEMM does, and the emitters below used to build five of
    # them per record of the residual's neighbour walks - more nodes than the
    # arithmetic they fed. Every one is now built by :meth:`_allocate`, before
    # any capture is open, where it is a plain eager view and free. The helpers
    # here are the memo tables that make that possible, and they are what the
    # emitters read.

    def _v(self, T):
        """``T`` as a VIEW, memoized; a no-op on something that is one already.

        :func:`~einsums.graph.grouped_batched_gemm`'s bindings take HOMOGENEOUS
        operand lists - every entry of a slot owning, or every entry a view -
        and the residual's families mix the two freely: a pair's amplitudes
        arrive as a slice of a padded store while its ``gamma`` is a tensor of
        its own. Coercing every operand through here makes the lists uniform
        without the caller having to know which it holds.
        """
        if isinstance(T, einsums.RuntimeTensorViewD):
            return T
        key = ("v", id(T))
        hit = self._view_cache.get(key)
        if hit is None:
            hit = T[tuple(slice(0, d) for d in ten.shape(T))]
            self._view_cache[key] = hit
            self._keep.append(hit)
        return hit

    def _sub(self, T, *extents):
        """The leading ``[:e0, :e1, ...]`` sub-block of ``T``, memoized.

        The per-pair scratch is sized to the widest partner its work list
        reaches and each record slices the part it needs, so this is the shape
        of every temporary in the walks below.
        """
        key = ("sub", id(T), extents)
        hit = self._view_cache.get(key)
        if hit is None:
            hit = T[tuple(slice(0, e) for e in extents)]
            self._view_cache[key] = hit
            self._keep.append(hit)
        return hit

    def _flat(self, T, shape_):
        """``T`` reshaped, memoized. The captured spelling of a reinterpretation."""
        key = ("flat", id(T), tuple(shape_))
        hit = self._view_cache.get(key)
        if hit is None:
            hit = T.reshape_view(list(shape_))
            self._view_cache[key] = hit
            self._keep.append(hit)
        return hit

    def _col(self, T, j=0):
        """Column ``j`` of a matrix as a rank-1 view, memoized.

        What an einsum writing a vector takes, where a GEMM takes the ``(n, 1)``
        matrix the column lives in.
        """
        key = ("col", id(T), j)
        hit = self._view_cache.get(key)
        if hit is None:
            hit = T[:, j]
            self._view_cache[key] = hit
            self._keep.append(hit)
        return hit

    def _sub_col(self, T, j):
        """Column ``j`` of a matrix as an ``(n, 1)`` MATRIX view, memoized.

        What a GEMM takes where :meth:`_col` gives what an einsum takes.
        """
        key = ("subcol", id(T), j)
        hit = self._view_cache.get(key)
        if hit is None:
            hit = T[:, j:j + 1]
            self._view_cache[key] = hit
            self._keep.append(hit)
        return hit

    def _mrow(self, T, i):
        """Row ``i`` of a matrix as a rank-1 view, memoized."""
        key = ("mrow", id(T), i)
        hit = self._view_cache.get(key)
        if hit is None:
            hit = T[i, :]
            self._view_cache[key] = hit
            self._keep.append(hit)
        return hit

    def _row(self, p, x):
        """The projection ``S(a_p, a_xx) t_x`` as a rank-1 view, memoized.

        The ``(1, n)`` form is what a GEMM contracts through ``trans_a``; this
        is the vector an einsum or a ``ger`` takes.
        """
        key = ("row", p, x)
        hit = self._view_cache.get(key)
        if hit is None:
            hit = self._proj_row[p, x][0, :]
            self._view_cache[key] = hit
            self._keep.append(hit)
        return hit

    def T2(self, ij):
        """Pair ``ij``'s doubles amplitudes as a numpy view, unpadded."""
        n = self.cc.n_pno[ij]
        return self.cc.pair_block(self.cc.T_all, ij)[:n, :n]

    def Tt2(self, ij):
        """Pair ``ij``'s antisymmetrized amplitudes ``2T - T^T``, as numpy."""
        n = self.cc.n_pno[ij]
        return self.cc.pair_block(self.cc.Tt_all, ij)[:n, :n]

    def _T2(self, ij):
        """Pair ``ij``'s doubles amplitudes as a captured view."""
        return self._pair(self.cc.T_all, ij)

    def _Tt2(self, ij):
        """Pair ``ij``'s antisymmetrized amplitudes as a captured view."""
        return self._pair(self.cc.Tt_all, ij)

    def _K2(self, ij):
        """The exchange operator ``(i a | j b)`` in pair ``ij``'s PNO basis."""
        return self._pair(self.cc.K_all, ij)

    def _T1(self, i):
        """LMO ``i``'s singles amplitudes as a captured ``(n, 1)`` view."""
        hit = self._T1_view.get(i)
        if hit is None:
            n = self.cc.n_pno[self.diag(i)]
            hit = self.T1_all[:n, i:i + 1]
            self._T1_view[i] = hit
        return hit

    # -- Jiang Eq. 70, and every other projection of a singles amplitude ----

    def _proj(self, p, x):
        """``S(a_p, a_xx) t_x^{a_xx}``, as a ``(1, n_p)`` captured view.

        Every term that reads a singles amplitude reads it in some pair's PNO
        basis rather than in its own, so the projection is the single most
        common operation in the iteration. psi4 rebuilds it at each use; here it
        is built once per distinct ``(pair, LMO)`` request and shared.

        Most requests are for an LMO in the pair's own neighbour list, which is
        exactly Jiang Eq. 70's ``T_{n_ij}^{a_ij}``, and those land as rows of
        that pair's ``T_n_ij`` block so the block also exists whole for the terms
        that contract it as a matrix. The rest - a pair against an LMO outside
        its neighbour list, which Eq. 83c and Eq. 88's second term both ask for -
        get a standalone row. Both are the same GEMM against the same overlap,
        so the emitter does not distinguish them.

        Returned as ``(1, n_p)`` rather than as a vector because both forms are
        wanted: ``[0, :]`` is the rank-1 operand ``ger`` and the einsums take,
        and the ``(1, n_p)`` form is what a GEMM contracts through ``trans_b``.
        """
        return self._proj_row[p, x]

    def _plan_projections(self):
        """Which singles projections the iteration reads, and where each lives.

        Walks every term that reads one and records the ``(pair, LMO)`` request.
        Deduplicating them here is what turns psi4's repeated
        ``S_PNO(...) * T_ia`` products into one GEMM apiece.
        """
        cc = self._cc_maps()

        # Eq. 70 proper: every pair against every LMO in its neighbour list.
        # EVERY row is registered, including those whose LMO has no PNOs of its
        # own - the block has a row for each neighbour either way, and the terms
        # that contract it as a matrix read the whole thing. A row with nothing
        # to project stays at the zero it was allocated with, which is what psi4
        # leaves there and what those terms want.
        rows = []
        for ij in cc["live"]:
            for n_ij, n in enumerate(cc["lmos"][ij]):
                rows.append((ij, n_ij, n, self.diag(n)))
        in_rows = {(ij, n) for ij, _, n, _ in rows}

        # A pair's own two LMOs are always in its neighbour list, and several
        # terms index the row by position rather than searching for it, so a
        # missing one would be read as numpy's -1 - the last row - rather than
        # raise. Stated here once instead of guarded at four call sites.
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            for x in (i, j):
                assert (ij, x) in in_rows, (
                    f"LMO {x} is outside its own pair {ij}'s neighbour list")

        # Everything else that projects a singles amplitude into a pair: a pair
        # against an LMO that is NOT one of its neighbours. Eq. 83c, Eq. 84c and
        # Eq. 88's second term each ask for one.
        requests = set()

        def want(p, x):
            if p != -1 and cc["n_pno"][p] and (p, x) not in in_rows:
                requests.add((p, x))

        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            for l in cc["lmos"][ij]:
                want(int(cc["i_j_to_ij"][i, l]), j)   # Eq. 83c, labelled by ki
                want(int(cc["i_j_to_ij"][l, j]), i)   # Eq. 84c, labelled by ik
                want(self.diag(i), l)                 # Eq. 88's second term

        extras = sorted(requests)
        self._n_proj_rows = len(rows)
        self._n_proj_extra = len(extras)
        # What actually gets a GEMM: a projection of an amplitude that exists.
        emit = [(ij, n, nn) for ij, _, n, nn in rows if cc["n_pno"][nn]]
        emit += [(p, x, self.diag(x)) for p, x in extras
                 if cc["n_pno"][self.diag(x)]]
        return rows, emit

    def _emit_projections(self):
        """Record Eq. 70 and its relatives: the whole family in ONE batch.

        ``dest`` is a ``(1, n_p)`` row, so the product is written transposed:
        ``t^T S^T`` rather than ``S t``. That is not a detour - the destination
        is a row of a column-major block either way, and writing it directly is
        cheaper than producing a column and permuting it.

        Every projection has its own destination row and reads only amplitudes
        and overlaps, so the family is one grouped batch with no stepping at all:
        a phase that was one GEMM per distinct ``(pair, LMO)`` request is one
        node.
        """
        _emit_all(self._bat["projections"])

    # -- Jiang Eq. 91-92 ---------------------------------------------------

    def _plan_t1_ints(self):
        """The pairs whose dressed three-index factors the residual reads.

        Strong pairs only: Eq. 75 and Eq. 82 are the only consumers and neither
        runs over a weak pair.
        """
        cc = self._cc_maps()
        out = []
        for ij in cc["live"]:
            if not self.is_strong(ij):
                continue
            i, _ = cc["ij_to_i_j"][ij]
            out.append((ij, int(cc["ij_to_ji"][ij]), i))
        return out

    def _emit_t1_ints(self):
        """The T1-dressed three-index factors.

        Eq. 91:  ``B~^Q_{ki} = B^Q_{ki} + B^Q_{ka} t_i^a``
        Eq. 92:  ``B~^Q_{ai} = B^Q_{ai} - t_k^a B~^Q_{ki} + B^Q_{ab} t_i^b``

        Dressing the integrals rather than the amplitudes is what makes this
        formulation worth having: the singles are folded into the factors once
        per iteration, and every term downstream then reads a plain integral.

        Emitted as three passes over the pair list rather than one, so that
        Eq. 92's ``- t_k^a B~^Q_{ki}`` - the only GEMM here - is one batch over
        every pair instead of one dispatch each. Splitting the walk changes no
        order that matters: each pass writes only its own record's factors, and
        the two einsums still read the factor the pass before them left.

        The undressed starts are the same statement one step further: both
        families are one run, ahead of every einsum that dresses them.
        """
        cc = self.cc
        # Eq. 91, and Eq. 92's undressed start.
        self._run["t1_ints_init"].emit()
        for ij, _ji, i in self._plan.t1_ints:
            einsums.einsum("qk <- qka ; a", self.i_Qk_t1[ij],
                           self.B.qma(cc.ij_to_ji, ij),
                           self._row(ij, i), c_pf=1.0)

        # Eq. 92's occupied term, (Q,k)(k,a), for every pair at once.
        _emit_all(self._bat["t1_ints"])

        # Eq. 92's virtual term.
        for ij, _ji, i in self._plan.t1_ints:
            einsums.einsum("qa <- qab ; b", self.i_Qa_t1[ij],
                           self.B.qab(cc.ij_to_ji, ij), self._row(ij, i),
                           c_pf=1.0)

    # -- Jiang Eq. 94-101 --------------------------------------------------

    def _plan_t1_fock(self):
        """The three work lists the dressed Fock matrices need.

        psi4 dresses in two passes, the first over CONTRACTED indices (the
        ``bar`` quantities, Eq. 98-101) and the second over free ones (the
        ``tilde`` quantities, Eq. 94-97). The pair sets differ between them and
        between the terms within them, which is the whole content of this
        method: Eq. 98 runs over ALL pairs because the occupied block picks up a
        contribution from every one of them, weak pairs included, while Eq. 99
        to 101 run over strong upper-triangle pairs only.
        """
        cc = self._cc_maps()
        occ = [(ij, *cc["ij_to_i_j"][ij], int(cc["ij_to_ji"][ij]))
               for ij in cc["live"]]

        bar = []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            if i > j or not self.is_strong(ij):
                continue
            bar.append((ij, i, j))

        # Eq. 94 wants Fkc_bar on the diagonal pair jj and the row of it
        # belonging to i, so i has to be one of jj's neighbours. It always is -
        # pair ij exists, so i and j interact - but reading a missing slot as
        # numpy's -1 would silently take the last row rather than fail.
        kj = []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            jj = self.diag(j)
            if cc["n_pno"][jj] == 0 or not self.is_strong(jj):
                continue
            i_jj = int(cc["lmos_dense"][jj, i])
            assert i_jj != -1, (
                f"LMO {i} is outside diagonal pair {j}{j}'s neighbour list, so "
                f"Eq. 94 has no F~_kc row to read for pair ({i},{j})")
            kj.append((ij, i, j, jj, i_jj))

        # Eq. 95, over ALL pairs: Fkc_bar is not built over weak pairs, so psi4
        # builds this one separately and from L_iajb.
        kc = []
        for ij in cc["live"]:
            i, _ = cc["ij_to_i_j"][ij]
            for k in cc["lmos"][ij]:
                ik = int(cc["i_j_to_ij"][i, k])
                kk = self.diag(k)
                if ik == -1 or not cc["n_pno"][ik] or not cc["n_pno"][kk]:
                    continue
                kc.append((ij, ik, k))
        return occ, bar, kj, kc

    def _emit_t1_fock_bar(self):
        """Eq. 98-101, the quantities dressed over contracted indices.

        Two intermediates carry the whole dressing. ``gamma_Q = B^Q_{me} t_m^e``
        is the Coulomb-like half of all three, one scalar per auxiliary
        function. The exchange-like half is a three-factor contraction, and
        WHICH index is contracted first decides how big the intermediate is:
        folding the amplitude into the occupied factor gives ``(Q, k, m)`` for
        Eq. 99 and ``(Q, a, m)`` for Eq. 101, both of which carry an occupied
        axis, where the obvious order carries a second virtual one and is a
        factor of ``n_pno / n_lmo`` larger and slower.
        """
        cc = self.cc
        # Eq. 101's start, the whole family in one run. Every term that
        # accumulates onto it is recorded below, so hoisting the copy out of the
        # walk reorders nothing a pair can see.
        self._run["fock_bar_init"].emit()
        for slot, (ij, i, j) in enumerate(self._plan.fock_bar):
            nq = ten.shape(self.B.i_Qk[ij])[0]
            nk, na = ten.shape(self.T_n[ij])
            T_n = self.T_n[ij]                                  # (k, e)
            Qma = self.B.qma(cc.ij_to_ji, ij)                   # (Q, k, a)
            Qab = self.B.qab(cc.ij_to_ji, ij)                   # (Q, a, b)

            gamma = self.gamma_Q[ij]
            einsums.einsum("q <- qme ; me", gamma, Qma, T_n)

            # Eq. 99, Coulomb-like then exchange-like.
            V = self._shared("V", [nq, nk, nk], slot)
            einsums.einsum("qkm <- qke ; me", V, Qma, T_n)
            Fkc = self.Fkc_bar[ij]
            einsums.einsum("kc <- q ; qkc", Fkc, gamma, Qma, ab_pf=2.0)
            einsums.einsum("kc <- qkm ; qmc", Fkc, V, Qma, c_pf=1.0, ab_pf=-1.0)

            # Eq. 101. The orbital energies are the undressed diagonal and are
            # constant, so they are written once at allocation and copied by the
            # run above.
            Fab = self.Fab_bar[ij]
            einsums.einsum("ab <- q ; qab", Fab, gamma, Qab, c_pf=1.0, ab_pf=2.0)
            U = self._shared("U", [nq, na, nk], slot)
            einsums.einsum("qam <- qae ; me", U, Qab, T_n)
            einsums.einsum("ab <- qam ; qmb", Fab, U, Qma, c_pf=1.0, ab_pf=-1.0)

            # Eq. 100, only on the diagonal pairs, where the singles live.
            if i == j:
                y = self._shared("y", [nq, na], slot)
                einsums.einsum("qe <- me ; qm", y, T_n, self.B.i_Qk[ij])
                Fai = self._col(self.Fai_bar[i])
                einsums.einsum("a <- q ; qa", Fai, gamma, self.B.i_Qa[ij],
                               ab_pf=2.0)
                einsums.einsum("a <- qae ; qe", Fai, Qab, y,
                               c_pf=1.0, ab_pf=-1.0)

    def _emit_t1_fock_occupied(self):
        """Eq. 98: the dressed occupied Fock matrix, over ALL pairs.

        One element of ``F_bar`` per pair, so this is thousands of scalar
        reductions landing in thousands of distinct matrix elements. Each one is
        a dot product into a length-1 tensor and an accumulate into a ``(1, 1)``
        view of the matrix - the captured spelling of ``+=`` on a scalar, and
        the shape of every scalar accumulation below.

        This phase used to keep the serial executor for that reason - every
        record writes into the one occupied matrix, so there is no per-pair
        output to spread over cores. Under one graph it needs no exception:
        every pair's record lands in a DIFFERENT element of ``F_bar``, so the
        records are independent memory and the team may run them at once, and
        the two terms of one element are ordered against each other by a
        write-after-write on it. What it does not gain is much - a dot product
        into a scalar has no work in it to spread.

        **Which is exactly why the whole family is now three nodes.** There was
        never any parallelism to win here, only dispatch to lose, so the
        reductions go out as one :class:`_DotRun` and the accumulations as one
        :class:`_AxpbyRun`: the same kernels on the same operands, one after
        another inside a node, with each element's two terms still added in
        psi4's order because that is the order they were added to the run.
        """
        la.axpby(1.0, self.cc.F_lmo, 0.0, self.Fij_bar)
        self._run["fock_occ_dot"].emit()
        self._run["fock_occ_acc"].emit()

    def _emit_t1_fock_tilde(self):
        """Eq. 94-97, the quantities dressed over free indices.

        The one thing to be careful of is that Eq. 96's last term reads the
        *bar* occupied Fock matrix, not the fully dressed one.

        Eq. 94 accumulates one scalar per pair into a ``(1, 1)`` element view of
        the shared ``F~`` occupied matrix, exactly as Eq. 98 does, and that is
        what used to hold this phase on the serial executor while Eq. 95 to 97
        below it - which write per-pair quantities and would thread - were held
        with it. Under one graph the distinction is the hazard scan's to make
        rather than the phase's, and it makes it per element: one per pair, so
        they are independent.

        Eq. 95 is the phase's whole node count and it is now two batches and a
        step walk. Its first product depends on the pair ``ik`` alone and writes
        its own temporary, so the entire family goes out as one node; its second
        accumulates into pair ``ij``'s ``F~_{kc}``, so it is one node per step of
        the neighbour walk, which is as few as the accumulation allows.

        Eq. 94's reduction and accumulation are a run each, as Eq. 98's are, and
        Eq. 96's and Eq. 97's undressed starts share a third: both are beta = 0
        writes of a per-pair quantity and every term that dresses them is
        recorded after the run.
        """
        cc = self.cc

        # Eq. 94.
        la.axpby(1.0, self.Fij_bar, 0.0, self.Fkj)
        self._run["fock_kj_dot"].emit()
        self._run["fock_kj_acc"].emit()

        # Eq. 95.
        for ij in self._plan.live:
            la.scale(0.0, self.Fkc[ij])
        _emit_all(self._bat["fock_kc"])

        # Eq. 97 and Eq. 96.
        self._run["fock_tilde_init"].emit()
        for ij, i, j in self._plan.fock_bar:
            la.gemm(-1.0, self.T_n[ij], self.Fkc_bar[ij], 1.0, self.Fab[ij],
                    trans_a=True)
            if i != j:
                continue
            Fai = self.Fai[i]
            la.gemm(1.0, self.Fab_bar[ij], self._T1(i), 1.0, Fai)
            tmp = self.fock_ai_tmp[i]
            la.gemm(1.0, self.Fkc_bar[ij], self._T1(i), 0.0, tmp)
            la.gemm(-1.0, self.T_n[ij], tmp, 1.0, Fai, trans_a=True)
            # The last term reads the BAR occupied matrix, not Fkj.
            col = self.fock_ai_col[i]
            la.gather(col, self.Fij_bar, [list(cc.lmopair_to_lmos[ij]), [i]])
            la.gemm(-1.0, self.T_n[ij], col, 1.0, Fai, trans_a=True)

    # -- Jiang Eq. 82 -------------------------------------------------------

    def _plan_beta(self):
        """Eq. 82 runs over the strong pairs, which are the only ones Eq. 77 reads."""
        cc = self._cc_maps()
        return [(ij, int(cc["ij_to_ji"][ij])) for ij in cc["live"]
                if self.is_strong(ij)]

    def _emit_beta(self):
        """Eq. 82: ``beta_{ij}^{kl} = B~^Q_{ki} B~^Q_{lj} + t_{ij}^{cd} B^Q_{kc} B^Q_{ld}``.

        The first term is one batch over every strong pair, since each writes its
        own ``beta`` with no accumulation; the second is the pooled rank-3
        intermediate and stays a pair at a time.
        """
        cc = self.cc
        _emit_all(self._bat["beta"])
        for slot, (ij, ji) in enumerate(self._plan.beta):
            Qma = self.B.qma(cc.ij_to_ji, ij)                   # (Q, k, c)
            nq, nk, na = ten.shape(Qma)
            W = self._shared("bW", [nq, nk, na], slot)
            einsums.einsum("qkd <- qkc ; cd", W, Qma, self._T2(ij))
            einsums.einsum("kl <- qkd ; qld", self.beta[ij], W, Qma, c_pf=1.0)

    # -- Jiang Eq. 83 -------------------------------------------------------

    def _plan_gamma(self):
        """Eq. 83, labelled by the pair ``ki``, over ALL pairs.

        Four terms. 83a and 83b are per pair; 83c and 83d walk the pair's
        neighbour list and each needs a different set of bridges, so the plan
        records the resolved partner pairs rather than resolving them again on
        every replay.
        """
        cc = self._cc_maps()
        head, tail = [], []
        for ki in cc["live"]:
            k, i = cc["ij_to_i_j"][ki]
            ii = self.diag(i)
            head.append((ki, k, i, ii,
                         bool(cc["n_pno"][ii] and self.B.K_ivvv[ki] is not None)))
            for l in cc["lmos"][ki]:
                kl = int(cc["i_j_to_ij"][k, l])
                ll = self.diag(l)
                li = int(cc["i_j_to_ij"][l, i])
                c = (kl != -1 and cc["n_pno"][kl] and cc["n_pno"][ll]
                     and cc["n_pno"][ii])
                d = (kl != -1 and li != -1 and cc["n_pno"][kl] and cc["n_pno"][li])
                if c or d:
                    tail.append((ki, l, kl, ll, li, bool(c), bool(d)))
        return head, tail

    def _emit_gamma(self):
        """Eq. 83, four terms.

        83b is where the rank-3 ``K_ivvv`` earns itself: it is ``t_i^b (k b | a
        c)``, so ``b`` pairs with ``k`` and ``(a, c)`` are the free indices.
        Written against a flattened block the two would have to be separated by
        a reshape whose order depends on the storage layout.

        83c is the term the milestone's defect was in. ``b`` pairs with ``k``,
        which is the FIRST index of pair ``kl``'s exchange operator, hence the
        transpose; the untransposed form is ``(k c | l b) t_i^b``, wrong by
        exactly a swap of which occupied each virtual attaches to, and equal to
        it only on the diagonal pairs.

        **The neighbour walk is emitted step by step rather than record by
        record.** 83c is a three-product chain and 83d a four-product one, both
        accumulating into pair ``ki``'s ``gamma`` through that pair's own
        temporaries, so the records of one ``ki`` are a chain and the records of
        different ``ki`` are independent. :func:`_by_step` transposes the walk so
        that each product of each term is one batch over the pairs, which leaves
        every accumulation in its original order and takes the phase from about
        thirteen nodes a record to seven a step.

        83c's rank-1 update is written as the GEMM it is: a ``k = 1`` product of
        the projection column into the bridge row. At a prefactor of exactly
        ``-1`` that is the same arithmetic to the last bit - the scaling is
        exact - and unlike ``ger`` it batches.
        """
        # 83a: -t_l^a (k i | l c), every pair in one batch.
        _emit_all(self._bat["gamma_head"])
        # 83b: +t_i^b (k b | a c)
        for ki, k, i, ii, has_vvv in self._plan.gamma_head:
            if has_vvv:
                einsums.einsum("ac <- bac ; b", self.gamma[ki],
                               self.B.K_ivvv[ki], self._row(ki, i), c_pf=1.0)
        # 83c and 83d, the neighbour walk.
        _emit_all(self._bat["gamma_tail"])

    # -- Jiang Eq. 84 -------------------------------------------------------

    def _plan_delta(self):
        """Eq. 84, labelled by the pair ``ik``, over ALL pairs.

        psi4 calls this one "a doozy" and it is: four terms, two of them
        re-blockings of the three-external block in opposite directions.
        """
        cc = self._cc_maps()
        head, tail = [], []
        for ik in cc["live"]:
            i, k = cc["ij_to_i_j"][ik]
            ki = int(cc["ij_to_ji"][ik])
            ii = self.diag(i)
            head.append((ik, i, k, ki, ii,
                         bool(cc["n_pno"][ii] and self.B.K_ivvv[ki] is not None)))
            for l in cc["lmos"][ik]:
                ll = self.diag(l)
                lk = int(cc["i_j_to_ij"][l, k])
                il = int(cc["i_j_to_ij"][i, l])
                c = (lk != -1 and cc["n_pno"][lk] and cc["n_pno"][ll]
                     and cc["n_pno"][ii])
                d = (il != -1 and lk != -1 and cc["n_pno"][il] and cc["n_pno"][lk])
                if c or d:
                    tail.append((ik, l, ll, lk, il, bool(c), bool(d)))
        return head, tail

    def _emit_delta(self):
        """Eq. 84, four terms.

        84b's two halves differ only in which slot the singles amplitude
        contracts into, which is the whole reason ``K_ivvv`` is stored rank 3:
        against a flat block the two re-blockings run in opposite directions and
        neither is expressible without knowing the storage order.

        **84c's two products are taken transposed, and that is not cosmetic.**
        Written the way the equation reads, they multiply a ``(1, n)`` row into
        a matrix; Eq. 83c, which is the same term with the indices exchanged,
        multiplies a matrix into an ``(n, 1)`` column. A GEMM with one row is a
        degenerate case, and this one was worth a factor of two on the whole
        iteration: at a six-monomer chain on ten threads it took this phase from
        68 ms to 419 and the LCCSD row from 12.2 s to 24.0. Since the result is
        only ever consumed as the rank-1 operand of the outer product below,
        taking it as a column costs nothing.

        The interaction with the thread count is not reproduced by timing the
        two shapes in isolation, where the row form is slow but slow equally at
        one thread and at ten. Do not go looking for it in a microbenchmark; the
        phase-by-phase replay in the design notes is what localized it.

        Batched exactly as Eq. 83 is, and for the same reason: see
        :meth:`_emit_gamma` on the step walk and on 84c's rank-1 update becoming
        a ``k = 1`` GEMM.
        """
        # 84a: -t_l^a [2 (i l | k c) - (i k | l c)], whose integral combination
        # is constant and was formed once at allocation. One batch.
        _emit_all(self._bat["delta_head"])
        # 84b: +t_i^b [2 (k c | a b) - (k b | c a)]
        for ik, i, k, ki, ii, has_vvv in self._plan.delta_head:
            if has_vvv:
                K3 = self.B.K_ivvv[ki]                 # (e, a, f) for pair ki
                T_i = self._row(ik, i)
                einsums.einsum("ac <- cab ; b", self.delta[ik], K3, T_i,
                               c_pf=1.0, ab_pf=2.0)
                einsums.einsum("ac <- bac ; b", self.delta[ik], K3, T_i,
                               c_pf=1.0, ab_pf=-1.0)
        # 84c and 84d, the neighbour walk.
        _emit_all(self._bat["delta_tail"])

    # -- Jiang Eq. 86 -------------------------------------------------------

    def _plan_Fkj_double_tilde(self):
        """Eq. 86 walks every pair against its own neighbour list."""
        cc = self._cc_maps()
        out = []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            for l in cc["lmos"][ij]:
                li = int(cc["i_j_to_ij"][l, i])
                lj = int(cc["i_j_to_ij"][l, j])
                if li == -1 or lj == -1 or not cc["n_pno"][li] or not cc["n_pno"][lj]:
                    continue
                out.append((ij, i, j, l, li, lj))
        return out

    def _emit_Fkj_double_tilde(self):
        """Eq. 86: ``F''_{ij} = F~_{ij} + u_{lj}^{cd} (l c | i d)``.

        This is the one phase whose scalar accumulations genuinely collide: the
        two products are per pair and per neighbour, but they land as one more
        scalar in pair ``ij``'s element of the shared ``F''``, so a pair's
        neighbours all write the same ``(1, 1)`` view. The hazard scan orders
        them by a write-after-write on that element, in emission order, which is
        why the sum over neighbours keeps its term order and why folding this
        phase into the iteration graph moves no bit. The products in front of
        them are batched and independent, so what serializes is the accumulate
        alone rather than the whole record.

        The congruence transform's two products need no stepping at all: both
        write a temporary that belongs to the record rather than to the pair, so
        each is one batch over the whole walk.

        What was left per record - the contraction into a scalar and the
        accumulate - is now a run each, and this is the phase where the run's
        sequential semantics carry the argument rather than merely permit it.
        ``dot`` has no batched form; writing it as a ``m = n = 1`` GEMM would
        reassociate the sum and was never worth a last bit of the energy, and a
        run needs no such trade: it is the SAME ``dot``, entry after entry.
        The accumulate is the collision described above, and the run applies its
        entries in the order they were added, which is the order the unbatched
        emitter walked, so the sum over a pair's neighbours arrives unchanged.
        """
        la.axpby(1.0, self.Fkj, 0.0, self.Fkj2)
        _emit_all(self._bat["fkj2"])
        self._run["fkj2_dot"].emit()
        self._run["fkj2_acc"].emit()

    # -- Jiang Eq. 87a, 88, 89, 90 -----------------------------------------

    def _plan_R_ia(self):
        """The singles residual's three work lists.

        Eq. 88's A and Eq. 90's C are driven by the pair ``ik``, Eq. 89's B by
        the pair ``kl``, which is why psi4 walks the pair list twice rather than
        nesting.
        """
        cc = self._cc_maps()
        ik_list, a2_list, b_list = [], [], []
        for ik in cc["live"]:
            i, k = cc["ij_to_i_j"][ik]
            ki = int(cc["ij_to_ji"][ik])
            ii = self.diag(i)
            if not cc["n_pno"][ii]:
                continue
            ik_list.append((ik, i, k, ki, ii,
                            self.B.K_ivvv[ki] is not None,
                            bool(cc["n_pno"][ki])))
            for l in cc["lmos"][ik]:
                kl = int(cc["i_j_to_ij"][k, l])
                ll = self.diag(l)
                if kl == -1 or not cc["n_pno"][kl] or not cc["n_pno"][ll]:
                    continue
                a2_list.append((ik, i, ki, ii, l, kl))
        for kl in cc["live"]:
            members = []
            for i_kl, i in enumerate(cc["lmos"][kl]):
                ii = self.diag(i)
                if cc["n_pno"][ii]:
                    members.append((i_kl, i, ii))
            if members:
                b_list.append((kl, members))
        return ik_list, a2_list, b_list

    def _emit_R_ia(self):
        """The singles residual.

        Eq. 87a is the initialization from the dressed ``F_ai``; the three named
        contributions are A (Eq. 88), B (Eq. 89) and C (Eq. 90).

        All three accumulate into the singles residual of an LMO, so the LMO is
        the owner every batch here steps over, and the three families keep their
        order relative to each other because they are emitted in that order.
        Within a family the products that write a record's own temporary go out
        as one batch over the whole family; only the accumulation into ``R1``
        steps.
        """
        la.scale(0.0, self.R1_all)
        if self.clamp_singles:
            return
        self._run["r_ia_fai"].emit()

        # Eq. 88 A1: u_{ki}^{cd} (k c | d a). K_ivvv[ki] is (e, a, f) with e the
        # index paired with k, so (k c | d a) is K3[c, d, a].
        for ik, i, k, ki, ii, has_vvv, has_fkc in self._plan.r_ia_ik:
            if has_vvv:
                einsums.einsum("a <- cd ; cda", self._col(self.r_ia_a1[ik]),
                               self._Tt2(ki), self.B.K_ivvv[ki])
        # Eq. 90 C's product, then A1 and C into R1 step by step.
        _emit_all(self._bat["r_ia_ik"])

        # Eq. 88 A2: -u_{ki}^{cd} (k c | l d) t_l^a. Both congruence products
        # write the record's own temporary and go out as one batch each; the
        # reductions are one run, as Eq. 86's are and for the same reason.
        _emit_all(self._bat["r_ia_a2_pre"])
        self._run["r_ia_a2_dot"].emit()
        _emit_all(self._bat["r_ia_a2"])

        # Eq. 89 B: -u_{kl}^{ac} [(k i | l c) + t_i^b (k b | l c)].
        self._run["r_ia_b_init"].emit()
        _emit_all(self._bat["r_ia_b"])

    # -- Jiang Eq. 19, 75-81 -----------------------------------------------

    def _plan_R_iajb(self):
        """The doubles residual's four work lists.

        psi4 assembles the residual in two containers, and which container a
        term goes in is a property of the term rather than of the pair: ``R``
        collects the terms that are ALREADY symmetric under ``(ij, ab) <-> (ji,
        ba)`` and so are computed once for ``i <= j`` and mirrored, while ``Rn``
        collects the rest, which are symmetrized at the end by ``R[ij] += Rn[ij]
        + Rn[ji]^T``. Getting a term into the wrong container double-counts it.
        """
        cc = self._cc_maps()
        sym, sym_kl, non, non_g = [], [], [], []
        for ij in cc["live"]:
            i, j = cc["ij_to_i_j"][ij]
            if not self.is_strong(ij):
                continue
            if i <= j:
                sym.append((ij, i, j, int(cc["ij_to_ji"][ij])))
                lmos = cc["lmos"][ij]
                for k_ij, k in enumerate(lmos):
                    for l_ij, l in enumerate(lmos):
                        kl = int(cc["i_j_to_ij"][k, l])
                        if kl != -1 and cc["n_pno"][kl]:
                            sym_kl.append((ij, k_ij, l_ij, kl))
            # Eq. 78 (C) and Eq. 79 (D) share the neighbour walk but resolve
            # different partner pairs from it.
            cs, ds = [], []
            for k_ij, k in enumerate(cc["lmos"][ij]):
                ki = int(cc["i_j_to_ij"][k, i])
                kj = int(cc["i_j_to_ij"][k, j])
                if (ki != -1 and kj != -1 and cc["n_pno"][ki] and cc["n_pno"][kj]
                        and self.B.J_ikac[ij] is not None
                        and self.B.J_ikac[ij][k_ij] is not None):
                    cs.append((k_ij, ki, kj))
                ik = int(cc["i_j_to_ij"][i, k])
                jk = int(cc["i_j_to_ij"][j, k])
                if (ik != -1 and jk != -1 and cc["n_pno"][ik] and cc["n_pno"][jk]
                        and self.B.K_iakc[ij] is not None
                        and self.B.K_iakc[ij][k_ij] is not None
                        and self.B.J_ikac[ij][k_ij] is not None):
                    ds.append((k_ij, ik, jk))
            non.append((ij, i, j, cs, ds))
            gs = []
            for k in range(self.naocc):
                ik = int(cc["i_j_to_ij"][i, k])
                if ik != -1 and cc["n_pno"][ik]:
                    gs.append((k, ik))
            non_g.append((ij, i, j, gs))
        return sym, sym_kl, non, non_g

    def _plan_r_combine_perm(self, maps):
        """Per bucket, the slot the pair involution ``ij -> ji`` sends each slot to.

        Eq. 19's transposed half is ``R[ji] += Rn[ij]^T`` over the pair list, and
        that is a whole-store statement rather than a per-pair one because pair
        ``ji`` carries the same PNO count as ``ij`` - the PNO transform builds the
        upper triangle and mirrors it - so the two share a bucket and the
        involution restricted to a bucket permutes its slots. Returning that
        permutation is what lets :meth:`_emit_R_iajb_combine` spell the family as
        one accumulating scatter per bucket.

        EVERY live pair gets a slot here, weak ones included, and that rests on
        strength being a property of the unordered pair: a weak pair's ``Rn``
        block is never written, so its slot contributes exactly ``+0.0``, but
        only if ``ji`` is weak whenever ``ij`` is. Eq. 75-77 already relies on
        the same invariant when it mirrors ``R[ij]`` onto ``R[ji]`` without
        asking whether ``ji`` is strong, so it is checked here rather than
        assumed: ``is_strong`` reads a pair energy that each ordering computes
        independently, and a pair sitting on the ``T_CUT_PAIRS`` knife edge in
        one ordering and not the other would silently unfreeze a weak pair's
        amplitudes.
        """
        layout = self.cc.layout
        n_pno = maps["n_pno"]
        ij_to_ji = maps["ij_to_ji"]
        perms = []
        for b, members in enumerate(layout.bucket_members):
            perm = []
            for ij in members:
                ji = int(ij_to_ji[ij])
                if n_pno[ji] != n_pno[ij] or layout.bucket_of[ji] != b:
                    raise RuntimeError(
                        f"pair {ij} has {n_pno[ij]} PNOs in bucket {b} but its "
                        f"transpose {ji} has {n_pno[ji]} in bucket "
                        f"{layout.bucket_of[ji]}; Eq. 19 assumes they share one")
                if self.is_strong(ij) != self.is_strong(ji):
                    raise RuntimeError(
                        f"pair {ij} is {'strong' if self.is_strong(ij) else 'weak'} "
                        f"but its transpose {ji} is not; the residual assumes "
                        f"strength is a property of the unordered pair")
                perm.append(layout.slot_of[ji])
            perms.append(perm)
        return perms

    def _emit_R_iajb_zero(self):
        """Clear both residual containers.

        Four operations, and they used to be a graph of their own so that
        nothing could race them. Inside the iteration graph the same guarantee
        is a write-after-write: every later accumulation into ``R`` or ``Rn``
        goes through a slice of the store this clears whole, and the scan orders
        a slice write after a covering write of its parent.
        """
        for R, Rn in zip(self.R_all, self.Rn_all):
            la.scale(0.0, R)
            la.scale(0.0, Rn)

    def _emit_R_iajb_symmetric(self):
        """Eq. 75-77 and Eq. 80: the terms already symmetric in ``(ij, ab)``.

        All four are accumulated into one per-pair buffer and then added to
        ``R[ij]`` and, transposed, to ``R[ji]``, rather than added four times to
        each. Eq. 77 and Eq. 85 share the double walk over the pair's
        neighbours, and ``u_{kl} (k c | l d)^T`` in Eq. 85 depends only on
        ``kl``, so it is hoisted to one product per pair.

        **The double walk is the iteration's largest family and is emitted step
        by step.** Its five products per record - two congruence transforms, the
        rank-1 update that carries Eq. 77's ``beta`` prefactor, and Eq. 85's two -
        all run through pair ``ij``'s own two scratch buffers and accumulate into
        that pair's ``B`` and ``F''_{bc}``, so records of one pair are a chain and
        records of different pairs are independent. Transposing the walk with
        :func:`_by_step` makes each of the five one batch per step over the pairs.

        Eq. 77's prefactor is an element of ``beta`` rather than a constant, and
        the update is written as the ``k = 1`` GEMM it is: the flattened block
        against that one element. At ``alpha = 1`` it is bit for bit the ``ger``
        it replaces, and unlike ``ger`` it batches.
        """
        cc = self.cc
        # The Eq. 85 factor, shared by every pair whose neighbour walk reaches
        # kl. One batch: each pair writes its own.
        _emit_all(self._bat["E_kl"])

        # Eq. 75: the dressed exchange operator, one batch over the pairs.
        _emit_all(self._bat["r_sym_75"])

        # Eq. 85's F''_bc start. The copy of ``Qab`` below cannot join it - it
        # writes a slice of the POOLED rank-3 scratch, whose next hand-out has to
        # wait for this pair's einsums to read it - but F''_bc is that pair's own
        # buffer, and the double walk that fills it is recorded afterwards.
        self._run["r_sym_fbc"].emit()

        for slot, (ij, i, j, ji) in enumerate(self._plan.r_sym):
            acc = self.r_sym_acc[ij]
            Qma = self.B.qma(cc.ij_to_ji, ij)
            Qab = self.B.qab(cc.ij_to_ji, ij)
            nq, nk, na = ten.shape(Qma)

            # Eq. 76, with the Eq. 93 dressing B~^Q_{ab} = B^Q_{ab} - t_k^a B^Q_{kb}.
            Qab_t1 = self._shared("Qab_t1", [nq, na, na], slot)
            la.axpby(1.0, Qab, 0.0, Qab_t1)
            einsums.einsum("qab <- ka ; qkb", Qab_t1, self.T_n[ij], Qma,
                           c_pf=1.0, ab_pf=-1.0)
            W = self._shared("AW", [nq, na, na], slot)
            einsums.einsum("qad <- qac ; cd", W, Qab_t1, self._T2(ij))
            einsums.einsum("ab <- qad ; qbd", acc, W, Qab_t1, c_pf=1.0)

            # Eq. 77's accumulator, which the double walk below fills.
            la.scale(0.0, self.r_sym_B[ij])

        # Eq. 77 (B) and Eq. 85 (F_bc), the shared double walk.
        _emit_all(self._bat["r_sym_kl"])

        self._run["r_sym_B"].emit()
        # Eq. 80, already carrying its own permutation. Two batches, in the
        # order the two products accumulated in before.
        _emit_all(self._bat["r_sym_80"])
        # Eq. 19's mirrored half. The transposing views have to be built inside
        # the capture - ``permute_view`` is capture-only - so this run is built
        # here rather than by :meth:`_build_batches`, and it is two runs rather
        # than one because the operand lists a binding takes are homogeneous:
        # the untransposed half reads the accumulator itself and the mirrored
        # half reads a view of it. Splitting them reorders nothing, because the
        # two halves write disjoint slots: pair ``ij`` here always has
        # ``i <= j``, so ``ji`` is never itself a member unless ``i == j``, in
        # which case only the first half runs.
        direct, mirror = _AxpbyRun(), _AxpbyRun()
        for ij, i, j, ji in self._plan.r_sym:
            acc = self.r_sym_acc[ij]
            direct.add(1.0, acc, 1.0, self._pair(self.R_all, ij))
            if i != j:
                mirror.add(1.0, cg.permute_view(acc, [1, 0]), 1.0,
                           self._pair(self.R_all, ji))
        direct.emit()
        mirror.emit()

    def _emit_R_iajb_nonsymmetric(self):
        """Eq. 78, 79 and 81: the terms that carry their own permutation.

        All three walk a pair's neighbour list through that pair's own three
        scratch buffers and accumulate into that pair's ``C``, ``D`` or ``G``, so
        all three are emitted step by step over the pairs; see
        :meth:`_emit_gamma` for what that means and why it moves no bit.

        Eq. 78's and Eq. 79's ``total`` starts from a constant integral
        combination, and that initialization is one copy for the whole family
        rather than one per record, because the buffers are packed into a store
        laid out identically to a store holding the constants. See
        :meth:`_build_batches`.

        Eq. 81's prefactor is an element of ``F''`` rather than a constant, so its
        accumulation is a ``k = 1`` GEMM against a ``(1, 1)`` slice - the batchable
        spelling of "scalar times matrix", and at ``alpha = -1`` the same
        arithmetic as the ``ger`` it replaces.

        All three accumulators are cleared and then added into ``Rn`` a STORE at
        a time, on Eq. 19's reading of the same rule: every live pair has a slot,
        the padding is zero on both sides, and no sum runs across pairs.
        """
        for C, D in zip(self.C_non_all, self.D_non_all):
            la.scale(0.0, C)
            la.scale(0.0, D)

        # Eq. 78 (C) and Eq. 79 (D): the family's initialization, then one step
        # of each neighbour walk at a time as four batches.
        self._run["r_non_reset"].emit()
        _emit_all(self._bat["r_non"])

        # The three accumulations into ``Rn`` are THREE runs and not one, so that
        # each bucket still receives them in the order ``C``, ``C^T``, ``D``: a
        # run holds one entry per bucket, and the runs are recorded in that
        # order. Merging them would put every bucket's ``C`` before any
        # bucket's ``C^T``, which reorders the sum on each ``Rn``.
        c_direct, c_mirror, d_direct = _AxpbyRun(), _AxpbyRun(), _AxpbyRun()
        for C, D, Rn in zip(self.C_non_all, self.D_non_all, self.Rn_all):
            c_direct.add(0.5, C, 1.0, Rn)
            c_mirror.add(1.0, self._keepv(cg.permute_view(C, [1, 0, 2])),
                         1.0, Rn)
            d_direct.add(0.5, D, 1.0, Rn)
        c_direct.emit()
        c_mirror.emit()
        d_direct.emit()

        # Eq. 81 (G). Its store is packed rather than padded, so it clears in one
        # operation but lands in ``Rn`` a pair at a time.
        la.scale(0.0, self.G_store)
        _emit_all(self._bat["r_non_g"])
        self._run["r_non_g_into_Rn"].emit()

    def _emit_R_iajb_combine(self):
        """``R[ij] += Rn[ij] + Rn[ji]^T``, both halves per BUCKET.

        The untransposed half is one operation per bucket because both
        containers share a layout, weak pairs were never written into ``Rn``,
        and the padding of both is identically zero, so adding the whole store
        is the same arithmetic as adding pair by pair.

        The transposed half is the same statement composed with the pair
        involution ``ij -> ji``. Pair ``ji`` carries the same PNO count as ``ij``
        and therefore sits in the SAME bucket, so within a bucket the involution
        is a permutation of slots and the whole family is one accumulating
        scatter of a transposing view of the store into that permutation:
        ``R[.., .., perm[t]] += Rn[.., .., t]^T``, which is
        :func:`~einsums.linalg.scatter_add` because the destination is indexed
        and accumulated rather than overwritten. ``gather`` can permute and
        transpose in one pass (it takes an ``axes``) but it assigns, so it would
        need a whole extra store to land in; ``scatter_add`` accumulates
        straight onto ``R`` and needs no scratch at all.

        Bit for bit the same as the pair loop it replaces, and by construction
        rather than by measurement: every block addition is elementwise onto its
        own pair's block, so no sum is reassociated and no operand is touched
        twice. What the whole-store form adds is the padding, which is
        identically zero on both sides, and the weak slots, whose ``Rn`` block
        is never written. Both contribute exactly ``+0.0``.

        :meth:`_plan_r_combine_perm` derives the permutation and states the
        invariant it rests on.

        The unit the team spreads here is the bucket rather than the pair: each
        bucket's two accumulations onto its own ``R`` store are serialized
        against each other by the hazard scan and independent of every other
        bucket's, and they are full passes over the store, so there is bandwidth
        for a team to use.
        """
        perms = self._plan.r_combine_perm
        for b, (R, Rn) in enumerate(zip(self.R_all, self.Rn_all)):
            la.axpby(1.0, Rn, 1.0, R)
            perm = perms[b]
            if not perm:
                continue
            rows = list(range(self.cc.layout.bucket_dims[b]))
            la.scatter_add(R, self._keepv(cg.permute_view(Rn, [1, 0, 2])),
                           [rows, rows, perm])

    # -- Jiang Eq. 45 -------------------------------------------------------

    def _emit_energy(self):
        """Eq. 45: ``E = (t_{ij}^{ab} + t_i^a t_j^b) L_{ij}^{ab}``.

        The weak pairs' contribution is accumulated separately rather than
        skipped, because psi4 computes it here and then subtracts it: their
        doubles are frozen at the converged LMP2 values, but their SINGLES
        contribution is live, which is why the published ``DLPNO LMP2 WEAK PAIR
        ENERGY`` is not the value ``recompute_pnos`` printed earlier.

        Every pair adds into the SAME two scalars, so the accumulation is the
        one family here whose order is the energy itself. It is one run, and its
        entries are added in the order the pair walk visits them, so the sum is
        psi4's to the last bit - both sums, since a weak pair's two entries stay
        adjacent and neither scalar sees the other's.
        """
        la.scale(0.0, self.e_total)
        la.scale(0.0, self.e_weak)
        self._run["energy_tau"].emit()
        # The singles half, one batch: an outer product of two projections is a
        # k = 1 GEMM, and at alpha = 1 it is the ger it replaces to the last bit.
        _emit_all(self._bat["energy"])
        self._run["energy_dot"].emit()
        self._run["energy_acc"].emit()

    # -- Jiang Eq. 103-104 --------------------------------------------------

    def _emit_step(self):
        """The Jacobi update, both amplitude families, over whole stores.

        The step is materialized rather than fused into the update because it is
        DIIS's error vector and has to exist as its own tensor for
        :func:`einsums.graph.diis` to snapshot between replays. The padding
        stays exactly zero: the residual's padding is zero and the denominator's
        is one.
        """
        la.direct_division(-1.0, self.R1_all, self.D1_all, 0.0, self.step1_all)
        la.axpby(1.0, self.step1_all, 1.0, self.T1_all)
        for R, D, S, T in zip(self.R_all, self.D_all, self.step_all,
                              self.cc.T_all):
            la.direct_division(-1.0, R, D, 0.0, S)
            la.axpby(1.0, S, 1.0, T)

    def _emit_antisymmetrize(self):
        """``Tt = 2 T - T^T``, every pair in two operations per bucket."""
        for Tt, T in zip(self.cc.Tt_all, self.cc.T_all):
            einsums.permute("abp <- bap", Tt, T)
            la.axpby(2.0, T, -1.0, Tt)

    # -- planning ------------------------------------------------------------

    def _cc_maps(self):
        """The sparsity maps the plan reads, resolved once."""
        cc = self.cc
        return {
            "live": [ij for ij in range(cc.n_lmo_pairs) if cc.n_pno[ij]],
            "n_pno": cc.n_pno,
            "ij_to_i_j": cc.ij_to_i_j,
            "i_j_to_ij": cc.i_j_to_ij,
            "ij_to_ji": cc.ij_to_ji,
            "lmos": cc.lmopair_to_lmos,
            "lmos_dense": cc.lmopair_to_lmos_dense,
        }

    def plan(self):
        """Derive every work list, once, as pure integer bookkeeping.

        Not one floating-point array is touched here and nothing depends on the
        value of an integral: this is where the method's index logic lives, and
        separating it from the emission is what lets the emission be replaced
        wholesale by a batched one without re-deriving a single map.

        Each list is also classified by shape class, on the same key the LMP2
        planner uses - ``(bucket of ij, bucket of the partner)`` - extended by
        the neighbour count where a term's shape depends on it. The batched
        emitter does not read them, and the reason is that it turned out not to
        need them: ``grouped_batched_gemm`` derives its own groups from the
        operands' dims and leading dimensions when it records the node, so a
        family goes out as ONE call whatever shapes it holds. What the classes
        are still good for is the report - how many distinct shapes a term
        actually presents - and that is what ``_report`` prints them as.
        """
        t0 = time.perf_counter()
        maps = self._cc_maps()
        p = _Plan()
        p.live = maps["live"]
        p.projection_rows, p.projections = self._plan_projections()
        p.t1_ints = self._plan_t1_ints()
        (p.fock_occupied, p.fock_bar, p.fock_kj, p.fock_kc) = self._plan_t1_fock()
        p.beta = self._plan_beta()
        p.gamma_head, p.gamma_tail = self._plan_gamma()
        p.delta_head, p.delta_tail = self._plan_delta()
        p.fkj2 = self._plan_Fkj_double_tilde()
        p.r_ia_ik, p.r_ia_a2, p.r_ia_b = self._plan_R_ia()
        p.r_sym, p.r_sym_kl, p.r_non, p.r_non_g = self._plan_R_iajb()
        # Eq. 19 is emitted from the per-BUCKET slot permutation rather than
        # from the pair family, but the family stays: it is the statement of
        # which pairs the phase sums over, it is what the permutation is checked
        # against in ``test_lccsd.py``, and it is what a reader comparing this
        # against psi4's loop is looking for.
        p.r_combine = [(ij, *maps["ij_to_i_j"][ij], int(maps["ij_to_ji"][ij]))
                       for ij in maps["live"] if self.is_strong(ij)]
        p.r_combine_perm = self._plan_r_combine_perm(maps)
        p.energy = [(ij, i, j, self.is_strong(ij),
                     bool(maps["n_pno"][self.diag(i)]
                          and maps["n_pno"][self.diag(j)]))
                    for ij in maps["live"]
                    for i, j in [maps["ij_to_i_j"][ij]]]
        p.classify(self.cc.layout.bucket_of, self.cc.n_pno,
                   self.cc.lmopair_to_lmos)
        self._plan = p
        self.t_plan = time.perf_counter() - t0
        return p

    # -- allocation ----------------------------------------------------------

    def _shared(self, role, shape_, slot):
        """A slice of one of the pooled buffers of its role.

        The rank-3 intermediates carry the auxiliary index, so one per pair is
        of order 15 MiB times the pair count - larger than every integral in the
        calculation put together. So they come from a pool of
        ``_SCRATCH_POOL`` buffers per role, each sized to the largest pair, and
        ``slot`` - the position of the record being emitted in its phase's work
        list - picks one round-robin.

        The round-robin is the whole point. With ONE buffer per role the hazard
        scan correctly serializes every chain that touches it, which is safe but
        is exactly the parallelism the phase has to offer; with ``K`` of them
        consecutive records are independent and an executor can run ``K`` chains
        at once. Records ``slot`` and ``slot + K`` still share, and the scan
        still orders those two, which is why the pool is about as deep as the
        core count and no deeper.
        """
        key = ("shared", role, tuple(shape_), slot % _SCRATCH_POOL)
        hit = self._view_cache.get(key)
        if hit is None:
            pool = self._shared_buf[role]
            buf = pool[slot % len(pool)]
            flat = int(np.prod(shape_))
            hit = self._keepv(buf[:flat].reshape_view(list(shape_)))
            self._view_cache[key] = hit
        return hit

    def _keepv(self, v):
        self._keep.append(v)
        return v

    def _element(self, M, i, j):
        """The ``(1, 1)`` view of one matrix element, memoized."""
        key = (id(M), i, j)
        hit = self._view_cache.get(key)
        if hit is None:
            hit = M[i:i + 1, j:j + 1]
            self._view_cache[key] = hit
        return hit

    def _R1(self, i):
        """LMO ``i``'s singles residual as a captured ``(n, 1)`` view."""
        hit = self._R1_view.get(i)
        if hit is None:
            n = self.cc.n_pno[self.diag(i)]
            hit = self.R1_all[:n, i:i + 1]
            self._R1_view[i] = hit
        return hit

    def _allocate(self):
        """Every tensor the iteration writes, allocated once.

        The emitters below create only views and record operations; nothing here
        is reallocated per iteration, which is the standing advantage over
        psi4's per-iteration ``make_shared`` traffic in ``lccsd_iterations``.
        """
        cc = self.cc
        p = self._plan
        naocc = self.naocc
        n_pno = cc.n_pno
        F = self.F_lmo_np

        self._S_cache = {}
        self._view_cache = {}
        self._T1_view = {}
        self._R1_view = {}

        def z(name, shape_):
            return ten.zeros(name, shape_)

        # -- the singles, on the diagonal pairs -----------------------------
        # One padded block store rather than a list of blocks, so DIIS
        # extrapolates the singles the same way it extrapolates the doubles and
        # the Jacobi step is one operation for the whole family.
        diag = [self.diag(i) for i in range(naocc)]
        M1 = max([n_pno[ii] for ii in diag] + [1])
        self.T1_all = z("T1", [M1, naocc])
        self.R1_all = z("R1", [M1, naocc])
        self.step1_all = z("step T1", [M1, naocc])
        self.D1_all = z("D1", [M1, naocc])
        d1 = ten.view(self.D1_all)
        d1[...] = 1.0
        for i, ii in enumerate(diag):
            n = n_pno[ii]
            if n:
                d1[:n, i] = ten.view(cc.e_pno[ii]) - F[i, i]
        #: The singles as numpy views, one per LMO, for callers that inject or
        #: read amplitudes (``check_ccsd_defect.py``'s bridge).
        t1 = ten.view(self.T1_all)
        self.T_ia = [t1[:n_pno[ii], i:i + 1] for i, ii in enumerate(diag)]

        # -- the doubles residual, and both Jacobi denominators --------------
        self.R_all = cc.new_pair_stores("R")
        self.Rn_all = cc.new_pair_stores("Rn")
        self.step_all = cc.new_pair_stores("step T2")
        self.D_all = cc.new_pair_stores("D")
        for D in self.D_all:
            ten.view(D)[...] = 1.0
        for ij, (i, j) in enumerate(cc.ij_to_i_j):
            n = n_pno[ij]
            if not n:
                continue
            e = ten.view(cc.e_pno[ij])
            cc.pair_block(self.D_all, ij)[:n, :n] = (
                e[:, None] + e[None, :] - F[i, i] - F[j, j])

        # -- Eq. 70 and the other projections -------------------------------
        self.T_n = {}
        for ij in p.live:
            nk = len(cc.lmopair_to_lmos[ij])
            self.T_n[ij] = z(f"T_n ({ij})", [max(nk, 1), n_pno[ij]])
        self._proj_row = {}
        for ij, n_ij, n, _nn in p.projection_rows:
            self._proj_row[ij, n] = self._keepv(self.T_n[ij][n_ij:n_ij + 1, :])
        for ij, x, _xx in p.projections:
            if (ij, x) not in self._proj_row:
                t = z(f"t({x}) in {ij}", [1, n_pno[ij]])
                self._keep.append(t)
                self._proj_row[ij, x] = t

        # -- Eq. 91-92 -------------------------------------------------------
        self.i_Qk_t1 = [None] * cc.n_lmo_pairs
        self.i_Qa_t1 = [None] * cc.n_lmo_pairs
        for ij, _ji, _i in p.t1_ints:
            self.i_Qk_t1[ij] = z(f"B~(Q|k i) {ij}", ten.shape(self.B.i_Qk[ij]))
            self.i_Qa_t1[ij] = z(f"B~(Q|a i) {ij}", ten.shape(self.B.i_Qa[ij]))

        # -- Eq. 94-101 -------------------------------------------------------
        self.Fij_bar = z("F_bar (occ)", [naocc, naocc])
        self.Fkj = z("F~ (occ)", [naocc, naocc])
        self.Fkj2 = z("F'' (occ)", [naocc, naocc])
        self.fock_occ_scalar = {ij: (ten.scalar(), ten.scalar())
                                for ij, _, _, _ in p.fock_occupied}
        self.fock_kj_scalar = {ij: ten.scalar() for ij, *_ in p.fock_kj}
        self.Fkc_bar = [None] * cc.n_lmo_pairs
        self.Fab_bar = [None] * cc.n_lmo_pairs
        self.Fab = [None] * cc.n_lmo_pairs
        self.e_diag = [None] * cc.n_lmo_pairs
        self.Fai_bar = [None] * naocc
        self.Fai = [None] * naocc
        self.gamma_Q = {}
        self.fock_ai_tmp = [None] * naocc
        self.fock_ai_col = [None] * naocc
        for ij, i, j in p.fock_bar:
            nq = ten.shape(self.B.i_Qk[ij])[0]
            nk, na = ten.shape(self.T_n[ij])
            self.Fkc_bar[ij] = z(f"F_bar (kc) {ij}", [nk, na])
            self.Fab_bar[ij] = z(f"F_bar (ab) {ij}", [na, na])
            self.Fab[ij] = z(f"F~ (ab) {ij}", [na, na])
            self.e_diag[ij] = z(f"e_pno diag {ij}", [na, na])
            ten.view(self.e_diag[ij])[...] = np.diag(ten.view(cc.e_pno[ij]))
            self.gamma_Q[ij] = z(f"gamma_Q {ij}", [nq])
            if i == j:
                self.Fai_bar[i] = z(f"F_bar (ai) {i}", [na, 1])
                self.Fai[i] = z(f"F~ (ai) {i}", [na, 1])
                self.fock_ai_tmp[i] = z(f"F~ (ai) tmp {i}", [nk, 1])
                self.fock_ai_col[i] = z(f"F_bar column {i}", [nk, 1])
        self.Fkc = [None] * cc.n_lmo_pairs
        for ij in p.live:
            self.Fkc[ij] = z(f"F~ (kc) {ij}", [n_pno[ij], 1])
        self.fock_kc_tmp = {(ij, ik): z(f"F~ (kc) tmp {ij},{ik}", [n_pno[ik], 1])
                            for ij, ik, _k in p.fock_kc}

        # -- Eq. 82-86 --------------------------------------------------------
        self.beta = [None] * cc.n_lmo_pairs
        for ij, _ji in p.beta:
            nk = ten.shape(self.T_n[ij])[0]
            self.beta[ij] = z(f"beta {ij}", [nk, nk])
        self.gamma = [None] * cc.n_lmo_pairs
        self.delta = [None] * cc.n_lmo_pairs
        self.M_delta = [None] * cc.n_lmo_pairs
        for ij in p.live:
            self.gamma[ij] = z(f"gamma {ij}", [n_pno[ij], n_pno[ij]])
            self.delta[ij] = z(f"delta {ij}", [n_pno[ij], n_pno[ij]])
            # Eq. 84a's integral combination is constant.
            self.M_delta[ij] = ten.from_numpy(
                f"2K-J {ij}",
                2.0 * ten.view(self.B.K_mibj[ij]) - ten.view(self.B.J_ijmb[ij]))

        # Per-pair scratch is sized to the WIDEST partner that pair's work list
        # reaches, and each use slices the part it needs. One buffer per pair
        # and role rather than one per record: the records of a pair accumulate
        # into that pair's intermediate and are therefore serialized by the
        # dependency anyway, so sharing costs no parallelism and saves an
        # allocation per neighbour.
        gw = _widest(p.gamma_tail, 0, [(2, n_pno), (4, n_pno)])
        self.gamma_tmp, self.gamma_tmp_d = {}, {}
        for ki, (wide, tall) in gw.items():
            n = n_pno[ki]
            self.gamma_tmp[ki] = (z(f"83c a {ki}", [wide, 1]),
                                  z(f"83c b {ki}", [n, 1]),
                                  z(f"83d c {ki}", [n, wide]))
            self.gamma_tmp_d[ki] = (z(f"83d a {ki}", [tall, wide]),
                                    z(f"83d b {ki}", [tall, wide]))
        dw = _widest(p.delta_tail, 0, [(3, n_pno), (4, n_pno)])
        self.delta_tmp, self.delta_tmp_d = {}, {}
        for ik, (wide, tall) in dw.items():
            n = n_pno[ik]
            # Column-shaped, mirroring 83c's; see _emit_delta on why.
            self.delta_tmp[ik] = (z(f"84c a {ik}", [wide, 1]),
                                  z(f"84c b {ik}", [n, 1]),
                                  z(f"84d c {ik}", [n, wide]))
            self.delta_tmp_d[ik] = (z(f"84d a {ik}", [tall, wide]),
                                    z(f"84d b {ik}", [tall, wide]))

        self.fkj2_tmp = {}
        for ij, _i, _j, l, li, lj in p.fkj2:
            self.fkj2_tmp[ij, l] = (z(f"86 a {ij},{l}", [n_pno[li], n_pno[lj]]),
                                    z(f"86 b {ij},{l}", [n_pno[li], n_pno[li]]),
                                    ten.scalar())

        # -- Eq. 87a-90 -------------------------------------------------------
        self.r_ia_a1 = {ik: z(f"88 A1 {ik}", [n_pno[ki], 1])
                        for ik, _i, _k, ki, _ii, has, _f in p.r_ia_ik if has}
        self.r_ia_c = {ik: z(f"90 C {ik}", [n_pno[ik], 1])
                       for ik, _i, _k, _ki, _ii, _h, has in p.r_ia_ik if has}
        self.r_ia_a2_tmp = {}
        for ik, _i, ki, _ii, l, kl in p.r_ia_a2:
            self.r_ia_a2_tmp[ik, l] = (z(f"88 A2 a {ik},{l}", [n_pno[kl], n_pno[ki]]),
                                       z(f"88 A2 b {ik},{l}", [n_pno[kl], n_pno[kl]]),
                                       ten.scalar())
        self.r_ia_b_tmp = {}
        for kl, _members in p.r_ia_b:
            nk, na = ten.shape(self.T_n[kl])
            self.r_ia_b_tmp[kl] = (z(f"89 K {kl}", [nk, na]),
                                   z(f"89 B {kl}", [na, nk]))

        # -- Eq. 19, 75-81 -----------------------------------------------------
        self.E_kl = [None] * cc.n_lmo_pairs
        for ij in p.live:
            self.E_kl[ij] = z(f"85 u K^T {ij}", [n_pno[ij], n_pno[ij]])
        self.r_sym_acc = {}
        self.r_sym_B = {}
        self.F_bc = {}
        self.r_sym_kl_tmp = {}
        for ij, i, j, _ji in p.r_sym:
            n = n_pno[ij]
            self.r_sym_acc[ij] = z(f"R sym {ij}", [n, n])
            self.r_sym_B[ij] = z(f"77 B {ij}", [n, n])
            self.F_bc[ij] = z(f"85 F'' bc {ij}", [n, n])
        for ij, (wide,) in _widest(p.r_sym_kl, 0, [(3, n_pno)]).items():
            n = n_pno[ij]
            self.r_sym_kl_tmp[ij] = (z(f"77 h1 {ij}", [n, wide]),
                                     z(f"85 h2 {ij}", [n, wide]),
                                     z(f"77 tmp {ij}", [n, n]))

        # Eq. 78's and Eq. 79's accumulators live in PAIR STORES rather than one
        # tensor per pair, and for the same reason Eq. 19 does: each is cleared
        # once and then added whole into ``Rn``, and both of those are statements
        # about the store. Eq. 78 runs over EVERY live pair, so a store-wide
        # clear or add is the pair loop it replaces plus the padding, which is
        # identically zero on both sides and contributes exactly ``+0.0``. That
        # takes five operations per pair down to five per BUCKET.
        #
        # Eq. 81's accumulator cannot be one of them, because its update carries
        # an element of ``F''`` as its prefactor and is therefore a ``k = 1``
        # GEMM against the block FLATTENED, which a padded slice of a store
        # cannot be: the two axes being merged are not adjacent in memory. It
        # gets a store of its own holding each pair's block packed at its exact
        # extent, which flattens and which still clears in one operation.
        #
        # Both names carry the ``non``, because ``self.D_all`` is already the
        # Jacobi denominator: shadowing it replaces a store of ones with a store
        # of zeros, and what that looks like is every amplitude coming back a NaN
        # two phases later with nothing in between to say why.
        self.C_non_all = cc.new_pair_stores("78 C")
        self.D_non_all = cc.new_pair_stores("79 D")
        g_off, g_elem = {}, 0
        for ij, _i, _j, _gs in p.r_non_g:
            g_off[ij] = g_elem
            g_elem += n_pno[ij] ** 2
        self.G_store = ten.zeros("81 G", [max(g_elem, 1)])

        self.r_non_C, self.r_non_D, self.r_non_G = {}, {}, {}
        self.r_non_bridge, self.r_non_half = {}, {}
        self.r_non_g_tmp = {}
        for ij, _i, _j, cs, ds in p.r_non:
            n = n_pno[ij]
            self.r_non_C[ij] = self._pair(self.C_non_all, ij)
            self.r_non_D[ij] = self._pair(self.D_non_all, ij)
            wide = max([n_pno[r[2]] for r in cs] + [n_pno[r[2]] for r in ds]
                       + [n_pno[r[1]] for r in cs] + [n_pno[r[1]] for r in ds]
                       + [1])
            self.r_non_bridge[ij] = z(f"78/79 bridge {ij}", [n, wide])
            self.r_non_half[ij] = z(f"78/79 half {ij}", [n, wide])
        for ij, _i, _j, gs in p.r_non_g:
            n = n_pno[ij]
            self.r_non_G[ij] = self._keepv(
                self.G_store[g_off[ij]:g_off[ij] + n * n].reshape_view([n, n]))
            wide = max([n_pno[ik] for _k, ik in gs] + [1])
            self.r_non_g_tmp[ij] = (z(f"81 a {ij}", [n, wide]),
                                    z(f"81 tmp {ij}", [n, n]))

        # -- Eq. 45 ------------------------------------------------------------
        self.tau = {ij: z(f"tau {ij}", [n_pno[ij], n_pno[ij]]) for ij in p.live}
        self.energy_scalar = {ij: ten.scalar() for ij in p.live}
        self.e_total = ten.scalar("E(iteration)")
        self.e_weak = ten.scalar("E(weak)")

        # -- the pooled rank-3 scratch -----------------------------------------
        # One pool per role rather than one buffer, so the records of a phase
        # can be in flight at once; see _shared for why the depth is the core
        # count. Every use overwrites the slice it takes in full, so the buffers
        # are allocated uninitialized and nothing reads across a hand-out.
        self._shared_buf = {}
        for role, dims in self._shared_sizes().items():
            self._shared_buf[role] = [
                ten.empty(f"scratch {role} [{k}]", [dims])
                for k in range(_SCRATCH_POOL)]

        self._build_batches()

    def _build_batches(self):
        """Every grouped GEMM the iteration replays, and every view it reads.

        Runs with NO capture open, which is the whole point of doing it here:
        a view built inside an emitter is recorded as a node, and this walks the
        same work lists the emitters walk and builds their operands eagerly. What
        the emitters are left with is one
        :func:`~einsums.graph.grouped_batched_gemm` per stage and per step, plus
        the einsums, dots and copies that have no batched form.

        The grouping rule is stated on :class:`_Batch` and the transposition of
        the neighbour walks on :func:`_by_step`. Every batch here either has one
        destination per member outright, or is one step of a walk whose members
        belong to distinct pairs; nothing in it needs a shared scratch buffer, so
        the pool of :meth:`_shared` is untouched by all of this - it carries the
        rank-3 einsum intermediates, which are not GEMMs and are still emitted a
        pair at a time.
        """
        cc = self.cc
        p = self._plan
        n_pno = cc.n_pno
        bat = self._bat = {}
        run = self._run = {}

        def batches(*specs):
            """One :class:`_Batch` per ``(alpha, beta, trans_a, trans_b)``."""
            return [_Batch(*spec) for spec in specs]

        def runs(cls, *names):
            """One run of class ``cls`` per name, registered under it."""
            for name in names:
                run[name] = cls()
            return [run[name] for name in names]

        # -- Eq. 70, one node for every projection in the iteration ----------
        one = _Batch(1.0, 0.0, trans_a=True, trans_b=True)
        for pair, x, xx in p.projections:
            one.add(self._v(self._T1(x)), self._v(self._S(pair, xx)),
                    self._v(self._proj_row[pair, x]))
        bat["projections"] = [one]

        # -- Eq. 91-92 -------------------------------------------------------
        #
        # Both undressed starts go out as ONE run, ahead of the einsums that
        # dress them. Hoisting them out of the walk changes no order that
        # matters: each is a beta = 0 write of its own record's factor, and the
        # einsum that then accumulates onto that factor is recorded after the
        # run, so every record still reads what its own copy left.
        init, = runs(_AxpbyRun, "t1_ints_init")
        one = _Batch(-1.0, 1.0)
        for ij, _ji, i in p.t1_ints:
            self._row(ij, i)
            init.add(1.0, self._v(self.B.i_Qk[ij]), 0.0,
                     self._v(self.i_Qk_t1[ij]))
            init.add(1.0, self._v(self.B.i_Qa[ij]), 0.0,
                     self._v(self.i_Qa_t1[ij]))
            one.add(self._v(self.i_Qk_t1[ij]), self._v(self.T_n[ij]),
                    self._v(self.i_Qa_t1[ij]))
        bat["t1_ints"] = [one]

        # -- the scalar families ---------------------------------------------
        #
        # A reduction into a scalar and a scaled accumulation of it into one
        # matrix element, thousands of each, and neither has any arithmetic in
        # it to spread. Each family is one :class:`_DotRun` and one
        # :class:`_AxpbyRun`; both run their entries in sequence inside a single
        # node, in the order added, so what the merge removes is the dispatch
        # and nothing else. See those classes for why that is bit-exact even
        # where a family's entries collide on their destination.
        dot, = runs(_DotRun, "fock_occ_dot")
        acc, = runs(_AxpbyRun, "fock_occ_acc")
        for ij, i, j, ji in p.fock_occupied:
            s1, s2 = self.fock_occ_scalar[ij]
            T_n = self._v(self.T_n[ij])
            dest = self._element(self.Fij_bar, i, j)
            # Eq. 98's two terms land in ONE element, the first with a factor of
            # two and the second with a minus sign, so both belong to one run
            # and the run's order is what keeps their sum in psi4's order.
            dot.add(s1, T_n, self._v(self.B.J_ijmb[ij]))
            dot.add(s2, T_n, self._v(self.B.K_mibj[ji]))
            acc.add(2.0, self._flat(s1, [1, 1]), 1.0, dest)
            acc.add(-1.0, self._flat(s2, [1, 1]), 1.0, dest)

        dot, = runs(_DotRun, "fock_kj_dot")
        acc, = runs(_AxpbyRun, "fock_kj_acc")
        for ij, i, j, jj, i_jj in p.fock_kj:
            s = self.fock_kj_scalar[ij]
            dot.add(s, self._mrow(self.Fkc_bar[jj], i_jj),
                    self._col(self._T1(j)))
            acc.add(1.0, self._flat(s, [1, 1]), 1.0,
                    self._element(self.Fkj, i, j))

        # Eq. 101's orbital-energy start and Eq. 96-97's two, all beta = 0
        # writes of a per-pair quantity, hoisted for the same reason Eq. 91's
        # are: the terms that accumulate onto them are recorded afterwards.
        bar_init, tilde_init = runs(_AxpbyRun, "fock_bar_init",
                                    "fock_tilde_init")
        for ij, i, j in p.fock_bar:
            bar_init.add(1.0, self._v(self.e_diag[ij]), 0.0,
                         self._v(self.Fab_bar[ij]))
            tilde_init.add(1.0, self._v(self.Fab_bar[ij]), 0.0,
                           self._v(self.Fab[ij]))
            if i == j:
                self._col(self.Fai_bar[i])
                tilde_init.add(1.0, self._v(self.Fai_bar[i]), 0.0,
                               self._v(self.Fai[i]))

        # -- Eq. 95 ----------------------------------------------------------
        pre = _Batch(1.0, 0.0, trans_b=True)
        for ij, ik, k in p.fock_kc:
            pre.add(self._v(self.B.L_iajb[ik]), self._v(self._proj(ik, k)),
                    self._v(self.fock_kc_tmp[ij, ik]))
        bat["fock_kc"] = [pre]
        for step in _by_step(p.fock_kc, 0):
            acc = _Batch(1.0, 1.0)
            for ij, ik, k in step:
                acc.add(self._v(self._S(ij, ik)),
                        self._v(self.fock_kc_tmp[ij, ik]), self._v(self.Fkc[ij]))
            bat["fock_kc"].append(acc)

        # -- Eq. 82's first term ---------------------------------------------
        one = _Batch(1.0, 0.0, trans_a=True)
        for ij, ji in p.beta:
            one.add(self._v(self.i_Qk_t1[ij]), self._v(self.i_Qk_t1[ji]),
                    self._v(self.beta[ij]))
        bat["beta"] = [one]

        # -- Eq. 83a, and Eq. 83b's operand ----------------------------------
        one = _Batch(-1.0, 0.0, trans_a=True)
        for ki, k, i, ii, has_vvv in p.gamma_head:
            one.add(self._v(self.T_n[ki]), self._v(self.B.J_ijmb[ki]),
                    self._v(self.gamma[ki]))
            if has_vvv:
                self._row(ki, i)
        bat["gamma_head"] = [one]

        # -- Eq. 83c and 83d, the neighbour walk -----------------------------
        bat["gamma_tail"] = []
        for step in _by_step(p.gamma_tail, 0):
            c1, c2, c3, d1, d2, d3, d4 = batches(
                (1.0, 0.0, True, True), (1.0, 0.0), (-1.0, 1.0, True, True),
                (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (-0.5, 1.0))
            for ki, l, kl, ll, li, do_c, do_d in step:
                t1, t2, t3 = self.gamma_tmp[ki]
                n_kl, n_ki = n_pno[kl], n_pno[ki]
                if do_c:
                    _, i_of_ki = cc.ij_to_i_j[ki]
                    a = self._sub(t1, n_kl, 1)
                    c1.add(self._v(self._K2(kl)),
                           self._v(self._proj(kl, i_of_ki)), a)
                    b = self._sub(t2, n_ki, 1)
                    c2.add(self._v(self._S(ki, kl)), a, b)
                    c3.add(self._v(self._proj(ki, l)), b, self._v(self.gamma[ki]))
                if do_d:
                    n_li = n_pno[li]
                    a = self._sub(self.gamma_tmp_d[ki][0], n_li, n_kl)
                    d1.add(self._v(self._T2(li)), self._v(self._S(li, kl)), a)
                    b = self._sub(self.gamma_tmp_d[ki][1], n_li, n_kl)
                    d2.add(a, self._v(self._K2(kl)), b)
                    c = self._sub(t3, n_ki, n_kl)
                    d3.add(self._v(self._S(ki, li)), b, c)
                    d4.add(c, self._v(self._S(kl, ki)), self._v(self.gamma[ki]))
            bat["gamma_tail"] += [c1, c2, c3, d1, d2, d3, d4]

        # -- Eq. 84a, and Eq. 84b's operand ----------------------------------
        one = _Batch(-1.0, 0.0, trans_a=True)
        for ik, i, k, ki, ii, has_vvv in p.delta_head:
            one.add(self._v(self.T_n[ik]), self._v(self.M_delta[ik]),
                    self._v(self.delta[ik]))
            if has_vvv:
                self._row(ik, i)
        bat["delta_head"] = [one]

        # -- Eq. 84c and 84d, the neighbour walk -----------------------------
        bat["delta_tail"] = []
        for step in _by_step(p.delta_tail, 0):
            c1, c2, c3, d1, d2, d3, d4 = batches(
                (1.0, 0.0, True, True), (1.0, 0.0, True, False),
                (-1.0, 1.0, True, True),
                (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.5, 1.0))
            for ik, l, ll, lk, il, do_c, do_d in step:
                t1, t2, t3 = self.delta_tmp[ik]
                n_lk, n_ik = n_pno[lk], n_pno[ik]
                if do_c:
                    i_of_ik, _ = cc.ij_to_i_j[ik]
                    a = self._sub(t1, n_lk, 1)
                    c1.add(self._v(self.B.L_iajb[lk]),
                           self._v(self._proj(lk, i_of_ik)), a)
                    b = self._sub(t2, n_ik, 1)
                    c2.add(self._v(self._S(lk, ik)), a, b)
                    c3.add(self._v(self._proj(ik, l)), b, self._v(self.delta[ik]))
                if do_d:
                    n_il = n_pno[il]
                    a = self._sub(self.delta_tmp_d[ik][0], n_il, n_lk)
                    d1.add(self._v(self._Tt2(il)), self._v(self._S(il, lk)), a)
                    b = self._sub(self.delta_tmp_d[ik][1], n_il, n_lk)
                    d2.add(a, self._v(self.B.L_iajb[lk]), b)
                    c = self._sub(t3, n_ik, n_lk)
                    d3.add(self._v(self._S(ik, il)), b, c)
                    d4.add(c, self._v(self._S(lk, ik)), self._v(self.delta[ik]))
            bat["delta_tail"] += [c1, c2, c3, d1, d2, d3, d4]

        # -- Eq. 86's congruence transform -----------------------------------
        b1, b2 = batches((1.0, 0.0), (1.0, 0.0))
        dot, = runs(_DotRun, "fkj2_dot")
        acc, = runs(_AxpbyRun, "fkj2_acc")
        for ij, i, j, l, li, lj in p.fkj2:
            a, b, s = self.fkj2_tmp[ij, l]
            a_v = self._sub(a, n_pno[li], n_pno[lj])
            b1.add(self._v(self._S(li, lj)), self._v(self._Tt2(lj)), a_v)
            b2.add(a_v, self._v(self._S(lj, li)), self._v(b))
            dot.add(s, self._v(self._K2(li)), self._v(b))
            # A pair's neighbours ALL land in its one element of F'', so this is
            # the run whose order carries the sum over neighbours. Adding in the
            # order the unbatched emitter walked keeps it exactly.
            acc.add(1.0, self._flat(s, [1, 1]), 1.0,
                    self._element(self.Fkj2, i, j))
        bat["fkj2"] = [b1, b2]

        # -- Eq. 88, 89, 90 --------------------------------------------------
        acc, = runs(_AxpbyRun, "r_ia_fai")
        for i in range(self.naocc):
            if n_pno[self.diag(i)]:
                self._R1(i)
            if self.Fai[i] is not None:
                acc.add(1.0, self._v(self.Fai[i]), 1.0, self._R1(i))
        pre = _Batch(1.0, 0.0)
        for ik, i, k, ki, ii, has_vvv, has_fkc in p.r_ia_ik:
            if has_vvv:
                self._col(self.r_ia_a1[ik])
            if has_fkc:
                pre.add(self._v(self._Tt2(ik)), self._v(self.Fkc[ki]),
                        self._v(self.r_ia_c[ik]))
        bat["r_ia_ik"] = [pre]
        for step in _by_step(p.r_ia_ik, 1):
            a1, cc_b = batches((1.0, 1.0, True, False), (1.0, 1.0, True, False))
            for ik, i, k, ki, ii, has_vvv, has_fkc in step:
                if has_vvv:
                    a1.add(self._v(self._S(ki, ii)), self._v(self.r_ia_a1[ik]),
                           self._v(self._R1(i)))
                if has_fkc:
                    cc_b.add(self._v(self._S(ik, ii)), self._v(self.r_ia_c[ik]),
                             self._v(self._R1(i)))
            bat["r_ia_ik"] += [a1, cc_b]

        b1, b2 = batches((1.0, 0.0), (1.0, 0.0))
        dot, = runs(_DotRun, "r_ia_a2_dot")
        for ik, i, ki, ii, l, kl in p.r_ia_a2:
            a, b, s = self.r_ia_a2_tmp[ik, l]
            a_v = self._sub(a, n_pno[kl], n_pno[ki])
            b1.add(self._v(self._S(kl, ki)), self._v(self._Tt2(ki)), a_v)
            b2.add(a_v, self._v(self._S(ki, kl)), self._v(b))
            dot.add(s, self._v(b), self._v(self._K2(kl)))
        bat["r_ia_a2_pre"] = [b1, b2]
        bat["r_ia_a2"] = []
        for step in _by_step(p.r_ia_a2, 1):
            one = _Batch(-1.0, 1.0, trans_a=True)
            for ik, i, ki, ii, l, kl in step:
                _a, _b, s = self.r_ia_a2_tmp[ik, l]
                one.add(self._v(self._proj(ii, l)), self._flat(s, [1, 1]),
                        self._v(self._R1(i)))
            bat["r_ia_a2"].append(one)

        b1, b2 = batches((1.0, 1.0), (1.0, 0.0, False, True))
        init, = runs(_AxpbyRun, "r_ia_b_init")
        members = []
        for kl, mem in p.r_ia_b:
            K_kilc, B_ai = self.r_ia_b_tmp[kl]
            init.add(1.0, self._v(self.B.K_mibj[kl]), 0.0, self._v(K_kilc))
            b1.add(self._v(self.T_n[kl]), self._v(self._K2(kl)), self._v(K_kilc))
            b2.add(self._v(self._Tt2(kl)), self._v(K_kilc), self._v(B_ai))
            for i_kl, i, ii in mem:
                members.append((i, kl, i_kl, ii))
        bat["r_ia_b"] = [b1, b2]
        for step in _by_step(members, 0):
            one = _Batch(-1.0, 1.0, trans_a=True)
            for i, kl, i_kl, ii in step:
                one.add(self._v(self._S(kl, ii)),
                        self._sub_col(self.r_ia_b_tmp[kl][1], i_kl),
                        self._v(self._R1(i)))
            bat["r_ia_b"].append(one)

        # -- Eq. 19, 75-81 ---------------------------------------------------
        one = _Batch(1.0, 0.0, trans_b=True)
        for kl in p.live:
            one.add(self._v(self._Tt2(kl)), self._v(self._K2(kl)),
                    self._v(self.E_kl[kl]))
        bat["E_kl"] = [one]

        one = _Batch(1.0, 0.0, trans_a=True)
        # Eq. 85's F''_bc start, hoisted ahead of the double walk that fills it
        # for the same reason Eq. 91's copies are hoisted; and Eq. 77's
        # accumulator folded into Eq. 76's, which is its own standalone walk.
        fbc, into_acc = runs(_AxpbyRun, "r_sym_fbc", "r_sym_B")
        for ij, i, j, ji in p.r_sym:
            one.add(self._v(self.i_Qa_t1[ij]), self._v(self.i_Qa_t1[ji]),
                    self._v(self.r_sym_acc[ij]))
            self._pair(self.R_all, ij)
            self._pair(self.R_all, ji)
            fbc.add(1.0, self._v(self.Fab[ij]), 0.0, self._v(self.F_bc[ij]))
            into_acc.add(1.0, self._v(self.r_sym_B[ij]), 1.0,
                         self._v(self.r_sym_acc[ij]))
        bat["r_sym_75"] = [one]

        bat["r_sym_kl"] = []
        for step in _by_step(p.r_sym_kl, 0):
            s1, s2, s3, s4, s5 = batches(
                (1.0, 0.0, True, False), (1.0, 0.0), (1.0, 1.0),
                (1.0, 0.0, True, False), (-1.0, 1.0))
            for ij, k_ij, l_ij, kl in step:
                n_ij, n_kl = n_pno[ij], n_pno[kl]
                h1, h2, tmp = self.r_sym_kl_tmp[ij]
                S_kl = self._v(self._S(kl, ij))
                a = self._sub(h1, n_ij, n_kl)
                s1.add(S_kl, self._v(self._T2(kl)), a)
                s2.add(a, S_kl, self._v(tmp))
                s3.add(self._flat(tmp, [n_ij * n_ij, 1]),
                       self._element(self.beta[ij], k_ij, l_ij),
                       self._flat(self.r_sym_B[ij], [n_ij * n_ij, 1]))
                b = self._sub(h2, n_ij, n_kl)
                s4.add(S_kl, self._v(self.E_kl[kl]), b)
                s5.add(b, S_kl, self._v(self.F_bc[ij]))
            bat["r_sym_kl"] += [s1, s2, s3, s4, s5]

        b1, b2 = batches((1.0, 1.0, False, True), (1.0, 1.0))
        for ij, i, j, ji in p.r_sym:
            b1.add(self._v(self._T2(ij)), self._v(self.F_bc[ij]),
                   self._v(self.r_sym_acc[ij]))
            b2.add(self._v(self.F_bc[ij]), self._v(self._T2(ij)),
                   self._v(self.r_sym_acc[ij]))
        bat["r_sym_80"] = [b1, b2]

        # -- Eq. 78 and 79, the two neighbour walks --------------------------
        #
        # ``total`` gets ONE buffer per record rather than one per pair, packed
        # end to end in a flat store with a second store beside it holding the
        # constant integral combination each record starts from. That is what
        # makes the reset batchable: the family's whole initialization is one
        # copy of one store onto another, where per-pair buffers forced a copy
        # per record and left it the phase's largest node family. It costs no
        # padding - the records are packed at their exact extents - and moves no
        # bit, since each record still starts from its own constant and
        # accumulates its own product on top.
        bat["r_non"] = []
        self._r_non_reset = []
        walks = [("78", [(ij, k_ij, ki, kj) for ij, _i, _j, cs, _ds in p.r_non
                         for k_ij, ki, kj in cs],
                  self.gamma, self._T2, -1.0, self.r_non_C,
                  lambda ij, k: ten.view(self.B.J_ikac[ij][k])),
                 ("79", [(ij, k_ij, ik, jk) for ij, _i, _j, _cs, ds in p.r_non
                         for k_ij, ik, jk in ds],
                  self.delta, self._Tt2, 1.0, self.r_non_D,
                  lambda ij, k: (2.0 * ten.view(self.B.K_iakc[ij][k])
                                 - ten.view(self.B.J_ikac[ij][k])))]
        for tag, records, mid, amp, sign, dest, constant in walks:
            offsets, n_elem = [], 0
            for ij, _k_ij, _near, far in records:
                offsets.append(n_elem)
                n_elem += n_pno[ij] * n_pno[far]
            const = ten.zeros(f"Eq. {tag} constant", [max(n_elem, 1)])
            store = ten.empty(f"Eq. {tag} total", [max(n_elem, 1)])
            packed = ten.view(const)
            totals = {}
            for off, (ij, k_ij, _near, far) in zip(offsets, records):
                n_ij, n_far = n_pno[ij], n_pno[far]
                packed[off:off + n_ij * n_far] = \
                    constant(ij, k_ij).reshape(n_ij * n_far, order="F")
                totals[ij, k_ij] = self._keepv(
                    store[off:off + n_ij * n_far].reshape_view([n_ij, n_far]))
            self._keep += [const, store]
            self._r_non_reset.append((const, store))
            run.setdefault("r_non_reset", _AxpbyRun()).add(
                1.0, self._v(const), 0.0, self._v(store))
            for step in _by_step(records, 0):
                s1, s2, s3, s4 = batches((1.0, 0.0), (1.0, 1.0),
                                         (1.0, 0.0, False, True), (sign, 1.0))
                for ij, k_ij, near, far in step:
                    n_ij, n_far, n_near = n_pno[ij], n_pno[far], n_pno[near]
                    total = totals[ij, k_ij]
                    bridge = self._sub(self.r_non_bridge[ij], n_ij, n_near)
                    half = self._sub(self.r_non_half[ij], n_ij, n_far)
                    s1.add(self._v(self._S(ij, near)), self._v(mid[near]), bridge)
                    s2.add(bridge, self._v(self._S(near, far)), total)
                    s3.add(total, self._v(amp(far)), half)
                    s4.add(half, self._v(self._S(far, ij)), self._v(dest[ij]))
                bat["r_non"] += [s1, s2, s3, s4]

        # -- Eq. 81 ----------------------------------------------------------
        bat["r_non_g"] = []
        g_records = [(ij, j, k, ik) for ij, _i, j, gs in p.r_non_g
                     for k, ik in gs]
        acc, = runs(_AxpbyRun, "r_non_g_into_Rn")
        for ij, _i, _j, _gs in p.r_non_g:
            acc.add(1.0, self._v(self.r_non_G[ij]), 1.0,
                    self._pair(self.Rn_all, ij))
        for step in _by_step(g_records, 0):
            s1, s2, s3 = batches((1.0, 0.0), (1.0, 0.0), (-1.0, 1.0))
            for ij, j, k, ik in step:
                n_ij, n_ik = n_pno[ij], n_pno[ik]
                a, tmp = self.r_non_g_tmp[ij]
                a_v = self._sub(a, n_ij, n_ik)
                s1.add(self._v(self._S(ij, ik)), self._v(self._T2(ik)), a_v)
                s2.add(a_v, self._v(self._S(ik, ij)), self._v(tmp))
                s3.add(self._flat(tmp, [n_ij * n_ij, 1]),
                       self._element(self.Fkj2, k, j),
                       self._flat(self.r_non_G[ij], [n_ij * n_ij, 1]))
            bat["r_non_g"] += [s1, s2, s3]

        # -- Eq. 45 -----------------------------------------------------------
        one = _Batch(1.0, 1.0, trans_a=True)
        dot, = runs(_DotRun, "energy_dot")
        tau, acc = runs(_AxpbyRun, "energy_tau", "energy_acc")
        for ij, i, j, strong, has_singles in p.energy:
            tau.add(1.0, self._T2(ij), 0.0, self._v(self.tau[ij]))
            if has_singles:
                one.add(self._v(self._proj(ij, i)), self._v(self._proj(ij, j)),
                        self._v(self.tau[ij]))
            dot.add(self.energy_scalar[ij], self._v(self.tau[ij]),
                    self._v(self.B.L_iajb[ij]))
            # Every pair adds into the SAME two scalars, so this run's order is
            # the order of the energy's sum over pairs, and it is the walk's.
            acc.add(1.0, self.energy_scalar[ij], 1.0, self.e_total)
            if not strong:
                acc.add(1.0, self.energy_scalar[ij], 1.0, self.e_weak)
        bat["energy"] = [one]

        # -- the pooled rank-3 scratch's views, which are not GEMM operands --
        # Warmed here for the same reason everything else is: the emitters that
        # take them run inside a capture, where building the view would record
        # one more node per hand-out.
        for slot, (ij, _i, _j) in enumerate(p.fock_bar):
            nq = ten.shape(self.B.i_Qk[ij])[0]
            nk, na = ten.shape(self.T_n[ij])
            self._shared("V", [nq, nk, nk], slot)
            self._shared("U", [nq, na, nk], slot)
            self._shared("y", [nq, na], slot)
        for slot, (ij, _ji) in enumerate(p.beta):
            nq, nk, na = ten.shape(self.B.qma(cc.ij_to_ji, ij))
            self._shared("bW", [nq, nk, na], slot)
        for slot, (ij, _i, _j, _ji) in enumerate(p.r_sym):
            nq, na, _ = ten.shape(self.B.qab(cc.ij_to_ji, ij))
            self._shared("Qab_t1", [nq, na, na], slot)
            self._shared("AW", [nq, na, na], slot)

    def _shared_sizes(self):
        """How large each pooled rank-3 buffer has to be, in elements."""
        cc = self.cc
        p = self._plan
        need = {"V": 1, "U": 1, "y": 1, "bW": 1, "Qab_t1": 1, "AW": 1}
        for ij, _i, _j in p.fock_bar:
            nq = ten.shape(self.B.i_Qk[ij])[0]
            nk, na = ten.shape(self.T_n[ij])
            need["V"] = max(need["V"], nq * nk * nk)
            need["U"] = max(need["U"], nq * na * nk)
            need["y"] = max(need["y"], nq * na)
        for ij, _ji in p.beta:
            nq, nk, na = ten.shape(self.B.qma(cc.ij_to_ji, ij))
            need["bW"] = max(need["bW"], nq * nk * na)
        for ij, _i, _j, _ji in p.r_sym:
            nq, na, _ = ten.shape(self.B.qab(cc.ij_to_ji, ij))
            need["Qab_t1"] = max(need["Qab_t1"], nq * na * na)
            need["AW"] = max(need["AW"], nq * na * na)
        return need

    # -- capture -------------------------------------------------------------

    def phases(self):
        """The residual phases, in the order they are recorded and replayed.

        psi4's order in ``lccsd_iterations``, and it stays a table for two
        reasons that outlived the fourteen graphs it used to build: it is the
        list :meth:`phase_graph` bisects against, and its third column records
        which phases WOULD take an OpenMP executor if they still had one each.
        That column is ``True`` exactly where every record writes its own pair's
        outputs and its own per-pair temporaries and nothing else.

        The four ``False`` entries are the phases that accumulate into a shared
        occupied matrix element by element (Eq. 98, Eq. 94-97, Eq. 86) and the
        one that is four operations long. Under one graph they are no longer a
        separate executor decision - :meth:`capture` says why the merge is safe
        for them and moves no bit - but which phases they are is still worth
        recording, because they are the ones whose records are ordered by a
        hazard edge rather than being independent.

        The scalar runs do not change that column and were not allowed to. What
        used to be a hazard edge per element inside those three phases is now an
        ordering inside ONE node, which is a stronger guarantee of the same
        thing and no more parallelism: the reductions they hold have no
        arithmetic to spread. So the entries stay ``False`` on the same reading,
        and no executor assignment moved.
        """
        return [
            ("Eq. 70 projections", self._emit_projections, True),
            ("Eq. 91-92 t1 ints", self._emit_t1_ints, True),
            ("Eq. 98 F_bar (occ)", self._emit_t1_fock_occupied, False),
            ("Eq. 99-101 F_bar", self._emit_t1_fock_bar, True),
            ("Eq. 94-97 F~", self._emit_t1_fock_tilde, False),
            ("Eq. 82 beta", self._emit_beta, True),
            ("Eq. 83 gamma", self._emit_gamma, True),
            ("Eq. 84 delta", self._emit_delta, True),
            ("Eq. 86 F''", self._emit_Fkj_double_tilde, False),
            ("Eq. 87-90 R_ia", self._emit_R_ia, True),
            ("residual containers", self._emit_R_iajb_zero, False),
            ("Eq. 75-77, 80, 85", self._emit_R_iajb_symmetric, True),
            ("Eq. 78-79, 81", self._emit_R_iajb_nonsymmetric, True),
            ("Eq. 19 combine", self._emit_R_iajb_combine, True),
        ]

    def phase_graph(self, name):
        """One phase of :meth:`phases`, recorded into a graph of its own.

        What the fourteen phase graphs were, on demand and one at a time. The
        emitters record into whatever capture is ambient, and every batch and
        every view they read was built by :meth:`_allocate`, so a phase can be
        re-recorded at any point after that for the price of its own nodes.

        It exists because two things were done with the phase graphs besides
        replaying them, and both are worth keeping. Replaying ONE phase is how
        ``test_lccsd.py`` checks Eq. 19's whole-store form against a per-pair
        reference and how it asserts the fat phases are batched rather than
        emitted per record. Bisecting a phase at a time is how the executor's
        last-bit effect was localized to Eq. 99-101 and Eq. 75-77 (see
        :class:`LCCSDSolver`), and a merged graph would have had no way to say
        which phase moved it.

        A phase recorded on its own does NOT reproduce the merged graph's
        schedule - the merged graph's levels are the whole iteration's - so a
        phase graph is a probe and a reference, never the thing that replays.
        """
        for label, emit, _threadable in self.phases():
            if label != name:
                continue
            g = cg.Graph(f"lccsd: {label}")
            with cg.capture(g):
                emit()
            return g
        raise KeyError(f"no phase named {name!r}; have "
                       f"{[label for label, _e, _t in self.phases()]}")

    def capture(self):
        """Record the whole iteration into ONE graph, once.

        The fourteen phases of :meth:`phases` are recorded back to back into a
        single graph, so what used to be fourteen graphs replayed in sequence by
        the host is fourteen stretches of one dependency graph. Two facts make
        that a trade rather than a leap.

        *The boundaries were never what carried the ordering.* A phase boundary
        orders everything before it against everything after it, which is
        strictly stronger than the equations need, so removing one can only let
        independent work overlap - it cannot let dependent work reorder. Every
        cross-phase order the sequence used to impose is a real data dependency
        the hazard scan finds instead: Eq. 94-97 reads the ``F_bar`` Eq. 98
        writes, Eq. 82 reads Eq. 91-92's dressed factors, Eq. 78-79 reads
        Eq. 83's and Eq. 84's intermediates, Eq. 19 reads what Eq. 78-79 leaves
        in ``Rn``, and the residual containers are cleared before Eq. 75-77
        accumulates into them by a write-after-write on the same store.

        *The four serial-by-design phases stay correct under one team, and stay
        bit-identical.* Their records accumulate scalars into ``(1, 1)`` element
        views of one occupied matrix, and there are only two cases. Records
        landing in DIFFERENT elements - Eq. 98's and Eq. 94's, one per pair -
        touch disjoint memory and may run at once. Records landing in the SAME
        element - Eq. 86's, one per neighbour of the pair - overlap, so the scan
        emits a write-after-write between them and they run in emission order on
        consecutive levels. Each element's sum therefore keeps its order term
        for term, which is what bit-identity needs, and it is the same argument
        :func:`_by_step` makes for the batched walks.

        One boundary is NOT folded away, and it is the load-bearing one. The
        amplitude step, the DIIS extrapolation, the antisymmetrization and the
        energy stay in graphs of their own after this one, because the legality
        of threading the residual is Jacobi's: within the residual a pair reads
        its neighbours' amplitudes and writes only its own output, which holds
        only as long as nothing in the same graph updates an amplitude. That is
        a property of the equations rather than of the hazard scan, so it is
        expressed as a boundary rather than trusted to an edge.

        The pooled rank-3 scratch needed no attention here, which is worth
        stating because it looks as though it should have: each of the six roles
        is used by exactly ONE phase (``V``, ``U`` and ``y`` by Eq. 99-101,
        ``bW`` by Eq. 82, ``Qab_t1`` and ``AW`` by Eq. 75-77), so the merge
        creates no chain through a shared buffer that a phase graph did not
        already have. The round-robin's serialization is unchanged.

        ``--einsums:graph:verify-levels`` is the check for the whole class of
        mistake a fold can make, and it passes on the merged graph on every
        fixture. It used to be unusable for a reason that was the checker's
        rather than the graph's: it modelled a ``view_rt`` node as writing the
        storage its view spans, where the node only fills in the view's own dims
        and strides, so any two views of one parent on one level were reported as
        a race. Idempotence (a second replay at fixed amplitudes, bit for bit)
        remains the check with the wider reach, since no level checker can see
        inside a batched node; ``test_lccsd.py`` runs both, and it also runs the
        merge against the sequence it replaced at fixed amplitudes.

        **One thing the fold could have changed silently, and did not.** A node
        alone on its level runs unwrapped and may thread its own contraction,
        where a node sharing a level runs inside the team with serial BLAS - so
        widening a level changes the internal summation order of whatever was
        alone on it, which is the effect ``use_executor`` above describes for
        Eq. 99-101 and Eq. 75-77. The level partition is not reachable from
        Python to be counted, but that effect is: it is exactly what moves the
        last bit at ten threads. Merged and sequential replays agree bit for bit
        at ten threads on water, the water dimer and methanol - same amplitudes,
        same residuals, same energy - so no node crossed that boundary.
        """
        t0 = time.perf_counter()
        g = cg.Graph("lccsd: iteration")
        with cg.capture(g):
            for _label, emit, _threadable in self.phases():
                emit()
        if self.use_executor:
            # An OpenMP team over the iteration's independent per-pair chains,
            # with serial BLAS inside each node. That is psi4's own parallelism
            # - `#pragma omp parallel for` over the pair list - and it is the
            # only safe way to get it from Python: a thread pool over the same
            # work returns silently wrong numbers against the OpenMP-built
            # OpenBLAS. See `base.py::_run`.
            g.set_executor(cg.OpenMPExecutor())
        #: The graphs replayed in order to evaluate the residual. One, now that
        #: the phases are folded; it stays a list because that is what
        #: :meth:`iterate` and :meth:`evaluate_residuals` walk, and because the
        #: step, the antisymmetrization and the energy are still graphs beside it.
        self._graphs = [g]

        self._g_step = cg.Graph("lccsd: Eq. 103-104 step")
        with cg.capture(self._g_step):
            self._emit_step()
        self._g_antisym = cg.Graph("lccsd: antisymmetrize")
        with cg.capture(self._g_antisym):
            self._emit_antisymmetrize()
        self._g_energy = cg.Graph("lccsd: Eq. 45 energy")
        with cg.capture(self._g_energy):
            self._emit_energy()

        self.t_capture = time.perf_counter() - t0
        return self

    def num_nodes(self):
        """Nodes across every captured graph, which is what the replay costs."""
        return (sum(g.num_nodes() for g in self._graphs)
                + self._g_step.num_nodes() + self._g_antisym.num_nodes()
                + self._g_energy.num_nodes())

    # -- replay ---------------------------------------------------------------

    def evaluate_residuals(self):
        """Replay everything up to but not including the amplitude update.

        Every phase before the Jacobi step is a pure function of the amplitudes,
        so this evaluates the port's equations at whatever amplitudes are
        currently in the stores. That is what makes the residual-level
        comparison against the canonical oracle possible at probe amplitudes
        rather than only at the solution; see ``check_ccsd_defect.py``.
        """
        self.prepare()
        for g in self._graphs:
            g.execute()
        return self.residuals()

    def residuals(self):
        """The current residuals as numpy, laid out the way the oracle wants."""
        cc = self.cc
        R1 = ten.view(self.R1_all)
        R_ia = [R1[:cc.n_pno[self.diag(i)], i:i + 1].copy()
                for i in range(self.naocc)]
        R_iajb = {}
        for ij in self._plan.live:
            n = cc.n_pno[ij]
            R_iajb[ij] = cc.pair_block(self.R_all, ij)[:n, :n].copy()
        return R_ia, R_iajb

    def _residual_norms(self):
        """Max RMS over the singles and over the doubles, on logical blocks only.

        The padding is identically zero and would otherwise dilute the norm.
        """
        cc = self.cc
        R1 = ten.view(self.R1_all)
        rms_s = 0.0
        for i in range(self.naocc):
            n = cc.n_pno[self.diag(i)]
            if n:
                block = R1[:n, i]
                rms_s = max(rms_s, float(np.sqrt(np.mean(block ** 2))))
        rms_d = 0.0
        for ij in self._plan.live:
            n = cc.n_pno[ij]
            block = cc.pair_block(self.R_all, ij)[:n, :n]
            rms_d = max(rms_d, float(np.sqrt(np.mean(block ** 2))))
        return rms_s, rms_d

    # -- psi4 DLPNOCCSD::lccsd_iterations ------------------------------------

    def prepare(self):
        """Plan, allocate and capture. Idempotent."""
        if self._graphs is None:
            self.plan()
            self._allocate()
            self.capture()
            self._report()
        return self

    def _report(self):
        p = self._plan
        self._print("\n  ==> Local CCSD <==\n")
        self._print(
            f"    plan:     {len(p.projections)} projections "
            f"({self._n_proj_extra} outside the neighbour lists), "
            f"{len(p.r_sym_kl)} Eq. 77/85 couplings, "
            f"{p.n_classes()} shape classes over {len(p.class_report())} terms")
        batches = sum(1 for group in self._bat.values() for b in group if b.a)
        members = sum(len(b.a) for group in self._bat.values() for b in group)
        self._print(f"    captured: {self.num_nodes()} nodes, "
                    f"{self._graphs[0].num_nodes()} of them in one iteration "
                    f"graph over {len(self.phases())} phases "
                    f"({self.t_plan * 1e3:.0f} ms planning, "
                    f"{self.t_capture * 1e3:.0f} ms capture)")
        self._print(f"    batched:  {members} GEMMs in {batches} grouped nodes")
        # The scalar families are reported beside the GEMMs because they were
        # the larger half of the node count and nothing else would show it: a
        # regression to one node per entry is bit-identical and silent.
        dots = sum(len(r) for r in self._run.values() if isinstance(r, _DotRun))
        adds = sum(len(r) for r in self._run.values()
                   if isinstance(r, _AxpbyRun))
        used = sum(1 for r in self._run.values() if len(r))
        self._print(f"    grouped:  {dots} dots and {adds} accumulations in "
                    f"{used} runs")
        where = ("the iteration graph over an OpenMP team; the step, the "
                 "antisymmetrization and the energy serial"
                 if self.use_executor else
                 "every graph serial (use_executor is off)")
        self._print(f"    executor: {where}, "
                    f"{_SCRATCH_POOL} buffers per shared scratch role")

    def iterate(self):
        """Drive the coupled-cluster equations to convergence.

        The doubles enter at the converged LMP2 values ``recompute_pnos`` left
        in the stores and the singles at zero, which is psi4's starting guess.

        The whole residual is one graph and one executor: an OpenMP team over
        the iteration's independent chains, which the hazard scan has already
        partitioned into levels across all fourteen phases rather than within
        each. :meth:`capture` states what the merge rests on. Nothing here nests
        an OpenMP team inside OpenBLAS's own threads - each node's BLAS call
        stays serial underneath the team, which is where the parallelism in this
        method actually is.

        What Python still does per iteration is the four things that cannot be
        graph nodes: the residual norms, the DIIS extrapolation between the step
        and the antisymmetrization, the convergence test, and the printed row.
        Folding those into an ``add_loop`` body would mean putting the amplitude
        step in the same graph as the residual, which is the one boundary
        :meth:`capture` keeps on purpose.
        """
        cc = self.cc
        self.prepare()

        if self.use_diis:
            self._diis = cg.diis(
                [(self.T1_all, self.step1_all)]
                + list(zip(cc.T_all, self.step_all)),
                k=cc.cut.diis_max_vecs)

        self._print(f"\n    {'iter':>4}  {'Corr. Energy':>18} {'Delta E':>12} "
                    f"{'Max R1':>10} {'Max R2':>10}")
        t0 = time.perf_counter()
        e_prev = 0.0
        for iteration in range(cc.cut.maxiter + 1):
            for g in self._graphs:
                g.execute()
            rms_s, rms_d = self._residual_norms()
            self._g_step.execute()
            if self._diis is not None:
                self._diis.step()
            self._g_antisym.execute()
            self._g_energy.execute()

            e_curr = float(ten.view(self.e_total)[0])
            e_weak = float(ten.view(self.e_weak)[0])
            self._print(f"    {iteration:>4}  {e_curr:>18.12f} "
                        f"{e_curr - e_prev:>12.3e} {rms_s:>10.3e} {rms_d:>10.3e}")
            converged = (iteration > 0
                         and abs(e_curr - e_prev) < cc.cut.e_convergence
                         and max(rms_s, rms_d) < cc.cut.r_convergence)
            e_prev = e_curr
            if converged:
                self.t_iterate = time.perf_counter() - t0
                self.n_iterations = iteration + 1
                self.e_lccsd = e_curr - e_weak
                self.de_weak = e_weak
                self._print(f"    replay:   {self.t_iterate:.3f} s over "
                            f"{self.n_iterations} iterations")
                return self
        raise RuntimeError("maximum DLPNO-CCSD iterations exceeded")


class _Batch:
    """One grouped GEMM: the operand lists, built once, replayed as one node.

    Built by :meth:`LCCSDSolver._build_batches` while no capture is open, so
    every view in the lists is a plain eager view, and emitted by a single
    :func:`~einsums.graph.grouped_batched_gemm` call.

    The prefactors and the transpose flags are properties of the CALL and not of
    the member, which is the one thing the grouping has to respect: a family
    whose records disagree on either needs more than one batch. The shapes need
    not agree at all - the grouped form sorts its members into uniform
    ``(m, n, k, lda, ldb, ldc)`` groups when it records the node - so nothing
    here consults the plan's shape classes.

    **What may share a batch, and why it matters.** The members of one grouped
    node may run CONCURRENTLY inside it, and the graph's hazard scan cannot see
    inside a node, so nothing that has to be ordered may share one. That means
    no two members may write the same destination, and no two may write the same
    scratch. Every batch below is built so that each member's destination and
    each member's temporaries belong to a different pair, which is also what
    makes an accumulation chain safe: a family that adds several terms into one
    pair's block is split into one batch per STEP of the walk (see
    :func:`_by_step`), never into one batch over the walk. The duplicate
    destination is additionally rejected by the binding, which is a check and not
    the argument.
    """

    __slots__ = ("alpha", "beta", "trans_a", "trans_b", "a", "b", "c")

    def __init__(self, alpha, beta, trans_a=False, trans_b=False):
        self.alpha = alpha
        self.beta = beta
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.a, self.b, self.c = [], [], []

    def add(self, a, b, c):
        self.a.append(a)
        self.b.append(b)
        self.c.append(c)

    def emit(self):
        """Record the batch, or nothing at all if it collected no member."""
        if self.a:
            cg.grouped_batched_gemm(self.alpha, self.a, self.b, self.beta,
                                    self.c, trans_a=self.trans_a,
                                    trans_b=self.trans_b)


class _DotRun:
    """One grouped dot: the operand lists, built once, replayed as one node.

    The counterpart of :class:`_Batch` for the reductions, and it exists for a
    different reason. A batch merges GEMMs to stop paying one OpenMP region per
    shape class; a run merges dots because a dot into a scalar has no arithmetic
    to spread at all and its whole cost is the node dispatch. The iteration
    reached about seventeen hundred of them.

    **What may share a run is a weaker condition than for a batch**, because
    :func:`~einsums.linalg.grouped_dot` runs its entries in SEQUENCE inside the
    node rather than possibly at once. So there is no distinct-destination rule
    here: entries land in the order they were added, exactly as the loop of
    single ``la.dot`` calls did, and each entry is the same kernel on the same
    operands. That is why merging a family moves no bit, and it is the property
    every gate in this port rests on.
    """

    __slots__ = ("r", "a", "b")

    def __init__(self):
        self.r, self.a, self.b = [], [], []

    def add(self, r, a, b):
        self.r.append(r)
        self.a.append(a)
        self.b.append(b)

    def __len__(self):
        return len(self.r)

    def emit(self):
        """Record the run, or nothing at all if it collected no entry."""
        if self.r:
            la.grouped_dot(self.r, self.a, self.b)


class _AxpbyRun:
    """One grouped axpby: ``Y_i = alpha_i X_i + beta_i Y_i``, as one node.

    The accumulating half of what :class:`_DotRun` reduces, and the prefactors
    are per entry, so terms that disagree on their coefficient still share a
    run.

    Sequential entries are what makes this usable on the families that
    genuinely collide. Eq. 86 adds one scalar per NEIGHBOUR into a single
    element of the shared ``F''``, and Eq. 45 adds one per pair into the energy;
    both sums keep their term order because the entries are applied in the order
    they were added and no two run at once. Where a family instead writes one
    destination per record the ordering is moot and the run is merely a node
    count.
    """

    __slots__ = ("alpha", "x", "beta", "y")

    def __init__(self):
        self.alpha, self.x, self.beta, self.y = [], [], [], []

    def add(self, alpha, x, beta, y):
        self.alpha.append(alpha)
        self.x.append(x)
        self.beta.append(beta)
        self.y.append(y)

    def __len__(self):
        return len(self.x)

    def emit(self):
        """Record the run, or nothing at all if it collected no entry."""
        if self.x:
            la.grouped_axpby(self.alpha, self.x, self.beta, self.y)


def _emit_all(batches):
    """Record a list of batches in order. One node each, at most."""
    for batch in batches:
        batch.emit()


def _by_step(records, owner):
    """Regroup a work list so step ``s`` holds each owner's ``s``-th record.

    The neighbour walks below all have the same shape: a record reads its
    partner pair and accumulates into its OWNER's intermediate, so the records
    of one owner are a chain and the records of different owners are
    independent. Transposing the walk that way - step outer, owner inner - makes
    every step a set of records with distinct owners, which is exactly a legal
    batch: distinct destinations, and distinct per-owner scratch.

    It also leaves every accumulation in its original order, which is why the
    energies do not move. Owner ``o``'s contributions still arrive as its record
    0, then its record 1, and so on, because step ``s`` is emitted before step
    ``s + 1`` and each owner appears at most once per step. Only the interleaving
    BETWEEN owners changes, and no sum runs across owners.

    Args:
        records: the work list, in the order the unbatched emitter walked it.
        owner: which field of a record identifies the accumulation it belongs to.
    """
    seen = {}
    steps = []
    for r in records:
        key = r[owner]
        s = seen.get(key, 0)
        seen[key] = s + 1
        if s == len(steps):
            steps.append([])
        steps[s].append(r)
    return steps


def _widest(records, owner, fields):
    """Per owner, the largest value of each field over its records.

    One pass rather than one scan of the whole list per owner. The obvious
    nesting is quadratic in the record count and at a six-monomer chain that is
    tens of millions of comparisons for an answer that is a running maximum.

    Args:
        records: the work list.
        owner: which field of a record identifies its owner.
        fields: ``(index, table)`` pairs; the width is ``table[record[index]]``,
            and a record whose index is ``-1`` does not contribute.
    """
    out = {}
    for r in records:
        row = out.get(r[owner])
        if row is None:
            row = out[r[owner]] = [1] * len(fields)
        for slot, (index, table) in enumerate(fields):
            key = r[index]
            if key != -1 and table[key] > row[slot]:
                row[slot] = table[key]
    return {k: tuple(v) for k, v in out.items()}


class _Plan:
    """The iteration's work lists, and the shape classes they fall into.

    A plain container rather than a dataclass because it is filled field by
    field as each term is planned, and because the only thing anything does with
    it is iterate over one of its lists.

    The shape classes describe how many distinct operand shapes each term
    presents. They were derived as the vocabulary a batched emitter would group
    on, and the emitter that arrived does not group on them - ``grouped_batched
    _gemm`` derives its own groups from the operands when it records the node, so
    one call covers every shape a family has. They stay because they are the
    plan's own account of the work and because the report prints them. The key is
    the LMP2 planner's - the PNO bucket of the pair and of its partner - extended
    by the neighbour count for the terms whose shape depends on it.
    """

    #: Per work list, the function from a record to its shape-class key.
    _KEYS = {
        "projections": lambda b, n, k, r: (b[r[0]], b[r[2]]),
        "t1_ints": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "fock_occupied": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "fock_bar": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "fock_kj": lambda b, n, k, r: (b[r[0]], b[r[3]]),
        "fock_kc": lambda b, n, k, r: (b[r[0]], b[r[1]]),
        "beta": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "gamma_head": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "gamma_tail": lambda b, n, k, r: (b[r[0]], b[r[2]]),
        "delta_head": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "delta_tail": lambda b, n, k, r: (b[r[0]], b[r[3]]),
        "fkj2": lambda b, n, k, r: (b[r[4]], b[r[5]]),
        "r_ia_ik": lambda b, n, k, r: (b[r[0]], b[r[4]]),
        "r_ia_a2": lambda b, n, k, r: (b[r[2]], b[r[5]]),
        "r_ia_b": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "r_sym": lambda b, n, k, r: (b[r[0]], len(k[r[0]])),
        "r_sym_kl": lambda b, n, k, r: (b[r[0]], b[r[3]]),
        "r_combine": lambda b, n, k, r: (b[r[0]],),
        "energy": lambda b, n, k, r: (b[r[0]],),
    }

    def __init__(self):
        self.live = []
        self.classes = {}
        self.cls = {}

    def classify(self, bucket_of, n_pno, lmos):
        """Intern each work list's shape classes and label every record."""
        for name, key in self._KEYS.items():
            records = getattr(self, name, None)
            if not records:
                continue
            seen, labels = {}, []
            for r in records:
                k = key(bucket_of, n_pno, lmos, r)
                labels.append(seen.setdefault(k, len(seen)))
            self.classes[name] = sorted(seen, key=seen.get)
            self.cls[name] = labels

    def n_classes(self):
        return sum(len(v) for v in self.classes.values())

    def class_report(self):
        """Per work list, how many records and how many shape classes."""
        return {name: (len(getattr(self, name)), len(cls))
                for name, cls in self.classes.items()}
