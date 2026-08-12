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
"""

import time

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import tensors as ten
from .base import DLPNOBase
from .cc_overlaps import build_overlaps

__all__ = ["LCCSDT", "permuter_spec"]

_run_setup_graph = DLPNOBase._run

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


class LCCSDT:
    """Iterate Eq. 111/112 to convergence over a prepared triplet list.

    Constructed from a :class:`~dlpno.triples.DLPNOCCSDT` and the
    :class:`~dlpno.lccsd_t0.LCCSDT0` that produced the semicanonical starting
    point with ``retain=True``, so ``W``, ``V`` and the initial amplitudes are
    already resident.
    """

    def __init__(self, cc, t0, verbose=True, use_diis=False, use_executor=True):
        self.cc = cc
        self.t0 = t0
        self.verbose = verbose
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
        self.F_lmo_np = np.asarray(ten.view(cc.F_lmo))
        self.e_ijk = np.array(t0.e_ijk, dtype=float)
        self.e_t = 0.0
        self.n_iterations = 0
        self.t_plan = 0.0
        self.t_iterate = 0.0
        self.n_nodes = 0
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
        use.
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
                    couplings.append((partner, -f, permuter_spec(*labels)))
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
        requests = {(r["ijk"], partner)
                    for r in self._plan for partner, _f, _spec in r["couplings"]}
        keys = sorted(requests)
        blocks = build_overlaps(list(cc.X_tno), cc.S_pao,
                               list(cc.lmotriplet_to_paos), list(cc.n_tno), keys)
        return dict(zip(keys, blocks))

    # -- allocation ----------------------------------------------------------

    def _allocate(self):
        """The residual and the denominator, per triplet.

        ``W``, ``V`` and the amplitudes come from the (T0) pass and are read
        in place. What is new is one residual block per triplet - the price of
        a Jacobi update - and one denominator, which is constant across
        iterations and so is built once rather than per pass.
        """
        cc = self.cc
        self.R = {}
        self.D = {}
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

    # -- emission ------------------------------------------------------------

    def _emit_residual(self, overlaps, records=None):
        """Eq. 111, for every triplet, or for the subset ``records`` names.

        The subset exists so the residual can be captured one triplet per
        graph, which is what lets a pass SKIP a triplet: a captured graph
        replays whole, so selective work needs selective graphs.

        ``R = W + T D`` and then one term per coupling. The coupled term is a
        rotation of the partner's amplitudes onto this triplet's TNOs, one
        GEMM per index, with the index permutation folded into the first of
        the three so the block is never permuted on its own.
        """
        cc = self.cc
        for r in (records if records is not None else self._plan):
            ijk, nt = r["ijk"], r["nt"]
            R, D = self.R[ijk], self.D[ijk]
            la.axpby(1.0, self.t0.W[ijk], 0.0, R)
            einsums.einsum("abc <- abc ; abc", R, self.t0.T[ijk], D, c_pf=1.0)

            live = [(p, w, s) for p, w, s in r["couplings"]
                    if overlaps.get((ijk, p)) is not None]
            if not live:
                continue

            # One scratch set per TRIPLET, not per coupling. Every coupling
            # writes these in full before reading them and then accumulates into
            # the same R, so they were already serialized by that dependency and
            # sharing costs no parallelism that existed. Per coupling it cost
            # three n_tno^3-scale blocks apiece - about ninety per triplet at
            # benchmark scale - every one of them captured and so alive for the
            # whole iteration.
            #
            # Sized to this triplet's widest partner and sliced per coupling,
            # taking a contiguous prefix and reshaping it rather than a strided
            # view, so each rotation still sees a dense block. Same idiom as
            # lccsd.py::_shared, and the same reason: the merged-axis reshapes
            # the rotations rely on are only valid on contiguous storage.
            widest = max(cc.n_tno[p] for p, _w, _s in live)
            b1 = ten.empty(f"S t (1) [{ijk}]", [nt * widest * widest])
            b2 = ten.empty(f"S t (2) [{ijk}]", [nt * nt * widest])
            b3 = ten.empty(f"S t (3) [{ijk}]", [nt * nt * nt])
            self._keep += [b1, b2, b3]

            for partner, weight, spec in live:
                S = overlaps[(ijk, partner)]
                np_ = cc.n_tno[partner]
                # The partner's block, permuted into this triplet's index
                # order and rotated one index at a time. The first einsum does
                # both: its right-hand side names the partner's axes in the
                # permuted order.
                t1 = b1[:nt * np_ * np_].reshape_view([nt, np_, np_])
                t2 = b2[:nt * nt * np_].reshape_view([nt, nt, np_])
                t3 = b3.reshape_view([nt, nt, nt])
                self._keep += [t1, t2, t3]
                einsums.einsum(f"xbc <- {spec} ; xa", t1, self.t0.T[partner], S)
                einsums.einsum("xyc <- xbc ; yb", t2, t1, S)
                einsums.einsum("xyz <- xyc ; zc", t3, t2, S)
                la.axpby(weight, t3, 1.0, R)

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

    def iterate(self):
        """Drive Eq. 111/112 to convergence and return the (T) energy."""
        cc = self.cc
        if self._plan is None:
            self.plan()
        if not self._plan:
            return 0.0

        self._allocate()
        overlaps = self._overlaps()

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

        skipping = bool(cc.cut.t_skip_converged) and float(cc.cut.t_cut_iter) > 0.0
        self._active = {r["ijk"]: True for r in self._plan}

        # Both phases are ONE graph each, with per-triplet work gated behind a
        # conditional when skipping is on. The obvious alternative - a graph
        # per triplet, driven from a Python loop - also skips, and was what
        # this did first, but it forfeits the thing that matters more: a Python
        # loop cannot be threaded (the OpenMP-built OpenBLAS returns silently
        # wrong numbers from caller-created threads), so the phase could skip
        # or thread but not both. One graph keeps the executor's view of every
        # triplet and lets it spread them over cores.
        #
        # Residual and step stay SEPARATE graphs, and that separation is what
        # makes the threading legal rather than merely convenient. Inside the
        # residual every triplet reads its neighbours' amplitudes and writes
        # only its own R, so the triplets are independent; a step in the same
        # graph would write T while another triplet's residual was reading it.
        # Jacobi is what buys the parallelism, and only across a phase
        # boundary.
        def _gate(ijk):
            return lambda: self._active[ijk]

        g_residual = cg.Graph("lccsd(t): Eq. 111 residual")
        g_step = cg.Graph("lccsd(t): Eq. 112 step")
        if skipping:
            for r in self._plan:
                branch, _ = g_residual.add_conditional(f"residual [{r['ijk']}]",
                                                       _gate(r["ijk"]))
                with cg.capture(branch):
                    self._emit_residual(overlaps, [r])
                branch, _ = g_step.add_conditional(f"step [{r['ijk']}]",
                                                   _gate(r["ijk"]))
                with cg.capture(branch):
                    self._emit_update([r])
        else:
            with cg.capture(g_residual):
                self._emit_residual(overlaps)
            with cg.capture(g_step):
                self._emit_update()
        self.n_nodes += g_residual.num_nodes() + g_step.num_nodes()

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

        self._print(f"\n  ==> Local CCSD(T) <==\n")
        self._print(f"    plan:     {len(self._plan)} triplets, "
                    f"{sum(len(r['couplings']) for r in self._plan)} couplings, "
                    f"{self.n_nodes} captured nodes"
                    + (f", skipping at t_cut_iter {cc.cut.t_cut_iter:.1e}"
                       if skipping else ""))
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
                active = [r for r in self._plan
                          if abs(self.e_ijk[r["ijk"]] - e_ijk_old[r["ijk"]])
                          >= abs(e_ijk_old[r["ijk"]] * cc.cut.t_cut_iter)]
                self.n_skipped += len(self._plan) - len(active)
                # The gates the conditionals read. Set before the replay, so
                # one graph execution does exactly the active triplets.
                live = {r["ijk"] for r in active}
                for ijk in self._active:
                    self._active[ijk] = ijk in live
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
                            f"{self.n_iterations} iterations")
                return e_curr
        raise RuntimeError("maximum DLPNO-(T) iterations exceeded")
