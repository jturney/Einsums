#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The DLPNO foundation, ported from psi4's ``dlpno/dlpno.cc``.

:class:`DLPNOBase` is the analogue of psi4's ``DLPNO`` class: it builds the
localized occupied orbitals and projected atomic orbitals, decides which LMO
pairs and which PAO/auxiliary domains matter, and transforms each surviving pair
into its own truncated pair-natural-orbital basis. The MP2, CCSD and (T) layers
all sit on top of exactly these quantities.

All three of psi4's truncations are here: differential-overlap PAO domains,
Mulliken auxiliary domains, and dipole pair prescreening
(``DLPNO::compute_overlap_ints``, ``DLPNO::compute_dipole_ints``,
``DLPNO::prep_sparsity``). Switching them off - which
:meth:`~dlpno.thresholds.Thresholds.untruncated` does, and which is also what
happens when no integration grid is available - gives every pair the full PAO
and auxiliary domain, and local MP2 becomes exactly canonical DF-MP2. That
equivalence is what pins the port down, so it is kept as a supported mode
rather than a historical stage.
"""

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from . import integrals
from . import sparse
from . import tensors as ten
from .layout import PairLayout
from .thresholds import Thresholds

__all__ = ["DLPNOBase", "plan_widths"]


def plan_widths(graph, dataflow=False):
    """Give a solver graph's nodes thread widths, and bind the executor that
    honors them.

    :meth:`~einsums.graph.Graph.plan_threads` chooses a width per node from the
    cost model, records the thread count it planned against, and arms one
    re-plan from the first replay's recorded timings; only
    :class:`~einsums.graph.DataflowExecutor` reads the result, because only its
    workers are separate OpenMP initial threads and so can each carry a
    different thread count. That is what replaces the all-or-nothing split the
    solvers used to live with: under
    :class:`~einsums.graph.OpenMPExecutor` a node ALONE on its execution level
    runs unwrapped and may thread its own kernel, while a node sharing a level
    runs inside the team with serial BLAS, so a purely structural accident
    decided whether a fat contraction got the machine.

    Planning can decline - a graph of nodes all below the planner's serial-time
    floor is not worth widening, and widening it would only pay fork costs - and
    that is the case this falls back for. A declined graph keeps the OpenMP
    executor it had before, which is not a consolation prize: the solo-level
    rule is exactly the right answer for the four one-node batched graphs of
    ``lccsd_t0.py``, and for a graph of many small independent nodes the team
    over nodes is the parallelism there is. So the graph is either moldable or
    it is what it already was, and never silently serial.

    Pass ``dataflow`` to take the dataflow executor even on a declined graph.
    That is the right choice for a graph whose parallelism is the COUNT of its
    nodes rather than any node's width - many independent chains, each one
    narrow - and it is not merely a preference for speed. The solo-level rule
    the fallback relies on decides a kernel's thread count from what else shares
    its level, so a declined graph of many nodes under the OpenMP team gets a
    thread count that moves with scheduling, and a reduction's thread count is
    its summation order. The dataflow executor honors the plan's width of one on
    every node instead, which pins the arithmetic.

    Returns whether any node was given a width above one, whichever executor was
    bound.
    """
    widened = graph.plan_threads()
    graph.set_executor(cg.DataflowExecutor() if widened or dataflow
                       else cg.OpenMPExecutor())
    return widened


class DLPNOBase:
    """Orbitals, domains, and pair natural orbitals for a DLPNO calculation."""

    def __init__(self, reference, thresholds=None, verbose=True, integral_source=None):
        self.ref = reference.validate()
        self.cut = thresholds if thresholds is not None else Thresholds()
        self.verbose = verbose
        # Where (Q|i u) comes from. The dense source transforms a materialized
        # (Q|mn) and is exact, which is what makes it the default and the oracle
        # anything else is checked against; see dlpno.integrals.
        # Explicit argument, then whatever the reference carries, then the dense
        # transform of a materialized (Q|mn). Only the last needs the array.
        if integral_source is not None:
            self.integrals = integral_source
        elif getattr(self.ref, "integral_source", None) is not None:
            self.integrals = self.ref.integral_source
        else:
            self.integrals = integrals.DenseSource(self.ref.eri_3index)

        # setup_orbitals
        self.C_lmo = self.F_lmo = None
        self.C_pao = self.S_pao = self.F_pao = None

        # compute_doi
        self.doi_ij = None
        self.doi_iu = None

        # prep_sparsity
        self.i_j_to_ij = None
        self.ij_to_i_j = []
        self.ij_to_ji = []
        self.lmo_to_paos = []
        self.lmo_to_ribfs = []
        self.lmopair_to_paos = []
        self.lmopair_to_ribfs = []
        self.lmopair_to_lmos = []
        self.lmopair_to_lmos_dense = None
        self.dipole_pair_e = None
        self.dipole_pair_e_bound = None
        #: Correlation energy of the pairs dropped by prescreening, added back
        #: to the total (psi4's ``de_dipole_``).
        self.de_dipole = 0.0
        #: One canonical object per distinct domain; see _intern.
        self._interned = {}

        # compute_metric / compute_qia / compute_qij / compute_qab
        self.metric = None
        self.q_ia = None
        self.q_ij = None
        self.q_ab = None

        # Per-domain results shared across pairs (see _fit_coefficients and
        # _canonical_pao_domain). Screening multiplies the number of distinct
        # domains, which is exactly why the domain lists are interned.
        # The fits are keyed by (domain, LMO) and hold ``(parent, position)``:
        # one solve produces a domain's whole block and each LMO's fit is a
        # column slice of it, so the entry carries the parent to keep it alive.
        self._fit_cache = {}
        self._pao_domain_cache = {}
        self._pass_manager = None

        # pno_transform. The per-pair amplitude/integral blocks live in flat
        # rank-3 stores with the PAIR INDEX TRAILING; see _allocate_pair_stores.
        self.npno_max = 0
        self.layout = None
        self.K_all = None
        self.T_all = None
        self.Tt_all = None
        self.X_pno = []
        self.e_pno = []
        self.n_pno = []
        self.de_pno = []
        self.de_pno_os = []
        self.de_pno_ss = []
        self.de_pno_total = 0.0
        self.de_pno_total_os = 0.0
        self.de_pno_total_ss = 0.0

    def pass_manager(self):
        """The default optimization pipeline, built once.

        ``PassManager.populate_default()`` costs ~40 ms, an order of magnitude
        more than applying the result to a graph that was already emitted in
        fused form (~2 ms). Building one per graph made pipeline construction
        the single largest line item in the overlap phase.
        """
        if self._pass_manager is None:
            pm = cg.PassManager()
            pm.populate_default()
            self._pass_manager = pm
        return self._pass_manager

    # -- reporting ---------------------------------------------------------

    def _print(self, *args):
        if self.verbose:
            print(*args, flush=True)

    @property
    def n_lmo_pairs(self):
        return len(self.ij_to_i_j)

    # -- psi4 DLPNO::setup_orbitals ----------------------------------------

    def setup_orbitals(self):
        """Localized occupied orbitals and projected atomic orbitals.

        The LMOs arrive already localized (see ``psi4_source.localize``); what
        happens here is the PAO construction, which removes the occupied space
        from the AO basis, ``C_pao = 1 - C_occ C_occ^T S``, normalizes the
        result, and builds the PAO overlap and Fock matrices. The PAOs are
        linearly dependent by construction (``nbf`` of them spanning ``nvir``
        dimensions); ``s_cut`` removes the redundancy per domain later.
        """
        ref = self.ref
        S = ten.from_numpy("S (AO)", ref.S)
        F = ten.from_numpy("F (AO)", ref.F)
        C_occ = ten.from_numpy("C (occ)", ref.C_occ)

        self.C_lmo = ten.from_numpy("C (LMO)", ref.C_lmo)
        self.F_lmo = ten.triplet(self.C_lmo, F, self.C_lmo, trans_a=True, name="F (LMO)")

        C_pao = einsums.eye(ref.nbf, dtype="float64", name="C (PAO)")
        la.axpby(-1.0, ten.triplet(C_occ, C_occ, S, trans_b=True), 1.0, C_pao)

        # Normalize each PAO against its own overlap, then rebuild S and F.
        S_pao = ten.triplet(C_pao, S, C_pao, trans_a=True, name="S (PAO)")
        self.C_pao = ten.scale_columns(
            C_pao, ten.view(ten.diagonal(S_pao)) ** -0.5, name="C (PAO)"
        )
        self.S_pao = ten.triplet(self.C_pao, S, self.C_pao, trans_a=True, name="S (PAO)")
        self.F_pao = ten.triplet(self.C_pao, F, self.C_pao, trans_a=True, name="F (PAO)")

        self._print(
            f"  orbitals: {ref.naocc} active occupied LMOs, "
            f"{ten.shape(self.C_pao)[1]} PAOs, {ref.naux} auxiliary functions"
        )
        return self

    # -- psi4 DLPNO::prep_sparsity -----------------------------------------

    def _intern(self, indices):
        """Return one canonical list object per distinct index list.

        Load-bearing rather than a micro-optimization. The domain caches
        (:meth:`_fit_coefficients`, :meth:`_canonical_pao_domain`) are keyed on
        ``id()`` of these lists, so two pairs with *equal* domains only share a
        cache entry if they are handed the *same* object. Without screening
        there was one domain and one object; merging per pair would produce
        thousands of equal-but-distinct lists and silently turn every cache hit
        into a miss, which is most of the setup cost.
        """
        key = tuple(indices)
        hit = self._interned.get(key)
        if hit is None:
            hit = list(indices)
            self._interned[key] = hit
        return hit

    # -- psi4 DLPNO::compute_overlap_ints ----------------------------------

    def compute_doi(self):
        """Differential overlap integrals, by numerical quadrature.

        ``DOI_ij = sqrt(int |phi_i(r)|^2 |phi_j(r)|^2 dr)`` measures how much
        two orbitals occupy the same space, which is what decides both the PAO
        domains and which LMO pairs survive. It is the one quantity here that
        cannot be had from the AO integrals alone, so it needs a grid; the
        blocks stream in from the bridge (see
        :func:`~dlpno.psi4_source.grid_block_provider`).

        Accumulated over blocks as two GEMMs each, which is the whole cost:
        ``(points x naocc)^T (points x naocc)`` and the same against the PAOs.

        The blocks are tiny - a six-monomer water chain streams 186 of them
        averaging 84 points - so the arithmetic is nowhere near the cost.
        Issued one block at a time this was seven eager dispatches per block
        against a few hundred microseconds of work, and a dispatch is several
        microseconds before it reaches a kernel. Capturing the same seven calls
        into a graph does not help, because capture costs what dispatch costs;
        the only thing that helps is making fewer, larger calls. So the whole
        stream runs as SEVEN calls total:

        * the two collocations stay per block, because a block's ``phi`` only
          covers the basis functions that are non-negligible on it, and that
          sparsity is worth more than uniformity. They go out as one
          :func:`~einsums.graph.grouped_batched_gemm` each, which takes members
          of differing shapes and runs the whole set under a single OpenMP
          region;
        * squaring and weighting are elementwise, so they run once over the
          concatenated stream rather than once per block;
        * the two contractions back down to ``naocc`` are again per block, as
          one grouped batch each, writing per-block contributions that
          :meth:`_reduce_doi_blocks` sums.

        Every block's arithmetic is the GEMM it always was, so this is bit for
        bit the sequential result, which is the bar: DOI decides which PAOs
        enter each domain and which pairs survive.
        """
        ref = self.ref
        naocc = ref.naocc
        npao = ten.shape(self.C_pao)[1]

        if self.doi_ij is not None:      # idempotent: prep_sparsity self-serves
            return self
        if ref.grid_blocks is None:
            self.doi_ij = self.doi_iu = None
            return self

        C_lmo = ten.view(self.C_lmo)
        C_pao = ten.view(self.C_pao)

        # Host side first: drain the stream, gathering each block's slice of the
        # coefficients. Nothing numeric happens in this loop, and draining is
        # safe here for the reason it is not in general - a block is a few tens
        # of kilobytes and :func:`~dlpno.tensors.from_numpy` copies it, so the
        # provider's buffer reuse (see grid_block_provider) is respected.
        phis, c_lmos, c_paos, weights, extents = [], [], [], [], []
        npoints = 0
        for phi_np, w_np, bf_map in ref.grid_blocks():
            npts = int(phi_np.shape[0])
            if npts == 0:
                continue
            # The gather seam again: the block's basis functions are an index
            # list into the global coefficients. Gathering the whole stream at
            # once instead was measured and is a wash - the cost is the strided
            # read, not the call - so this stays the simpler shape.
            phis.append(ten.from_numpy("phi", phi_np))
            c_lmos.append(ten.from_numpy("C_lmo blk", C_lmo[bf_map, :]))
            c_paos.append(ten.from_numpy("C_pao blk", C_pao[bf_map, :]))
            weights.append(w_np)
            extents.append((npoints, npoints + npts))
            npoints += npts

        # One tensor per quantity over the whole stream, with a view per block
        # where the per-block GEMMs write and read. The views carry the parent's
        # leading dimension, which changes no value the GEMM computes.
        w_all = ten.from_numpy("w", np.concatenate(weights) if weights else np.zeros(0))
        lmo = ten.empty("phi_i", [npoints, naocc])
        pao = ten.empty("phi_u", [npoints, npao])
        lmo2 = ten.empty("|phi_i|^2", [npoints, naocc])
        pao2 = ten.empty("|phi_u|^2", [npoints, npao])
        lmo2_w = ten.empty("w |phi_i|^2", [npoints, naocc])
        lmo_v = [lmo[a:b, :] for a, b in extents]
        pao_v = [pao[a:b, :] for a, b in extents]
        lmo2_v = [lmo2[a:b, :] for a, b in extents]
        pao2_v = [pao2[a:b, :] for a, b in extents]
        lmo2_w_v = [lmo2_w[a:b, :] for a, b in extents]
        part_ij = [ten.empty("DOI (i,j) block", [naocc, naocc]) for _ in extents]
        part_iu = [ten.empty("DOI (i,u) block", [naocc, npao]) for _ in extents]

        if extents:
            cg.grouped_batched_gemm(1.0, phis, c_lmos, 0.0, lmo_v)
            cg.grouped_batched_gemm(1.0, phis, c_paos, 0.0, pao_v)

            # Squared orbital values, and a weighted copy of the LMO side. The
            # quadrature weight belongs to exactly one of the two factors.
            einsums.einsum("pi <- pi ; pi", lmo2, lmo, lmo)
            einsums.einsum("pu <- pu ; pu", pao2, pao, pao)
            einsums.einsum("pi <- pi ; p", lmo2_w, lmo2, w_all)

            cg.grouped_batched_gemm(1.0, lmo2_w_v, lmo2_v, 0.0, part_ij, trans_a=True)
            cg.grouped_batched_gemm(1.0, lmo2_w_v, pao2_v, 0.0, part_iu, trans_a=True)

        doi_ij, doi_iu = self._reduce_doi_blocks(part_ij, part_iu, naocc, npao)

        # sqrt(abs(...)): both are correctly rounded and elementwise, so doing
        # them on the reduced arrays is the same arithmetic linalg.abs and
        # linalg.sqrt did on the tensors. (linalg.pow is a *matrix* power via
        # eigendecomposition, not this.)
        self.doi_ij = np.sqrt(np.abs(doi_ij))
        self.doi_iu = np.sqrt(np.abs(doi_iu))
        self._print(f"  DOI:      {len(extents)} grid blocks, {npoints} points")
        return self

    @staticmethod
    def _reduce_doi_blocks(part_ij, part_iu, naocc, npao):
        """Sum the per-block DOI contributions, in block order.

        Deliberately a plain in-order loop of adds rather than anything
        cleverer, and deliberately on the host. Only strict left-to-right
        accumulation reproduces what the sequential loop's ``c_pf=1.0`` einsums
        did, and reproducing it is what makes the rest of this phase checkable
        as a pure reordering. Every alternative reassociates: striping the
        blocks across threads, a contraction against a vector of ones (whose
        dot kernel carries several SIMD partial sums), and numpy's own ``sum``
        (pairwise) all give a different answer in the last bits.

        Staying in the tensor layer would mean two ``axpby`` calls per block,
        and per-call dispatch is the very cost this phase was rewritten to
        avoid: the adds themselves are under a millisecond for the whole
        stream, while 372 eager dispatches are several.
        """
        doi_ij = np.zeros((naocc, naocc))
        doi_iu = np.zeros((naocc, npao))
        for block_ij, block_iu in zip(part_ij, part_iu):
            doi_ij += ten.view(block_ij)
            doi_iu += ten.view(block_iu)
        return doi_ij, doi_iu

    # -- psi4 DLPNO::compute_dipole_ints -----------------------------------

    def _dipole_pair_energies(self):
        """Semicanonical MP2 pair energies from the dipole approximation.

        Two well-separated orbitals interact through their transition dipoles,
        so a pair energy can be estimated at O(1) cost per pair without any
        integrals over the pair. ``dipole_pair_e_bound`` is the same expression
        with the orientation factor replaced by its largest possible value, so
        it is an upper bound and is what the screening tests: dropping a pair
        because the *bound* is small cannot discard a large true energy.

        Returns ``(dipole_pair_e, dipole_pair_e_bound)``, both ``naocc^2``.
        """
        ref = self.ref
        naocc = ref.naocc
        F_lmo = ten.view(self.F_lmo)

        C_lmo, C_pao = self.C_lmo, self.C_pao
        dip_ii, dip_iu = [], []
        for x in range(3):
            D = ten.from_numpy(f"dipole {x} (AO)", np.ascontiguousarray(ref.dipole_ao[x]))
            dip_ii.append(
                ten.view(ten.diagonal(ten.triplet(C_lmo, D, C_lmo, trans_a=True))).copy())
            dip_iu.append(ten.view(ten.triplet(C_lmo, D, C_pao, trans_a=True)).copy())

        # Orbital centroids <i|r|i>.
        R_i = np.stack(dip_ii, axis=1)  # naocc x 3

        # Per-LMO transition dipoles into its own orthocanonical PAO domain.
        # A looser DOI cutoff than the real domains: this is a prescreen, and
        # it must not be more aggressive than what it is screening for.
        # Every domain below is known before any of the work runs: compute_doi
        # produced doi_iu already, and nothing here feeds back into it. So the
        # whole per-LMO sequence batches, which is worth doing for the reason
        # :meth:`_warm_pao_domains` gives - one domain-sized eigendecomposition
        # at a time gets SLOWER as threads are added, because each syev is small
        # enough to lose to the BLAS's own internal threading.
        live, dom_paos, S_doms, F_doms = [], [], [], []
        for i in range(naocc):
            paos = [u for u in range(self.doi_iu.shape[1])
                    if abs(self.doi_iu[i, u]) > self.cut.t_cut_do_pre]
            paos = sparse.contract_lists(paos, ref.atom_to_bf)
            if not paos:
                continue
            live.append(i)
            dom_paos.append(paos)
            S_doms.append(sparse.submatrix_rows_and_cols(self.S_pao, paos, paos))
            F_doms.append(sparse.submatrix_rows_and_cols(self.F_pao, paos, paos))

        canon = batched_orthocanonicalizer(S_doms, F_doms, self.cut.s_cut, self._run)

        # The transition dipoles, rotated into each domain's orthocanonical
        # basis. One GEMM per (LMO, Cartesian component), which is the same
        # 1 x npao_dom by npao_dom x nkeep call the sequential loop made - the
        # three components deliberately stay separate rather than stacking into
        # one operand, because a different GEMM shape is a different kernel and
        # a different summation order. Keeping the calls identical makes this
        # change a pure reordering, which the prescreen is strict enough to
        # want: it decides which pairs exist, so bit equality is checkable and
        # a near-miss is not.
        jobs = []
        for slot, i in enumerate(live):
            X_i = canon[slot][0]
            nkeep = ten.shape(X_i)[1]
            for x in range(3):
                d = ten.from_numpy("d", np.ascontiguousarray(
                    dip_iu[x][[i], :][:, dom_paos[slot]]))
                jobs.append((d, X_i, ten.zeros("dipole (orthocanonical)", [1, nkeep])))

        if jobs:
            g = cg.Graph("dipole prescreen projections")
            with cg.capture(g):
                for d, X_i, out in jobs:
                    la.gemm(1.0, d, X_i, 0.0, out)
            self._run(g)

        lmo_dr = [np.zeros((0, 3)) for _ in range(naocc)]
        lmo_e = [np.zeros(0) for _ in range(naocc)]
        for slot, i in enumerate(live):
            lmo_dr[i] = np.stack(
                [ten.view(jobs[3 * slot + x][2])[0] for x in range(3)], axis=1)
            lmo_e[i] = np.asarray(ten.view(canon[slot][1])).copy()

        e_pair = np.zeros((naocc, naocc))
        e_bound = np.zeros((naocc, naocc))
        for i in range(naocc):
            for j in range(i + 1, naocc):
                R_ij = R_i[i] - R_i[j]
                norm = np.linalg.norm(R_ij)
                if norm == 0.0 or lmo_dr[i].size == 0 or lmo_dr[j].size == 0:
                    continue
                Rh = R_ij / norm
                iu, jv = lmo_dr[i], lmo_dr[j]

                dot = iu @ jv.T                                   # nu x nv
                num_actual = (dot - 3.0 * np.outer(iu @ Rh, jv @ Rh)) ** 2
                num_linear = 4.0 * dot ** 2
                denom = (lmo_e[i][:, None] + lmo_e[j][None, :]
                         - (F_lmo[i, i] + F_lmo[j, j]))
                scale = -4.0 * norm ** -6

                e_pair[i, j] = e_pair[j, i] = scale * np.sum(num_actual / denom)
                e_bound[i, j] = e_bound[j, i] = scale * np.sum(num_linear / denom)
        return e_pair, e_bound

    # -- the two LMO domain constructions ----------------------------------
    #
    # Both take their threshold as an argument rather than reading ``self.cut``,
    # because a coupled-cluster triples calculation builds each of them THREE
    # times at three different thresholds: once for the pairs
    # (``t_cut_mkn`` / ``t_cut_do``), once for the triples prescreening pass
    # (``*_triples_pre``) and once for the production triples pass
    # (``*_triples``). psi4 duplicates the loops in ``triples_sparsity``; here
    # they are the same code called with different numbers, which is the only
    # way the two passes cannot drift apart.

    def mulliken_aux_domains(self, t_cut_mkn):
        """Per LMO, the auxiliary functions of the atoms it has population on.

        psi4's fitting-domain construction: build the Mulliken population
        matrix of one LMO, split each off-diagonal element between its two
        centres in proportion to their diagonal populations, and keep every
        atom above the threshold, all-or-nothing over that atom's auxiliary
        functions.

        With no grid available - the untruncated reference calculation - every
        LMO gets the full auxiliary basis, matching the pair path.
        """
        ref = self.ref
        naocc, naux = ref.naocc, ref.naux
        if self.doi_ij is None:
            return [self._intern(range(naux))] * naocc

        C_lmo = ten.view(self.C_lmo)
        S = np.asarray(ref.S)
        natom = len(ref.atom_to_bf)
        bf_to_atom = np.empty(ref.nbf, dtype=int)
        for a, bfs in enumerate(ref.atom_to_bf):
            bf_to_atom[list(bfs)] = a

        out = []
        for i in range(naocc):
            c = C_lmo[:, i]
            P = c[:, None] * S * c[None, :]
            d = np.diag(P)
            denom = d[:, None] + d[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                share_u = np.where(denom != 0.0, P * (d[:, None] / denom), 0.0)
                share_v = np.where(denom != 0.0, P * (d[None, :] / denom), 0.0)
            pop = np.zeros(natom)
            np.add.at(pop, bf_to_atom, share_u.sum(axis=1))
            np.add.at(pop, bf_to_atom, share_v.sum(axis=0))

            ribfs = []
            for a in range(natom):
                if abs(pop[a]) > t_cut_mkn:
                    ribfs.extend(ref.atom_to_ribf[a])
            out.append(self._intern(sorted(ribfs)))
        return out

    def doi_pao_domains(self, t_cut_do):
        """Per LMO, the PAOs it has differential overlap with, whole atoms.

        With no grid available every LMO gets every PAO, which is the
        untruncated calculation.
        """
        ref = self.ref
        naocc = ref.naocc
        npao = ten.shape(self.C_pao)[1]
        if self.doi_ij is None:
            return [self._intern(range(npao))] * naocc

        out = []
        for i in range(naocc):
            paos = [u for u in range(npao) if abs(self.doi_iu[i, u]) > t_cut_do]
            # Any PAO on an atom pulls in all of that atom's PAOs.
            out.append(self._intern(sparse.contract_lists(paos, ref.atom_to_bf)))
        return out

    # -- psi4 DLPNO::prep_sparsity -----------------------------------------

    def prep_sparsity(self, initial=True, final=False):
        """Significant LMO pairs and their PAO/auxiliary domains.

        This is where "domain-based local" stops being a name. Three independent
        truncations, all of them psi4's:

        * **PAO domains.** LMO ``i`` keeps the PAOs it has differential overlap
          with (``t_cut_do``), rounded up to whole atoms.
        * **Auxiliary domains.** LMO ``i`` keeps the fitting functions of atoms
          carrying Mulliken population from it (``t_cut_mkn``), again
          all-or-nothing per atom. This is what makes the density fitting local.
        * **Pair screening.** A pair survives if the two orbitals overlap
          (``t_cut_do_ij``) or their dipole-estimated energy bound is
          non-negligible (``t_cut_pre``). Diagonal pairs always survive. What is
          dropped is not ignored: its dipole pair energy is accumulated into
          ``de_dipole`` and added to the correlation energy.

        A pair's domains are the union of the two orbitals' domains, so the
        merged lists are interned (see :meth:`_intern`) to keep the per-domain
        caches effective.

        With no grid available, or with the thresholds switched off, every pair
        and every domain is complete and this reduces to the untruncated
        calculation that validates against canonical DF-MP2.

        **The two flags are psi4's, and a coupled-cluster calculation uses all
        three of their meaningful combinations.** MP2 calls this once, at the
        defaults, and gets everything. CC calls it three times, and what it
        wants each time is different:

        ``prep_sparsity(True, False)``
            The crude pass, at deliberately loosened thresholds. Everything is
            built, including the dipole screen that decides the pair list.

        ``prep_sparsity(False, False)``
            The refined pass, at full thresholds. Both domain families are
            rebuilt tighter, but the pair list is NOT: crude prescreening has
            already eliminated pairs from it, and rerunning the dipole screen
            would put them back.

        ``prep_sparsity(False, True)``
            The final pass, after the strong/weak split. Only the PAO domains
            and everything derived from the pair list are rebuilt. The auxiliary
            domains are deliberately left alone, because the three-index
            integrals already built are indexed against them and rebuilding one
            without the other would silently mismatch.

        Returns ``self``.

        Args:
            initial: Run the dipole prescreen and (re)build the pair list.
            final: Skip the auxiliary domains, keeping them consistent with the
                integrals already built against them.
        """
        ref = self.ref
        naocc = ref.naocc
        npao = ten.shape(self.C_pao)[1]
        naux = ref.naux

        # Self-serving rather than relying on the caller to have run
        # compute_doi first. Forgetting it would not raise; it would silently
        # disable every domain truncation and quietly solve a bigger problem,
        # which is exactly the failure that is hard to notice in a benchmark.
        if self.doi_ij is None:
            self.compute_doi()
        screening = self.doi_ij is not None

        # A rerun invalidates both domain memos. They are keyed by ``id()`` of
        # an interned domain list, and re-interning at a different threshold
        # yields different objects, so a stale entry cannot be *reached* - but
        # a domain that comes out identical WOULD be reached, and then the
        # entry's correctness rests on the fit and the PAO decomposition not
        # depending on anything that changed. That happens to hold, and it is
        # not a thing to depend on silently.
        if self.i_j_to_ij is not None:
            self._fit_cache.clear()
            self._pao_domain_cache.clear()

        # -- LMO -> auxiliary domain, by Mulliken population ----------------
        # Skipped in the final pass: see the flag discussion above.
        if not final:
            self.lmo_to_ribfs = self.mulliken_aux_domains(self.cut.t_cut_mkn)

        # -- LMO -> PAO domain, by differential overlap ---------------------
        self.lmo_to_paos = self.doi_pao_domains(self.cut.t_cut_do)

        # -- which (i, j) pairs survive -------------------------------------
        # Only in the initial pass. Later passes inherit the pair list, which by
        # then carries the crude prescreening's eliminations as well as the
        # dipole screen's.
        if initial:
            self.de_dipole = 0.0
            # The estimate is only worth building when a pair can actually be
            # rejected. Under `untruncated()` the thresholds are negative, so
            # every pair survives and it would be an expensive unused quantity:
            # one orthocanonicalization of the *full* PAO space per LMO.
            prescreen = screening and (self.cut.t_cut_pre >= 0.0 or self.cut.t_cut_do_ij >= 0.0)
            if prescreen:
                self.dipole_pair_e, self.dipole_pair_e_bound = self._dipole_pair_energies()
            else:
                self.dipole_pair_e = self.dipole_pair_e_bound = np.zeros((naocc, naocc))

            self.i_j_to_ij = np.full((naocc, naocc), -1, dtype=int)
            self.ij_to_i_j = []
            for i in range(naocc):
                for j in range(naocc):
                    keep = (i == j) or not prescreen or (
                        self.doi_ij[i, j] > self.cut.t_cut_do_ij
                        or abs(self.dipole_pair_e_bound[i, j]) > self.cut.t_cut_pre
                    )
                    if keep:
                        self.i_j_to_ij[i, j] = len(self.ij_to_i_j)
                        self.ij_to_i_j.append((i, j))
                    else:
                        self.de_dipole += self.dipole_pair_e[i, j]
            self.ij_to_ji = [int(self.i_j_to_ij[j, i]) for (i, j) in self.ij_to_i_j]
        elif self.i_j_to_ij is None:
            raise RuntimeError(
                "prep_sparsity(initial=False) needs a pair list from an earlier "
                "prep_sparsity(initial=True)"
            )

        self._build_pair_domains()

        n_pairs = self.n_lmo_pairs
        if screening:
            avg_pao = sum(len(d) for d in self.lmopair_to_paos) / max(n_pairs, 1)
            avg_aux = sum(len(d) for d in self.lmopair_to_ribfs) / max(n_pairs, 1)
            self._print(
                f"  domains:  {n_pairs} of {naocc}^2 LMO pairs survive "
                f"({100.0 * n_pairs / naocc**2:.1f}%), "
                f"{avg_pao:.1f}/{npao} PAOs and {avg_aux:.1f}/{naux} aux per pair, "
                f"{len(self._interned)} distinct domains"
            )
            if self.de_dipole:
                self._print(f"            dropped pairs contribute {self.de_dipole:.10f} Eh")
        else:
            self._print(
                f"  domains:  {n_pairs} LMO pairs "
                f"({naocc}^2, no screening), {npao} PAOs and {naux} aux per pair"
            )
        return self

    def _build_pair_domains(self):
        """Every per-pair domain, from the LMO domains and the pair list.

        Split out of :meth:`prep_sparsity` because it has to run whenever
        EITHER of its two inputs changes, and in a coupled-cluster calculation
        the pair list changes without the LMO domains doing so: crude
        prescreening deletes pairs, which invalidates ``lmopair_to_lmos`` for
        every pair that coupled through one of them.
        """
        naocc = self.ref.naocc

        # A pair's domain is the union of the two LMOs' domains. Interned, so
        # that pairs landing on equal domains share one object and the
        # per-domain caches keyed on ``id()`` stay effective.
        self.lmopair_to_paos = [
            self._intern(sparse.merge_lists(self.lmo_to_paos[i], self.lmo_to_paos[j]))
            for (i, j) in self.ij_to_i_j
        ]
        self.lmopair_to_ribfs = [
            self._intern(sparse.merge_lists(self.lmo_to_ribfs[i], self.lmo_to_ribfs[j]))
            for (i, j) in self.ij_to_i_j
        ]

        # LMOs m for which both (i,m) and (j,m) survived. The residual's
        # coupling sum runs over these, not over all naocc, and so does every
        # one-external index in the CC residual.
        self.lmopair_to_lmos = [
            [m for m in range(naocc)
             if self.i_j_to_ij[i, m] != -1 and self.i_j_to_ij[j, m] != -1]
            for (i, j) in self.ij_to_i_j
        ]
        # psi4's ``lmopair_to_lmos_dense_``: the position of a global LMO index
        # within a pair's list, or -1. The CC residual looks this up in inner
        # loops, where scanning the list would be quadratic in the pair count.
        self.lmopair_to_lmos_dense = np.full((self.n_lmo_pairs, naocc), -1, dtype=int)
        for ij, lmos in enumerate(self.lmopair_to_lmos):
            for slot, m in enumerate(lmos):
                self.lmopair_to_lmos_dense[ij, m] = slot
        return self

    # -- psi4 DLPNO::pno_transform -----------------------------------------

    # -- psi4 DLPNO::compute_metric / DLPNO::compute_qia -------------------

    def compute_metric(self):
        """The auxiliary Coulomb metric ``(P|Q)``."""
        self.metric = ten.from_numpy("(P|Q)", self.ref.metric)
        return self

    def compute_qia(self):
        """Three-index integrals ``(Q | i u)`` over LMOs and PAOs.

        psi4 builds these with a screened shell-triplet loop that transforms
        straight into each auxiliary atom's extended LMO/PAO domain, so the
        result is stored sparsely. Which is a statement about a *producer*, not
        about what this phase means, and that is why the work is not here: this
        declares what will be read and asks a :class:`~dlpno.integrals.ThreeIndexSource`
        for it. The default source transforms a dense ``(Q|mn)`` and ignores the
        declaration, which is exact and is what the psi4-free fixtures replay.

        The declaration is possible at all because :meth:`prep_sparsity` has
        already run: every domain the calculation will read is known before any
        integral is built. That ordering is the whole opportunity, and it is why
        a screened producer can slot in behind this without touching a consumer.

        No metric is applied: DLPNO fits per pair domain, not globally.
        """
        return self._build_integrals(("q_ia",))

    def compute_qij(self):
        """Three-index integrals ``(Q | i j)``, psi4's ``compute_qij``.

        The LMO/LMO class, read only by the coupled-cluster one-external
        integrals. Declared as its own phase rather than folded into
        :meth:`compute_qia` because the two do not run at the same points: a CC
        calculation builds ``(Q|i u)`` twice, once at crude domains and again at
        refined ones, and ``(Q|i j)`` once, after the pair list has settled.
        """
        return self._build_integrals(("q_ij",))

    def compute_qab(self):
        """Three-index integrals ``(Q | u v)``, psi4's ``compute_qab``.

        The PAO/PAO class, and the largest of the three by a wide margin: the
        PAOs span the whole AO basis, so this is ``naux * nbf^2`` where
        ``(Q|i u)`` is ``naux * naocc * nbf``. psi4 stores it sparsely, as the
        PAO pairs belonging to each auxiliary atom; the port's dense source
        materializes all of it, which is what makes it exact and what makes a
        screened producer the eventual production path (design decision 5).
        """
        return self._build_integrals(("q_ab",))

    def _build_integrals(self, kinds):
        """Declare *kinds* against the current domains and ask the source for them."""
        # Distinct domains only. Pairs share them heavily - with screening off
        # there is exactly one - and a producer wants the set, not the multiset.
        seen_aux, seen_pao = {}, {}
        for domain in self.lmopair_to_ribfs:
            seen_aux.setdefault(id(domain), domain)
        for domain in self.lmopair_to_paos:
            seen_pao.setdefault(id(domain), domain)
        atom_lmos, atom_paos = self._aux_atom_demand(seen_aux, seen_pao)
        demand = integrals.Demand(aux_domains=list(seen_aux.values()),
                                  pao_domains=list(seen_pao.values()),
                                  aux_atom_to_lmos=atom_lmos,
                                  aux_atom_to_paos=atom_paos,
                                  kinds=tuple(kinds))

        self.integrals.declare(integrals.Spaces(C_lmo=self.C_lmo, C_pao=self.C_pao), demand)
        self.integrals.build()
        for kind in kinds:
            setattr(self, kind, getattr(self.integrals, kind)())

        describe = getattr(self.integrals, "describe", None)
        self._print(f"  DF ints:  {describe()}" if describe is not None else
                    f"  DF ints:  {', '.join(kinds)} from {type(self.integrals).__name__}")
        return self

    def _aux_atom_demand(self, seen_aux, seen_pao):
        """Which LMOs and PAOs are read against each auxiliary ATOM's functions.

        The pairing the two flat domain lists throw away, and the thing a
        producer that builds AO integrals per atom cannot work without. It is
        recoverable here and nowhere downstream, because it is a property of the
        pair list rather than of any one read.

        Correct BY CONSTRUCTION rather than by appeal to how psi4 derives its
        own extended maps. Every read of ``q_ia`` anywhere downstream is
        ``submatrix(q_ia, [ribfs_ij, lmos, paos_ij])`` for some surviving pair
        ``ij``, with ``lmos`` a subset of ``{i, j}``: :meth:`_fit_operands`
        gathers the ``i`` of each pair over the domain, and both exchange builds
        gather a single ``j``. So if, for every auxiliary atom ``A``,

            lmos[A] contains i and j for every pair ij whose ribfs touch A
            paos[A] contains lmopair_to_paos[ij] for those same pairs

        then every element any consumer will ever read has been built, and the
        rest of the tensor can stay zero. Taking those unions exactly is what
        this does.

        Grouped by interned domain rather than per pair: pairs share domain
        objects heavily, and the PAO union is the expensive half.
        """
        atom_of_ribf = {}
        for atom, ribfs in enumerate(self.ref.atom_to_ribf):
            for q in ribfs:
                atom_of_ribf[q] = atom
        natom = len(self.ref.atom_to_ribf)

        # One atom set per distinct auxiliary domain, one frozenset per distinct
        # PAO domain, so the per-pair loop below only unions small things.
        aux_atoms = {key: {atom_of_ribf[q] for q in dom} for key, dom in seen_aux.items()}
        pao_sets = {key: frozenset(dom) for key, dom in seen_pao.items()}

        lmos = [set() for _ in range(natom)]
        pao_keys = [set() for _ in range(natom)]
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            pao_key = id(self.lmopair_to_paos[ij])
            for atom in aux_atoms[id(self.lmopair_to_ribfs[ij])]:
                lmos[atom].add(i)
                lmos[atom].add(j)
                pao_keys[atom].add(pao_key)

        paos = [sorted(frozenset().union(*(pao_sets[k] for k in keys))) if keys else []
                for keys in pao_keys]
        return [sorted(s) for s in lmos], paos

    def _domain_key(self, ij):
        """What a fit is shared across: the pair's two domains."""
        return (id(self.lmopair_to_ribfs[ij]), id(self.lmopair_to_paos[ij]))

    def _fit_demand(self):
        """``{domain: (representative pair, sorted LMOs)}`` for the whole run.

        Which right-hand sides the fits are actually asked for, and the reason
        this phase does not grow with the occupied space. The solve is per
        *domain*, so a domain shared by many pairs is factorized once - but its
        right-hand side only has to carry the LMOs that some pair over that
        domain will read, and :meth:`_pair_exchange` reads exactly one, the
        ``i`` of the pair it is building. Solving for all ``naocc`` regardless
        is correct, and it is what the shared domain of an unscreened run wants
        anyway, since there every ``i`` is asked for. Under screening it is
        almost all waste: domains become nearly distinct per pair, and each one
        is asked for 1.6 LMOs on average against ``naocc``, which is 13 at
        ethanol/cc-pVTZ and 30 at a six-monomer water chain. The waste therefore
        grows with the system while the useful work does not, which is why this
        phase used to grow 22x from n=3 to n=6 against 9x for the whole run.

        Only ``i <= j`` pairs appear, matching :meth:`pno_transform`'s loop;
        anything else falls through to the lazy path in
        :meth:`_fit_coefficients`.
        """
        demand = {}
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if i > j:
                continue
            key = self._domain_key(ij)
            hit = demand.get(key)
            if hit is None:
                demand[key] = (ij, {i})
            else:
                hit[1].add(i)
        return {key: (ij, sorted(lmos)) for key, (ij, lmos) in demand.items()}

    def _fit_operands(self, ij, lmos):
        """``(A, rhs)`` for one domain's fitting equations over ``lmos``.

        Split out of :meth:`_fit_coefficients` so the solves can be captured as
        a batch.
        """
        ribfs = self.lmopair_to_ribfs[ij]
        paos = self.lmopair_to_paos[ij]
        # Gather then reshape, both captured. row_major=True because the
        # flattening this replaces was numpy's, whose default order is C; the
        # column-major walk would transpose the (i, u) block silently.
        block = sparse.submatrix(self.q_ia, [ribfs, lmos, paos], name="(Q|i u) domain")
        rhs = ten.reshape(block, [len(ribfs), len(lmos) * len(paos)], name="(Q|i u) domain")
        A = sparse.submatrix_rows_and_cols(self.metric, ribfs, ribfs, name="(P|Q) domain")
        return A, rhs

    def _reshape_fit(self, ij, rhs, nlmo):
        """The solved right-hand side, back to ``(naux_dom, nlmo, npao_dom)``."""
        ribfs = self.lmopair_to_ribfs[ij]
        paos = self.lmopair_to_paos[ij]
        return ten.reshape(rhs, [len(ribfs), nlmo, len(paos)], name="J^-1 (Q|i u)")

    @staticmethod
    def _run(graph):
        """Replay a setup graph as an OpenMP team over its independent nodes.

        The setup phases are wide and shallow - one disconnected chain per pair,
        every chain independent - so the parallelism worth having is across
        nodes, with each node's BLAS call left serial underneath. That is what
        the OpenMP executor gives, and it is the only way to get it here: a
        ``ThreadPoolExecutor`` over the same work returns silently wrong numbers
        against the OpenMP-built OpenBLAS conda resolves by default, because
        that build indexes its internal scratch by ``omp_get_thread_num()``,
        which is 0 on every caller-created thread. An OpenMP parallel region is
        a real team with distinct thread numbers, so the same BLAS is safe
        underneath it. See docs/sphinx/building/blas_threading.rst.

        No optimization passes: these graphs are a few thousand independent
        chains with nothing to fuse across them, and the passes would cost more
        to run than they could find. The residual, which does have structure
        worth rewriting, builds its own pass pipeline.
        """
        graph.set_executor(cg.OpenMPExecutor())
        graph.execute()

    def precompute_fits(self):
        """Solve every distinct domain's fitting equations in one graph.

        Two reductions, both of which matter and neither of which changes a
        number the rest of the calculation sees.

        The solve is per *domain* rather than per pair, so a domain shared by
        many pairs is factorized once: 23 solves against 169 pairs at
        ethanol/cc-pVTZ.

        Each solve then carries only the right-hand sides some pair over that
        domain will read, which :meth:`_fit_demand` collects, rather than all
        ``naocc`` of them. Solving for every LMO was the right thing when there
        was one domain and every pair asked it for a different LMO; once
        screening splits the domains apart, nearly all of it is discarded. That
        is 3.5x off this phase at ethanol/cc-pVTZ and 15x at a six-monomer water
        chain, where the phase went from 46% of the run to 6%.

        They are mutually independent, so they go into one graph and run under
        the OpenMP executor. That is the safe way to get this parallelism.
        Driving the same loop from a Python thread pool is what an earlier
        attempt did, and against the OpenMP-built OpenBLAS that conda resolves
        by default it returns silently wrong numbers: that build indexes its
        internal buffers by ``omp_get_thread_num()``, which is 0 on every
        caller-created thread, so the workers overwrite each other's scratch.
        An OpenMP parallel region is a real team with distinct thread numbers,
        so the same BLAS is safe underneath it. See
        docs/sphinx/building/blas_threading.rst.

        One check is given up: ``gesv``'s ``info``, which the eager path
        inspects per solve. The domain metric is a Coulomb metric and positive
        definite, so a singular factorization would mean the auxiliary basis
        itself is broken; :meth:`_fit_coefficients` still checks on the lazy
        path that remains for anything this pass has not covered.
        """
        if self.q_ia is None:
            raise RuntimeError("precompute_fits() needs compute_qia() first")

        demand = self._fit_demand()
        if not demand:
            return self

        # Allocate every operand first, then capture the WHOLE assembly - the
        # two gathers, the reshape, the solve and the reshape back. Building
        # the operands eagerly and capturing only the solve, which is what this
        # did first, leaves the gathers running one at a time on the calling
        # thread: the primitives are capture-aware precisely so this phase does
        # not have to be, and calling them eagerly pays their cost without
        # collecting the parallelism.
        jobs = []
        for key, (ij, lmos) in demand.items():
            ribfs = self.lmopair_to_ribfs[ij]
            paos = self.lmopair_to_paos[ij]
            nq, nu, nk = len(ribfs), len(paos), len(lmos)
            # Uninitialized, not zeroed. Every one of these four is written in
            # full by the node that consumes it - the two gathers cover their
            # whole destination, and the reshapes copy element for element - so
            # zeroing them first is a second pass over memory that nothing
            # reads. It is not a rounding error at this scale: 756 allocations
            # on a six-monomer chain, 22 ms of the phase's 68.
            jobs.append((
                key, lmos,
                [list(ribfs), [int(i) for i in lmos], list(paos)],
                [list(ribfs), list(ribfs)],
                ten.empty("(P|Q) domain", [nq, nq]),
                ten.empty("(Q|i u) domain", [nq, nk, nu]),
                ten.empty("(Q|i u) domain flat", [nq, nk * nu]),
                ten.empty("J^-1 (Q|i u)", [nq, nk, nu]),
            ))

        g = cg.Graph("domain fits")
        with cg.capture(g):
            for _, _, q_idx, m_idx, A, block, rhs, fit in jobs:
                la.gather(A, self.metric, m_idx)
                la.gather(block, self.q_ia, q_idx)
                # row_major=True: this flattening replaces numpy's, whose
                # default order is C.
                la.reshape(rhs, block, True)
                la.gesv(A, rhs)
                la.reshape(fit, rhs, True)
        self._run(g)

        rhs_total = 0
        for key, lmos, _, _, _, _, _, fit in jobs:
            rhs_total += len(lmos)
            for pos, i in enumerate(lmos):
                self._fit_cache[(key, i)] = (fit, pos)
        self._print(f"  fits:     {len(jobs)} distinct domains, "
                    f"{rhs_total} of {len(jobs) * self.ref.naocc} LMO blocks, "
                    f"{g.num_nodes()} nodes in one graph")
        return self

    def _fit_coefficients(self, ij, i):
        """``J_dom^-1 (Q | i u)`` for LMO ``i`` over pair ``ij``'s domain.

        psi4's local robust fit solves ``J_dom x = (Q|ia)`` per pair. The matrix
        is the same for every pair sharing an auxiliary domain, so solving it
        per pair re-factorizes one matrix once per pair: with screening off that
        is 91 identical factorizations for ethanol, a quarter of this phase.

        Solved once per domain instead, for the set of LMOs that domain's pairs
        will ask for, so a shared domain costs one factorization and one
        right-hand-side block per distinct ``i``. Keyed by the (auxiliary, PAO)
        domain pair and the LMO, so it stays correct when screening makes the
        domains differ and simply yields more entries.
        """
        key = (self._domain_key(ij), i)
        hit = self._fit_cache.get(key)
        if hit is None:
            # Lazy fallback for anything precompute_fits() did not cover. It
            # also keeps gesv's info check, which the captured path cannot see.
            A_solve, rhs = self._fit_operands(ij, [i])
            hit = (self._reshape_fit(ij, ten.solve(A_solve, rhs), 1), 0)
            self._fit_cache[key] = hit
        fit, pos = hit
        return fit[:, pos, :]

    def _batched_eigh(self, mats, descending=True, label="eigh"):
        """Module-level :func:`batched_eigh` bound to this class's executor rule."""
        return batched_eigh(mats, self._run, descending=descending, label=label)

    def _warm_pao_domains(self, pairs):
        """Fill the PAO domain memo for every distinct domain, in two batches.

        :meth:`_canonical_pao_domain` computes one domain at a time, and each
        costs two eigendecompositions: one of the domain overlap and, after the
        orthogonalizer is built from it, one of the Fock matrix in that basis.
        Done pair by pair those are issued one at a time from the calling
        thread, which is the phase's whole cost on an extended system. At a
        six-monomer water chain the planning half spent 152 ms in 138 ``syev``
        calls of about 1.1 ms each, against 69 distinct domains - and it got
        SLOWER with threads, because a domain-sized eigendecomposition is small
        enough that threading one loses more than it gains.

        The domains are independent, so the right grain is a team across them
        with each decomposition serial underneath, which is exactly what
        :func:`batched_eigh` gives. The dependency between the two
        decompositions is what makes it two rounds rather than one: the second
        needs the orthogonalizer the first produces.

        Numerically this is the sequential path, reordered. The same matrices
        are decomposed by the same routine, so the memo it leaves is the memo
        :meth:`_canonical_pao_domain` would have left, and that method still
        works unchanged for anything this did not cover.
        """
        keys, domains = [], []
        for ij in pairs:
            key = id(self.lmopair_to_paos[ij])
            if key in self._pao_domain_cache or key in keys:
                continue
            keys.append(key)
            domains.append(self.lmopair_to_paos[ij])
        if not keys:
            return

        S_doms, F_doms = [], []
        for paos in domains:
            S_doms.append(sparse.submatrix_rows_and_cols(self.S_pao, paos, paos))
            F_doms.append(sparse.submatrix_rows_and_cols(self.F_pao, paos, paos))

        canon = batched_orthocanonicalizer(S_doms, F_doms, self.cut.s_cut, self._run)

        for key, (X_pao, e_pao), F_dom in zip(keys, canon, F_doms):
            F_can = ten.triplet(X_pao, F_dom, X_pao, trans_a=True, name="F (can PAO)")
            self._pao_domain_cache[key] = (X_pao, e_pao, F_can)

    def _canonical_pao_domain(self, ij):
        """``(X_pao, e_pao, F_can)`` for pair ``ij``'s PAO domain, memoized.

        The orthocanonical basis of a PAO domain depends only on the domain, so
        every pair sharing one gets the same answer. Computing it per pair costs
        two eigendecompositions each, and with screening off there is exactly one
        distinct domain: 182 of this phase's 364 ``syev`` calls produced the same
        result. Keyed by domain, so screening just adds entries.
        """
        key = id(self.lmopair_to_paos[ij])
        hit = self._pao_domain_cache.get(key)
        if hit is not None:
            return hit

        paos = self.lmopair_to_paos[ij]
        S_dom = sparse.submatrix_rows_and_cols(self.S_pao, paos, paos)
        F_dom = sparse.submatrix_rows_and_cols(self.F_pao, paos, paos)
        X_pao, e_pao = ten.orthocanonicalizer(S_dom, F_dom, self.cut.s_cut)
        F_can = ten.triplet(X_pao, F_dom, X_pao, trans_a=True, name="F (can PAO)")

        hit = (X_pao, e_pao, F_can)
        self._pao_domain_cache[key] = hit
        return hit

    def _pair_exchange(self, ij, i, j):
        """The exchange operator ``K_ij[a,b] = (i a | j b)`` over pair ``ij``'s
        PAO domain, density-fitted within the pair's auxiliary domain.

        The fit coefficients come from :meth:`_fit_coefficients`, which solves
        the domain metric once for every LMO some pair over that domain asks
        for; here it is one GEMM per pair.
        """
        ribfs = self.lmopair_to_ribfs[ij]
        paos = self.lmopair_to_paos[ij]

        fit_i = self._fit_coefficients(ij, i)
        # Gather straight into a rank-3 block and contract through its rank-2
        # view, rather than slicing the gathered block in numpy. gather needs
        # one index list per axis of the source, so selecting a single LMO
        # leaves a length-1 middle axis; dropping it in numpy would be a host
        # copy, and a host copy in the middle of this is what pins the whole
        # pair loop to the calling thread. The view costs nothing: (nq, 1, nu)
        # column major has exactly the layout of (nq, nu).
        blk = ten.zeros("(Q|j a)", [len(ribfs), 1, len(paos)])
        la.gather(blk, self.q_ia,
                  [[int(p) for p in ribfs], [int(j)], [int(p) for p in paos]])
        return ten.doublet(fit_i, blk[:, 0, :], trans_a=True, name="K (ia|jb)")

    # -- pair-block storage ------------------------------------------------

    def _allocate_pair_stores(self):
        """Flat rank-3 stores for the per-pair blocks, pair index trailing.

        This is the layout decision the whole calculation turns on. psi4 keeps
        every pair's block in its own ``SharedMatrix``, which forces the
        residual into a loop of individually-dispatched small GEMMs. Here each
        per-pair quantity is one contiguous ``(npno_max, npno_max, n_pairs)``
        tensor.

        Two consequences, both load-bearing:

        * Einsums tensors are column major, so a *trailing* index is the batch
          index a strided-batched GEMM wants. A contraction over a run of pairs
          is then a single rank-3 einsum, not a loop.
        * Everything elementwise over pairs (the amplitude update, the
          antisymmetrization, the energy dot products) collapses from O(n_pairs)
          graph nodes to exactly one, because it is just an operation on the
          whole store.

        Blocks are padded to their bucket's dimension and the padding is inert:
        the integrals and amplitudes are zero there, the overlaps are zero
        there, and the energy denominators are one, so padded components stay
        zero for the life of the calculation and contribute nothing to any dot
        product. How the buckets are chosen, and why the layout is a value
        rather than fields on this object, is :class:`~dlpno.layout.PairLayout`.
        """
        self.npno_max = max(self.n_pno) if self.n_pno else 0
        self._choose_buckets()
        self.K_all = self.new_pair_stores("K")
        self.T_all = self.new_pair_stores("T")
        self.Tt_all = self.new_pair_stores("Tt")

    def _choose_buckets(self):
        """Build this calculation's :class:`~dlpno.layout.PairLayout`.

        The bucket boundaries and the objective that picks them live in
        ``layout.py``, because a DLPNO-CCSD calculation needs two of them alive
        at once - the MP2-level PNOs the prescreening LMP2 runs in, and the
        tighter CC-level ones ``recompute_pnos`` produces - so the layout is a
        value rather than a set of fields on the calculation.
        """
        self.layout = PairLayout.choose(self.n_pno, self.cut, report=self._print)

    # The layout's fields and accessors, forwarded. Reading ``self.bucket_of``
    # rather than ``self.layout.bucket_of`` is what every existing consumer and
    # every promoted C++ stage signature does, and there is no reason to churn
    # them for a factoring whose point is that a SECOND layout can now exist.

    @property
    def bucket_dims(self):
        return self.layout.bucket_dims

    @property
    def bucket_of(self):
        return self.layout.bucket_of

    @property
    def slot_of(self):
        return self.layout.slot_of

    @property
    def bucket_members(self):
        return self.layout.bucket_members

    def new_pair_stores(self, name):
        """One ``(M_b, M_b, n_b)`` tensor per bucket."""
        return self.layout.new_stores(name)

    def pair_dim(self, ij):
        """The padded block dimension pair ``ij`` is stored at."""
        return self.layout.dim(ij)

    def pair_block(self, stores, ij):
        """The padded numpy view of pair ``ij``'s block within its bucket."""
        return self.layout.block(stores, ij)

    def pair_view(self, stores, ij):
        """The einsums (capture-safe) view of pair ``ij``'s block."""
        return self.layout.view(stores, ij)

    def pno_transform(self):
        """Build each pair's truncated, canonical PNO basis.

        For every pair ``i <= j``: form the exchange operator over the PAO
        domain, orthocanonicalize the domain's PAOs, take one semicanonical MP2
        step for the amplitudes, diagonalize the resulting pair density to get
        the PNOs, keep the ones whose occupation number clears ``T_CUT_PNO``,
        and recanonicalize what survives. The energy lost to that truncation is
        accumulated as ``de_pno_total``, psi4's PNO truncation correction.

        Three pieces since the M6 split: :meth:`plan_pno_transform` warms the
        domain memos and flattens the numerics' arguments, the ``transform_pnos``
        stage is the promotable numerics, and :meth:`_finish_pno_transform`
        mirrors the results onto the lower triangle and scatters them into the
        flat stores. The numerics goes through the stage registry, so whichever
        backend has been selected (``einsums.stages.apply_backend_spec``) runs,
        from every entry point; the Python one at
        :func:`dlpno.pno_xform.transform_pnos` is the default.

        PNOs defined in DOI 10.1063/1.3086717, equations 17 through 24.
        """
        from .stages import transform_pnos

        args = self.plan_pno_transform()
        self._finish_pno_transform(transform_pnos(**args))
        return self

    def pno_criteria(self):
        """The truncation criteria this method's PNO transform applies.

        MP2 has only the occupation cutoff. psi4's ``DLPNO::pno_transform``
        carries neither the trace nor the energy criterion and does not scale
        the pair density for diagonal pairs, while ``DLPNOCCSD`` does all three
        - so the difference between the branches is a returned value here
        rather than a second copy of the transform.

        ``diag_factor`` multiplies the occupation cutoff for a diagonal pair. It
        folds together psi4's two separate diagonal adjustments, which is exact
        rather than approximate: comparing ``|occ/2| >= s * tol`` is the same
        test as ``|occ| >= 2 * s * tol``, and the halved eigenvalues are read
        nowhere else - the trace criterion is a ratio and so is invariant, and
        the energy criterion does not involve them at all. Doing it this way
        avoids a second eigendecomposition input and keeps the MP2 path's
        arithmetic untouched.
        """
        return dict(t_cut_pno=float(self.cut.t_cut_pno),
                    t_cut_trace=0.0, t_cut_energy=0.0, diag_factor=1.0)

    def plan_pno_transform(self):
        """Warm the domain memos and pack the PNO numerics' arguments.

        The planning half of the PNO transform, and the half that stays
        Python. The memo warm has to come first and stay host-side: both
        caches end in something that cannot be captured - ``gesv``'s info
        check on the fits, and ``eigh``'s descending reorder on the PAO
        domains. What follows flattens per-upper-pair arrays and deduplicates
        everything domain-shaped exactly as the memos share it, so pairs on
        one domain hand the numerics one tensor, not one copy each.

        Splitting this out is what makes the numerics' contract narrow enough
        to state: as one phase this read 32 fields of ``self``; the numerics
        take fifteen parameters.
        """
        F_lmo = ten.view(self.F_lmo)
        crit = self.pno_criteria()
        upper = [ij for ij, (i, j) in enumerate(self.ij_to_i_j) if i <= j]

        self._warm_pao_domains(upper)
        for ij in upper:
            self._fit_coefficients(ij, self.ij_to_i_j[ij][0])
            self._canonical_pao_domain(ij)

        fits, fit_index = [], {}
        doms, dom_index = [], {}
        fit_of, fit_pos, dom_of = [], [], []
        ribfs, paos, lmo_j, shift, pno_scale = [], [], [], [], []
        for ij in upper:
            i, j = self.ij_to_i_j[ij]

            fit, pos = self._fit_cache[(self._domain_key(ij), i)]
            fo = fit_index.setdefault(id(fit), len(fits))
            if fo == len(fits):
                fits.append(fit)
            fit_of.append(fo)
            fit_pos.append(int(pos))

            key = id(self.lmopair_to_paos[ij])
            do = dom_index.setdefault(key, len(doms))
            if do == len(doms):
                doms.append(self._pao_domain_cache[key])
            dom_of.append(do)

            ribfs.append([int(p) for p in self.lmopair_to_ribfs[ij]])
            paos.append([int(p) for p in self.lmopair_to_paos[ij]])
            lmo_j.append(int(j))
            shift.append(float(F_lmo[i, i] + F_lmo[j, j]))
            scale = 1.0
            if i == j:
                scale *= crit["diag_factor"]
            if i < self.ref.n_core or j < self.ref.n_core:
                scale *= self.cut.t_cut_pno_core_scale
            pno_scale.append(float(scale))

        self._pno_upper = upper
        return dict(
            q_ia=self.q_ia,
            fits=fits, fit_of=fit_of, fit_pos=fit_pos,
            dom_X=[d[0] for d in doms], dom_e=[d[1] for d in doms],
            dom_F=[d[2] for d in doms], dom_of=dom_of,
            ribfs=ribfs, paos=paos, lmo_j=lmo_j,
            shift=shift, pno_scale=pno_scale,
            min_pnos=int(self.cut.min_pnos),
            t_cut_pno=crit["t_cut_pno"],
            t_cut_trace=crit["t_cut_trace"],
            t_cut_energy=crit["t_cut_energy"],
        )

    def _finish_pno_transform(self, result):
        """Mirror the numerics' upper-triangle results onto every pair.

        The lower triangle is ``K_ji = K_ij^T`` and ``T_ji = T_ij^T``: a host
        transpose at scatter time, where the pre-split method emitted one
        transpose node per lower pair into stage 3's graph. Everything else -
        the shared ``X_pno``/``e_pno`` objects, the doubled ``de_pno``
        accounting for off-diagonal pairs, the store scatter and the
        antisymmetrized amplitudes - is unchanged.
        """
        npairs = self.n_lmo_pairs
        upper = self._pno_upper

        # Read each contract list field ONCE. When the result comes from the
        # C++ backend, every attribute access converts the underlying
        # std::vector into a fresh Python list, so indexing result.K_pno[u]
        # inside the pair loop is quadratic in the pair count: 320 ms of pure
        # conversion at chain6's 403 upper pairs, against 4 ms this way. The
        # Python dataclass never punished the pattern, which is how it crept in.
        K_pno, T_pno = result.K_pno, result.T_pno
        X_pno, e_pno, n_pno = result.X_pno, result.e_pno, result.n_pno
        e_initial, e_os_initial = result.e_initial, result.e_os_initial
        e_trunc, e_os_trunc = result.e_trunc, result.e_os_trunc

        self.X_pno = [None] * npairs
        self.e_pno = [None] * npairs
        self.n_pno = [0] * npairs
        self.de_pno = [0.0] * npairs
        self.de_pno_os = [0.0] * npairs
        self.de_pno_ss = [0.0] * npairs
        for u, ij in enumerate(upper):
            i, j = self.ij_to_i_j[ij]
            ji = self.ij_to_ji[ij]
            # The truncation energies, from the scalar tensors the numerics'
            # graphs wrote (the contract cannot carry computed floats).
            e_ij_initial = float(ten.view(e_initial[u])[0])
            e_ij_os_initial = float(ten.view(e_os_initial[u])[0])
            e_ij_trunc = float(ten.view(e_trunc[u])[0])
            e_ij_os_trunc = float(ten.view(e_os_trunc[u])[0])
            de = e_ij_initial - e_ij_trunc
            de_os = e_ij_os_initial - e_ij_os_trunc
            de_ss = (e_ij_initial - e_ij_os_initial) - (e_ij_trunc - e_ij_os_trunc)
            for target in ((ij, ji) if i < j else (ij,)):
                self.X_pno[target] = X_pno[u]
                self.e_pno[target] = e_pno[u]
                self.n_pno[target] = n_pno[u]
                self.de_pno[target] = de
                self.de_pno_os[target] = de_os
                self.de_pno_ss[target] = de_ss

        # Scatter the per-pair blocks into the flat stores, then form the
        # antisymmetrized amplitudes for every pair with one rank-3 permute
        # plus one axpby instead of two operations per pair.
        self._allocate_pair_stores()
        for u, ij in enumerate(upper):
            n = self.n_pno[ij]
            if n == 0:
                continue
            i, j = self.ij_to_i_j[ij]
            K = ten.view(K_pno[u])
            T = ten.view(T_pno[u])
            self.pair_block(self.K_all, ij)[:n, :n] = K
            self.pair_block(self.T_all, ij)[:n, :n] = T
            if i < j:
                ji = self.ij_to_ji[ij]
                self.pair_block(self.K_all, ji)[:n, :n] = K.T
                self.pair_block(self.T_all, ji)[:n, :n] = T.T
        for Tt, T in zip(self.Tt_all, self.T_all):
            einsums.permute("abp <- bap", Tt, T)
            la.axpby(2.0, T, -1.0, Tt)

        self.de_pno_total = sum(self.de_pno)
        self.de_pno_total_os = sum(self.de_pno_os)
        self.de_pno_total_ss = sum(self.de_pno_ss)

        counts = self.n_pno
        self._print(
            f"  PNOs:     avg {sum(counts) / npairs:.1f}, min {min(counts)}, "
            f"max {max(counts)} per LMO pair"
        )
        self._print(f"  PNO truncation energy = {self.de_pno_total:.12f}")
        return self


def batched_eigh(mats, run, descending=True, label="eigh"):
    """Diagonalize many independent matrices in one graph.

    Per-pair ``syev`` is the largest remaining item in the PNO transform
    once the domain fits are batched, and the calls are mutually
    independent, so they are captured together and run under the OpenMP
    executor instead of being issued one at a time.

    Issuing them eagerly gets *worse* with threads, not better: 219 ms
    serial against 329 ms on ten threads at ethanol/cc-pVTZ, because each
    small ``syev`` is fighting the BLAS's own internal threading. One graph
    node per matrix with an OpenMP team over the nodes is the right grain -
    the parallelism is across matrices, and each matrix stays serial.

    A Python thread pool would be the obvious alternative and is not safe
    here; see :meth:`DLPNOBase.precompute_fits` for why.

    A module-level function so :mod:`dlpno.pno_xform` can use it without an
    instance; ``run`` is the graph runner, normally ``DLPNOBase._run``.
    """
    evecs = [ten.from_numpy(f"{label} vectors", ten.view(A)) for A in mats]
    evals = [ten.zeros(f"{label} values", [ten.shape(A)[0]]) for A in mats]
    if evecs:
        g = cg.Graph(label)
        with cg.capture(g):
            for V, w in zip(evecs, evals):
                la.syev(V, w, compute_eigenvectors=True)
        run(g)
    if descending:
        for V, w in zip(evecs, evals):
            v, e = ten.view(V), ten.view(w)
            v[...] = v[:, ::-1].copy()
            e[...] = e[::-1].copy()
    return evals, evecs


def batched_orthocanonicalizer(S_doms, F_doms, s_cut, run):
    """:func:`~dlpno.tensors.orthocanonicalizer` over many domains, in two graphs.

    Numerically this is the sequential routine, reordered. The same matrices go
    through the same ``syev`` in the same order within each domain, so the
    result is what calling ``orthocanonicalizer`` once per domain would give;
    only the issue order across domains changes.

    Reordering is the entire point. Each domain costs two eigendecompositions,
    and issued one at a time they are small enough that the BLAS's own internal
    threading loses more than it gains - the sequential loop gets SLOWER as
    threads are added. Batching them puts the parallelism across domains, where
    it belongs, with each decomposition serial underneath.

    Two rounds rather than one because the second decomposition needs the
    orthogonalizer the first produces. ``run`` is the graph runner, normally
    :meth:`DLPNOBase._run`.

    Returns ``[(X, e), ...]``, one per input domain. A domain with nothing above
    ``s_cut`` yields a zero-column ``X`` and an empty ``e``, matching
    ``orthocanonicalizer``'s own early return, and sits out the second round
    rather than entering it as an empty matrix.
    """
    s_vals, s_vecs = batched_eigh(S_doms, run, descending=True, label="overlap")

    # The orthogonalizers, which are host-side slicing and scaling rather than
    # linear algebra, then the Fock matrices they induce.
    orthos, fmos = [], []
    for sv, U, F_dom in zip(s_vals, s_vecs, F_doms):
        keep = int(np.count_nonzero(ten.view(sv) > s_cut))
        if keep == 0:
            orthos.append(None)
            fmos.append(None)
            continue
        X = ten.scale_columns(
            ten.from_numpy("orthogonalizer", ten.view(U)[:, :keep]),
            ten.view(sv)[:keep] ** -0.5,
            name="orthogonalizer",
        )
        orthos.append(X)
        fmos.append(ten.triplet(X, F_dom, X, trans_a=True, name="C^T F C"))

    # Round two: the Fock matrices in each orthogonal basis.
    live = [i for i, f in enumerate(fmos) if f is not None]
    e_vals, e_vecs = batched_eigh([fmos[i] for i in live], run,
                                  descending=True, label="orthocanonical")

    slot_of = {i: slot for slot, i in enumerate(live)}
    out = []
    for i, S_dom in enumerate(S_doms):
        slot = slot_of.get(i)
        if slot is None:
            out.append((ten.zeros("orthocanonical", [ten.shape(S_dom)[0], 0]),
                        ten.zeros("orthocanonical energies", [0])))
        else:
            out.append((ten.doublet(orthos[i], e_vecs[slot], name="orthocanonical"),
                        e_vals[slot]))
    return out
