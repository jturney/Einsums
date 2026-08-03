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

Screening is not implemented yet. :meth:`prep_sparsity` currently marks every
LMO pair significant and gives every pair the full PAO and auxiliary domain,
which makes local MP2 exactly equivalent to canonical DF-MP2 once PNO truncation
is also switched off. That equivalence is what pins the port down; the
differential-overlap and dipole screening from ``DLPNO::compute_overlap_ints``,
``DLPNO::compute_dipole_ints`` and ``DLPNO::prep_sparsity`` then layers on top
without changing anything below.

The per-pair data structures (``ij_to_i_j``, ``lmopair_to_paos``, ...) are the
real ones from the start even though they are trivially dense today, so turning
screening on is a change to :meth:`prep_sparsity` alone.
"""

import numpy as np

import einsums
from einsums import linalg as la

from . import sparse
from . import tensors as ten
from .thresholds import Thresholds

__all__ = ["DLPNOBase"]


class DLPNOBase:
    """Orbitals, domains, and pair natural orbitals for a DLPNO calculation."""

    def __init__(self, reference, thresholds=None, verbose=True):
        self.ref = reference.validate()
        self.cut = thresholds if thresholds is not None else Thresholds()
        self.verbose = verbose

        # setup_orbitals
        self.C_lmo = self.F_lmo = None
        self.C_pao = self.S_pao = self.F_pao = None

        # prep_sparsity
        self.i_j_to_ij = None
        self.ij_to_i_j = []
        self.ij_to_ji = []
        self.lmopair_to_paos = []
        self.lmopair_to_ribfs = []

        # compute_metric / compute_qia
        self.metric = None
        self.q_ia = None

        # Per-domain results shared across pairs (see _fit_coefficients and
        # _canonical_pao_domain). One entry each until screening lands.
        self._fit_cache = {}
        self._pao_domain_cache = {}

        # pno_transform. The per-pair amplitude/integral blocks live in flat
        # rank-3 stores with the PAIR INDEX TRAILING; see _allocate_pair_stores.
        self.npno_max = 0
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
            C_pao, np.diag(ten.view(S_pao)) ** -0.5, name="C (PAO)"
        )
        self.S_pao = ten.triplet(self.C_pao, S, self.C_pao, trans_a=True, name="S (PAO)")
        self.F_pao = ten.triplet(self.C_pao, F, self.C_pao, trans_a=True, name="F (PAO)")

        self._print(
            f"  orbitals: {ref.naocc} active occupied LMOs, "
            f"{ten.shape(self.C_pao)[1]} PAOs, {ref.naux} auxiliary functions"
        )
        return self

    # -- psi4 DLPNO::prep_sparsity -----------------------------------------

    def prep_sparsity(self):
        """Enumerate significant LMO pairs and their PAO/auxiliary domains.

        No screening yet: every ordered pair ``(i, j)`` is significant and every
        domain is complete. The maps built here have the same meaning and layout
        as psi4's, so enabling screening means shrinking these lists, not
        reshaping the code that consumes them.
        """
        naocc = self.ref.naocc
        npao = ten.shape(self.C_pao)[1]
        naux = self.ref.naux

        self.i_j_to_ij = np.full((naocc, naocc), -1, dtype=int)
        self.ij_to_i_j = []
        for i in range(naocc):
            for j in range(naocc):
                self.i_j_to_ij[i, j] = len(self.ij_to_i_j)
                self.ij_to_i_j.append((i, j))
        self.ij_to_ji = [int(self.i_j_to_ij[j, i]) for (i, j) in self.ij_to_i_j]

        all_paos = list(range(npao))
        all_ribfs = list(range(naux))
        self.lmopair_to_paos = [all_paos for _ in self.ij_to_i_j]
        self.lmopair_to_ribfs = [all_ribfs for _ in self.ij_to_i_j]

        self._print(
            f"  domains:  {self.n_lmo_pairs} LMO pairs "
            f"({naocc}^2, no screening), {npao} PAOs and {naux} aux per pair"
        )
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
        result is stored sparsely. Here the full ``(naux, naocc, npao)`` tensor
        is built in two GEMM-shaped contractions and the domains are taken as
        slices of it, which gives identical numbers and keeps the consumers
        unchanged when the sparse builder lands.

        No metric is applied: DLPNO fits per pair domain, not globally.
        """
        ref = self.ref
        Qmn = ten.from_numpy("(Q|mn)", ref.eri_3index)
        npao = ten.shape(self.C_pao)[1]

        half = ten.zeros("(Q|m u)", [ref.naux, ref.nbf, npao])
        einsums.einsum("Qmu <- Qmn ; nu", half, Qmn, self.C_pao)
        self.q_ia = ten.zeros("(Q|i u)", [ref.naux, ref.naocc, npao])
        einsums.einsum("Qiu <- Qmu ; mi", self.q_ia, half, self.C_lmo)

        self._print(
            f"  DF ints:  (Q|iu) is {ref.naux} x {ref.naocc} x {npao} "
            f"({ten.view(self.q_ia).nbytes / 2**20:.1f} MiB, dense)"
        )
        return self

    def _fit_coefficients(self, ij):
        """``J_dom^-1 (Q | i u)`` for every LMO at once, memoized by domain.

        psi4's local robust fit solves ``J_dom x = (Q|ia)`` per pair. The matrix
        is the same for every pair sharing an auxiliary domain, so solving it
        per pair re-factorizes one matrix once per pair: with screening off that
        is 91 identical factorizations for ethanol, a quarter of this phase.

        Solved once for all LMOs instead, which is a single ``gesv`` with
        ``naocc * npao`` right-hand sides. Keyed by the (auxiliary, PAO) domain
        pair, so it stays correct when screening makes the domains differ and
        simply yields more entries.
        """
        key = (id(self.lmopair_to_ribfs[ij]), id(self.lmopair_to_paos[ij]))
        hit = self._fit_cache.get(key)
        if hit is not None:
            return hit

        ribfs = self.lmopair_to_ribfs[ij]
        paos = self.lmopair_to_paos[ij]
        naocc = self.ref.naocc
        block = ten.view(self.q_ia)[np.ix_(ribfs, range(naocc), paos)]
        rhs = ten.from_numpy("(Q|i u) domain", block.reshape(len(ribfs), naocc * len(paos)))
        A_solve = sparse.submatrix_rows_and_cols(self.metric, ribfs, ribfs, name="(P|Q) domain")
        solved = ten.view(ten.solve(A_solve, rhs)).reshape(len(ribfs), naocc, len(paos))

        hit = ten.from_numpy("J^-1 (Q|i u)", solved)
        self._fit_cache[key] = hit
        return hit

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
        the domain metric once for all LMOs; here it is one GEMM per pair.
        """
        ribfs = self.lmopair_to_ribfs[ij]
        paos = self.lmopair_to_paos[ij]

        # The gather seam: einsums has no index-list gather, so the domain
        # restriction runs on the host view rather than as a captured op.
        fit = self._fit_coefficients(ij)
        fit_i = fit[:, i, :]
        j_qa = ten.from_numpy("(Q|j a)", ten.view(self.q_ia)[np.ix_(ribfs, [j], paos)][:, 0, :])
        return ten.doublet(fit_i, j_qa, trans_a=True, name="K (ia|jb)")

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

        Blocks are padded to ``npno_max`` and the padding is inert: the integrals
        and amplitudes are zero there, the overlaps are zero there, and the
        energy denominators are one, so padded components stay zero for the life
        of the calculation and contribute nothing to any dot product. Bucketing
        pairs by PNO count would avoid the wasted flops; padding is the simpler
        thing that keeps every pair in one batchable store, and the waste is
        small while the PNO counts are tight (18.1 average against a 19 maximum
        for water/cc-pVDZ).
        """
        self.npno_max = max(self.n_pno) if self.n_pno else 0
        self._choose_buckets()
        self.K_all = self.new_pair_stores("K")
        self.T_all = self.new_pair_stores("T")
        self.Tt_all = self.new_pair_stores("Tt")

    def _choose_buckets(self):
        """Partition pairs into ``n_buckets`` groups by PNO count.

        Padding every pair to the global maximum is what makes the coupling
        GEMMs batchable, but it is paid in cubic flops: on ethanol/cc-pVDZ the
        PNO counts run 5..48 against an average of 29, so a single store does
        4.3x the necessary work. Splitting into a few buckets, each padded to
        its own maximum, keeps the shapes uniform *within* a bucket (which is
        all the batching needs) while cutting most of that waste.

        Boundaries minimize the padded cubic cost. Each PNO count contributes
        independently once its bucket's maximum is fixed, so the objective is
        separable and a segmentation DP finds the exact optimum in O(n^2 B) over
        the distinct counts. Enumerating the boundary combinations instead is
        the same answer at C(n-1, B-1) cost, which for 32 distinct counts and 6
        buckets is 170k combinations: three seconds, dwarfing the PNO transform
        it is sizing storage for.
        """
        counts = sorted({n for n in self.n_pno if n})
        if not counts:
            self.bucket_dims, self.bucket_of, self.slot_of = [], [], []
            self.bucket_members = []
            return
        n_buckets = max(1, min(self.cut.n_buckets, len(counts)))

        weight = [0] * len(counts)
        index = {c: i for i, c in enumerate(counts)}
        for n in self.n_pno:
            if n:
                weight[index[n]] += 1
        prefix = [0] * (len(counts) + 1)
        for i, w in enumerate(weight):
            prefix[i + 1] = prefix[i] + w

        # seg(j, i): counts[j:i] in one bucket, padded to counts[i-1].
        def seg(j, i):
            return (prefix[i] - prefix[j]) * counts[i - 1] ** 3

        INF = float("inf")
        dp = [[INF] * (n_buckets + 1) for _ in range(len(counts) + 1)]
        back = [[0] * (n_buckets + 1) for _ in range(len(counts) + 1)]
        dp[0][0] = 0
        for i in range(1, len(counts) + 1):
            for b in range(1, n_buckets + 1):
                for j in range(i):
                    if dp[j][b - 1] == INF:
                        continue
                    cost = dp[j][b - 1] + seg(j, i)
                    if cost < dp[i][b]:
                        dp[i][b] = cost
                        back[i][b] = j

        edges, i = [], len(counts)
        for b in range(n_buckets, 0, -1):
            edges.append(counts[i - 1])
            i = back[i][b]
        self.bucket_dims = sorted(edges)

        self.bucket_of = [-1] * self.n_lmo_pairs
        self.slot_of = [-1] * self.n_lmo_pairs
        self.bucket_members = [[] for _ in self.bucket_dims]
        for ij, n in enumerate(self.n_pno):
            if n == 0:
                continue
            b = next(i for i, e in enumerate(self.bucket_dims) if e >= n)
            self.bucket_of[ij] = b
            self.slot_of[ij] = len(self.bucket_members[b])
            self.bucket_members[b].append(ij)

    def new_pair_stores(self, name):
        """One ``(M_b, M_b, n_b)`` tensor per bucket."""
        return [
            ten.zeros(f"{name} (bucket {b}, dim {M})", [M, M, len(members)])
            for b, (M, members) in enumerate(zip(self.bucket_dims, self.bucket_members))
        ]

    def pair_dim(self, ij):
        """The padded block dimension pair ``ij`` is stored at."""
        return self.bucket_dims[self.bucket_of[ij]]

    def pair_block(self, stores, ij):
        """The padded numpy view of pair ``ij``'s block within its bucket."""
        return ten.view(stores[self.bucket_of[ij]])[:, :, self.slot_of[ij]]

    def pair_view(self, stores, ij):
        """The einsums (capture-safe) view of pair ``ij``'s block."""
        return stores[self.bucket_of[ij]][:, :, self.slot_of[ij]]

    def pno_transform(self):
        """Build each pair's truncated, canonical PNO basis.

        For every pair ``i <= j``: form the exchange operator over the PAO
        domain, orthocanonicalize the domain's PAOs, take one semicanonical MP2
        step for the amplitudes, diagonalize the resulting pair density to get
        the PNOs, keep the ones whose occupation number clears ``T_CUT_PNO``,
        and recanonicalize what survives. The energy lost to that truncation is
        accumulated as ``de_pno_total``, psi4's PNO truncation correction.

        PNOs defined in DOI 10.1063/1.3086717, equations 17 through 24.
        """
        npairs = self.n_lmo_pairs
        F_lmo = ten.view(self.F_lmo)

        # Per-pair results are collected at their natural (unpadded) sizes and
        # only then scattered into the flat stores, so the setup code stays
        # readable and the padding lives in one place.
        K_pairs = [None] * npairs
        T_pairs = [None] * npairs
        self.X_pno = [None] * npairs
        self.e_pno = [None] * npairs
        self.n_pno = [0] * npairs
        self.de_pno = [0.0] * npairs
        self.de_pno_os = [0.0] * npairs
        self.de_pno_ss = [0.0] * npairs

        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if i > j:
                continue
            ji = self.ij_to_ji[ij]

            K_pao = self._pair_exchange(ij, i, j)

            # Canonicalize the pair's PAO domain, removing linear dependencies.
            # Shared by every pair on the same domain; see _canonical_pao_domain.
            X_pao, e_pao, F_pao_ij = self._canonical_pao_domain(ij)
            K_pao = ten.triplet(X_pao, K_pao, X_pao, trans_a=True, name="K (can PAO)")

            # One semicanonical MP2 step gives the amplitudes the PNOs come from.
            shift = float(F_lmo[i, i] + F_lmo[j, j])
            D_pao = ten.pair_denominator(e_pao, shift)
            T_pao = ten.zeros("T (can PAO)", ten.shape(K_pao))
            la.direct_division(1.0, K_pao, D_pao, 0.0, T_pao)
            Tt_pao = ten.antisymmetrize(T_pao)

            e_ij_initial = ten.vector_dot(K_pao, Tt_pao)
            e_ij_os_initial = ten.vector_dot(K_pao, T_pao)

            # Pair density; its eigenvectors are the PNOs, eigenvalues the
            # PNO occupation numbers.
            D_ij = ten.doublet(Tt_pao, T_pao, trans_b=True, name="D (pair)")
            la.axpby(1.0, ten.doublet(Tt_pao, T_pao, trans_a=True), 1.0, D_ij)
            pno_occ, X_pno = ten.eigh(D_ij, descending=True, name="PNO")

            nvir_ij = ten.shape(K_pao)[0]
            pno_scale = 1.0
            if i < self.ref.n_core or j < self.ref.n_core:
                pno_scale *= self.cut.t_cut_pno_core_scale

            keep = min(self.cut.min_pnos, nvir_ij)
            occ = ten.view(pno_occ)
            for a in range(keep, nvir_ij):
                if abs(occ[a]) >= pno_scale * self.cut.t_cut_pno:
                    keep += 1

            X_pno = sparse.submatrix_cols(X_pno, range(keep), name="X (PNO)")

            # Orthonormal but not canonical yet; rotate so F is diagonal.
            pno_canon, e_pno = ten.canonicalizer(X_pno, F_pao_ij, name="PNO")
            X_pno = ten.doublet(X_pno, pno_canon, name="X (PNO)")

            K_pno = ten.triplet(X_pno, K_pao, X_pno, trans_a=True, name="K (PNO)")
            T_pno = ten.triplet(X_pno, T_pao, X_pno, trans_a=True, name="T (PNO)")
            Tt_pno = ten.triplet(X_pno, Tt_pao, X_pno, trans_a=True, name="Tt (PNO)")

            e_ij_trunc = ten.vector_dot(K_pno, Tt_pno)
            e_ij_os_trunc = ten.vector_dot(K_pno, T_pno)

            de = e_ij_initial - e_ij_trunc
            de_os = e_ij_os_initial - e_ij_os_trunc
            de_ss = (e_ij_initial - e_ij_os_initial) - (e_ij_trunc - e_ij_os_trunc)

            # PAO domain -> canonical PNO, in one transform.
            X_pno_global = ten.doublet(X_pao, X_pno, name="X (PAO->PNO)")

            K_pairs[ij] = K_pno
            T_pairs[ij] = T_pno
            self.X_pno[ij] = X_pno_global
            self.e_pno[ij] = e_pno
            self.n_pno[ij] = keep
            self.de_pno[ij] = de
            self.de_pno_os[ij] = de_os
            self.de_pno_ss[ij] = de_ss

            if i < j:
                K_pairs[ji] = ten.transpose(K_pno, name="K (PNO)")
                T_pairs[ji] = ten.transpose(T_pno, name="T (PNO)")
                self.X_pno[ji] = X_pno_global
                self.e_pno[ji] = e_pno
                self.n_pno[ji] = keep
                self.de_pno[ji] = de
                self.de_pno_os[ji] = de_os
                self.de_pno_ss[ji] = de_ss

        # Scatter the per-pair blocks into the flat stores, then form the
        # antisymmetrized amplitudes for every pair with one rank-3 permute
        # plus one axpby instead of two operations per pair.
        self._allocate_pair_stores()
        for ij in range(npairs):
            n = self.n_pno[ij]
            if n == 0:
                continue
            self.pair_block(self.K_all, ij)[:n, :n] = ten.view(K_pairs[ij])
            self.pair_block(self.T_all, ij)[:n, :n] = ten.view(T_pairs[ij])
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
