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
        self.dipole_pair_e = None
        self.dipole_pair_e_bound = None
        #: Correlation energy of the pairs dropped by prescreening, added back
        #: to the total (psi4's ``de_dipole_``).
        self.de_dipole = 0.0
        #: One canonical object per distinct domain; see _intern.
        self._interned = {}

        # compute_metric / compute_qia
        self.metric = None
        self.q_ia = None

        # Per-domain results shared across pairs (see _fit_coefficients and
        # _canonical_pao_domain). Screening multiplies the number of distinct
        # domains, which is exactly why the domain lists are interned.
        self._fit_cache = {}
        self._pao_domain_cache = {}
        self._pass_manager = None

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
        doi_ij = ten.zeros("DOI (i,j)", [naocc, naocc])
        doi_iu = ten.zeros("DOI (i,u)", [naocc, npao])
        nblocks = npoints = 0

        for phi_np, w_np, bf_map in ref.grid_blocks():
            npts = phi_np.shape[0]
            if npts == 0:
                continue
            nblocks += 1
            npoints += npts

            # The gather seam again: the block's basis functions are an index
            # list into the global coefficients.
            phi = ten.from_numpy("phi", phi_np)
            lmo = ten.doublet(phi, ten.from_numpy("C_lmo blk", C_lmo[bf_map, :]))
            pao = ten.doublet(phi, ten.from_numpy("C_pao blk", C_pao[bf_map, :]))

            # Squared orbital values, and a weighted copy of the LMO side. The
            # quadrature weight belongs to exactly one of the two factors.
            lmo2 = ten.zeros("|phi_i|^2", [npts, naocc])
            pao2 = ten.zeros("|phi_u|^2", [npts, npao])
            einsums.einsum("pi <- pi ; pi", lmo2, lmo, lmo)
            einsums.einsum("pu <- pu ; pu", pao2, pao, pao)

            w = ten.from_numpy("w", np.ascontiguousarray(w_np))
            lmo2_w = ten.zeros("w |phi_i|^2", [npts, naocc])
            einsums.einsum("pi <- pi ; p", lmo2_w, lmo2, w)

            einsums.einsum("ij <- pi ; pj", doi_ij, lmo2_w, lmo2, c_pf=1.0, ab_pf=1.0)
            einsums.einsum("iu <- pi ; pu", doi_iu, lmo2_w, pao2, c_pf=1.0, ab_pf=1.0)

        self.doi_ij = np.sqrt(np.abs(ten.view(doi_ij)))
        self.doi_iu = np.sqrt(np.abs(ten.view(doi_iu)))
        self._print(f"  DOI:      {nblocks} grid blocks, {npoints} points")
        return self

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
            dip_ii.append(np.diag(ten.view(ten.triplet(C_lmo, D, C_lmo, trans_a=True))).copy())
            dip_iu.append(ten.view(ten.triplet(C_lmo, D, C_pao, trans_a=True)).copy())

        # Orbital centroids <i|r|i>.
        R_i = np.stack(dip_ii, axis=1)  # naocc x 3

        # Per-LMO transition dipoles into its own orthocanonical PAO domain.
        # A looser DOI cutoff than the real domains: this is a prescreen, and
        # it must not be more aggressive than what it is screening for.
        lmo_dr, lmo_e = [], []
        for i in range(naocc):
            paos = [u for u in range(self.doi_iu.shape[1])
                    if abs(self.doi_iu[i, u]) > self.cut.t_cut_do_pre]
            paos = sparse.contract_lists(paos, ref.atom_to_bf)
            if not paos:
                lmo_dr.append(np.zeros((0, 3)))
                lmo_e.append(np.zeros(0))
                continue
            S_dom = sparse.submatrix_rows_and_cols(self.S_pao, paos, paos)
            F_dom = sparse.submatrix_rows_and_cols(self.F_pao, paos, paos)
            X_i, e_i = ten.orthocanonicalizer(S_dom, F_dom, self.cut.s_cut)
            dr = np.stack(
                [ten.view(ten.doublet(
                    ten.from_numpy("d", np.ascontiguousarray(dip_iu[x][[i], :][:, paos])), X_i))[0]
                 for x in range(3)], axis=1)          # npao_i x 3
            lmo_dr.append(dr)
            lmo_e.append(np.asarray(ten.view(e_i)).copy())

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

    # -- psi4 DLPNO::prep_sparsity -----------------------------------------

    def prep_sparsity(self):
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

        # -- LMO -> auxiliary domain, by Mulliken population ----------------
        # Population bookkeeping rather than tensor algebra: the products are
        # elementwise and the result is an index list, not an operand.
        if screening:
            C_lmo = ten.view(self.C_lmo)
            S = np.asarray(ref.S)
            natom = len(ref.atom_to_bf)
            bf_to_atom = np.empty(ref.nbf, dtype=int)
            for a, bfs in enumerate(ref.atom_to_bf):
                bf_to_atom[list(bfs)] = a

            self.lmo_to_ribfs = []
            for i in range(naocc):
                c = C_lmo[:, i]
                P = c[:, None] * S * c[None, :]
                d = np.diag(P)
                denom = d[:, None] + d[None, :]
                # psi4 splits each off-diagonal population between the two
                # centres in proportion to their diagonal populations.
                with np.errstate(divide="ignore", invalid="ignore"):
                    share_u = np.where(denom != 0.0, P * (d[:, None] / denom), 0.0)
                    share_v = np.where(denom != 0.0, P * (d[None, :] / denom), 0.0)
                pop = np.zeros(natom)
                np.add.at(pop, bf_to_atom, share_u.sum(axis=1))
                np.add.at(pop, bf_to_atom, share_v.sum(axis=0))

                ribfs = []
                for a in range(natom):
                    if abs(pop[a]) > self.cut.t_cut_mkn:
                        ribfs.extend(ref.atom_to_ribf[a])
                self.lmo_to_ribfs.append(self._intern(sorted(ribfs)))
        else:
            self.lmo_to_ribfs = [self._intern(range(naux))] * naocc

        # -- LMO -> PAO domain, by differential overlap ---------------------
        if screening:
            self.lmo_to_paos = []
            for i in range(naocc):
                paos = [u for u in range(npao) if abs(self.doi_iu[i, u]) > self.cut.t_cut_do]
                # Any PAO on an atom pulls in all of that atom's PAOs.
                self.lmo_to_paos.append(
                    self._intern(sparse.contract_lists(paos, ref.atom_to_bf)))
        else:
            self.lmo_to_paos = [self._intern(range(npao))] * naocc

        # -- which (i, j) pairs survive -------------------------------------
        self.de_dipole = 0.0
        # Only worth building when a pair can actually be rejected. Under
        # `untruncated()` the thresholds are negative, so every pair survives
        # and the estimate would be an expensive unused quantity: one
        # orthocanonicalization of the *full* PAO space per LMO.
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

        # -- pair domains are the union of the two LMO domains --------------
        self.lmopair_to_paos = [
            self._intern(sparse.merge_lists(self.lmo_to_paos[i], self.lmo_to_paos[j]))
            for (i, j) in self.ij_to_i_j
        ]
        self.lmopair_to_ribfs = [
            self._intern(sparse.merge_lists(self.lmo_to_ribfs[i], self.lmo_to_ribfs[j]))
            for (i, j) in self.ij_to_i_j
        ]

        # LMOs m for which both (i,m) and (j,m) survived. The residual's
        # coupling sum runs over these, not over all naocc.
        self.lmopair_to_lmos = [
            [m for m in range(naocc)
             if self.i_j_to_ij[i, m] != -1 and self.i_j_to_ij[j, m] != -1]
            for (i, j) in self.ij_to_i_j
        ]

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

    def _domain_key(self, ij):
        """What :meth:`_fit_coefficients` caches on: the pair's two domains."""
        return (id(self.lmopair_to_ribfs[ij]), id(self.lmopair_to_paos[ij]))

    def _fit_operands(self, ij):
        """``(A, rhs)`` for one domain's fitting equations, gathered on the host.

        Split out of :meth:`_fit_coefficients` so the solves can be captured as
        a batch. The gathers stay eager: they are index lists, and there is no
        einsums primitive for "take these rows and these columns".
        """
        ribfs = self.lmopair_to_ribfs[ij]
        paos = self.lmopair_to_paos[ij]
        naocc = self.ref.naocc
        block = ten.view(self.q_ia)[np.ix_(ribfs, range(naocc), paos)]
        rhs = ten.from_numpy("(Q|i u) domain", block.reshape(len(ribfs), naocc * len(paos)))
        A = sparse.submatrix_rows_and_cols(self.metric, ribfs, ribfs, name="(P|Q) domain")
        return A, rhs

    def _reshape_fit(self, ij, rhs):
        """The solved right-hand side, back to ``(naux_dom, naocc, npao_dom)``."""
        ribfs = self.lmopair_to_ribfs[ij]
        paos = self.lmopair_to_paos[ij]
        solved = ten.view(rhs).reshape(len(ribfs), self.ref.naocc, len(paos))
        return ten.from_numpy("J^-1 (Q|i u)", solved)

    def precompute_fits(self):
        """Solve every distinct domain's fitting equations in one graph.

        The single largest item in the PNO transform: 31% of the phase at
        ethanol/cc-pVTZ, and only 23 calls, because the solve is per *domain*
        rather than per pair and each one carries ``naocc * npao`` right-hand
        sides.

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

        first = {}
        for ij in range(self.n_lmo_pairs):
            first.setdefault(self._domain_key(ij), ij)
        if not first:
            return self

        jobs = [(key, ij) + self._fit_operands(ij) for key, ij in first.items()]
        g = cg.Graph("domain fits")
        with cg.capture(g):
            for _, _, A, rhs in jobs:
                la.gesv(A, rhs)
        g.set_executor(cg.OpenMPExecutor())
        g.execute()

        for key, ij, _, rhs in jobs:
            self._fit_cache[key] = self._reshape_fit(ij, rhs)
        self._print(f"  fits:     {len(jobs)} distinct domains solved in one graph")
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
        key = self._domain_key(ij)
        hit = self._fit_cache.get(key)
        if hit is not None:
            return hit

        # Lazy fallback for anything precompute_fits() did not cover. It also
        # keeps gesv's info check, which the captured path cannot see.
        A_solve, rhs = self._fit_operands(ij)
        solved = ten.view(ten.solve(A_solve, rhs)).reshape(
            len(self.lmopair_to_ribfs[ij]), self.ref.naocc, len(self.lmopair_to_paos[ij]))

        hit = ten.from_numpy("J^-1 (Q|i u)", solved)
        self._fit_cache[key] = hit
        return hit

    def _batched_eigh(self, mats, descending=True, label="eigh"):
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
        here; see :meth:`precompute_fits` for why.
        """
        evecs = [ten.from_numpy(f"{label} vectors", ten.view(A)) for A in mats]
        evals = [ten.zeros(f"{label} values", [ten.shape(A)[0]]) for A in mats]
        if evecs:
            g = cg.Graph(label)
            with cg.capture(g):
                for V, w in zip(evecs, evals):
                    la.syev(V, w, compute_eigenvectors=True)
            g.set_executor(cg.OpenMPExecutor())
            g.execute()
        if descending:
            for V, w in zip(evecs, evals):
                v, e = ten.view(V), ten.view(w)
                v[...] = v[:, ::-1].copy()
                e[...] = e[::-1].copy()
        return evals, evecs

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

        # Three stages rather than one loop, so the two eigendecompositions per
        # pair can be batched. Each stage is the same arithmetic in the same
        # order as the per-pair loop it replaces; only the issue order changes.
        #
        # Stage 1: everything up to the pair density, which depends on nothing
        # outside its own pair.
        upper = [ij for ij, (i, j) in enumerate(self.ij_to_i_j) if i <= j]
        st = {}
        for ij in upper:
            i, j = self.ij_to_i_j[ij]
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

            # Pair density; its eigenvectors are the PNOs, eigenvalues the
            # PNO occupation numbers.
            D_ij = ten.doublet(Tt_pao, T_pao, trans_b=True, name="D (pair)")
            la.axpby(1.0, ten.doublet(Tt_pao, T_pao, trans_a=True), 1.0, D_ij)

            st[ij] = dict(
                K_pao=K_pao, T_pao=T_pao, Tt_pao=Tt_pao, D_ij=D_ij,
                X_pao=X_pao, F_pao_ij=F_pao_ij,
                e_ij_initial=ten.vector_dot(K_pao, Tt_pao),
                e_ij_os_initial=ten.vector_dot(K_pao, T_pao),
            )

        occs, vecs = self._batched_eigh([st[ij]["D_ij"] for ij in upper],
                                        descending=True, label="PNO")

        # Stage 2: the truncation decision, which is data dependent and so
        # cannot be captured, then the Fock matrix in each surviving subspace.
        for ij, pno_occ, X_pno in zip(upper, occs, vecs):
            i, j = self.ij_to_i_j[ij]
            nvir_ij = ten.shape(st[ij]["K_pao"])[0]
            pno_scale = 1.0
            if i < self.ref.n_core or j < self.ref.n_core:
                pno_scale *= self.cut.t_cut_pno_core_scale

            keep = min(self.cut.min_pnos, nvir_ij)
            occ = ten.view(pno_occ)
            for a in range(keep, nvir_ij):
                if abs(occ[a]) >= pno_scale * self.cut.t_cut_pno:
                    keep += 1

            X_pno = sparse.submatrix_cols(X_pno, range(keep), name="X (PNO)")
            st[ij]["X_pno"] = X_pno
            st[ij]["keep"] = keep
            # Orthonormal but not canonical yet; rotate so F is diagonal.
            st[ij]["Fmo"] = ten.triplet(X_pno, st[ij]["F_pao_ij"], X_pno,
                                        trans_a=True, name="C^T F C")

        e_pnos, canons = self._batched_eigh([st[ij]["Fmo"] for ij in upper],
                                            descending=True, label="PNO canon")

        # Stage 3: rotate into the canonical PNO basis and record the pair.
        for ij, e_pno, pno_canon in zip(upper, e_pnos, canons):
            i, j = self.ij_to_i_j[ij]
            ji = self.ij_to_ji[ij]
            K_pao, T_pao, Tt_pao = st[ij]["K_pao"], st[ij]["T_pao"], st[ij]["Tt_pao"]
            X_pao, keep = st[ij]["X_pao"], st[ij]["keep"]
            e_ij_initial = st[ij]["e_ij_initial"]
            e_ij_os_initial = st[ij]["e_ij_os_initial"]

            X_pno = ten.doublet(st[ij]["X_pno"], pno_canon, name="X (PNO)")

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
