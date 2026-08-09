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
        # The fits are keyed by (domain, LMO) and hold ``(parent, position)``:
        # one solve produces a domain's whole block and each LMO's fit is a
        # column slice of it, so the entry carries the parent to keep it alive.
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

        # sqrt(abs(...)) without leaving the tensor layer: linalg.abs and the
        # new element-wise linalg.sqrt. (linalg.pow is a *matrix* power via
        # eigendecomposition, not this.)
        for src in (doi_ij, doi_iu):
            la.abs(src, src)
            la.sqrt(src, src)
        self.doi_ij = np.asarray(ten.view(doi_ij)).copy()
        self.doi_iu = np.asarray(ten.view(doi_iu)).copy()
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
            dip_ii.append(
                ten.view(ten.diagonal(ten.triplet(C_lmo, D, C_lmo, trans_a=True))).copy())
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

        # Occupied index first. Both orders give the same answer, but the
        # half-transform carries whichever index has already been contracted,
        # and there are naocc of those against npao of the other - 13 against
        # 174 at ethanol/cc-pVTZ, since the PAOs span the whole AO basis. Doing
        # the PAO index first makes the dominant contraction naux*nbf^2*npao
        # instead of naux*nbf^2*naocc and leaves a half-transform the size of
        # the integrals themselves: 4.79 GFLOP and 97.7 MiB against 0.67 GFLOP
        # and 7.3 MiB. The gap is npao/naocc, so it widens with basis set.
        half = ten.zeros("(Q|i n)", [ref.naux, ref.naocc, ref.nbf])
        einsums.einsum("Qin <- Qmn ; mi", half, Qmn, self.C_lmo)
        self.q_ia = ten.zeros("(Q|i u)", [ref.naux, ref.naocc, npao])
        einsums.einsum("Qiu <- Qin ; nu", self.q_ia, half, self.C_pao)

        self._print(
            f"  DF ints:  (Q|iu) is {ref.naux} x {ref.naocc} x {npao} "
            f"({ten.view(self.q_ia).nbytes / 2**20:.1f} MiB, dense)"
        )
        return self

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
            jobs.append((
                key, lmos,
                [list(ribfs), [int(i) for i in lmos], list(paos)],
                [list(ribfs), list(ribfs)],
                ten.zeros("(P|Q) domain", [nq, nq]),
                ten.zeros("(Q|i u) domain", [nq, nk, nu]),
                ten.zeros("(Q|i u) domain flat", [nq, nk * nu]),
                ten.zeros("J^-1 (Q|i u)", [nq, nk, nu]),
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

        Three pieces since the M6 split: :meth:`plan_pno_transform` warms the
        domain memos and flattens the numerics' arguments,
        :func:`dlpno.pno_xform.transform_pnos` is the promotable numerics, and
        :meth:`_finish_pno_transform` mirrors the results onto the lower
        triangle and scatters them into the flat stores. This method is the
        composition, so the non-stage path is unchanged.

        PNOs defined in DOI 10.1063/1.3086717, equations 17 through 24.
        """
        from .pno_xform import transform_pnos

        args = self.plan_pno_transform()
        self._finish_pno_transform(transform_pnos(**args))
        return self

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
        upper = [ij for ij, (i, j) in enumerate(self.ij_to_i_j) if i <= j]

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
            t_cut_pno=float(self.cut.t_cut_pno),
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
            e_ij_initial = float(ten.view(result.e_initial[u])[0])
            e_ij_os_initial = float(ten.view(result.e_os_initial[u])[0])
            e_ij_trunc = float(ten.view(result.e_trunc[u])[0])
            e_ij_os_trunc = float(ten.view(result.e_os_trunc[u])[0])
            de = e_ij_initial - e_ij_trunc
            de_os = e_ij_os_initial - e_ij_os_trunc
            de_ss = (e_ij_initial - e_ij_os_initial) - (e_ij_trunc - e_ij_os_trunc)
            for target in ((ij, ji) if i < j else (ij,)):
                self.X_pno[target] = result.X_pno[u]
                self.e_pno[target] = result.e_pno[u]
                self.n_pno[target] = result.n_pno[u]
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
            K = ten.view(result.K_pno[u])
            T = ten.view(result.T_pno[u])
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
