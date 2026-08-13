#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""DLPNO-CCSD(T), ported from psi4's ``dlpno/triples.cc``.

The perturbative triples correction of Jiang et al. (JCP 161, 082502, 2024),
sitting on top of the converged DLPNO-CCSD of :mod:`dlpno.ccsd`.

**The shape of the calculation is a second, smaller version of the pair
cascade.** Where CCSD screened pairs twice and built PNOs twice, the triples
screen triplets twice and build triplet natural orbitals twice:

1. *Prescreening.* Enumerate every triplet that can be formed from surviving
   pairs, build deliberately cheap TNOs at ``t_cut_tno_pre``, and compute the
   whole (T0) correction with them. The number that comes out is thrown away;
   what is kept is the per-triplet breakdown.
2. *Screening.* Triplets whose prescreening energy falls below
   ``t_cut_triples_weak`` leave the list, and their estimated energy is booked
   once as ``de_lccsd_t_screened`` - the exact analogue of what
   ``de_lmp2_eliminated`` is for pairs.
3. *Production.* Rebuild the TNOs for the survivors at the full ``t_cut_tno``
   and compute the correction again. That is the (T0) energy.

Two things about the correction itself are worth stating before reading any
code, because both are easy to mistake for defects.

**(T0) is not the same number as (T), and it is not invariant to a rotation of
the occupied orbitals.** Its energy denominator carries only the DIAGONAL of the
occupied Fock matrix, ``F_ii + F_jj + F_kk``, so in the localized basis this
port works in it drops the off-diagonal ``F_il`` coupling between triplets. On
water/cc-pVDZ that is worth 1.5e-4 Eh against the canonical value, which is why
the (T0) gate has to be run in two occupied bases and
``test_lccsd_t0.py`` does.

:mod:`dlpno.lccsd_t` puts the coupling back by iterating, and that restores the
invariance: untruncated in the localized basis the iterative (T) reproduces
canonical DF-CCSD(T) to 1e-12 with no canonicalized rerun. ``t0_approximation``
selects between the two, as psi4's option of that name does.

**Triplets with i == j == k are skipped, and that is exact.** All six
permutations of the ``P_ijk^abc`` operator map such a triplet onto itself, so
``W``, ``V`` and the denominator come out fully symmetric in ``abc`` and the
energy's six coefficients ``8 - 4 - 4 - 4 + 2 + 2`` cancel identically.
"""

import time

import numpy as np

from einsums import linalg as la
import einsums.graph as cg

from . import sparse
from . import tensors as ten
from .base import batched_eigh, batched_orthocanonicalizer
from .cc_overlaps import OverlapHalfCache, build_overlaps
from .ccsd import DLPNOCCSD
from .lccsd_t import LCCSDT
from .lccsd_t0 import LCCSDT0

__all__ = ["DLPNOCCSDT"]


class DLPNOCCSDT(DLPNOCCSD):
    """DLPNO-CCSD(T): the triplet cascade on top of the pair one.

    Derives from :class:`~dlpno.ccsd.DLPNOCCSD`, as psi4's ``DLPNOCCSD_T``
    does, because it genuinely is that calculation plus a phase: every pair
    quantity it reads - the converged amplitudes, the PNO bases, the pair
    domains - is left behind by ``compute_energy``'s cascade.
    """

    def __init__(self, reference, thresholds=None, verbose=True, use_diis=True,
                 integral_source=None):
        super().__init__(reference, thresholds, verbose, use_diis=use_diis,
                         integral_source=integral_source)

        #: Unique triplets ``i <= j <= k``, and the reverse lookup over all six
        #: orderings of each. Dense rather than psi4's hash map: the (T0) inner
        #: loops index it, and ``naocc^3`` integers is small next to one TNO
        #: block.
        self.ijk_to_i_j_k = []
        self.i_j_k_to_ijk = None

        #: Per triplet, its auxiliary, LMO and PAO domains.
        self.lmotriplet_to_ribfs = []
        self.lmotriplet_to_lmos = []
        self.lmotriplet_to_paos = []

        #: Per triplet, the PAO-domain-to-TNO transform, the TNO energies and
        #: the count. Rebuilt from scratch by every :meth:`tno_transform`.
        self.X_tno = []
        self.e_tno = []
        self.n_tno = []
        #: Per triplet, the multiplier on ``t_cut_tno``. All ones until
        #: :meth:`sort_triplets` splits triplets into strong and weak for the
        #: iterative (T).
        self.tno_scale = []
        #: Per triplet, whether :meth:`sort_triplets` called it strong.
        self.is_strong_triplet = []

        #: Per triplet, its (T0) energy. What the screening pass tests and what
        #: the energy is the sum of.
        self.e_ijk = np.zeros(0)

        self.e_t0 = 0.0
        self.e_t0_pre = 0.0
        #: The iterative pass, in its own looser TNO space.
        self.e_t0_crude = 0.0
        self.e_t_raw = 0.0
        self.de_t = 0.0
        self.lccsd_t = None
        self.lccsd_t0 = None
        #: Energy of the triplets screening dropped, psi4's
        #: ``de_lccsd_t_screened_``.
        self.de_lccsd_t_screened = 0.0
        self.e_lccsd_t = 0.0

        #: The singles amplitudes, snapshotted out of the CCSD solver before it
        #: is dropped; see :meth:`_release_ccsd`.
        self.T_ia = None
        #: Timings, so ``bench_vs_psi4.py`` can price the phases psi4 prices.
        self.t_sparsity = 0.0
        self.t_tno = 0.0
        self.t_t0 = 0.0
        #: Captured nodes and chunks of the last (T0) pass.
        self.t0_nodes = 0
        self.t0_chunks = 0
        #: The pair-side overlap halves, shared by every (T0) pass of this
        #: calculation; see :meth:`compute_lccsd_t0` for why it is created
        #: there rather than here.
        self._overlap_halves = None

    @property
    def n_lmo_triplets(self):
        return len(self.ijk_to_i_j_k)

    # -- psi4 DLPNOCCSD_T::triples_sparsity ---------------------------------

    def triples_sparsity(self, prescreening):
        """Enumerate or screen the triplets, then build their three domains.

        The two halves of psi4's function, which share nothing but the domain
        construction at the end.

        *Prescreening* forms a triplet ``i <= j <= k`` for every surviving pair
        ``ij`` and every ``k`` in that pair's neighbour list, subject to one
        rule: a triplet may contain at most ``triples_max_weak_pairs`` weak
        pairs among ``ij``, ``ik`` and ``kj``. That is the only place the
        strong/weak split from the CCSD cascade reaches the triples.

        *Screening* keeps the triplets whose prescreening energy clears
        ``t_cut_triples_weak`` and books the rest into
        ``de_lccsd_t_screened``. Like the pair elimination and unlike the
        strong/weak split, this REMOVES triplets: the domains are rebuilt
        without them.

        Both passes then rebuild all three domain families, at the thresholds
        belonging to the pass. Those are the triples' own
        (``t_cut_do_triples``, ``t_cut_mkn_triples``), looser than the pairs'
        by design, and they come from the same two constructions the pair path
        uses - see :meth:`~dlpno.base.DLPNOBase.mulliken_aux_domains`.
        """
        t0 = time.perf_counter()
        naocc = self.ref.naocc

        if prescreening:
            self.ijk_to_i_j_k = self._enumerate_triplets()
            self.de_lccsd_t_screened = 0.0
        else:
            self.ijk_to_i_j_k, self.de_lccsd_t_screened = self._screen_triplets()

        n_trip = len(self.ijk_to_i_j_k)
        self.i_j_k_to_ijk = np.full((naocc, naocc, naocc), -1, dtype=int)
        for ijk, (i, j, k) in enumerate(self.ijk_to_i_j_k):
            for p, q, r in ((i, j, k), (i, k, j), (j, i, k),
                            (j, k, i), (k, i, j), (k, j, i)):
                self.i_j_k_to_ijk[p, q, r] = ijk
        self.tno_scale = [1.0] * n_trip

        # -- the three domain families ---------------------------------------
        t_mkn = (self.cut.t_cut_mkn_triples_pre if prescreening
                 else self.cut.t_cut_mkn_triples)
        t_do = (self.cut.t_cut_do_triples_pre if prescreening
                else self.cut.t_cut_do_triples)
        lmo_to_ribfs = self.mulliken_aux_domains(t_mkn)
        lmo_to_paos = self.doi_pao_domains(t_do)

        self.lmotriplet_to_ribfs = []
        self.lmotriplet_to_paos = []
        self.lmotriplet_to_lmos = []
        for i, j, k in self.ijk_to_i_j_k:
            self.lmotriplet_to_ribfs.append(self._intern(sparse.merge_lists(
                lmo_to_ribfs[i], sparse.merge_lists(lmo_to_ribfs[j],
                                                    lmo_to_ribfs[k]))))
            self.lmotriplet_to_paos.append(self._intern(sparse.merge_lists(
                lmo_to_paos[i], sparse.merge_lists(lmo_to_paos[j],
                                                   lmo_to_paos[k]))))
            # LMOs forming a surviving pair with all three of i, j and k.
            self.lmotriplet_to_lmos.append([
                l for l in range(naocc)
                if self.i_j_to_ij[i, l] != -1 and self.i_j_to_ij[j, l] != -1
                and self.i_j_to_ij[k, l] != -1])

        possible = (naocc + 2) * (naocc + 1) * naocc // 6 - naocc
        label = "crude" if prescreening else "screened"
        self._print(
            f"  triplets: {n_trip} of {possible} possible ({label}), "
            f"{sum(len(d) for d in self.lmotriplet_to_paos) / max(n_trip, 1):.1f} PAOs "
            f"and {sum(len(d) for d in self.lmotriplet_to_ribfs) / max(n_trip, 1):.1f} aux "
            f"per triplet")
        if not prescreening:
            self._print("            screened triplets contribute "
                        f"{self.de_lccsd_t_screened:.12f} Eh")
        self.t_sparsity += time.perf_counter() - t0
        return self

    def _enumerate_triplets(self):
        """Every triplet formable from the surviving pairs, psi4's first pass.

        ``i <= j <= k`` with not all three equal, ``k`` drawn from pair
        ``ij``'s neighbour list so a triplet only forms where all three of its
        pairs survived, and at most ``triples_max_weak_pairs`` of those three
        weak.
        """
        max_weak = int(self.cut.triples_max_weak_pairs)
        out = []
        for ij, (i, j) in enumerate(self.ij_to_i_j):
            if i > j:
                continue
            for k in self.lmopair_to_lmos[ij]:
                if i > k or j > k:
                    continue
                if i == j and j == k:
                    continue
                weak = sum(1 for (p, q) in ((i, j), (i, k), (k, j))
                           if self.i_j_to_ij_weak[p, q] != -1)
                if weak > max_weak:
                    continue
                out.append((i, j, k))
        return out

    def _screen_triplets(self):
        """Drop the triplets the prescreening pass found negligible.

        Returns ``(survivors, dropped_energy)``. The dropped energy is the
        prescreening estimate, which is the whole point of computing a (T0)
        nobody reads: an estimated triplet costs one cheap TNO space instead of
        a full one.
        """
        keep, dropped = [], 0.0
        for ijk, (i, j, k) in enumerate(self.ijk_to_i_j_k):
            if abs(self.e_ijk[ijk]) >= self.cut.t_cut_triples_weak:
                keep.append((i, j, k))
            else:
                dropped += float(self.e_ijk[ijk])
        return keep, dropped

    # -- psi4 DLPNOCCSD_T::sort_triplets ------------------------------------

    def sort_triplets(self, e_total):
        """Split triplets into strong and weak by their share of the energy.

        Walk the triplets in descending order of ``|e_ijk|`` and call them
        strong until the running total passes 90% of ``e_total``; the rest are
        weak. The split does not remove anything - it sets ``tno_scale``, a
        MULTIPLIER on ``t_cut_tno``, so a weak triplet gets a looser TNO space
        and costs less rather than being dropped.

        The direction is worth stating because the naming invites the opposite
        reading: both scales are above one (10 and 100 at psi4's defaults), so
        both classes get a LOOSER cutoff than ``t_cut_tno`` alone, and weak
        looser still. The iterative (T) is expensive enough that psi4 pays for
        it in a deliberately smaller space than the (T0) it corrects.
        """
        n_trip = self.n_lmo_triplets
        order = sorted(range(n_trip), key=lambda ijk: -abs(self.e_ijk[ijk]))

        strong = float(self.cut.t_cut_tno_strong_scale)
        weak = float(self.cut.t_cut_tno_weak_scale)
        self.is_strong_triplet = [False] * n_trip
        self.tno_scale = [weak] * n_trip

        running, n_strong = 0.0, 0
        for ijk in order:
            self.is_strong_triplet[ijk] = True
            self.tno_scale[ijk] = strong
            running += self.e_ijk[ijk]
            n_strong += 1
            # psi4 breaks AFTER promoting, so the triplet that crosses the
            # line is strong. Reproduced rather than tidied: the counts are
            # compared against psi4's.
            if e_total != 0.0 and running / e_total > 0.9:
                break

        self._print(
            f"  triplets: {n_strong} strong, {n_trip - n_strong} weak "
            f"({100.0 * n_strong / max(n_trip, 1):.1f}% strong); "
            f"t_cut_tno scaled by {strong:g} and {weak:g}")
        return self

    # -- psi4 DLPNOCCSD_T::tno_transform ------------------------------------

    def tno_transform(self, t_cut_tno):
        """Build each triplet's truncated, canonical TNO basis.

        The triples analogue of ``recompute_pnos``, and structurally the same
        construction: form a density from the converged amplitudes, diagonalize
        it, keep the leading occupations, recanonicalize what survives. What
        differs is where the density comes from - a triplet has no amplitudes
        of its own, so psi4 averages the three PAIR densities of ``ij``, ``jk``
        and ``ik``, each first projected into the triplet's own orthocanonical
        PAO space.

        Three things are done once here that psi4 does once per triplet, and
        all three are pure sharing rather than a change of arithmetic:

        * the orthocanonicalization of a triplet PAO domain, which depends only
          on the domain and which the interned domain lists make shareable;
        * the pair densities ``Tt T^T + Tt^T T``, which depend only on the
          pair, where psi4 rebuilds them for each triplet the pair appears in;
        * the pair-to-triplet overlaps, which depend on the pair and the
          triplet's DOMAIN rather than on the triplet.

        The eigendecompositions go out in batches for the reason
        :func:`~dlpno.base.batched_eigh` gives: issued one at a time they get
        slower as threads are added.
        """
        t0 = time.perf_counter()
        n_trip = self.n_lmo_triplets
        if n_trip == 0:
            self.X_tno, self.e_tno, self.n_tno = [], [], []
            return self

        domains, dom_of = self._triplet_domains()
        canon = self._warm_triplet_domains(domains)
        D_pair = self._pair_densities()
        S = self._pair_to_domain_overlaps(domains, dom_of,
                                          [c[0] for c in canon])

        # -- the triplet densities -------------------------------------------
        D_ijk = []
        g = cg.Graph("TNO densities")
        with cg.capture(g):
            for ijk, (i, j, k) in enumerate(self.ijk_to_i_j_k):
                dom = dom_of[ijk]
                n_can = ten.shape(canon[dom][0])[1]
                D = ten.zeros(f"D (triplet {ijk})", [n_can, n_can])
                for pair in (int(self.i_j_to_ij[i, j]), int(self.i_j_to_ij[j, k]),
                             int(self.i_j_to_ij[i, k])):
                    block = D_pair[pair]
                    if block is None:
                        continue
                    bridge = S[pair, dom]
                    # S^T D S, accumulated. One third, applied per term rather
                    # than to the sum, because the three are added into the
                    # same destination and there is no sum to scale.
                    half = ten.doublet(bridge, block, trans_a=True,
                                       name="S^T D")
                    la.gemm(1.0 / 3.0, half, bridge, 1.0, D)
                D_ijk.append(D)
        self._run(g)

        occs, vecs = batched_eigh(D_ijk, self._run, descending=True,
                                  label="TNO")

        # -- truncation, then recanonicalization ------------------------------
        keep = [self._keep_tnos(ten.view(occs[ijk]), self.tno_scale[ijk],
                                t_cut_tno)
                for ijk in range(n_trip)]
        X_trunc = [vecs[ijk][:, :keep[ijk]] for ijk in range(n_trip)]

        Fmo = []
        g2 = cg.Graph("TNO canonicalization")
        with cg.capture(g2):
            for ijk in range(n_trip):
                F_can = canon[dom_of[ijk]][2]
                Fmo.append(ten.triplet(X_trunc[ijk], F_can, X_trunc[ijk],
                                       trans_a=True, name="C^T F C"))
        self._run(g2)
        self.e_tno, tno_canon = batched_eigh(Fmo, self._run, descending=True,
                                             label="TNO canon")

        self.X_tno = [None] * n_trip
        g3 = cg.Graph("TNO transforms")
        with cg.capture(g3):
            for ijk in range(n_trip):
                X_pao = canon[dom_of[ijk]][0]
                # Composed onto the domain's orthocanonicalizer, so X_tno maps
                # the triplet's raw PAOs straight to canonical TNOs and every
                # consumer needs one transform rather than two.
                self.X_tno[ijk] = ten.doublet(
                    X_pao, ten.doublet(X_trunc[ijk], tno_canon[ijk],
                                       name="TNO (canonical)"),
                    name="X (PAO->TNO)")
        self._run(g3)
        self.n_tno = [ten.shape(X)[1] for X in self.X_tno]

        counts = self.n_tno
        self._print(
            f"  TNOs:     avg {sum(counts) / n_trip:.1f}, min {min(counts)}, "
            f"max {max(counts)} per LMO triplet at t_cut_tno {t_cut_tno:.3e}")
        self.t_tno += time.perf_counter() - t0
        return self

    def _triplet_domains(self):
        """The distinct triplet PAO domains, and which one each triplet uses.

        Distinct by identity, which the interning in
        :meth:`~dlpno.base.DLPNOBase._intern` makes equivalent to distinct by
        value. Triplets sharing a domain share their whole orthocanonical
        basis, and on a compact molecule most of them do.
        """
        index, domains, dom_of = {}, [], []
        for paos in self.lmotriplet_to_paos:
            slot = index.setdefault(id(paos), len(domains))
            if slot == len(domains):
                domains.append(paos)
            dom_of.append(slot)
        return domains, dom_of

    def _warm_triplet_domains(self, domains):
        """``(X_pao, e_pao, F_can)`` per distinct triplet PAO domain."""
        S_doms = [sparse.submatrix_rows_and_cols(self.S_pao, d, d) for d in domains]
        F_doms = [sparse.submatrix_rows_and_cols(self.F_pao, d, d) for d in domains]
        canon = batched_orthocanonicalizer(S_doms, F_doms, self.cut.s_cut,
                                           self._run)
        out = []
        g = cg.Graph("triplet domain Fock matrices")
        with cg.capture(g):
            for (X, e), F_dom in zip(canon, F_doms):
                out.append((X, e, ten.triplet(X, F_dom, X, trans_a=True,
                                              name="F (can PAO)")))
        self._run(g)
        return out

    def _pair_densities(self):
        """``D_ij = Tt_ij T_ij^T + Tt_ij^T T_ij``, once per pair.

        psi4's ``tno_transform`` builds these inside its per-triplet loop, so a
        pair appearing in fifty triplets pays for fifty identical
        decompositions of the same amplitudes. They depend on nothing but the
        pair.

        The diagonal pairs carry psi4's factor of one half, which is there
        because ``T_ii`` is symmetric and the two products above are then the
        same matrix counted twice.
        """
        wanted = set()
        for i, j, k in self.ijk_to_i_j_k:
            wanted.update((int(self.i_j_to_ij[i, j]), int(self.i_j_to_ij[j, k]),
                           int(self.i_j_to_ij[i, k])))

        out = [None] * self.n_lmo_pairs
        # The views outlive the capture deliberately: a view handed to a
        # captured operation is held by the graph as a slot pointer, so one
        # that dies with the loop iteration that made it leaves the replay
        # reading freed memory.
        views = []
        g = cg.Graph("pair densities for the TNOs")
        with cg.capture(g):
            for pair in sorted(wanted):
                if pair == -1 or not self.n_pno[pair]:
                    continue
                i, j = self.ij_to_i_j[pair]
                n = self.n_pno[pair]
                # Logical rather than padded: the density is bridged into the
                # triplet's basis by an overlap sized to the pair, so the
                # padding a bucket adds would simply not fit.
                T = self.layout.logical_view(self.T_all, pair)
                Tt = self.layout.logical_view(self.Tt_all, pair)
                views += [T, Tt]
                # One half on the diagonal pairs, which is psi4's: ``T_ii`` is
                # symmetric, so the two products below are the same matrix and
                # the unscaled sum would count it twice.
                scale = 0.5 if i == j else 1.0
                D = ten.zeros(f"D (pair {pair})", [n, n])
                la.gemm(scale, Tt, T, 0.0, D, trans_b=True)
                la.gemm(scale, Tt, T, 1.0, D, trans_a=True)
                out[pair] = D
        self._run(g)
        self._density_views = views
        return out

    def _pair_to_domain_overlaps(self, domains, dom_of, canon_X):
        """``X_pno[ij]^T S_pao X_pao[domain]`` for every pair a triplet reads.

        Keyed by ``(pair, domain slot)`` rather than by ``(pair, triplet)``:
        the right factor is the domain's orthocanonical PAO basis, which many
        triplets share, so the request set is far smaller than the triplet
        count.

        Built through the same :func:`~dlpno.cc_overlaps.build_overlaps` the
        pair families use, with the domains appended to the pair list as extra
        entities. psi4 instead restricts ``S_pao`` to a merged "extended"
        domain and indexes into it; scattering each transform onto the full PAO
        axis makes the restriction implicit and gives the identical matrix.
        """
        npairs = self.n_lmo_pairs
        X_all = list(self.X_pno) + list(canon_X)
        paos_all = list(self.lmopair_to_paos) + list(domains)
        n_all = list(self.n_pno) + [ten.shape(X)[1] for X in canon_X]

        requests = set()
        for ijk, (i, j, k) in enumerate(self.ijk_to_i_j_k):
            dom = npairs + dom_of[ijk]
            for pair in (int(self.i_j_to_ij[i, j]), int(self.i_j_to_ij[j, k]),
                         int(self.i_j_to_ij[i, k])):
                if pair != -1 and self.n_pno[pair]:
                    requests.add((pair, dom))
        keys = sorted(requests)
        blocks = build_overlaps(X_all, self.S_pao, paos_all, n_all, keys)
        return {(pair, dom - npairs): block
                for (pair, dom), block in zip(keys, blocks)}

    def _keep_tnos(self, occ, scale, t_cut_tno):
        """How many TNOs survive, by psi4's selection loop.

        Counted rather than found as a prefix, which is psi4's own spelling and
        is not the same thing when an occupation number comes out negative: the
        eigenvalues are sorted by VALUE and the test is on their absolute
        value, and psi4 then takes that many columns from the front whichever
        ones satisfied it.
        """
        cut = scale * t_cut_tno
        min_tnos = int(self.cut.min_tnos)
        kept = sum(1 for a, value in enumerate(occ)
                   if abs(value) >= cut or a < min_tnos)
        return min(len(occ), max(1, kept))

    # -- psi4 DLPNOCCSD_T::compute_lccsd_t0 ---------------------------------

    def compute_lccsd_t0(self, retain=False):
        """The semicanonical triples correction; see :mod:`dlpno.lccsd_t0`.

        Sets :attr:`e_ijk` per triplet and returns the total. ``retain`` keeps
        ``W``, ``V`` and the amplitudes for the iterative pass, and is psi4's
        ``save_memory`` argument; the solver itself is kept on
        :attr:`lccsd_t0` so :meth:`lccsd_t_iterations` can read them.

        The pair-side overlap halves are shared across the two or three passes
        this method is called for, because every pair's transform, PAO domain and
        PNO count are fixed by the time the first one runs: the passes differ in
        their TNO space, which is the side the cache deliberately excludes. The
        cache is created on FIRST USE rather than in ``__init__`` for the one
        thing that would invalidate it, the CCSD PNO rebuild, which replaces
        ``X_pno`` and runs long before any triples pass. That ordering is checked
        rather than trusted anyway - each entry carries the objects it was built
        from - so this is belt and braces on a cache whose lifetime is one
        calculation.
        """
        t0 = time.perf_counter()
        if self._overlap_halves is None:
            self._overlap_halves = OverlapHalfCache(self.n_lmo_pairs)
        solver = LCCSDT0(self, verbose=self.verbose,
                         halves=self._overlap_halves)
        energy = solver.run(retain=retain)
        self.lccsd_t0 = solver if retain else None
        self.e_ijk = solver.e_ijk
        # Kept for the same reason as ``ccsd_stats``: the solver is dropped
        # here and every driver reports what the phase cost.
        self.t0_nodes = solver.n_nodes
        self.t0_chunks = solver.n_chunks
        self.t_t0 += time.perf_counter() - t0
        return energy

    # -- psi4 DLPNOCCSD_T::estimate_memory ----------------------------------

    def estimate_memory(self):
        """Report what the iterative (T) will hold, and refuse if it will not fit.

        psi4 prints the same breakdown and then spills: over 90% of available
        memory it moves W and V to disk, then the amplitudes, then throws.
        Design decision 10 puts the disk path out of scope here, so this
        reports and refuses - but it must REPORT, because "in core only" is
        only a defensible scope decision if the point where it stops working
        arrives as a number rather than as an afternoon of paging. That is not
        hypothetical: it is how this method came to exist.

        Counts the ``n_tno^3`` stores only, per store rather than as one total
        so a run that does not fit says which store to attack. The
        ``n_tno^2``-scale things - the pair and triplet overlaps, the
        contracted integrals - are an order smaller and deliberately left out;
        the budget is not a precise allocation forecast, it is the line past
        which this phase must refuse rather than page.
        """
        cube = 8 * sum(n ** 3 for n in self.n_tno)
        # The rotation scratch is sized to the WIDEST partner a triplet
        # couples to, which is not known until the couplings are planned. The
        # global maximum is an upper bound, and an upper bound is the only
        # safe direction for a check whose whole purpose is to refuse early.
        widest = max(self.n_tno, default=0)
        scratch = 8 * sum(n * widest ** 2 + n * n * widest + n ** 3
                          for n in self.n_tno)
        stores = {
            "W_{ijk}^{abc}": cube,
            "V_{ijk}^{abc}": cube,
            "T_{ijk}^{abc}": cube,
            "R_{ijk}^{abc} (Jacobi)": cube,
            "denominators": cube,
            "Eq. 53 bracket": cube,
            "rotation scratch (bound)": scratch,
        }
        total = sum(stores.values())
        budget = int(self.cut.in_core_memory)

        self._print("\n  ==> DLPNO-(T) memory <==\n")
        for name, size in stores.items():
            self._print(f"    {name:<24} {size / 2**30:8.3f} GiB")
        self._print(f"    {'total required':<24} {total / 2**30:8.3f} GiB")
        self._print(f"    {'budget':<24} {budget / 2**30:8.3f} GiB")

        if total > budget:
            raise MemoryError(
                f"the iterative (T) needs {total / 2**30:.2f} GiB over "
                f"{self.n_lmo_triplets} triplets ({min(self.n_tno)}-"
                f"{max(self.n_tno)} TNOs) against a budget of "
                f"{budget / 2**30:.2f} GiB. There is no disk path (design "
                "decision 10), so the options are: raise "
                "Thresholds.in_core_memory if the machine has the memory, "
                "loosen t_cut_tno or t_cut_tno_weak_scale to shrink the TNO "
                "spaces, or set t0_approximation to stop at (T0), which holds "
                "nothing per triplet and is what the semicanonical passes "
                "already do.")
        return total

    # -- psi4 DLPNOCCSD_T::lccsd_t_iterations -------------------------------

    def lccsd_t_iterations(self):
        """Iterate Eq. 111/112; return the NET contribution over (T0).

        See :mod:`dlpno.lccsd_t`. The net rather than the absolute energy
        because the iterative pass runs in a DIFFERENT TNO space from the
        production (T0) - looser, and looser still for the weak triplets - so
        its own semicanonical energy is recomputed there and only the
        difference is carried across. Comparing the two (T0) values is what
        makes that legitimate: they bracket the same quantity in two spaces,
        and the correction transfers where the absolute number would not.

        A method rather than an inline block so it can be billed as one phase,
        matching psi4's timer of the same name.
        """
        self.lccsd_t = LCCSDT(self, self.lccsd_t0, verbose=self.verbose,
                              use_diis=self.cut.t_use_diis)
        self.e_t_raw = self.lccsd_t.iterate()
        self.de_t = self.e_t_raw - self.e_t0_crude
        self.e_ijk = self.lccsd_t.e_ijk
        self._print(
            f"\n    (T0) at the looser cutoff: {self.e_t0_crude:16.12f}\n"
            f"    (T)  at the looser cutoff: {self.e_t_raw:16.12f}\n"
            f"    net iterative contribution:{self.de_t:16.12f}")
        return self.de_t

    # -- psi4 DLPNOCCSD_T::compute_energy -----------------------------------

    def _release_ccsd(self):
        """Snapshot what the triples need, then drop what they do not.

        psi4 clears every CCSD integral and overlap family before starting the
        triples, and the reason is memory rather than tidiness: the pair-local
        integrals are the largest store in the calculation (603 MiB on
        ethanol/cc-pVTZ) and no triples term reads one. What the triples DO
        read is the converged amplitudes - the doubles from the pair stores and
        the singles from the solver - so the singles are copied out first,
        because they live inside the object about to be dropped.
        """
        #: What the CCSD solve cost, kept because the object that knows is
        #: about to be dropped and every driver reports it.
        self.ccsd_stats = dict(
            iterations=self.lccsd.n_iterations, nodes=self.lccsd.num_nodes(),
            t_plan=self.lccsd.t_plan, t_capture=self.lccsd.t_capture,
            t_iterate=self.lccsd.t_iterate,
            lmp2_iterations=self.lmp2.n_iterations if self.lmp2 else 0)

        self.T_ia = []
        for i in range(self.ref.naocc):
            ii = self.diag(i)
            n = self.n_pno[ii]
            block = np.asarray(self.lccsd.T_ia[i]).reshape(n, 1) if n else np.zeros((0, 1))
            self.T_ia.append(ten.from_numpy(f"t({i})", block))

        self.integral_blocks = None
        self.overlaps = None
        self.lccsd = None
        self.lmp2 = None

    def diag(self, i):
        """The diagonal pair ``ii``, which always exists."""
        return int(self.i_j_to_ij[i, i])

    def compute_energy(self, optimize=True, session=None, method="ccsd(t)"):
        """Run the DLPNO-CCSD(T) pipeline and return the total energy.

        Args:
            method: ``"ccsd(t)"`` for the full calculation, or anything
                :meth:`~dlpno.ccsd.DLPNOCCSD.compute_energy` accepts to stop
                before the triples.

        Only the ``T0_APPROXIMATION`` branch exists at this milestone: the
        iterative (T) that recovers the off-diagonal occupied Fock coupling is
        M6. A calculation asking for the full (T) gets the (T0) energy and a
        printed line saying so, rather than a silently mislabelled number.
        """
        if method != "ccsd(t)":
            return super().compute_energy(optimize=optimize, session=session,
                                          method=method)

        super().compute_energy(optimize=optimize, session=session, method="ccsd")
        self._release_ccsd()

        self._print("\n  ==> DLPNO-(T0) <==\n")
        self._print("    prescreening pass at "
                    f"t_cut_tno {self.cut.t_cut_tno_pre:.3e}")
        self.triples_sparsity(prescreening=True)
        self.tno_transform(self.cut.t_cut_tno_pre)
        self.e_t0_pre = self.compute_lccsd_t0()

        self._print("\n    production pass at "
                    f"t_cut_tno {self.cut.t_cut_tno:.3e}")
        self.triples_sparsity(prescreening=False)
        self.tno_transform(self.cut.t_cut_tno)
        self.e_t0 = self.compute_lccsd_t0()

        self.e_lccsd_t = self.e_lccsd + self.e_t0 + self.de_lccsd_t_screened

        if not self.cut.t0_approximation:
            self._print("\n  ==> Iterative (T) <==\n")
            self.sort_triplets(self.e_t0)
            self.tno_transform(self.cut.t_cut_tno)
            self.estimate_memory()
            self.e_t0_crude = self.compute_lccsd_t0(retain=True)
            self.e_lccsd_t += self.lccsd_t_iterations()

        # No pass past here reads a pair-side overlap half: the iterative (T)
        # bridges triplet to triplet. Dropped rather than left to the object's
        # lifetime, so the store's residency matches the span the budget in
        # LCCSDT0._chunks charged for it.
        self._overlap_halves = None

        self.e_corr = (self.e_lccsd_t + self.de_weak + self.de_lmp2_eliminated
                       + self.de_dipole + self.de_pno_total)
        label = "(T0)" if self.cut.t0_approximation else "(T)"
        self._print(
            f"\n  Total DLPNO-CCSD{label} Correlation Energy: "
            f"{self.e_corr:16.12f}\n"
            f"    CCSD Correlation Energy:          {self.e_lccsd:16.12f}\n"
            f"    (T) Contribution:                 {self.e_lccsd_t - self.e_lccsd - self.de_lccsd_t_screened:16.12f}\n"
            f"    Screened Triplets Contribution:   {self.de_lccsd_t_screened:16.12f}\n"
            f"    Weak Pair Contribution:           {self.de_weak:16.12f}\n"
            f"    Eliminated Pair Correction:       {self.de_lmp2_eliminated:16.12f}\n"
            f"    Dipole Pair Correction:           {self.de_dipole:16.12f}\n"
            f"    PNO Truncation Correction:        {self.de_pno_total:16.12f}"
        )
        return self.ref.e_scf + self.e_corr
