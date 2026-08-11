#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The substrate DLPNO-CCSD needs from the base class, before any CC code runs.

Three things arrived together and none of them has a solver to exercise it yet:
psi4's coupled-cluster threshold presets, the three-mode ``prep_sparsity`` its
prescreening cascade calls, and the ``(Q|i j)`` / ``(Q|u v)`` integral classes.
Each is checkable on its own, and each has a failure mode that would otherwise
surface as a wrong energy several milestones later.

No psi4. These run against a saved fixture:

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_cc_substrate.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno import integrals
from dlpno.mp2 import DLPNOMP2
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")
DIMER = os.path.join(FIXTURES, "water-dimer-ccpvdz.npz")


def _setup(path=WATER, thresholds=None):
    """Everything up to the point where domains exist, and nothing after."""
    reference, _ = load_reference(path)
    mp2 = DLPNOMP2(reference, thresholds or Thresholds.preset("NORMAL", method="cc"),
                   verbose=False)
    mp2.setup_orbitals()
    mp2.compute_doi()
    return mp2


# => thresholds <= #


@pytest.mark.parametrize("preset", ["LOOSE", "NORMAL", "TIGHT", "VERY_TIGHT"])
def test_the_cc_branch_is_looser_in_the_pnos_and_tighter_in_the_pairs(preset):
    """The CC preset table is not the MP2 one, and differs in a specific way.

    Both statements matter for a different reason. The PNO cutoff is looser
    because CC rebuilds its PNOs from converged amplitudes rather than
    semicanonical ones, so the first set only has to be good enough to converge
    the LMP2 that feeds the rebuild. The MP2-level cutoff underneath it is
    tighter than the MP2 method's own, because those amplitudes decide the
    strong/weak split and an amplitude truncated there is never recovered.
    """
    mp2 = Thresholds.preset(preset, method="mp2")
    cc = Thresholds.preset(preset, method="cc")
    assert cc.t_cut_pno > mp2.t_cut_pno
    assert cc.t_cut_pno_mp2 < cc.t_cut_pno


def test_the_derived_threshold_rules_follow_t_cut_pairs():
    """psi4 derives two thresholds from ``T_CUT_PAIRS`` unless told otherwise."""
    tight = Thresholds.preset("VERY_TIGHT", method="cc")
    assert tight.t_cut_pairs == 1e-6
    assert tight.t_cut_pre == pytest.approx(0.01 * tight.t_cut_pairs)
    assert tight.t_cut_pairs_mp2 == pytest.approx(0.1 * tight.t_cut_pairs)


def test_an_explicit_value_suppresses_the_derived_rule():
    """psi4 gates both rules on ``has_changed()``; an override is that statement."""
    pinned = Thresholds.preset("VERY_TIGHT", method="cc", t_cut_pre=1e-3)
    assert pinned.t_cut_pre == 1e-3


def test_untruncated_switches_off_every_cc_and_triples_criterion():
    """The validation anchor: nothing may survive as a live truncation.

    Each field is checked against what makes it vacuous in the comparison it
    appears in, which is not the same constant for all of them - see
    ``Thresholds.untruncated``.
    """
    cut = Thresholds.untruncated()
    # ">=" tests against an absolute value: zero already admits everything.
    assert cut.t_cut_pno == 0.0
    assert cut.t_cut_pno_mp2 == 0.0
    assert cut.t_cut_tno == 0.0
    # Recovery fractions: zero switches the criterion off rather than on.
    assert cut.t_cut_trace == 0.0
    assert cut.t_cut_energy == 0.0
    assert cut.t_cut_trace_mp2 == 0.0
    assert cut.t_cut_energy_mp2 == 0.0
    # ">" tests: negative, so an exactly-vanishing quantity still survives.
    assert cut.t_cut_pairs < 0.0
    assert cut.t_cut_pairs_mp2 < 0.0
    assert cut.t_cut_triples_weak < 0.0
    assert cut.min_pnos == 0


# => the three-mode prep_sparsity <= #


def test_the_crude_pass_gives_smaller_domains_than_the_refined_one():
    """psi4 scales ``T_CUT_MKN`` by 100x and ``T_CUT_DO`` by 2x for the crude pass.

    Both are thresholds a quantity must EXCEED to keep its atom or PAO, so
    raising them discards more and the crude domains come out smaller. That is
    the direction that makes the pass worth having: its only job is to reach a
    pair list cheaply, and it is thrown away afterwards. (The design note called
    these "looser domains", which reads the wrong way round: the tolerance
    loosens, the domain shrinks.)
    """
    from dataclasses import replace

    mp2 = _setup(DIMER)
    crude = replace(mp2.cut, t_cut_mkn=mp2.cut.t_cut_mkn * 100,
                    t_cut_do=mp2.cut.t_cut_do * 2)
    mp2.cut = crude
    mp2.prep_sparsity(True, False)
    crude_aux = sum(len(d) for d in mp2.lmopair_to_ribfs)
    crude_pao = sum(len(d) for d in mp2.lmopair_to_paos)

    mp2.cut = Thresholds.preset("NORMAL", method="cc")
    mp2.prep_sparsity(False, False)
    assert sum(len(d) for d in mp2.lmopair_to_ribfs) > crude_aux
    assert sum(len(d) for d in mp2.lmopair_to_paos) >= crude_pao


def test_the_refined_pass_keeps_the_pair_list_the_crude_one_left():
    """``initial=False`` must not re-run the dipole screen.

    By the time psi4 calls it, crude prescreening has already deleted pairs from
    the list; rerunning the screen would put every one of them back, silently
    undoing the elimination and double-counting ``de_lmp2_eliminated``.
    """
    mp2 = _setup(DIMER)
    mp2.prep_sparsity(True, False)

    # Stand in for filter_pairs<true>: drop an off-diagonal pair by hand.
    victim = next(ij for ij, (i, j) in enumerate(mp2.ij_to_i_j) if i != j)
    keep = [ij for ij in range(mp2.n_lmo_pairs) if ij != victim]
    naocc = mp2.ref.naocc
    mp2.i_j_to_ij = np.full((naocc, naocc), -1, dtype=int)
    mp2.ij_to_i_j = [mp2.ij_to_i_j[ij] for ij in keep]
    for new, (i, j) in enumerate(mp2.ij_to_i_j):
        mp2.i_j_to_ij[i, j] = new
    mp2.ij_to_ji = [int(mp2.i_j_to_ij[j, i]) for (i, j) in mp2.ij_to_i_j]
    n_after_filter = mp2.n_lmo_pairs

    mp2.prep_sparsity(False, False)
    assert mp2.n_lmo_pairs == n_after_filter
    assert len(mp2.lmopair_to_paos) == n_after_filter
    assert len(mp2.lmopair_to_lmos) == n_after_filter


def test_the_final_pass_leaves_the_auxiliary_domains_alone():
    """``final=True`` must not rebuild ``lmo_to_ribfs``.

    The three-index integrals have already been built and are indexed against
    those domains. Rebuilding one without the other is a silent mismatch rather
    than an error, which is why psi4 skips the block outright.
    """
    from dataclasses import replace

    mp2 = _setup(DIMER)
    mp2.prep_sparsity(True, False)
    before = [list(d) for d in mp2.lmo_to_ribfs]

    # Tighten hard enough that a rebuild could not possibly agree.
    mp2.cut = replace(mp2.cut, t_cut_mkn=1e-1, t_cut_do=1e-1)
    mp2.prep_sparsity(False, True)
    assert [list(d) for d in mp2.lmo_to_ribfs] == before
    # ...while the PAO domains, which final mode does rebuild, have moved.
    assert sum(len(d) for d in mp2.lmopair_to_paos) < sum(
        len(mp2.lmo_to_paos[i]) + len(mp2.lmo_to_paos[j]) for i, j in mp2.ij_to_i_j) + 1


def test_prep_sparsity_without_a_pair_list_refuses():
    mp2 = _setup()
    with pytest.raises(RuntimeError, match="needs a pair list"):
        mp2.prep_sparsity(False, False)


def test_the_dense_lmo_lookup_inverts_the_sparse_one():
    """``lmopair_to_lmos_dense`` is psi4's quick-lookup analogue.

    The CC residual reads it in inner loops where scanning the list would be
    quadratic in the pair count, so it has to agree with the list exactly - a
    slot that disagrees reads the wrong LMO's block and is not detectable
    downstream except as a wrong energy.
    """
    mp2 = _setup(DIMER)
    mp2.prep_sparsity()
    naocc = mp2.ref.naocc
    for ij, lmos in enumerate(mp2.lmopair_to_lmos):
        for m in range(naocc):
            slot = mp2.lmopair_to_lmos_dense[ij, m]
            if m in lmos:
                assert lmos[slot] == m
            else:
                assert slot == -1


def test_lmopair_to_lmos_is_the_pairs_that_survive_on_both_indices():
    """psi4's definition, restated: m is in ij's list iff im and jm both exist."""
    mp2 = _setup(DIMER)
    mp2.prep_sparsity()
    for ij, (i, j) in enumerate(mp2.ij_to_i_j):
        expect = [m for m in range(mp2.ref.naocc)
                  if mp2.i_j_to_ij[i, m] != -1 and mp2.i_j_to_ij[j, m] != -1]
        assert mp2.lmopair_to_lmos[ij] == expect


# => the two new integral classes <= #


def _reference_blocks(mp2):
    """``(Q|i u)``, ``(Q|i j)`` and ``(Q|u v)`` straight from the AO integrals."""
    eri = np.asarray(mp2.ref.eri_3index)
    C_lmo = np.asarray(mp2.C_lmo)
    C_pao = np.asarray(mp2.C_pao)
    return (np.einsum("Qmn,mi,nu->Qiu", eri, C_lmo, C_pao),
            np.einsum("Qmn,mi,nj->Qij", eri, C_lmo, C_lmo),
            np.einsum("Qmn,mu,nv->Quv", eri, C_pao, C_pao))


def test_the_dense_source_builds_all_three_classes_exactly():
    """The oracle has to BE one: a brute-force transform must reproduce it."""
    mp2 = _setup(thresholds=Thresholds.untruncated())
    mp2.prep_sparsity()
    mp2.compute_metric()
    mp2.compute_qia()
    mp2.compute_qij()
    mp2.compute_qab()

    want_ia, want_ij, want_ab = _reference_blocks(mp2)
    assert np.abs(np.asarray(mp2.q_ia) - want_ia).max() < 1e-12
    assert np.abs(np.asarray(mp2.q_ij) - want_ij).max() < 1e-12
    assert np.abs(np.asarray(mp2.q_ab) - want_ab).max() < 1e-12


def test_the_two_pao_and_lmo_classes_are_index_symmetric():
    """``(Q|i j)`` and ``(Q|u v)`` are symmetric in their two orbital indices.

    Not a tautology of how they are built here - the two contractions are
    written separately and could disagree - and it is the property psi4 leans on
    to store only the lower triangle of ``q_ab``.
    """
    mp2 = _setup(thresholds=Thresholds.untruncated())
    mp2.prep_sparsity()
    mp2.compute_metric()
    mp2.compute_qij()
    mp2.compute_qab()
    q_ij = np.asarray(mp2.q_ij)
    q_ab = np.asarray(mp2.q_ab)
    assert np.abs(q_ij - q_ij.transpose(0, 2, 1)).max() < 1e-13
    assert np.abs(q_ab - q_ab.transpose(0, 2, 1)).max() < 1e-13


def test_an_undeclared_kind_is_refused_rather_than_returned_stale():
    mp2 = _setup(thresholds=Thresholds.untruncated())
    mp2.prep_sparsity()
    mp2.compute_metric()
    mp2.compute_qia()
    with pytest.raises(RuntimeError, match="not declared"):
        mp2.integrals.q_ab()


def test_a_source_that_cannot_build_a_kind_says_so_at_declare_time():
    """The refusal has to name the source and the kind, and arrive early.

    Two of the three sources build ``(Q|i u)`` only, and the coupled-cluster
    layers ask for three. Finding that out from an ``AttributeError`` inside a
    contraction three phases later is the failure this exists to prevent.
    """
    class OnlyQia:
        def q_ia(self):
            return None

    with pytest.raises(NotImplementedError, match="OnlyQia cannot build q_ab"):
        integrals.check_kinds(OnlyQia(), integrals.Demand(kinds=("q_ia", "q_ab")))


def test_an_unknown_kind_is_a_typo_rather_than_a_missing_feature():
    with pytest.raises(ValueError, match="unknown integral kind"):
        integrals.check_kinds(object(), integrals.Demand(kinds=("q_xy",)))


# => the coupled-cluster PNO overlaps <= #


@pytest.fixture(scope="module")
def cascade():
    """The dimer through the whole prescreening cascade, built once.

    Module-scoped because it is the most expensive thing here by an order of
    magnitude and every overlap test wants the same converged state.
    """
    from dlpno.ccsd import DLPNOCCSD

    reference, _ = load_reference(DIMER)
    cc = DLPNOCCSD(reference, Thresholds.preset("NORMAL", method="cc"), verbose=False)
    cc.compute_energy(method="mp2")
    return cc


def _overlap_oracle(cc, ij, mn):
    """``X[ij]^T S_pao[paos_ij, paos_mn] X[mn]``, straight from the definition."""
    S_pao = np.asarray(cc.S_pao)
    rows, cols = cc.lmopair_to_paos[ij], cc.lmopair_to_paos[mn]
    return (np.asarray(cc.X_pno[ij]).T
            @ S_pao[np.ix_(rows, cols)]
            @ np.asarray(cc.X_pno[mn]))


def test_every_overlap_the_dispatcher_returns_matches_the_definition(cascade):
    """The dispatcher's five branches all have to name the right stored block.

    The branches exist because ``ij`` and ``ji`` share a PNO basis, so several
    distinct requests are answered by one stored matrix under a different name.
    Getting a branch wrong returns a real overlap of the wrong pair, which is
    the kind of error that survives every norm check and shows up only as a
    wrong correlation energy.
    """
    cc = cascade
    rng = np.random.default_rng(0)
    worst, checked = 0.0, 0
    for _ in range(2000):
        ij = int(rng.integers(cc.n_lmo_pairs))
        mn = int(rng.integers(cc.n_lmo_pairs))
        if not (cc.n_pno[ij] and cc.n_pno[mn]):
            continue
        got = cc.S_PNO(ij, mn)
        if got is None:
            continue
        worst = max(worst, float(np.abs(np.asarray(got)
                                        - _overlap_oracle(cc, ij, mn)).max()))
        checked += 1
    assert checked > 500, f"only {checked} live requests drawn; the draw is not covering"
    assert worst < 1e-12, f"dispatcher returned a block off by {worst:.3e}"


def test_a_pairs_overlap_with_itself_is_the_identity(cascade):
    """The PNO basis is orthonormal, so ``S(ij, ij)`` must be exactly ``I``.

    Independent of the oracle above, which shares this code's assumptions: this
    one only assumes the PNOs are what they claim to be.
    """
    cc = cascade
    for ij in range(cc.n_lmo_pairs):
        n = cc.n_pno[ij]
        if not n:
            continue
        block = np.asarray(cc.S_PNO(ij, ij))
        assert np.abs(block - np.eye(n)).max() < 1e-12


def test_the_overlap_of_a_pair_with_its_transpose_is_also_the_identity(cascade):
    """``ij`` and ``ji`` share one PNO basis, which is what four branches rest on."""
    cc = cascade
    for ij, (i, j) in enumerate(cc.ij_to_i_j):
        if i >= j or not cc.n_pno[ij]:
            continue
        block = np.asarray(cc.S_PNO(ij, cc.ij_to_ji[ij]))
        assert np.abs(block - np.eye(cc.n_pno[ij])).max() < 1e-12


def test_the_stored_families_cover_the_sparsity_the_residual_reads(cascade):
    """No on-the-fly build for any request inside ``lmopair_to_lmos``.

    The three families are sized to exactly the pattern psi4's residual reads:
    a pair against its own neighbours' pairs. If a request in that pattern falls
    through to the fallback, the families are mis-sized and the memory figures
    in the design note are wrong.
    """
    cc = cascade
    before = cc.overlaps._on_the_fly
    for ij, (i, j) in enumerate(cc.ij_to_i_j):
        if i > j or not cc.n_pno[ij]:
            continue
        for m in cc.lmopair_to_lmos[ij]:
            for n in cc.lmopair_to_lmos[ij]:
                mn = int(cc.i_j_to_ij[m, n])
                if mn != -1 and cc.n_pno[mn]:
                    cc.S_PNO(ij, mn)
    assert cc.overlaps._on_the_fly == before, (
        f"{cc.overlaps._on_the_fly - before} requests inside the neighbour "
        "sparsity fell through to an on-the-fly build"
    )
