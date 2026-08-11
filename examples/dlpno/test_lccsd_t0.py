#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The captured (T0) correction, checked without psi4.

``dlpno/lccsd_t0.py`` is the triples analogue of ``lccsd.py`` and is checked the
same three ways, adapted to a correction that is a one-shot function of the CCSD
amplitudes rather than an iteration.

*The equations can be wrong.* ``test_lccsd.py`` compares residuals at arbitrary
amplitudes, because a converged energy hides a term that is small near the
solution. (T0) has no residual - it is evaluated once at whatever the CCSD
converged to - so the equivalent sharp test is to compare the correction itself
against an independent implementation AT THE PORT'S OWN AMPLITUDES, which is
what :func:`test_untruncated_matches_the_oracle_at_the_ports_amplitudes` does.
Both sides then see identical inputs and any difference is in the expression.

*The capture can be wrong while the equations are right.* The (T0) phase shares
two rank-3 buffers across every triplet in a chunk and reuses four more within
each triplet, so a buffer read before it is written, or an accumulation not
cleared, would show up as an answer that depends on which triplets happened to
share a chunk. Varying the chunk size is therefore this phase's idempotence
test, and it is sharper than replaying twice would be.

*The gate against a canonical value needs canonical orbitals.* (T0) is not
invariant to a rotation of the occupied space - see
``test_canonical_triples.py`` - so "untruncated equals canonical DF-CCSD(T)"
only holds with the occupied orbitals canonicalized. Both forms are run here:
the localized one pins the equations, and the canonical one closes the loop to
the textbook number.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_lccsd_t0.py
"""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_ccsd_defect import Bridge
from dlpno.canonical_ccsd import CanonicalCCSD
from dlpno.canonical_triples import t0_energy
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds
from dlpno.triples import DLPNOCCSDT

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")
DIMER = os.path.join(FIXTURES, "water-dimer-ccpvdz.npz")

#: Canonical DF-CCSD(T) for water/cc-pVDZ, all-electron, cc-pvdz-ri. Pinned in
#: ``test_canonical_triples.py`` by two independent implementations agreeing to
#: 3e-18; reproduced here by the port with canonical occupied orbitals.
CANONICAL_DF_T = -0.003072147091


def _untruncated(reference):
    """One decade tighter than the default, for the reason ``test_lccsd.py`` gives.

    (T0) inherits whatever error the CCSD amplitudes carry, so a gate on the
    correction is a gate on the convergence unless the amplitudes are converged
    past it.
    """
    cut = Thresholds.untruncated()
    return replace(cut, r_convergence=cut.r_convergence * 0.1,
                   e_convergence=cut.e_convergence * 0.1, maxiter=100)


def _canonical_occupied(reference):
    """The same reference with its occupied orbitals rotated to diagonalize F.

    (T0)'s denominator carries only the diagonal of the occupied Fock matrix,
    so this is the basis - and the only basis - in which it is the exact (T).
    """
    F = np.asarray(reference.F)
    C = np.asarray(reference.C_lmo)
    _, U = np.linalg.eigh(C.T @ F @ C)
    return replace(reference, C_lmo=C @ U)


@pytest.fixture(scope="module")
def solved_localized():
    """The port run untruncated in its own localized basis."""
    reference, _ = load_reference(WATER)
    cc = DLPNOCCSDT(reference, _untruncated(reference), verbose=False)
    cc.compute_energy()
    return cc


@pytest.fixture(scope="module")
def solved_canonical():
    """The same, with the occupied orbitals canonicalized first."""
    reference, _ = load_reference(WATER)
    reference = _canonical_occupied(reference)
    cc = DLPNOCCSDT(reference, _untruncated(reference), verbose=False)
    cc.compute_energy()
    return cc


# => the equations <= #


def test_untruncated_matches_the_oracle_at_the_ports_amplitudes(solved_localized):
    """The sharp test: both sides evaluate (T0) at the SAME amplitudes.

    The port's converged doubles and singles are rotated into the oracle's
    virtual basis and handed back to it, so the two differ in nothing but the
    expression they evaluate. A total that agreed only because two errors
    cancelled across triplets would still fail the per-triplet comparison
    below.
    """
    cc = solved_localized
    oracle = CanonicalCCSD(cc.ref, occupied="localized")
    bridge = Bridge(cc, oracle.basis)
    bridge.check()

    t1 = np.zeros((oracle.nocc, oracle.nvir))
    t2 = np.zeros((oracle.nocc, oracle.nocc, oracle.nvir, oracle.nvir))
    for i in range(oracle.nocc):
        ii = int(cc.i_j_to_ij[i, i])
        t1[i] = (bridge.U[ii] @ np.asarray(cc.T_ia[i])).ravel()
    for ij, (i, j) in enumerate(cc.ij_to_i_j):
        n = cc.n_pno[ij]
        t2[i, j] = (bridge.U[ij] @ cc.pair_block(cc.T_all, ij)[:n, :n]
                    @ bridge.U[ij].T)

    want, per_triplet = t0_energy(oracle.basis, t1, t2, per_triplet=True)
    assert cc.e_t0 == pytest.approx(want, abs=1e-13)

    worst = 0.0
    for ijk, key in enumerate(cc.ijk_to_i_j_k):
        reference = per_triplet[key]
        worst = max(worst, abs(cc.e_ijk[ijk] - reference)
                    / max(abs(reference), 1e-12))
    assert worst < 1e-10, f"worst per-triplet disagreement {worst:.3e} relative"


def test_untruncated_canonical_is_canonical_df_ccsd_t(solved_canonical):
    """With canonical occupied orbitals, untruncated (T0) IS (T).

    Two truncations are off at once here and both matter: no TNO truncation, so
    each triplet's space is the whole virtual space, and no occupied
    off-diagonal Fock coupling to drop, so the semicanonical denominator is
    exact. What is left is the textbook correction.
    """
    assert solved_canonical.e_t0 == pytest.approx(CANONICAL_DF_T, abs=1e-11)


def test_the_localized_and_canonical_runs_differ(solved_localized,
                                                 solved_canonical):
    """The port reproduces (T0)'s basis dependence, rather than smoothing it.

    The CCSD energies underneath agree - CCSD is rotation-invariant - so the
    whole difference is the correction, and it has to be there: a port that got
    the same number in both bases would be computing (T), not (T0), and would
    be wrong against psi4.
    """
    assert solved_localized.e_lccsd == pytest.approx(solved_canonical.e_lccsd,
                                                     abs=1e-10)
    assert abs(solved_localized.e_t0 - solved_canonical.e_t0) > 1e-5


# => the capture <= #


def test_the_chunking_does_not_change_the_answer():
    """Bit-identical per-triplet energies however the triplets are grouped.

    This phase's idempotence test. Two buffers are shared by every triplet in a
    chunk and four more are reused inside each triplet, so a read before a
    write, or an accumulation that should have been an overwrite, gives an
    answer that depends on which triplets share a chunk. Neither failure is
    visible in a single run: a stale intermediate produces a plausible energy.

    Bit-identical rather than close, because nothing about the chunk boundary
    changes an operand or a summation order - each triplet's arithmetic is the
    same arithmetic in the same sequence.
    """
    reference, _ = load_reference(WATER)
    cut = _untruncated(reference)

    whole = DLPNOCCSDT(reference, cut, verbose=False)
    whole.compute_energy()

    # Small enough to force one triplet per chunk at this size, so every
    # triplet is both first and last in its chunk.
    split = DLPNOCCSDT(reference, replace(cut, t0_chunk_memory=6 * 2 ** 20),
                       verbose=False)
    split.compute_energy()

    assert split.n_lmo_triplets == whole.n_lmo_triplets
    assert np.array_equal(whole.e_ijk, split.e_ijk), (
        "the per-triplet energies moved when the chunk size changed")


def test_a_triplet_too_large_for_the_budget_reports_itself():
    """Design decision 10: in core, with a measured failure.

    There is no disk path, so the useful behaviour when a triplet does not fit
    is a message naming the triplet, what it needs and what to change - not a
    silent thrash or a MemoryError from the allocator with no context.
    """
    reference, _ = load_reference(WATER)
    cut = replace(_untruncated(reference), t0_chunk_memory=1024)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    with pytest.raises(MemoryError, match="TNOs.*against a budget"):
        cc.compute_energy()


# => the truncated path <= #


@pytest.fixture(scope="module")
def truncated_dimer():
    """The water dimer at psi4's CC ``NORMAL``, where the truncations bite."""
    reference, _ = load_reference(DIMER)
    cc = DLPNOCCSDT(reference, Thresholds.preset("NORMAL", method="cc"),
                    verbose=False)
    cc.compute_energy()
    return cc


def test_the_truncated_run_exercises_the_domain_and_tno_truncations(
        truncated_dimer):
    """Two of the three triples truncations, on a fixture that populates them.

    The untruncated gates above are structurally blind to all of this: with
    every threshold off, every triplet forms, every TNO survives and the
    domains are complete, so the maps that decide each are never exercised
    sparse.

    The dimer at NORMAL does exercise the triplet enumeration (88 of 210
    possible triplets form) and the TNO truncation (18 to 36 TNOs against a
    full space of 38). It does NOT exercise the energy screen; see
    :func:`test_the_energy_screen_books_what_it_drops` for why that needs its
    own setup.
    """
    cc = truncated_dimer
    naocc = cc.ref.naocc
    possible = (naocc + 2) * (naocc + 1) * naocc // 6 - naocc
    assert 0 < cc.n_lmo_triplets < possible, "every possible triplet formed"
    assert min(cc.n_tno) < max(cc.n_tno), "every triplet kept the same TNO count"
    assert max(cc.n_tno) < min(len(d) for d in cc.lmotriplet_to_paos), (
        "no triplet's TNO space was truncated below its PAO domain")

    # The domains are the triples' own, at their own thresholds, so they do not
    # simply agree with the pairs'.
    assert any(len(t) != len(p) for t, p in
               zip(cc.lmotriplet_to_paos, cc.lmopair_to_paos))


def test_the_energy_screen_books_what_it_drops():
    """The screened-triplet correction, on a screen loose enough to fire.

    **A census result, recorded because it decides how this can be tested at
    all: none of the six fixtures screens a single triplet at NORMAL.** The
    smallest per-triplet energy across the dimer, methanol and the far dimer is
    1.6e-7 Eh against a ``t_cut_triples_weak`` of 1e-8, so the whole screening
    path - the second ``triples_sparsity`` and ``de_lccsd_t_screened`` - never
    runs on a molecule this size. That is a statement about the fixtures rather
    than about the threshold, and the design's rule is to add no molecule until
    a code path demands one; a raised threshold demands nothing and exercises
    the same code.

    What is asserted is the bookkeeping, which is what could actually be wrong:
    the dropped triplets' estimated energy has to stand in for the energy they
    would have contributed. Screening at 1e-4 removes most of the dimer's
    triplets, and the total must still land close to the unscreened one.
    """
    reference, _ = load_reference(DIMER)
    base = Thresholds.preset("NORMAL", method="cc")

    unscreened = DLPNOCCSDT(reference, base, verbose=False)
    unscreened.compute_energy()

    screened = DLPNOCCSDT(reference, replace(base, t_cut_triples_weak=1e-4),
                          verbose=False)
    screened.compute_energy()

    assert screened.n_lmo_triplets < unscreened.n_lmo_triplets, (
        "the raised screen dropped nothing")
    assert screened.de_lccsd_t_screened != 0.0
    # The estimate stands in for what was dropped, so the totals agree far more
    # closely than the dropped fraction alone would suggest. Loose on purpose:
    # this is a bookkeeping check, not an accuracy claim about the screen.
    total = screened.e_t0 + screened.de_lccsd_t_screened
    assert total == pytest.approx(unscreened.e_t0, rel=0.05)


def test_the_plan_classifies_every_triplet():
    """Every triplet carries a shape class, which is what M8 would group on.

    Nothing reads the classes yet. Deriving them now is what makes a batched
    emitter a change to ``_emit_*`` alone, exactly as ``lccsd.py``'s plan does
    for the pairs.
    """
    from dlpno.lccsd_t0 import LCCSDT0

    reference, _ = load_reference(WATER)
    cc = DLPNOCCSDT(reference, Thresholds.preset("NORMAL", method="cc"),
                    verbose=False)
    cc.compute_energy()

    solver = LCCSDT0(cc, verbose=False)
    solver.plan()
    triplets, classes = solver.class_report()
    assert triplets == cc.n_lmo_triplets
    assert 0 < classes <= triplets
