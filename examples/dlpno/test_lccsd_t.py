#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The iterative triples, checked without psi4.

``dlpno/lccsd_t.py`` restores the one thing (T0) drops: the off-diagonal
occupied Fock coupling between triplets, Eq. 111's three ``- f_il T_ljk`` sums.
That single difference is what this suite is about, and it makes the gate
sharper than (T0)'s rather than looser.

**(T) is invariant to a rotation of the occupied orbitals and (T0) is not.**
``test_canonical_triples.py`` asserts the second half of that, because it is
what forced the (T0) gate to be run in two bases. Here it pays off: untruncated
in the port's OWN localized basis, (T) must reproduce canonical DF-CCSD(T)
exactly, with no canonicalized rerun needed. The number it has to hit is the one
(T0) misses by 1.5e-4 Eh, so a port that quietly computed (T0) here would fail by
five percent of the correction rather than by something subtle.

That also means this milestone needed no new oracle. The value was already
pinned by two independent implementations in ``canonical_triples.py``.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_lccsd_t.py
"""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.lccsd_t import permuter_spec
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds
from dlpno.triples import DLPNOCCSDT

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")
DIMER = os.path.join(FIXTURES, "water-dimer-ccpvdz.npz")

#: Canonical DF-CCSD(T) for water/cc-pVDZ, all-electron, cc-pvdz-ri. Pinned in
#: ``test_canonical_triples.py`` by two independent implementations agreeing to
#: 3e-18, and reached here by the iterative (T) in the LOCALIZED basis.
CANONICAL_DF_T = -0.003072147091

#: What (T0) gives in the localized basis, from ``test_lccsd_t0.py``. The
#: distance between the two is the whole of the iterative correction.
LOCALIZED_T0 = -0.002917162965


@pytest.fixture(scope="module")
def solved():
    """One untruncated iterative (T), in the port's own localized basis."""
    reference, _ = load_reference(WATER)
    cut = Thresholds.untruncated()
    cut = replace(cut, r_convergence=cut.r_convergence * 0.1,
                  e_convergence=cut.e_convergence * 0.1, maxiter=100)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    cc.compute_energy()
    return cc


def test_untruncated_iterative_t_is_canonical_df_ccsd_t(solved):
    """The gate, and it needs no canonicalized rerun.

    Eq. 111's coupling terms are exactly what makes the correction invariant to
    the occupied basis, so untruncated in the localized basis this IS the
    textbook (T). (T0) cannot reach it from here by any tolerance.
    """
    correction = solved.e_lccsd_t - solved.e_lccsd
    assert correction == pytest.approx(CANONICAL_DF_T, abs=1e-10)


def test_the_iteration_recovers_what_t0_was_missing(solved):
    """The correction is the (T0)-to-(T) gap, not a small refinement.

    Worth asserting as a size rather than only as a final value: it is five
    percent of the correction, and a port that skipped the coupling entirely
    would still look plausible on a total energy.
    """
    assert solved.e_t0 == pytest.approx(LOCALIZED_T0, abs=1e-9)
    correction = solved.e_lccsd_t - solved.e_lccsd
    assert abs(correction - solved.e_t0) > 1e-4


def test_the_residual_actually_converges(solved):
    """The iteration converged rather than ran out of patience.

    ``iterate`` raises on exceeding ``maxiter``, so reaching here at all says
    it stopped on the convergence test; this pins that it did so in a sane
    number of passes, which is what would degrade first if a coupling term
    were wrong in sign.
    """
    assert 1 < solved.lccsd_t.n_iterations < 100


# => the permuter <= #


def test_the_permuter_spec_matches_psi4s_selection():
    """psi4's ``triples_permuter`` index chain, case by case.

    An amplitude block is stored once in its own ``i <= j <= k`` ordering and
    read in whatever ordering the requesting triplet needs, so this mapping is
    the whole of the cross-triplet bookkeeping. The two CYCLIC orderings are
    the ones psi4 carries a ``reverse`` flag for, because permuting a tensor
    and permuting the request run in opposite directions for a 3-cycle - which
    is exactly where an off-by-one-permutation bug would hide.
    """
    expected = {
        (0, 1, 2): "abc", (0, 2, 1): "acb", (1, 0, 2): "bac",
        (1, 2, 0): "cab", (2, 0, 1): "bca", (2, 1, 0): "cba",
    }
    labels = (10, 20, 30)
    for order, spec in expected.items():
        assert permuter_spec(*(labels[p] for p in order)) == spec


def test_the_permuter_spec_is_a_genuine_permutation():
    """Applying the spec to a labelled cube reproduces the requested ordering.

    The mapping above is asserted against psi4's table; this asserts it against
    what it is supposed to MEAN, so the two cannot drift into agreeing on a
    consistent mistake.
    """
    labels = (10, 20, 30)
    block = np.arange(27.0).reshape(3, 3, 3)
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        spec = permuter_spec(*(labels[p] for p in order))
        got = np.einsum(f"{spec}->abc", block)
        # Reading the block in the requested ordering means axis n of the
        # result is axis order[n] of the original.
        assert np.array_equal(got, block.transpose(order))


# => the truncated path <= #


def test_the_strong_weak_triplet_split_fires():
    """``sort_triplets`` classifies, and both classes are populated.

    The split is the only thing ``tno_scale`` reads, and it decides the size of
    the space the iteration runs in. A run where every triplet came out strong
    would still converge to a plausible number in a space psi4 never used.
    """
    reference, _ = load_reference(DIMER)
    cc = DLPNOCCSDT(reference, Thresholds.preset("NORMAL", method="cc"),
                    verbose=False)
    cc.compute_energy()

    strong = sum(cc.is_strong_triplet)
    assert 0 < strong < cc.n_lmo_triplets, "the triplets did not split"
    # Strong triplets get the tighter of the two scales, so their TNO spaces
    # are the larger ones; the whole point of the split is that both exist.
    assert min(cc.n_tno) < max(cc.n_tno)
    assert cc.de_t != 0.0, "the iterative pass contributed nothing"


def test_the_iterative_pass_refuses_a_budget_it_cannot_meet():
    """Design decision 10: in core, with a MEASURED failure.

    The iterative (T) is the first phase in the port whose stores are not
    bounded by a chunk: it keeps W, V and the amplitudes for every triplet at
    once, because Eq. 111 reads its neighbours'. Bounding only the chunk is how
    this came to page instead of refusing - an ethanol/cc-pVTZ run sat at 12%
    CPU with 28 GB of swap in use, which looks exactly like a slow benchmark
    and is not one.

    psi4 spills to disk at that point. There is no disk path here, so what is
    asserted is that the refusal happens, names the requirement, and says what
    to change - because "in core only" is a defensible scope decision only if
    the point where it stops working arrives as a number.
    """
    reference, _ = load_reference(DIMER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  in_core_memory=20 * 2 ** 20)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    with pytest.raises(MemoryError, match="iterative \\(T\\) needs"):
        cc.compute_energy()


def test_stopping_at_t0_needs_no_per_triplet_storage():
    """The escape the refusal names has to actually work.

    (T0) retains nothing per triplet, so the same budget that refuses the
    iterative pass must let the semicanonical one through. A refusal message
    offering an option that also fails would be worse than no message.
    """
    reference, _ = load_reference(DIMER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  in_core_memory=20 * 2 ** 20, t0_approximation=True)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    cc.compute_energy()
    assert cc.e_t0 != 0.0
