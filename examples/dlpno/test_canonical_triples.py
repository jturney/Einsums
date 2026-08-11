#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The canonical (T) oracle, checked before anything is judged against it.

``dlpno/canonical_triples.py`` is to the triples port what
``dlpno/canonical_ccsd.py`` is to the CCSD one, and it is pinned down the same
way: an independent implementation of the same correction, compared at
ARBITRARY amplitudes rather than only at a converged set.

The one thing this suite exists to state, which the CCSD oracle had no analogue
of: **(T0) is not invariant to a unitary rotation of the occupied space.** CCSD
is, and ``test_canonical_ccsd.py`` asserts that invariance as a correctness
property. The semicanonical triples correction drops the off-diagonal ``F_il``
coupling between triplets, so it is a different number in a localized basis than
in a canonical one - on water/cc-pVDZ, 1.5e-4 Eh different, which is fifty times
the size of the whole correction's last three digits. That gap is the
approximation, not a defect, and it is what the iterative (T) of milestone M6
puts back.

The consequence for the port's gate is that "untruncated (T0) equals canonical
DF-CCSD(T)" is only true with CANONICAL occupied orbitals. In the port's own
localized basis the right comparison is against :func:`t0_energy` evaluated in
that same basis. ``test_lccsd_t0.py`` runs both.

    python -m pytest examples/dlpno/test_canonical_triples.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.canonical_ccsd import CanonicalCCSD
from dlpno.canonical_triples import (spin_orbital_t_energy, t0_energy,
                                     triplet_blocks, unique_triplets)
from dlpno.reference_io import load_reference

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")

#: Canonical DF-CCSD(T) for water/cc-pVDZ, all-electron, cc-pvdz-ri, on top of
#: the CCSD energy ``test_canonical_ccsd.py`` pins at -0.213602181936. Two
#: independent implementations in this package agree here to 3e-18: the
#: closed-shell semicanonical form in the canonical basis, where semicanonical
#: is exact, and the spin-orbital form of Raghavachari's correction.
CANONICAL_DF_T = -0.003072147091


@pytest.fixture(scope="module")
def solved():
    """One converged canonical CCSD, and the same solve in the localized basis.

    Both are needed by nearly every test here, and neither is cheap enough to
    repeat: the point of the suite is the difference between them.
    """
    reference, _ = load_reference(WATER)
    canonical = CanonicalCCSD(reference, occupied="canonical")
    canonical.solve()
    localized = CanonicalCCSD(reference, occupied="localized")
    localized.solve()
    return canonical, localized


def test_the_two_implementations_agree_at_converged_amplitudes(solved):
    """The closed-shell (T0) and the spin-orbital (T), in the canonical basis.

    They are the same quantity there and nowhere else, and they share nothing
    but the integrals: one is written over spatial orbitals with psi4's six
    permutations, the other over spin orbitals with Crawford's ``P(i/jk)``
    operators. A prefactor wrong in either cannot survive this.
    """
    canonical, _ = solved
    closed_shell = t0_energy(canonical.basis, canonical.t1, canonical.t2)
    spin_orbital = spin_orbital_t_energy(canonical.basis, canonical.t1,
                                         canonical.t2)
    assert closed_shell == pytest.approx(spin_orbital, abs=1e-14)
    assert closed_shell == pytest.approx(CANONICAL_DF_T, abs=1e-11)


def test_the_two_implementations_agree_at_random_amplitudes(solved):
    """The same comparison away from any solution, which is the sharp one.

    The triples correction is a one-shot function of the CCSD amplitudes, so it
    has no residual to compare - the analogue of ``test_lccsd.py``'s probe test
    is to evaluate the correction itself at amplitudes that solve nothing. A
    term that is small near the CCSD solution is not small here.
    """
    canonical, _ = solved
    basis = canonical.basis
    rng = np.random.default_rng(11)
    t1 = 0.05 * rng.standard_normal((basis.nocc, basis.nvir))
    t2 = 0.04 * rng.standard_normal((basis.nocc, basis.nocc,
                                     basis.nvir, basis.nvir))
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))

    closed_shell = t0_energy(basis, t1, t2)
    spin_orbital = spin_orbital_t_energy(basis, t1, t2)
    assert abs(closed_shell) > 0.1, "the probe should be far from any solution"
    assert closed_shell == pytest.approx(spin_orbital, rel=1e-12)


def test_t0_is_not_invariant_to_an_occupied_rotation(solved):
    """The semicanonical approximation, asserted rather than assumed.

    CCSD lands on the same energy from any basis of the occupied space; (T0)
    does not, because its denominator carries only the diagonal of ``F_oo``.
    Asserting the gap exists is what stops a future change from quietly
    "fixing" the localized number onto the canonical one - which would mean the
    port had stopped computing (T0) and started computing something else.
    """
    canonical, localized = solved
    assert localized.e_corr == pytest.approx(canonical.e_corr, abs=1e-10)

    e_canonical = t0_energy(canonical.basis, canonical.t1, canonical.t2)
    e_localized = t0_energy(localized.basis, localized.t1, localized.t2)
    assert abs(e_localized - e_canonical) > 1e-5, (
        "(T0) came out rotation-invariant, which it is not")


def test_the_spin_orbital_form_refuses_a_localized_basis(solved):
    """Raghavachari's (T) is only itself where the Fock matrix is diagonal.

    A silent wrong answer is the failure mode worth preventing: the expression
    evaluates perfectly happily on a non-diagonal Fock matrix and returns a
    number that is neither (T) nor (T0).
    """
    _, localized = solved
    with pytest.raises(ValueError, match="diagonal Fock"):
        spin_orbital_t_energy(localized.basis, localized.t1, localized.t2)


def test_diagonal_triplets_contribute_nothing(solved):
    """``iii`` is skipped by psi4 and by the enumeration, and that is exact.

    For ``i == j == k`` every permutation of the ``P`` operator maps the triplet
    onto itself, so ``W``, ``V`` and the denominator are all fully symmetric in
    ``abc`` and the six energy coefficients ``8 - 4 - 4 - 4 + 2 + 2`` cancel. If
    that ever stopped holding, dropping the triplet would become an
    approximation rather than a saving.
    """
    canonical, _ = solved
    basis = canonical.basis
    f_o, f_v = np.diag(basis.F_oo), np.diag(basis.F_vv)
    worst = 0.0
    for i in range(basis.nocc):
        W, V = triplet_blocks(basis, canonical.t1, canonical.t2, i, i, i)
        D = (f_v[:, None, None] + f_v[None, :, None] + f_v[None, None, :]
             - 3.0 * f_o[i])
        T = -W / D
        bracket = (8.0 * V - 4.0 * V.transpose(2, 1, 0)
                   - 4.0 * V.transpose(0, 2, 1) - 4.0 * V.transpose(1, 0, 2)
                   + 2.0 * V.transpose(1, 2, 0) + 2.0 * V.transpose(2, 0, 1))
        worst = max(worst, abs(float(np.sum(bracket * T)) / 6.0))
    assert worst < 1e-14, f"an iii triplet contributed {worst:.3e}"


def test_the_enumeration_is_every_unique_triplet(solved):
    """``(n + 2)(n + 1)n / 6 - n`` triplets, each ordered and each once.

    The count is psi4's own printed "Max Number of Possible (Unique) LMO
    Triplets", so a port whose enumeration drifts shows up against psi4's
    ratio line rather than only in an energy.
    """
    canonical, _ = solved
    n = canonical.basis.nocc
    triplets = unique_triplets(n)
    assert len(triplets) == (n + 2) * (n + 1) * n // 6 - n
    assert len(set(triplets)) == len(triplets)
    assert all(i <= j <= k for i, j, k in triplets)
