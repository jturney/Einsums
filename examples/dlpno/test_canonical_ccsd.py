#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The canonical CCSD oracle, checked before anything is judged against it.

``dlpno/canonical_ccsd.py`` exists to be the independent value the local
coupled-cluster port is compared to, so it has to be pinned down harder than the
thing it is judging. Four checks, in the order a failure would want to be read:

1. the MP2 starting guess reproduces the psi4 DF-MP2 energy recorded in the
   fixture, which says the orbitals, the auxiliary fit and the Fock matrix are
   right before any coupled-cluster equation is involved;
2. the closed-shell residuals agree with the textbook spin-orbital ones at
   RANDOM amplitudes, which is the only check that covers the equations away
   from their own solution - and away from the solution is where this module
   does its real work, evaluating somebody else's amplitudes;
3. the converged CCSD correlation energy, against the value psi4's ``dfocc``
   DF-CCSD and the untruncated port both report for this system;
4. the energy is unchanged by a unitary rotation of the occupied orbitals,
   which is what certifies that the non-diagonal ``F_oo`` terms are all there.
   A missing one shows up here and nowhere else, because the canonical basis it
   would otherwise be validated in is exactly the basis that hides it.

Nothing here imports einsums: the oracle is numpy only, deliberately.

    python -m pytest examples/dlpno/test_canonical_ccsd.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.canonical_ccsd import (CanonicalCCSD, build_orbital_basis,
                                  spin_orbital_residuals)
from dlpno.reference_io import load_reference

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")

#: Canonical DF-CCSD for this system, all-electron, cc-pvdz-ri. Four
#: independent calculations land here: this oracle in both occupied bases, the
#: untruncated DLPNO port, and psi4's ``dfocc`` DF-CCSD (-0.213602181924).
CANONICAL_DF_CCSD = -0.213602181936

#: psi4's OTHER DF-CCSD, ``fnocc`` with ``cc_type df``, which reports
#: -0.213602211142 for the same problem - 2.92e-8 away from everything else.
#:
#: Not a gate, and deliberately not asserted. fnocc uses two auxiliary bases in
#: one calculation: the amplitude integrals come from ``DF_BASIS_CC`` (RIFIT)
#: while ``DFCoupledCluster::T1Fock`` rebuilds the T1-dressed Fock matrix from
#: the SCF's three-index integrals over ``DF_BASIS_SCF`` (JKFIT). The two agree
#: at ``T1 = 0``, which is why every module's DF-MP2 matches to 1e-12, and
#: disagree once the singles turn on. Setting ``df_basis_cc cc-pvdz-jkfit`` so
#: one basis does both makes fnocc and this oracle agree to 4.3e-13, which is
#: the measurement that identifies the mechanism. See ``check_ccsd_defect.py``.
PSI4_FNOCC_DF_CCSD = -0.213602211142


@pytest.fixture(scope="module")
def water():
    reference, extras = load_reference(WATER)
    return reference, extras


def test_the_mp2_guess_is_psi4_df_mp2(water):
    """The starting guess, before any coupled-cluster term is involved.

    First because it separates the two things a wrong CCSD energy could mean.
    The guess amplitudes are ``<ij|ab> / D``, so this is DF-MP2 exactly, and it
    reads the auxiliary fit, the virtual space construction and the Fock matrix
    without reading a single CC equation.
    """
    reference, extras = water
    cc = CanonicalCCSD(reference, occupied="canonical")
    assert abs(cc.mp2_energy() - extras["energies"]["psi4_df_mp2"]) < 1e-11


@pytest.mark.parametrize("occupied", ["canonical", "localized"])
def test_the_closed_shell_residuals_match_the_spin_orbital_ones(water, occupied):
    """The spin-adapted equations against Stanton's, at random amplitudes.

    Random rather than converged, and that is the point: the converged energy
    check below only says the equations are right AT their solution, and this
    module's other job is to evaluate residuals at amplitudes that solve
    somebody else's. A term with a wrong prefactor can still have a fixed point
    in the right place if it vanishes there; it cannot survive this.

    Run in both occupied bases because the two differ precisely in the
    off-diagonal ``F_oo`` terms.
    """
    reference, _ = water
    cc = CanonicalCCSD(reference, occupied=occupied)
    rng = np.random.default_rng(7)
    t1 = 0.05 * rng.standard_normal((cc.nocc, cc.nvir))
    t2 = 0.04 * rng.standard_normal((cc.nocc, cc.nocc, cc.nvir, cc.nvir))
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))

    r1, r2 = cc.residuals(t1, t2)
    s1, s2 = spin_orbital_residuals(cc.basis, t1, t2)
    assert np.abs(r1 - s1).max() < 1e-12 * max(1.0, np.abs(s1).max())
    assert np.abs(r2 - s2).max() < 1e-12 * max(1.0, np.abs(s2).max())


def test_the_converged_energy(water):
    """The converged correlation energy, in the canonical occupied basis.

    The tolerance is 1e-10, four orders below the 2.92e-8 that separates this
    value from psi4's fnocc number, so a run that drifted onto fnocc's answer
    would fail here rather than pass quietly.
    """
    reference, _ = water
    cc = CanonicalCCSD(reference, occupied="canonical")
    e = cc.solve()
    assert abs(e - CANONICAL_DF_CCSD) < 1e-10
    assert abs(e - PSI4_FNOCC_DF_CCSD) > 1e-8


def test_the_energy_is_invariant_to_an_occupied_rotation(water):
    """The same energy from a random unitary rotation of the occupied orbitals.

    CCSD is invariant to a unitary rotation within the occupied space, so a
    localized calculation and a canonical one must land on the same number. The
    invariance is not automatic in a program: it holds only if every
    off-diagonal ``F_oo`` term is present, since a canonical run has none of
    them to get wrong. This is therefore the check that licenses running the
    oracle in the port's own localized basis, which is what makes an amplitude
    by amplitude comparison possible at all.
    """
    reference, _ = water
    naocc = reference.naocc
    rng = np.random.default_rng(19)
    U, _ = np.linalg.qr(rng.standard_normal((naocc, naocc)))

    canonical = CanonicalCCSD(reference, occupied="canonical").solve()
    localized = CanonicalCCSD(reference, occupied="localized").solve()
    rotated = CanonicalCCSD(reference, occupied_rotation=U).solve()
    assert abs(localized - canonical) < 1e-10
    assert abs(rotated - canonical) < 1e-10


def test_the_virtual_space_is_the_whole_complement(water):
    """``nvir = nbf - nocc``, and it is orthonormal and Fock-diagonal.

    A quiet failure mode of the projector construction is dropping a
    near-dependent virtual, which would leave a correct-looking calculation in a
    slightly smaller space and a correlation energy wrong in the sixth decimal.
    """
    reference, _ = water
    basis = build_orbital_basis(reference, occupied="localized")
    S = np.asarray(reference.S)
    assert basis.nvir == reference.nbf - reference.C_occ.shape[1]
    assert np.abs(basis.C_vir.T @ S @ basis.C_vir - np.eye(basis.nvir)).max() < 1e-10
    assert np.abs(basis.C_occ.T @ S @ basis.C_vir).max() < 1e-10
    off = basis.F_vv - np.diag(np.diag(basis.F_vv))
    assert np.abs(off).max() < 1e-10
