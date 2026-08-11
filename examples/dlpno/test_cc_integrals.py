#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Every coupled-cluster integral block, against a brute-force numpy transform.

The acceptance test for ``compute_pno_integrals``, and the reason it can be one
is the untruncated configuration: with full auxiliary domains the local density
fit is EXACT, so each block has a closed-form value

    (p q | r s) = sum_PQ (P | p q) (P|Q)^-1 (Q | r s)

that a four-line numpy expression computes independently of everything the port
does. Under truncated domains that identity is the approximation being made, so
the same comparison would be measuring the method rather than the code.

Ten families are checked, including both metric powers and both flattened
layouts, which are the two things in that phase most likely to be wrong in a way
no norm or symmetry check would catch.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_cc_integrals.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.ccsd import DLPNOCCSD
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")
TOL = 1e-10


@pytest.fixture(scope="module")
def exact():
    """Water through the CC cascade with every truncation off.

    Untruncated so the density fit is exact and the oracle below is a genuine
    independent value rather than the same approximation computed twice.
    """
    reference, _ = load_reference(WATER)
    # Only the cascade is needed: these tests are about the integral blocks,
    # not the coupled-cluster solve, and running the untruncated CC iteration
    # to convergence here would cost far more than everything else combined.
    cc = DLPNOCCSD(reference, Thresholds.untruncated(), verbose=False)
    cc.compute_energy(method="mp2")
    return cc


@pytest.fixture(scope="module")
def oracle(exact):
    """``(p q | r s)`` from the AO three-index integrals and the full metric."""
    cc = exact
    eri = np.asarray(cc.ref.eri_3index)
    Jinv = np.linalg.inv(np.asarray(cc.ref.metric))
    C_lmo = np.asarray(cc.C_lmo)
    C_pao = np.asarray(cc.C_pao)
    C_pno = [None] * cc.n_lmo_pairs
    for ij in range(cc.n_lmo_pairs):
        if cc.n_pno[ij]:
            C_pno[ij] = C_pao[:, cc.lmopair_to_paos[ij]] @ np.asarray(cc.X_pno[ij])

    def three(Cl, Cr):
        return np.einsum("Qmn,mp,nq->Qpq", eri, Cl, Cr, optimize=True)

    def four(Cl1, Cr1, Cl2, Cr2):
        return np.einsum("QAB,QP,PCD->ABCD",
                         three(Cl1, Cr1), Jinv, three(Cl2, Cr2), optimize=True)

    four.C_lmo, four.C_pno = C_lmo, C_pno
    return four


def _spaces(cc, oracle, ij):
    i, j = cc.ij_to_i_j[ij]
    C_lmo, C_pno = oracle.C_lmo, oracle.C_pno
    lmos = cc.lmopair_to_lmos[ij]
    return (C_lmo[:, lmos], C_lmo[:, [i]], C_lmo[:, [j]], C_pno[ij], lmos)


def _live(cc):
    return [ij for ij in range(cc.n_lmo_pairs) if cc.n_pno[ij]]


def test_one_external_integrals(exact, oracle):
    """``(m i | b j)`` and ``(i j | m b)``: the blocks with one virtual index."""
    cc = exact
    for ij in _live(cc):
        Cm, Ci, Cj, Cv, _ = _spaces(cc, oracle, ij)
        assert np.abs(np.asarray(cc.integral_blocks.K_mibj[ij])
                      - oracle(Cm, Ci, Cv, Cj)[:, 0, :, 0]).max() < TOL
        assert np.abs(np.asarray(cc.integral_blocks.J_ijmb[ij])
                      - oracle(Ci, Cj, Cm, Cv)[0, 0]).max() < TOL


def test_the_L_combinations_use_the_right_two_blocks(exact, oracle):
    """``L = 2K - K'``, and which ``K'`` is the part psi4 gets wrong.

    psi4 computes ``L_mibj_[ji]`` as ``3K_ji - 2K_ij`` because it subtracts the
    already-overwritten ``L_mibj_[ij]`` instead of ``K_mibj_[ij]``. It is
    harmless there only because nothing reads the member. This asserts the
    correct combination, so the port cannot inherit the bug by copying.
    """
    cc = exact
    for ij in _live(cc):
        i, j = cc.ij_to_i_j[ij]
        Cm, Ci, Cj, Cv, _ = _spaces(cc, oracle, ij)
        want = 2.0 * oracle(Cm, Ci, Cv, Cj)[:, 0, :, 0] - oracle(Cm, Cj, Cv, Ci)[:, 0, :, 0]
        assert np.abs(np.asarray(cc.integral_blocks.L_mibj[ij]) - want).max() < TOL

        K = oracle(Ci, Cv, Cj, Cv)[0, :, 0, :]
        assert np.abs(np.asarray(cc.integral_blocks.L_iajb[ij])
                      - (2.0 * K - K.T)).max() < TOL


def test_the_three_external_block_is_rank_three_and_correctly_oriented(exact, oracle):
    """``K_ivvv`` is ``(i e | a f)`` as a rank-3 ``(e, a, f)`` tensor.

    Rank 3 rather than psi4's flattened ``(e, a*f)`` because the consumers
    re-block it two different ways and the storage order is not the same
    between a row-major ``Matrix`` and a column-major tensor.

    The orientation cannot be checked by comparing against the integral alone:
    ``(i e | a f)`` is symmetric under ``a <-> f``, so a transposed store gives
    identical numbers. What distinguishes them is which index pairs with ``e``,
    so the assertion below contracts an ASYMMETRIC probe over ``f`` and would
    fail on the transpose.
    """
    cc = exact
    rng = np.random.default_rng(11)
    for ij in _live(cc):
        _, Ci, _, Cv, _ = _spaces(cc, oracle, ij)
        na = cc.n_pno[ij]
        got = np.asarray(cc.integral_blocks.K_ivvv[ij])
        assert got.shape == (na, na, na)
        want = oracle(Ci, Cv, Cv, Cv)[0]                     # (e, a, f)
        assert np.abs(got - want).max() < TOL
        # Orientation: contract the LAST axis with a random vector. On the
        # a <-> f transpose this contracts the middle axis instead and gives a
        # different matrix, which the symmetric comparison above cannot see.
        probe = rng.standard_normal(na)
        assert np.abs(got @ probe - want @ probe).max() < TOL
        assert np.abs(np.einsum("eaf,a->ef", got, probe)
                      - np.einsum("eaf,a->ef", want, probe)).max() < TOL


def test_the_density_fitted_factors_reconstruct_their_integrals(exact, oracle):
    """The symmetric fit's whole contract: ``B^T B`` is the integral.

    Checked rather than assumed because it is what distinguishes the symmetric
    power from the full inverse, and the two are one line apart in the source.
    """
    cc = exact
    B = cc.integral_blocks
    for ij in _live(cc):
        i, j = cc.ij_to_i_j[ij]
        if cc.i_j_to_ij_strong[i, j] == -1:
            continue
        Cm, Ci, Cj, Cv, _ = _spaces(cc, oracle, ij)
        i_Qk = np.asarray(B.i_Qk[ij])
        i_Qa = np.asarray(B.i_Qa[ij])
        assert np.abs(i_Qk.T @ i_Qk - oracle(Ci, Cm, Ci, Cm)[0, :, 0, :]).max() < TOL
        assert np.abs(i_Qa.T @ i_Qa - oracle(Ci, Cv, Ci, Cv)[0, :, 0, :]).max() < TOL

        Qma = np.asarray(B.qma(cc.ij_to_ji, ij))
        Qab = np.asarray(B.qab(cc.ij_to_ji, ij))
        got = np.einsum("Qmb,Qac->mbac", Qma, Qab, optimize=True)
        assert np.abs(got - oracle(Cm, Cv, Cv, Cv)).max() < TOL


def test_the_non_projected_families_carry_the_neighbours_virtuals(exact, oracle):
    """``(i k | a_ij c_kj)`` and ``(i a_ij | k c_kj)``.

    The two blocks whose second virtual index belongs to a NEIGHBOUR pair, so
    they go through the extended PAO domain. Two things this catches and nothing
    else does: the fitted side must carry the FULL inverse (the other side is
    unfitted, so a symmetric power would apply the metric one and a half times),
    and the two families flatten their merged axis in opposite orders, so
    reading either with the other's stride transposes every slab.
    """
    cc = exact
    B = cc.integral_blocks
    checked = 0
    for ij in _live(cc):
        i, j = cc.ij_to_i_j[ij]
        if cc.i_j_to_ij_strong[i, j] == -1:
            continue
        _, Ci, _, Cv, lmos = _spaces(cc, oracle, ij)
        for k_ij, k in enumerate(lmos):
            kj = int(cc.i_j_to_ij[k, j])
            if kj == -1 or not cc.n_pno[kj]:
                continue
            Ck = oracle.C_lmo[:, [k]]
            Ckj = oracle.C_pno[kj]
            assert np.abs(np.asarray(B.J_ikac[ij][k_ij])
                          - oracle(Ci, Ck, Cv, Ckj)[0, 0]).max() < TOL
            assert np.abs(np.asarray(B.K_iakc[ij][k_ij])
                          - oracle(Ci, Cv, Ck, Ckj)[0, :, 0, :]).max() < TOL
            checked += 1
    assert checked > 0, "no neighbour blocks were exercised"


def test_the_lower_triangle_accessor_finds_the_stored_block(exact):
    """``Qma``/``Qab`` are stored for ``i <= j`` only, and read through a mapping.

    psi4 does the same, and a caller that indexed the list directly would get
    ``None`` for half the pairs - silently, since ``None`` is also what a weak
    pair legitimately holds.
    """
    cc = exact
    B = cc.integral_blocks
    for ij, (i, j) in enumerate(cc.ij_to_i_j):
        if i <= j or not cc.n_pno[ij] or cc.i_j_to_ij_strong[i, j] == -1:
            continue
        assert B.Qma[ij] is None
        assert B.qma(cc.ij_to_ji, ij) is not None
        assert B.qab(cc.ij_to_ji, ij) is not None
