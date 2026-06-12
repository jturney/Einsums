# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::gemm.

After the 2026-05-20 refactor, gemm takes independent template parameters for
A, B, and C (with a SameUnderlying dtype constraint), so all 2**3 = 8 cells
of the (A, B, C) x (owning, view) matrix are bound for each dtype and each
(trans_a, trans_b) pair. This file exercises every cell once with double
inputs; the dtype dimension is orthogonal (codegen produces one overload
per dtype x combination), and dtype-specific correctness is covered by the
existing test_blas_python.py gemm suite.

Cell labels use O = owning, V = view, in (A, B, C) order:

    OOO  OOV  OVO  OVV  VOO  VOV  VVO  VVV

Each test builds inputs and an expected numpy result, captures the gemm
into a graph, executes, and asserts that writes through any view land in
the parent tensor as expected.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# Cell OOO, baseline, all-owning
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_OOO_baseline():
    A = einsums.create_random_tensor("A", [4, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("g-OOO")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)
    g.execute()

    expected = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cell OOV, owning A, owning B, view C (write into a sub-block of bigger C)
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_OOV_C_is_view_into_parent():
    A = einsums.create_random_tensor("A", [4, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("g-OOV")
    with cg.capture(g):
        Cv = cg.view(big_C, [(1, 5), (2, 5)])  # 4x3 sub-region
        einsums.linalg.gemm(1.0, A, B, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 6))
    expected[1:5, 2:5] = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cell OVO, owning A, view B, owning C
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_OVO_B_is_view():
    A = einsums.create_random_tensor("A", [4, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 6])  # B = big_B[:, :3]
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("g-OVO")
    with cg.capture(g):
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        einsums.linalg.gemm(1.0, A, Bv, 0.0, C)
    g.execute()

    expected = np.asarray(A) @ np.asarray(big_B)[:, :3]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cell OVV, owning A, view B, view C
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_OVV_only_A_is_owning():
    A = einsums.create_random_tensor("A", [4, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 6])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("g-OVV")
    with cg.capture(g):
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        Cv = cg.view(big_C, [(0, 4), (3, 6)])
        einsums.linalg.gemm(1.0, A, Bv, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 6))
    expected[0:4, 3:6] = np.asarray(A) @ np.asarray(big_B)[:, :3]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cell VOO, view A, owning B, owning C
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_VOO_A_is_view():
    big_A = einsums.create_random_tensor("big_A", [6, 5])  # A = big_A[1:5, :]
    B = einsums.create_random_tensor("B", [5, 3])
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("g-VOO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        einsums.linalg.gemm(1.0, Av, B, 0.0, C)
    g.execute()

    expected = np.asarray(big_A)[1:5, :] @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cell VOV, view A, owning B, view C
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_VOV_A_and_C_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    big_C = einsums.create_zero_tensor("big_C", [5, 5])

    g = cg.Graph("g-VOV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        Cv = cg.view(big_C, [(1, 5), (1, 4)])
        einsums.linalg.gemm(1.0, Av, B, 0.0, Cv)
    g.execute()

    expected = np.zeros((5, 5))
    expected[1:5, 1:4] = np.asarray(big_A)[1:5, :] @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cell VVO, view A, view B, owning C
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_VVO_A_and_B_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 6])
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("g-VVO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        Bv = cg.view(big_B, [(-1, -1), (2, 5)])
        einsums.linalg.gemm(1.0, Av, Bv, 0.0, C)
    g.execute()

    expected = np.asarray(big_A)[1:5, :] @ np.asarray(big_B)[:, 2:5]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cell VVV, all three are views
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_VVV_all_three_views_with_trans_a():
    big_A = einsums.create_random_tensor("big_A", [5, 6])  # A_view = big_A[:, :3] (5x3)
    big_B = einsums.create_random_tensor("big_B", [5, 6])  # B_view = big_B[:, :3] (5x3)
    big_C = einsums.create_zero_tensor("big_C", [5, 5])  # C_view = big_C[:3, :3] (3x3)

    g = cg.Graph("g-VVV-T")
    with cg.capture(g):
        Av = cg.view(big_A, [(-1, -1), (0, 3)])
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        Cv = cg.view(big_C, [(0, 3), (0, 3)])
        # trans_a: A^T (3x5) * B (5x3) = 3x3
        einsums.linalg.gemm(1.0, Av, Bv, 0.0, Cv, trans_a=True)
    g.execute()

    expected = np.zeros((5, 5))
    expected[:3, :3] = np.asarray(big_A)[:, :3].T @ np.asarray(big_B)[:, :3]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# trans_b coverage (one mixed-view cell with trans_b=True)
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_density_pattern_VVO_with_trans_b():
    """SCF density build via views: D = 2 * C_occ @ C_occ^T, C_occ a view of C."""
    nbf, nocc = 5, 3
    C = einsums.create_random_tensor("C", [nbf, nbf])
    D = einsums.create_zero_tensor("D", [nbf, nbf])

    g = cg.Graph("density-views-direct")
    with cg.capture(g):
        C_occ = cg.view(C, [(-1, -1), (0, nocc)])
        einsums.linalg.gemm(2.0, C_occ, C_occ, 0.0, D, trans_b=True)
    g.execute()

    C_np = np.asarray(C)
    expected = 2.0 * C_np[:, :nocc] @ C_np[:, :nocc].T
    np.testing.assert_allclose(np.asarray(D), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Both bool flags set on a view cell
# ──────────────────────────────────────────────────────────────────────────


def test_gemm_OVV_with_trans_a_and_trans_b():
    A = einsums.create_random_tensor("A", [5, 4])  # A^T is 4x5
    big_B = einsums.create_random_tensor("big_B", [3, 6])  # B = big_B[:, :5]; B^T is 5x3
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("g-OVV-TT")
    with cg.capture(g):
        Bv = cg.view(big_B, [(-1, -1), (0, 5)])  # 3x5
        Cv = cg.view(big_C, [(0, 4), (0, 3)])  # 4x3
        einsums.linalg.gemm(1.0, A, Bv, 0.0, Cv, trans_a=True, trans_b=True)
    g.execute()

    expected = np.zeros((6, 6))
    expected[0:4, 0:3] = np.asarray(A).T @ np.asarray(big_B)[:, :5].T
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)
