# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::symm_gemm.

symm_gemm computes C = op(B)^T * op(A) * op(B). All three tensors must be
rank 2 with matching dtypes. The trans_a / trans_b kwargs apply op() to A
and B respectively.

symm_gemm had independent template parameters for A, B, C from the start,
so all 2**3 = 8 cells of the (A, B, C) x (owning, view) matrix are bound
per dtype and per (TransA, TransB) bool pair.

Cell labels: O = owning, V = view, in (A, B, C) order.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg


def _ref(A, B, trans_a=False, trans_b=False):
    """Reference: C = op(B)^T * op(A) * op(B)."""
    opA = A.T if trans_a else A
    opB = B.T if trans_b else B
    return opB.T @ opA @ opB


def test_symm_gemm_OOO_baseline():
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("sgm-OOO")
    with cg.capture(g):
        einsums.linalg.symm_gemm(A, B, C)
    g.execute()

    expected = _ref(np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_symm_gemm_OOV_C_is_view():
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 3])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("sgm-OOV")
    with cg.capture(g):
        Cv = cg.view(big_C, [(1, 4), (2, 5)])
        einsums.linalg.symm_gemm(A, B, Cv)
    g.execute()

    expected = np.zeros((6, 6))
    expected[1:4, 2:5] = _ref(np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_symm_gemm_OVO_B_is_view():
    A = einsums.create_random_tensor("A", [4, 4])
    big_B = einsums.create_random_tensor("big_B", [4, 6])  # B = big_B[:, :3]
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("sgm-OVO")
    with cg.capture(g):
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        einsums.linalg.symm_gemm(A, Bv, C)
    g.execute()

    expected = _ref(np.asarray(A), np.asarray(big_B)[:, :3])
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_symm_gemm_OVV_B_and_C_are_views():
    A = einsums.create_random_tensor("A", [4, 4])
    big_B = einsums.create_random_tensor("big_B", [4, 6])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("sgm-OVV")
    with cg.capture(g):
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        Cv = cg.view(big_C, [(0, 3), (3, 6)])
        einsums.linalg.symm_gemm(A, Bv, Cv)
    g.execute()

    expected = np.zeros((6, 6))
    expected[0:3, 3:6] = _ref(np.asarray(A), np.asarray(big_B)[:, :3])
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_symm_gemm_VOO_A_is_view():
    big_A = einsums.create_random_tensor("big_A", [6, 6])  # A = big_A[1:5, 1:5]
    B = einsums.create_random_tensor("B", [4, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("sgm-VOO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (1, 5)])
        einsums.linalg.symm_gemm(Av, B, C)
    g.execute()

    expected = _ref(np.asarray(big_A)[1:5, 1:5], np.asarray(B))
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_symm_gemm_VOV_A_and_C_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 6])
    B = einsums.create_random_tensor("B", [4, 3])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("sgm-VOV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (1, 5)])
        Cv = cg.view(big_C, [(1, 4), (2, 5)])
        einsums.linalg.symm_gemm(Av, B, Cv)
    g.execute()

    expected = np.zeros((6, 6))
    expected[1:4, 2:5] = _ref(np.asarray(big_A)[1:5, 1:5], np.asarray(B))
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_symm_gemm_VVO_A_and_B_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 6])
    big_B = einsums.create_random_tensor("big_B", [4, 6])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("sgm-VVO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (1, 5)])
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        einsums.linalg.symm_gemm(Av, Bv, C)
    g.execute()

    expected = _ref(np.asarray(big_A)[1:5, 1:5], np.asarray(big_B)[:, :3])
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_symm_gemm_VVV_all_three_views_with_trans_a():
    big_A = einsums.create_random_tensor("big_A", [6, 6])
    big_B = einsums.create_random_tensor("big_B", [4, 6])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("sgm-VVV-T")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (1, 5)])
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        Cv = cg.view(big_C, [(2, 5), (3, 6)])
        einsums.linalg.symm_gemm(Av, Bv, Cv, trans_a=True)
    g.execute()

    expected = np.zeros((6, 6))
    expected[2:5, 3:6] = _ref(np.asarray(big_A)[1:5, 1:5], np.asarray(big_B)[:, :3], trans_a=True)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)
