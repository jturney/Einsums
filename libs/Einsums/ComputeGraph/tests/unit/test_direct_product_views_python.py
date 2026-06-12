# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::direct_product.

direct_product computes C = alpha * (A ⊙ B) + beta * C, element-wise
multiplication of A and B accumulated into C with prefactors. All three
tensors must have the same shape and dtype.

direct_product had independent template parameters for A, B, C from the
start, so all 2**3 = 8 cells of the (A, B, C) x (owning, view) matrix are
bound per dtype.

Cell labels: O = owning, V = view, in (A, B, C) order.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg


def test_direct_product_OOO_baseline():
    A = einsums.create_random_tensor("A", [4, 5])
    B = einsums.create_random_tensor("B", [4, 5])
    C = einsums.create_zero_tensor("C", [4, 5])

    g = cg.Graph("dp-OOO")
    with cg.capture(g):
        einsums.linalg.direct_product(1.5, A, B, 0.0, C)
    g.execute()

    expected = 1.5 * np.asarray(A) * np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_OOV_C_is_view():
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [3, 4])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])

    g = cg.Graph("dp-OOV")
    with cg.capture(g):
        Cv = cg.view(big_C, [(1, 4), (2, 6)])
        einsums.linalg.direct_product(2.0, A, B, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 8))
    expected[1:4, 2:6] = 2.0 * np.asarray(A) * np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_direct_product_OVO_B_is_view():
    A = einsums.create_random_tensor("A", [3, 4])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("dp-OVO")
    with cg.capture(g):
        Bv = cg.view(big_B, [(0, 3), (0, 4)])
        einsums.linalg.direct_product(1.0, A, Bv, 0.0, C)
    g.execute()

    expected = np.asarray(A) * np.asarray(big_B)[:3, :4]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_OVV_B_and_C_are_views():
    A = einsums.create_random_tensor("A", [3, 4])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])

    g = cg.Graph("dp-OVV")
    with cg.capture(g):
        Bv = cg.view(big_B, [(0, 3), (0, 4)])
        Cv = cg.view(big_C, [(2, 5), (3, 7)])
        einsums.linalg.direct_product(1.0, A, Bv, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 8))
    expected[2:5, 3:7] = np.asarray(A) * np.asarray(big_B)[:3, :4]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_direct_product_VOO_A_is_view():
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    B = einsums.create_random_tensor("B", [3, 4])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("dp-VOO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 4), (2, 6)])
        einsums.linalg.direct_product(1.0, Av, B, 0.0, C)
    g.execute()

    expected = np.asarray(big_A)[1:4, 2:6] * np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_VOV_A_and_C_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    B = einsums.create_random_tensor("B", [3, 4])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])

    g = cg.Graph("dp-VOV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 4), (2, 6)])
        Cv = cg.view(big_C, [(2, 5), (3, 7)])
        einsums.linalg.direct_product(1.0, Av, B, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 8))
    expected[2:5, 3:7] = np.asarray(big_A)[1:4, 2:6] * np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_direct_product_VVO_A_and_B_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("dp-VVO")
    with cg.capture(g):
        Av = cg.view(big_A, [(0, 3), (0, 4)])
        Bv = cg.view(big_B, [(1, 4), (2, 6)])
        einsums.linalg.direct_product(1.0, Av, Bv, 0.0, C)
    g.execute()

    expected = np.asarray(big_A)[:3, :4] * np.asarray(big_B)[1:4, 2:6]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_VVV_all_three_views_with_accumulation():
    """All three views + nonzero beta to confirm read-and-write of C-view."""
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])
    np.asarray(big_C)[...] = 1.0
    big_C_before = np.asarray(big_C).copy()

    g = cg.Graph("dp-VVV")
    with cg.capture(g):
        Av = cg.view(big_A, [(0, 3), (0, 4)])
        Bv = cg.view(big_B, [(1, 4), (2, 6)])
        Cv = cg.view(big_C, [(2, 5), (3, 7)])
        einsums.linalg.direct_product(1.0, Av, Bv, 0.5, Cv)
    g.execute()

    expected = big_C_before.copy()
    expected[2:5, 3:7] = np.asarray(big_A)[:3, :4] * np.asarray(big_B)[1:4, 2:6] + 0.5 * big_C_before[2:5, 3:7]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)
