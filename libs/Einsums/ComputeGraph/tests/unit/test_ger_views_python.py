# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::ger.

ger(alpha, X, Y, A): A += alpha * X * Y^T. X and Y are rank-1; A is rank-2.
Template parameters AType, XType, YType are independent (with SameUnderlying
across them), so all 2**3 = 8 cells of the (X, Y, A) x (owning, view) matrix
are bound per dtype.

Cell labels use O = owning, V = view in (X, Y, A) order. Each test verifies
that writes through a view-A land in its parent.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg


def test_ger_OOO_baseline():
    X = einsums.create_random_tensor("X", [4])
    Y = einsums.create_random_tensor("Y", [5])
    A = einsums.create_zero_tensor("A", [4, 5])

    g = cg.Graph("ger-OOO")
    with cg.capture(g):
        einsums.linalg.ger(1.5, X, Y, A)
    g.execute()

    expected = 1.5 * np.outer(np.asarray(X), np.asarray(Y))
    np.testing.assert_allclose(np.asarray(A), expected, rtol=1e-5)


def test_ger_OOV_A_is_view():
    X = einsums.create_random_tensor("X", [4])
    Y = einsums.create_random_tensor("Y", [5])
    big_A = einsums.create_zero_tensor("big_A", [6, 8])

    g = cg.Graph("ger-OOV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (2, 7)])
        einsums.linalg.ger(1.0, X, Y, Av)
    g.execute()

    expected = np.zeros((6, 8))
    expected[1:5, 2:7] = np.outer(np.asarray(X), np.asarray(Y))
    np.testing.assert_allclose(np.asarray(big_A), expected, rtol=1e-5)


def test_ger_OVO_Y_is_view():
    X = einsums.create_random_tensor("X", [4])
    big_Y = einsums.create_random_tensor("big_Y", [10])
    A = einsums.create_zero_tensor("A", [4, 5])

    g = cg.Graph("ger-OVO")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(2, 7)])
        einsums.linalg.ger(1.0, X, Yv, A)
    g.execute()

    expected = np.outer(np.asarray(X), np.asarray(big_Y)[2:7])
    np.testing.assert_allclose(np.asarray(A), expected, rtol=1e-5)


def test_ger_OVV_Y_and_A_are_views():
    X = einsums.create_random_tensor("X", [4])
    big_Y = einsums.create_random_tensor("big_Y", [10])
    big_A = einsums.create_zero_tensor("big_A", [6, 8])

    g = cg.Graph("ger-OVV")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(0, 5)])
        Av = cg.view(big_A, [(0, 4), (3, 8)])
        einsums.linalg.ger(1.0, X, Yv, Av)
    g.execute()

    expected = np.zeros((6, 8))
    expected[0:4, 3:8] = np.outer(np.asarray(X), np.asarray(big_Y)[:5])
    np.testing.assert_allclose(np.asarray(big_A), expected, rtol=1e-5)


def test_ger_VOO_X_is_view():
    big_X = einsums.create_random_tensor("big_X", [10])
    Y = einsums.create_random_tensor("Y", [5])
    A = einsums.create_zero_tensor("A", [4, 5])

    g = cg.Graph("ger-VOO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(2, 6)])
        einsums.linalg.ger(1.0, Xv, Y, A)
    g.execute()

    expected = np.outer(np.asarray(big_X)[2:6], np.asarray(Y))
    np.testing.assert_allclose(np.asarray(A), expected, rtol=1e-5)


def test_ger_VOV_X_and_A_are_views():
    big_X = einsums.create_random_tensor("big_X", [10])
    Y = einsums.create_random_tensor("Y", [5])
    big_A = einsums.create_zero_tensor("big_A", [6, 8])

    g = cg.Graph("ger-VOV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(2, 6)])
        Av = cg.view(big_A, [(1, 5), (3, 8)])
        einsums.linalg.ger(1.0, Xv, Y, Av)
    g.execute()

    expected = np.zeros((6, 8))
    expected[1:5, 3:8] = np.outer(np.asarray(big_X)[2:6], np.asarray(Y))
    np.testing.assert_allclose(np.asarray(big_A), expected, rtol=1e-5)


def test_ger_VVO_X_and_Y_are_views():
    big_X = einsums.create_random_tensor("big_X", [10])
    big_Y = einsums.create_random_tensor("big_Y", [10])
    A = einsums.create_zero_tensor("A", [4, 5])

    g = cg.Graph("ger-VVO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 4)])
        Yv = cg.view(big_Y, [(5, 10)])
        einsums.linalg.ger(1.0, Xv, Yv, A)
    g.execute()

    expected = np.outer(np.asarray(big_X)[:4], np.asarray(big_Y)[5:10])
    np.testing.assert_allclose(np.asarray(A), expected, rtol=1e-5)


def test_ger_VVV_all_three_views_with_alpha():
    big_X = einsums.create_random_tensor("big_X", [10])
    big_Y = einsums.create_random_tensor("big_Y", [10])
    big_A = einsums.create_zero_tensor("big_A", [6, 8])

    g = cg.Graph("ger-VVV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 4)])
        Yv = cg.view(big_Y, [(5, 10)])
        Av = cg.view(big_A, [(1, 5), (2, 7)])
        einsums.linalg.ger(2.5, Xv, Yv, Av)
    g.execute()

    expected = np.zeros((6, 8))
    expected[1:5, 2:7] = 2.5 * np.outer(np.asarray(big_X)[:4], np.asarray(big_Y)[5:10])
    np.testing.assert_allclose(np.asarray(big_A), expected, rtol=1e-5)
