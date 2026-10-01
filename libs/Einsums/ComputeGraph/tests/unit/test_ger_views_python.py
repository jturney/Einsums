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
from einsums.testing import assert_close


def _pf(dtype, re, im):
    """A prefactor for ``dtype``: complex(re, im) for a complex dtype, re otherwise."""
    return complex(re, im) if np.dtype(dtype).kind == "c" else re
import einsums.graph as cg


def test_ger_OOO_baseline(dtype):
    pf_1_5 = _pf(dtype, 1.5, 0.75)
    X = einsums.create_random_tensor("X", [4], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [5], dtype=dtype)
    A = einsums.create_zero_tensor("A", [4, 5], dtype=dtype)

    g = cg.Graph("ger-OOO")
    with cg.capture(g):
        einsums.linalg.ger(pf_1_5, X, Y, A)
    g.execute()

    expected = pf_1_5 * np.outer(np.asarray(X), np.asarray(Y))
    assert_close(np.asarray(A), expected)


def test_ger_OOV_A_is_view(dtype):
    X = einsums.create_random_tensor("X", [4], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [5], dtype=dtype)
    big_A = einsums.create_zero_tensor("big_A", [6, 8], dtype=dtype)

    g = cg.Graph("ger-OOV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (2, 7)])
        einsums.linalg.ger(1.0, X, Y, Av)
    g.execute()

    expected = np.zeros((6, 8), dtype=dtype)
    expected[1:5, 2:7] = np.outer(np.asarray(X), np.asarray(Y))
    assert_close(np.asarray(big_A), expected)


def test_ger_OVO_Y_is_view(dtype):
    X = einsums.create_random_tensor("X", [4], dtype=dtype)
    big_Y = einsums.create_random_tensor("big_Y", [10], dtype=dtype)
    A = einsums.create_zero_tensor("A", [4, 5], dtype=dtype)

    g = cg.Graph("ger-OVO")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(2, 7)])
        einsums.linalg.ger(1.0, X, Yv, A)
    g.execute()

    expected = np.outer(np.asarray(X), np.asarray(big_Y)[2:7])
    assert_close(np.asarray(A), expected)


def test_ger_OVV_Y_and_A_are_views(dtype):
    X = einsums.create_random_tensor("X", [4], dtype=dtype)
    big_Y = einsums.create_random_tensor("big_Y", [10], dtype=dtype)
    big_A = einsums.create_zero_tensor("big_A", [6, 8], dtype=dtype)

    g = cg.Graph("ger-OVV")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(0, 5)])
        Av = cg.view(big_A, [(0, 4), (3, 8)])
        einsums.linalg.ger(1.0, X, Yv, Av)
    g.execute()

    expected = np.zeros((6, 8), dtype=dtype)
    expected[0:4, 3:8] = np.outer(np.asarray(X), np.asarray(big_Y)[:5])
    assert_close(np.asarray(big_A), expected)


def test_ger_VOO_X_is_view(dtype):
    big_X = einsums.create_random_tensor("big_X", [10], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [5], dtype=dtype)
    A = einsums.create_zero_tensor("A", [4, 5], dtype=dtype)

    g = cg.Graph("ger-VOO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(2, 6)])
        einsums.linalg.ger(1.0, Xv, Y, A)
    g.execute()

    expected = np.outer(np.asarray(big_X)[2:6], np.asarray(Y))
    assert_close(np.asarray(A), expected)


def test_ger_VOV_X_and_A_are_views(dtype):
    big_X = einsums.create_random_tensor("big_X", [10], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [5], dtype=dtype)
    big_A = einsums.create_zero_tensor("big_A", [6, 8], dtype=dtype)

    g = cg.Graph("ger-VOV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(2, 6)])
        Av = cg.view(big_A, [(1, 5), (3, 8)])
        einsums.linalg.ger(1.0, Xv, Y, Av)
    g.execute()

    expected = np.zeros((6, 8), dtype=dtype)
    expected[1:5, 3:8] = np.outer(np.asarray(big_X)[2:6], np.asarray(Y))
    assert_close(np.asarray(big_A), expected)


def test_ger_VVO_X_and_Y_are_views(dtype):
    big_X = einsums.create_random_tensor("big_X", [10], dtype=dtype)
    big_Y = einsums.create_random_tensor("big_Y", [10], dtype=dtype)
    A = einsums.create_zero_tensor("A", [4, 5], dtype=dtype)

    g = cg.Graph("ger-VVO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 4)])
        Yv = cg.view(big_Y, [(5, 10)])
        einsums.linalg.ger(1.0, Xv, Yv, A)
    g.execute()

    expected = np.outer(np.asarray(big_X)[:4], np.asarray(big_Y)[5:10])
    assert_close(np.asarray(A), expected)


def test_ger_VVV_all_three_views_with_alpha(dtype):
    pf_2_5 = _pf(dtype, 2.5, 0.75)
    big_X = einsums.create_random_tensor("big_X", [10], dtype=dtype)
    big_Y = einsums.create_random_tensor("big_Y", [10], dtype=dtype)
    big_A = einsums.create_zero_tensor("big_A", [6, 8], dtype=dtype)

    g = cg.Graph("ger-VVV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 4)])
        Yv = cg.view(big_Y, [(5, 10)])
        Av = cg.view(big_A, [(1, 5), (2, 7)])
        einsums.linalg.ger(pf_2_5, Xv, Yv, Av)
    g.execute()

    expected = np.zeros((6, 8), dtype=dtype)
    expected[1:5, 2:7] = pf_2_5 * np.outer(np.asarray(big_X)[:4], np.asarray(big_Y)[5:10])
    assert_close(np.asarray(big_A), expected)
