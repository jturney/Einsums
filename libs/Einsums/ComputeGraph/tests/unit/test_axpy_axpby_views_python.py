# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::axpy and cg::axpby.

After the 2026-05-20 refactor, both ops take independent XType and YType
template parameters (with SameUnderlying), so all 2**2 = 4 cells of the
(X, Y) x (owning, view) matrix are bound for each dtype.

Cell labels: O = owning, V = view, in (X, Y) order. Each test verifies that
writes through a view-Y land in its parent and other regions are untouched.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# axpy: Y += alpha * X
# ──────────────────────────────────────────────────────────────────────────


def test_axpy_OO_baseline():
    X = einsums.create_random_tensor("X", [4, 5])
    Y = einsums.create_random_tensor("Y", [4, 5])
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpy-OO")
    with cg.capture(g):
        einsums.linalg.axpy(2.0, X, Y)
    g.execute()

    expected = Y_before + 2.0 * np.asarray(X)
    np.testing.assert_allclose(np.asarray(Y), expected, rtol=1e-5)


def test_axpy_OV_Y_is_view():
    X = einsums.create_random_tensor("X", [3, 4])
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8])
    np.asarray(big_Y)[...] = 1.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpy-OV")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(1, 4), (2, 6)])  # 3x4 slab
        einsums.linalg.axpy(0.5, X, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[1:4, 2:6] += 0.5 * np.asarray(X)
    np.testing.assert_allclose(np.asarray(big_Y), expected, rtol=1e-5)


def test_axpy_VO_X_is_view():
    big_X = einsums.create_random_tensor("big_X", [6, 8])
    Y = einsums.create_zero_tensor("Y", [3, 4])
    np.asarray(Y)[...] = 5.0
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpy-VO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(1, 4), (2, 6)])
        einsums.linalg.axpy(1.5, Xv, Y)
    g.execute()

    expected = Y_before + 1.5 * np.asarray(big_X)[1:4, 2:6]
    np.testing.assert_allclose(np.asarray(Y), expected, rtol=1e-5)


def test_axpy_VV_both_views():
    big_X = einsums.create_random_tensor("big_X", [6, 8])
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8])
    np.asarray(big_Y)[...] = 2.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpy-VV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 3), (0, 4)])
        Yv = cg.view(big_Y, [(2, 5), (3, 7)])
        einsums.linalg.axpy(-1.0, Xv, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[2:5, 3:7] -= np.asarray(big_X)[:3, :4]
    np.testing.assert_allclose(np.asarray(big_Y), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# axpby: Y = alpha * X + beta * Y
# ──────────────────────────────────────────────────────────────────────────


def test_axpby_OO_baseline():
    X = einsums.create_random_tensor("X", [4, 5])
    Y = einsums.create_random_tensor("Y", [4, 5])
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpby-OO")
    with cg.capture(g):
        einsums.linalg.axpby(2.0, X, 3.0, Y)
    g.execute()

    expected = 2.0 * np.asarray(X) + 3.0 * Y_before
    np.testing.assert_allclose(np.asarray(Y), expected, rtol=1e-5)


def test_axpby_OV_Y_is_view():
    X = einsums.create_random_tensor("X", [3, 4])
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8])
    np.asarray(big_Y)[...] = 4.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpby-OV")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(1, 4), (2, 6)])
        einsums.linalg.axpby(1.0, X, 0.25, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[1:4, 2:6] = np.asarray(X) + 0.25 * big_Y_before[1:4, 2:6]
    np.testing.assert_allclose(np.asarray(big_Y), expected, rtol=1e-5)


def test_axpby_VO_X_is_view():
    big_X = einsums.create_random_tensor("big_X", [6, 8])
    Y = einsums.create_zero_tensor("Y", [3, 4])
    np.asarray(Y)[...] = 1.5
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpby-VO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(1, 4), (2, 6)])
        einsums.linalg.axpby(-0.5, Xv, 2.0, Y)
    g.execute()

    expected = -0.5 * np.asarray(big_X)[1:4, 2:6] + 2.0 * Y_before
    np.testing.assert_allclose(np.asarray(Y), expected, rtol=1e-5)


def test_axpby_VV_both_views():
    big_X = einsums.create_random_tensor("big_X", [6, 8])
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8])
    np.asarray(big_Y)[...] = 7.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpby-VV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 3), (0, 4)])
        Yv = cg.view(big_Y, [(2, 5), (3, 7)])
        einsums.linalg.axpby(0.5, Xv, 0.5, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[2:5, 3:7] = 0.5 * np.asarray(big_X)[:3, :4] + 0.5 * big_Y_before[2:5, 3:7]
    np.testing.assert_allclose(np.asarray(big_Y), expected, rtol=1e-5)
