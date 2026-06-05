# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::gemv.

gemv had independent template parameters for A (matrix), x (input vector),
and y (output vector) from the start, so all 2**3 = 8 cells of the
(A, x, y) x (owning, view) matrix are bound per dtype and per trans_a flag.

Cell labels: O = owning, V = view, in (A, x, y) order. Each test runs the
captured graph and verifies that writes through a view-y land in its parent.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg


def test_gemv_OOO_baseline():
    A = einsums.create_random_tensor("A", [4, 5])
    z = einsums.create_random_tensor("z", [5])
    y = einsums.create_zero_tensor("y", [4])

    g = cg.Graph("gv-OOO")
    with cg.capture(g):
        einsums.linalg.gemv(1.0, A, z, 0.0, y)
    g.execute()

    expected = np.asarray(A) @ np.asarray(z)
    np.testing.assert_allclose(np.asarray(y), expected, rtol=1e-5)


def test_gemv_OOV_y_is_view():
    A = einsums.create_random_tensor("A", [4, 5])
    z = einsums.create_random_tensor("z", [5])
    big_y = einsums.create_zero_tensor("big_y", [10])

    g = cg.Graph("gv-OOV")
    with cg.capture(g):
        yv = cg.view(big_y, [(2, 6)])
        einsums.linalg.gemv(1.0, A, z, 0.0, yv)
    g.execute()

    expected = np.zeros(10)
    expected[2:6] = np.asarray(A) @ np.asarray(z)
    np.testing.assert_allclose(np.asarray(big_y), expected, rtol=1e-5)


def test_gemv_OVO_x_is_view():
    A = einsums.create_random_tensor("A", [4, 5])
    big_z = einsums.create_random_tensor("big_z", [10])
    y = einsums.create_zero_tensor("y", [4])

    g = cg.Graph("gv-OVO")
    with cg.capture(g):
        zv = cg.view(big_z, [(3, 8)])
        einsums.linalg.gemv(1.0, A, zv, 0.0, y)
    g.execute()

    expected = np.asarray(A) @ np.asarray(big_z)[3:8]
    np.testing.assert_allclose(np.asarray(y), expected, rtol=1e-5)


def test_gemv_OVV_x_and_y_are_views():
    A = einsums.create_random_tensor("A", [4, 5])
    big_z = einsums.create_random_tensor("big_z", [10])
    big_y = einsums.create_zero_tensor("big_y", [8])

    g = cg.Graph("gv-OVV")
    with cg.capture(g):
        zv = cg.view(big_z, [(0, 5)])
        yv = cg.view(big_y, [(2, 6)])
        einsums.linalg.gemv(1.0, A, zv, 0.0, yv)
    g.execute()

    expected = np.zeros(8)
    expected[2:6] = np.asarray(A) @ np.asarray(big_z)[:5]
    np.testing.assert_allclose(np.asarray(big_y), expected, rtol=1e-5)


def test_gemv_VOO_A_is_view():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    z = einsums.create_random_tensor("z", [5])
    y = einsums.create_zero_tensor("y", [4])

    g = cg.Graph("gv-VOO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        einsums.linalg.gemv(1.0, Av, z, 0.0, y)
    g.execute()

    expected = np.asarray(big_A)[1:5, :] @ np.asarray(z)
    np.testing.assert_allclose(np.asarray(y), expected, rtol=1e-5)


def test_gemv_VOV_A_and_y_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    z = einsums.create_random_tensor("z", [5])
    big_y = einsums.create_zero_tensor("big_y", [10])

    g = cg.Graph("gv-VOV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        yv = cg.view(big_y, [(3, 7)])
        einsums.linalg.gemv(1.0, Av, z, 0.0, yv)
    g.execute()

    expected = np.zeros(10)
    expected[3:7] = np.asarray(big_A)[1:5, :] @ np.asarray(z)
    np.testing.assert_allclose(np.asarray(big_y), expected, rtol=1e-5)


def test_gemv_VVO_A_and_x_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    big_z = einsums.create_random_tensor("big_z", [12])
    y = einsums.create_zero_tensor("y", [4])

    g = cg.Graph("gv-VVO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (2, 7)])
        zv = cg.view(big_z, [(0, 5)])
        einsums.linalg.gemv(1.0, Av, zv, 0.0, y)
    g.execute()

    expected = np.asarray(big_A)[1:5, 2:7] @ np.asarray(big_z)[:5]
    np.testing.assert_allclose(np.asarray(y), expected, rtol=1e-5)


def test_gemv_VVV_all_three_views_with_trans_a():
    big_A = einsums.create_random_tensor("big_A", [6, 8])  # A_view = big_A[1:6, 2:7] (5x5)
    big_z = einsums.create_random_tensor("big_z", [10])     # z_view = big_z[0:5]
    big_y = einsums.create_zero_tensor("big_y", [10])       # y_view = big_y[2:7]

    g = cg.Graph("gv-VVV-T")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 6), (2, 7)])
        zv = cg.view(big_z, [(0, 5)])
        yv = cg.view(big_y, [(2, 7)])
        # trans_a: A^T @ z, both 5-vectors
        einsums.linalg.gemv(1.0, Av, zv, 0.0, yv, trans_a=True)
    g.execute()

    expected = np.zeros(10)
    expected[2:7] = np.asarray(big_A)[1:6, 2:7].T @ np.asarray(big_z)[:5]
    np.testing.assert_allclose(np.asarray(big_y), expected, rtol=1e-5)
