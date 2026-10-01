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
import pytest

import einsums
import einsums.graph as cg


def _pf(dtype, re, im):
    """A prefactor for ``dtype``: complex(re, im) for a complex dtype, re otherwise."""
    return complex(re, im) if np.dtype(dtype).kind == "c" else re
from einsums.testing import ALL_DTYPES, assert_close


# ──────────────────────────────────────────────────────────────────────────
# axpy: Y += alpha * X
# ──────────────────────────────────────────────────────────────────────────


def test_axpy_OO_baseline(dtype):
    pf_2_0 = _pf(dtype, 2.0, 0.75)
    X = einsums.create_random_tensor("X", [4, 5], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [4, 5], dtype=dtype)
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpy-OO")
    with cg.capture(g):
        einsums.linalg.axpy(pf_2_0, X, Y)
    g.execute()

    expected = Y_before + pf_2_0 * np.asarray(X)
    assert_close(np.asarray(Y), expected)


def test_axpy_OV_Y_is_view(dtype):
    pf_0_5 = _pf(dtype, 0.5, 0.75)
    X = einsums.create_random_tensor("X", [3, 4], dtype=dtype)
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8], dtype=dtype)
    np.asarray(big_Y)[...] = 1.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpy-OV")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(1, 4), (2, 6)])  # 3x4 slab
        einsums.linalg.axpy(pf_0_5, X, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[1:4, 2:6] += pf_0_5 * np.asarray(X)
    assert_close(np.asarray(big_Y), expected)


def test_axpy_VO_X_is_view(dtype):
    pf_1_5 = _pf(dtype, 1.5, 0.75)
    big_X = einsums.create_random_tensor("big_X", [6, 8], dtype=dtype)
    Y = einsums.create_zero_tensor("Y", [3, 4], dtype=dtype)
    np.asarray(Y)[...] = 5.0
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpy-VO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(1, 4), (2, 6)])
        einsums.linalg.axpy(pf_1_5, Xv, Y)
    g.execute()

    expected = Y_before + pf_1_5 * np.asarray(big_X)[1:4, 2:6]
    assert_close(np.asarray(Y), expected)


def test_axpy_VV_both_views(dtype):
    pf_m1_0 = _pf(dtype, -1.0, 0.75)
    big_X = einsums.create_random_tensor("big_X", [6, 8], dtype=dtype)
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8], dtype=dtype)
    np.asarray(big_Y)[...] = 2.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpy-VV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 3), (0, 4)])
        Yv = cg.view(big_Y, [(2, 5), (3, 7)])
        einsums.linalg.axpy(pf_m1_0, Xv, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[2:5, 3:7] += pf_m1_0 * np.asarray(big_X)[:3, :4]
    assert_close(np.asarray(big_Y), expected)


# ──────────────────────────────────────────────────────────────────────────
# axpby: Y = alpha * X + beta * Y
# ──────────────────────────────────────────────────────────────────────────


def test_axpby_OO_baseline(dtype):
    pf_2_0 = _pf(dtype, 2.0, 0.75)
    pf_3_0 = _pf(dtype, 3.0, 0.75)
    X = einsums.create_random_tensor("X", [4, 5], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [4, 5], dtype=dtype)
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpby-OO")
    with cg.capture(g):
        einsums.linalg.axpby(pf_2_0, X, pf_3_0, Y)
    g.execute()

    expected = pf_2_0 * np.asarray(X) + pf_3_0 * Y_before
    assert_close(np.asarray(Y), expected)


def test_axpby_OV_Y_is_view(dtype):
    pf_0_25 = _pf(dtype, 0.25, 0.75)
    X = einsums.create_random_tensor("X", [3, 4], dtype=dtype)
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8], dtype=dtype)
    np.asarray(big_Y)[...] = 4.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpby-OV")
    with cg.capture(g):
        Yv = cg.view(big_Y, [(1, 4), (2, 6)])
        einsums.linalg.axpby(1.0, X, pf_0_25, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[1:4, 2:6] = np.asarray(X) + pf_0_25 * big_Y_before[1:4, 2:6]
    assert_close(np.asarray(big_Y), expected)


def test_axpby_VO_X_is_view(dtype):
    pf_m0_5 = _pf(dtype, -0.5, 0.75)
    pf_2_0 = _pf(dtype, 2.0, 0.75)
    big_X = einsums.create_random_tensor("big_X", [6, 8], dtype=dtype)
    Y = einsums.create_zero_tensor("Y", [3, 4], dtype=dtype)
    np.asarray(Y)[...] = 1.5
    Y_before = np.asarray(Y).copy()

    g = cg.Graph("axpby-VO")
    with cg.capture(g):
        Xv = cg.view(big_X, [(1, 4), (2, 6)])
        einsums.linalg.axpby(pf_m0_5, Xv, pf_2_0, Y)
    g.execute()

    expected = pf_m0_5 * np.asarray(big_X)[1:4, 2:6] + pf_2_0 * Y_before
    assert_close(np.asarray(Y), expected)


def test_axpby_VV_both_views(dtype):
    pf_0_5 = _pf(dtype, 0.5, 0.75)
    big_X = einsums.create_random_tensor("big_X", [6, 8], dtype=dtype)
    big_Y = einsums.create_zero_tensor("big_Y", [6, 8], dtype=dtype)
    np.asarray(big_Y)[...] = 7.0
    big_Y_before = np.asarray(big_Y).copy()

    g = cg.Graph("axpby-VV")
    with cg.capture(g):
        Xv = cg.view(big_X, [(0, 3), (0, 4)])
        Yv = cg.view(big_Y, [(2, 5), (3, 7)])
        einsums.linalg.axpby(pf_0_5, Xv, pf_0_5, Yv)
    g.execute()

    expected = big_Y_before.copy()
    expected[2:5, 3:7] = pf_0_5 * np.asarray(big_X)[:3, :4] + pf_0_5 * big_Y_before[2:5, 3:7]
    assert_close(np.asarray(big_Y), expected)


# ──────────────────────────────────────────────────────────────────────────
# X and Y are overlapping views of ONE parent
# ──────────────────────────────────────────────────────────────────────────
#
# Two views of one parent that share elements but are not the same view once
# reached vendor ?axpy aliased, which BLAS forbids. Each output element reads a
# neighbour the kernel had already updated, so the result is wrong for real
# prefactors as well as complex ones; only X that IS Y (same start, same
# layout) was guarded. The expected value reads every input from the parent as
# it was before the call.

_SHIFTS = {
    "row-shifted": ((0, 3), (0, 3), (1, 4), (0, 3)),
    "column-shifted": ((0, 3), (0, 3), (0, 3), (1, 4)),
    "diagonal-shifted": ((1, 4), (1, 4), (0, 3), (0, 3)),
}


def _view(P, rows, cols, mode):
    """A view of @p P: a slice eagerly (``cg.view`` is capture-only), ``cg.view`` in a graph."""
    return P[slice(*rows), slice(*cols)] if mode == "eager" else cg.view(P, [rows, cols])


def _parent(dtype):
    """A 4x4 parent of small exact values, genuinely complex on a complex dtype."""
    vals = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    if np.dtype(dtype).kind == "c":
        vals = vals + 1j * (np.arange(16).reshape(4, 4) % 3 - 1)
    P = einsums.create_zero_tensor("P", [4, 4], dtype=dtype)
    np.asarray(P)[...] = vals
    return P


def _prefactor(dtype, value):
    return value if np.dtype(dtype).kind == "c" else value.real


def _run_overlapping(mode, body):
    if mode == "eager":
        body()
        return
    g = cg.Graph("overlap")
    with cg.capture(g):
        body()
    if mode == "graph+passes":
        g.apply(cg.default_pass_manager())
    g.execute()


@pytest.mark.parametrize("mode", ["eager", "graph", "graph+passes"])
@pytest.mark.parametrize("shift", list(_SHIFTS))
@pytest.mark.parametrize("alpha", [2.0, 0.5 - 0.75j], ids=["real-alpha", "complex-alpha"])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_axpy_overlapping_views_of_one_parent(dtype, alpha, shift, mode):
    (xr, xc, yr, yc) = _SHIFTS[shift]
    alpha = _prefactor(dtype, alpha)
    P = _parent(dtype)
    before = np.asarray(P).copy()

    def body():
        einsums.linalg.axpy(alpha, _view(P, xr, xc, mode), _view(P, yr, yc, mode))

    _run_overlapping(mode, body)

    expected = before.copy()
    expected[slice(*yr), slice(*yc)] += alpha * before[slice(*xr), slice(*xc)]
    np.testing.assert_array_equal(np.asarray(P), expected)


@pytest.mark.parametrize("mode", ["eager", "graph", "graph+passes"])
@pytest.mark.parametrize("shift", list(_SHIFTS))
@pytest.mark.parametrize("beta", [0.0, 0.5])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_axpby_overlapping_views_of_one_parent(dtype, beta, shift, mode):
    (xr, xc, yr, yc) = _SHIFTS[shift]
    alpha = _prefactor(dtype, 0.5 - 0.75j)
    P = _parent(dtype)
    before = np.asarray(P).copy()

    def body():
        einsums.linalg.axpby(alpha, _view(P, xr, xc, mode), beta, _view(P, yr, yc, mode))

    _run_overlapping(mode, body)

    expected = before.copy()
    expected[slice(*yr), slice(*yc)] = alpha * before[slice(*xr), slice(*xc)] + beta * before[slice(*yr), slice(*yc)]
    np.testing.assert_array_equal(np.asarray(P), expected)
