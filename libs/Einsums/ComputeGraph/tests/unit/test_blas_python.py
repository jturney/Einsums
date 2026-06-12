# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for cg::axpy / axpby / gemv / ger bindings.

Mirrors the relevant slices of OperationsCoverage.cpp + MixedOps.cpp on the
Python side, exercising each operation in both eager mode (no graph) and
captured-then-executed mode. Numerical results are cross-checked against a
direct numpy computation so the binding is verified end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES, assert_close


# ──────────────────────────────────────────────────────────────────────────
# axpy: Y += alpha * X
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_axpy_eager(dtype):
    X = einsums.create_random_tensor("X", [4, 3], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [4, 3], dtype=dtype)
    expected = np.asarray(Y) + 2.0 * np.asarray(X)
    einsums.linalg.axpy(2.0, X, Y)
    assert_close(Y, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_axpy_in_graph_capture(dtype):
    X = einsums.create_random_tensor("X", [5], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [5], dtype=dtype)
    expected = np.asarray(Y) + 0.5 * np.asarray(X)

    g = cg.Graph("axpy")
    with cg.capture(g):
        einsums.linalg.axpy(0.5, X, Y)
    assert g.num_nodes() == 1
    g.execute()

    assert_close(Y, expected)


def test_axpy_complex_alpha_on_complex_tensor():
    X = einsums.create_random_tensor("X", [4], dtype="complex128")
    Y = einsums.create_random_tensor("Y", [4], dtype="complex128")
    alpha = 1.0 + 2.0j
    expected = np.asarray(Y) + alpha * np.asarray(X)
    einsums.linalg.axpy(alpha, X, Y)
    assert_close(Y, expected)


# ──────────────────────────────────────────────────────────────────────────
# axpby: Y = alpha * X + beta * Y
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_axpby_eager(dtype):
    X = einsums.create_random_tensor("X", [3, 4], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [3, 4], dtype=dtype)
    expected = 2.0 * np.asarray(X) + 3.0 * np.asarray(Y)
    einsums.linalg.axpby(2.0, X, 3.0, Y)
    assert_close(Y, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_axpby_zero_beta_overwrites_y(dtype):
    """beta=0 makes axpby equivalent to Y := alpha * X (scalar copy)."""
    X = einsums.create_random_tensor("X", [4, 4], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [4, 4], dtype=dtype)
    expected = 1.5 * np.asarray(X)
    einsums.linalg.axpby(1.5, X, 0.0, Y)
    assert_close(Y, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_axpby_in_graph_capture(dtype):
    X = einsums.create_random_tensor("X", [3, 3], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [3, 3], dtype=dtype)
    expected = 0.5 * np.asarray(X) + 0.25 * np.asarray(Y)

    g = cg.Graph("axpby")
    with cg.capture(g):
        einsums.linalg.axpby(0.5, X, 0.25, Y)
    g.execute()
    assert_close(Y, expected)


# ──────────────────────────────────────────────────────────────────────────
# gemv: y = alpha * op(A) * x + beta * y
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_gemv_eager_no_transpose(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    x = einsums.create_random_tensor("x", [5], dtype=dtype)
    y = einsums.create_zero_tensor("y", [4], dtype=dtype)
    expected = np.asarray(A) @ np.asarray(x)
    einsums.linalg.gemv(1.0, A, x, 0.0, y)
    assert_close(y, expected)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_gemv_eager_transpose(dtype):
    A = einsums.create_random_tensor("A", [5, 4], dtype=dtype)
    x = einsums.create_random_tensor("x", [5], dtype=dtype)
    y = einsums.create_zero_tensor("y", [4], dtype=dtype)
    expected = np.asarray(A).T @ np.asarray(x)
    einsums.linalg.gemv(1.0, A, x, 0.0, y, trans_a=True)
    assert_close(y, expected)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_gemv_in_graph_capture(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    x = einsums.create_random_tensor("x", [4], dtype=dtype)
    y = einsums.create_zero_tensor("y", [3], dtype=dtype)
    expected = 2.0 * (np.asarray(A) @ np.asarray(x))

    g = cg.Graph("gemv")
    with cg.capture(g):
        einsums.linalg.gemv(2.0, A, x, 0.0, y)
    assert g.num_nodes() == 1
    g.execute()

    assert_close(y, expected)


def test_gemv_accumulating_beta():
    """beta != 0 means y is accumulated into rather than overwritten."""
    A = einsums.create_random_tensor("A", [3, 3])
    x = einsums.create_random_tensor("x", [3])
    y = einsums.create_random_tensor("y", [3])
    y_before = np.asarray(y).copy()
    expected = np.asarray(A) @ np.asarray(x) + 0.5 * y_before
    einsums.linalg.gemv(1.0, A, x, 0.5, y)
    assert_close(y, expected)


# ──────────────────────────────────────────────────────────────────────────
# ger: A += alpha * X * Y^T (rank-1 update)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_ger_eager(dtype):
    X = einsums.create_random_tensor("X", [4], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [5], dtype=dtype)
    A = einsums.create_zero_tensor("A", [4, 5], dtype=dtype)
    expected = np.outer(np.asarray(X), np.asarray(Y))
    einsums.linalg.ger(1.0, X, Y, A)
    assert_close(A, expected)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_ger_in_graph_capture(dtype):
    X = einsums.create_random_tensor("X", [3], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [4], dtype=dtype)
    A = einsums.create_zero_tensor("A", [3, 4], dtype=dtype)
    expected = 2.0 * np.outer(np.asarray(X), np.asarray(Y))

    g = cg.Graph("ger")
    with cg.capture(g):
        einsums.linalg.ger(2.0, X, Y, A)
    assert g.num_nodes() == 1
    g.execute()

    assert_close(A, expected)


def test_ger_accumulates_into_a():
    """ger does ``A += alpha * X * Y^T``, initial A is preserved."""
    X = einsums.create_random_tensor("X", [3])
    Y = einsums.create_random_tensor("Y", [3])
    A = einsums.create_random_tensor("A", [3, 3])
    A_before = np.asarray(A).copy()
    expected = A_before + np.outer(np.asarray(X), np.asarray(Y))
    einsums.linalg.ger(1.0, X, Y, A)
    assert_close(A, expected)
