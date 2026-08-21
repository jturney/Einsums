# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for ``getrf``/``getrs`` and the ``LuPivots`` handle.

The pair exists so that one factorization can serve many right-hand sides:
``gesv`` consumes its coefficient matrix, so a caller solving repeatedly
against one matrix has to copy it per solve, and copying is what these two
remove. The checks below therefore care about three things beyond the numbers -
that the factorization SURVIVES a solve, that a captured factorization is
ordered before the solves reading it, and that the pivots outlive the handle
the caller held, since both executors bake in the shared buffer rather than the
Python object.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
from einsums.testing import ALL_DTYPES, assert_close


def _well_conditioned(n, dtype, seed):
    """A matrix no LU has to work hard on, so the gate is on the binding."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    if np.dtype(dtype).kind == "c":
        A = A + 1j * rng.random((n, n))
    return (A + n * np.eye(n)).astype(dtype)


def _tensor(name, arr):
    T = einsums.create_zero_tensor(name, list(arr.shape), dtype=str(arr.dtype))
    np.asarray(T)[...] = arr
    return T


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_eager_getrf_getrs_matches_numpy(dtype):
    n = 6
    A = _well_conditioned(n, dtype, 11)
    B = np.asarray(np.random.default_rng(12).random((n, 3)), dtype=dtype)

    Ta, Tb = _tensor("A", A), _tensor("B", B)
    pivots = la.LuPivots()

    assert la.getrf(Ta, pivots) == 0
    assert pivots.size == n
    assert la.getrs(Ta, pivots, Tb) == 0

    assert_close(np.asarray(Tb), np.linalg.solve(A, B), dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_captured_getrf_getrs_matches_eager(dtype):
    n = 5
    A = _well_conditioned(n, dtype, 21)
    B = np.asarray(np.random.default_rng(22).random((n, 2)), dtype=dtype)

    eager_a, eager_b = _tensor("A", A), _tensor("B", B)
    eager_pivots = la.LuPivots()
    la.getrf(eager_a, eager_pivots)
    la.getrs(eager_a, eager_pivots, eager_b)

    graph_a, graph_b = _tensor("A", A), _tensor("B", B)
    pivots = la.LuPivots()
    g = cg.Graph("lu")
    with cg.capture(g):
        la.getrf(graph_a, pivots)
        la.getrs(graph_a, pivots, graph_b)

    assert g.num_nodes() == 2
    # Capture records; it does not run. The pivots are sized by the executor.
    assert pivots.size == 0

    g.execute()

    assert pivots.size == n
    assert_close(np.asarray(graph_b), np.asarray(eager_b), dtype=dtype)
    assert_close(np.asarray(graph_a), np.asarray(eager_a), dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_one_factorization_serves_many_right_hand_sides(dtype):
    n = 5
    A = _well_conditioned(n, dtype, 31)
    rng = np.random.default_rng(32)
    rhs = [np.asarray(rng.random((n, k)), dtype=dtype) for k in (1, 3, 2)]

    Ta = _tensor("A", A)
    tensors = [_tensor(f"B{i}", b) for i, b in enumerate(rhs)]
    pivots = la.LuPivots()

    g = cg.Graph("lu-many")
    with cg.capture(g):
        la.getrf(Ta, pivots)
        for T in tensors:
            la.getrs(Ta, pivots, T)
    g.execute()

    for T, b in zip(tensors, rhs):
        assert_close(np.asarray(T), np.linalg.solve(A, b), dtype=dtype)


def test_getrs_leaves_the_factorization_alone():
    n = 4
    A = _well_conditioned(n, "float64", 41)
    B = np.asarray(np.random.default_rng(42).random((n, 2)))

    Ta, Tb = _tensor("A", A), _tensor("B", B)
    pivots = la.LuPivots()
    la.getrf(Ta, pivots)

    factored = np.array(np.asarray(Ta), copy=True)
    la.getrs(Ta, pivots, Tb)

    assert np.array_equal(np.asarray(Ta), factored)


def test_the_factorization_orders_the_solve():
    """Ordering rides on the tensor, because the pivots are not a slot."""
    n = 4
    A = _well_conditioned(n, "float64", 51)
    B = np.asarray(np.random.default_rng(52).random((n, 2)))

    Ta, Tb = _tensor("A", A), _tensor("B", B)
    pivots = la.LuPivots()

    g = cg.Graph("lu-order")
    with cg.capture(g):
        la.getrf(Ta, pivots)
        la.getrs(Ta, pivots, Tb)

    g.execute()
    assert_close(np.asarray(Tb), np.linalg.solve(A, B), rtol=1e-12, atol=1e-12)


def test_the_pivots_outlive_the_handle():
    """Both executors hold the buffer, not the Python object that named it."""
    n = 4
    A = _well_conditioned(n, "float64", 61)
    B = np.asarray(np.random.default_rng(62).random((n, 2)))

    Ta, Tb = _tensor("A", A), _tensor("B", B)

    g = cg.Graph("lu-lifetime")
    pivots = la.LuPivots()
    with cg.capture(g):
        la.getrf(Ta, pivots)
        la.getrs(Ta, pivots, Tb)
    del pivots

    g.execute()
    assert_close(np.asarray(Tb), np.linalg.solve(A, B), rtol=1e-12, atol=1e-12)


def test_replaying_a_captured_lu_reproduces_the_solve():
    n = 4
    A = _well_conditioned(n, "float64", 71)
    B = np.asarray(np.random.default_rng(72).random((n, 2)))
    want = np.linalg.solve(A, B)

    Ta, Tb = _tensor("A", A), _tensor("B", B)
    pivots = la.LuPivots()

    g = cg.Graph("lu-replay")
    with cg.capture(g):
        la.getrf(Ta, pivots)
        la.getrs(Ta, pivots, Tb)

    for _ in range(3):
        np.asarray(Ta)[...] = A
        np.asarray(Tb)[...] = B
        g.execute()
        assert_close(np.asarray(Tb), want, rtol=1e-12, atol=1e-12)


def test_getrs_solves_a_rank_one_right_hand_side():
    n = 5
    A = _well_conditioned(n, "float64", 81)
    b = np.asarray(np.random.default_rng(82).random(n))

    Ta, Tb = _tensor("A", A), _tensor("b", b)
    pivots = la.LuPivots()

    g = cg.Graph("lu-vector")
    with cg.capture(g):
        la.getrf(Ta, pivots)
        la.getrs(Ta, pivots, Tb)
    g.execute()

    assert_close(np.asarray(Tb), np.linalg.solve(A, b), rtol=1e-12, atol=1e-12)


def test_zero_extent_lu_is_a_quick_return():
    """An empty domain is a valid domain; LAPACK no-ops and so must the graph."""
    empty_a = einsums.create_zero_tensor("A", [0, 0], dtype="float64")
    empty_b = einsums.create_zero_tensor("B", [0, 3], dtype="float64")
    pivots = la.LuPivots()

    g = cg.Graph("lu-empty")
    with cg.capture(g):
        la.getrf(empty_a, pivots)
        la.getrs(empty_a, pivots, empty_b)
    g.execute()

    assert pivots.size == 0

    # An order the pivots cover, with no right-hand side to solve for.
    A = _well_conditioned(3, "float64", 91)
    Ta = _tensor("A", A)
    no_rhs = einsums.create_zero_tensor("B", [3, 0], dtype="float64")
    wide = la.LuPivots()

    g2 = cg.Graph("lu-no-rhs")
    with cg.capture(g2):
        la.getrf(Ta, wide)
        la.getrs(Ta, wide, no_rhs)
    g2.execute()

    assert wide.size == 3


def test_getrs_refuses_a_mismatched_factorization():
    Ta = _tensor("A", _well_conditioned(4, "float64", 101))
    Tb = _tensor("B", np.zeros((5, 2)))
    pivots = la.LuPivots()
    la.getrf(Ta, pivots)

    with pytest.raises(Exception):
        la.getrs(Ta, pivots, Tb)


def test_getrf_refuses_a_rank_three_tensor():
    T = einsums.create_zero_tensor("A", [2, 2, 2], dtype="float64")
    with pytest.raises(Exception):
        la.getrf(T, la.LuPivots())
