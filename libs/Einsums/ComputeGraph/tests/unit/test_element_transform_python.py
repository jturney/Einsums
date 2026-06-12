# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for cg::element_transform bindings.

Verifies ``einsums.linalg.element_transform(tensor, callable)`` applies a Python
callable element-wise across all four bound dtypes, both eagerly and when
recorded inside ``with cg.capture(g):``. The eager and captured paths must
agree.

This binding unblocks graph-only SCF/MP2 paths that need unary maps such as
``1/sqrt(x)`` for symmetric orthogonalization and ``1/x`` for MP2 energy
denominators.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES, assert_close


# ──────────────────────────────────────────────────────────────────────────
# Eager mode: callable runs against every element immediately
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_element_transform_eager_doubles(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    before = np.asarray(A).copy()
    einsums.linalg.element_transform(A, lambda x: 2 * x)
    assert_close(A, 2 * before)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_element_transform_eager_reciprocal_sqrt(dtype):
    """The S^{-1/2} use case: f(x) = 1 / sqrt(x) on positive eigenvalues."""
    # Build a strictly-positive vector (eigenvalue-shaped).
    s = einsums.create_random_tensor("s", [6], dtype=dtype)
    arr = np.asarray(s)
    arr[:] = np.abs(arr) + 0.5  # bump above zero
    before = arr.copy()

    einsums.linalg.element_transform(s, lambda x: 1.0 / math.sqrt(x))

    expected = 1.0 / np.sqrt(before)
    np.testing.assert_allclose(np.asarray(s), expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_element_transform_eager_mp2_denominator_pattern(dtype):
    """The MP2 use case: f(x) = 1 / x on a precomputed denominator tensor."""
    delta = einsums.create_random_tensor("delta", [3, 3, 4, 4], dtype=dtype)
    arr = np.asarray(delta)
    # Avoid zeros so 1/x is well-defined; sign doesn't matter.
    arr[:] = np.where(np.abs(arr) < 0.1, 0.5, arr)
    before = arr.copy()

    einsums.linalg.element_transform(delta, lambda x: 1.0 / x)

    np.testing.assert_allclose(np.asarray(delta), 1.0 / before, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Capture mode: records a graph node; lambda runs at execute()
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_element_transform_captured_records_node(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    g = cg.Graph("et")
    with cg.capture(g):
        einsums.linalg.element_transform(A, lambda x: x + 1.0)

    # One node, one tensor.
    assert g.num_nodes() == 1
    assert g.num_tensors() == 1


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_element_transform_captured_matches_eager(dtype):
    """Eager and captured-then-executed produce the same result."""
    eager = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    capt = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    # Same starting data on both.
    np.asarray(capt)[...] = np.asarray(eager)

    fn = lambda x: x * x - 0.25  # noqa: E731, exercising an arbitrary nonlinear unary

    # Eager
    einsums.linalg.element_transform(eager, fn)

    # Captured
    g = cg.Graph("et-vs-eager")
    with cg.capture(g):
        einsums.linalg.element_transform(capt, fn)
    g.execute()

    np.testing.assert_allclose(np.asarray(eager), np.asarray(capt), rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_element_transform_deferred_until_execute(dtype):
    """Recording into a graph must NOT mutate the tensor, only execute() does."""
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    before = np.asarray(A).copy()

    g = cg.Graph("deferred")
    with cg.capture(g):
        einsums.linalg.element_transform(A, lambda x: 99.0)

    # Tensor unchanged after capture exits, the lambda has not run yet.
    np.testing.assert_array_equal(np.asarray(A), before)

    g.execute()
    np.testing.assert_array_equal(np.asarray(A), np.full_like(before, 99.0))


# ──────────────────────────────────────────────────────────────────────────
# Higher-rank coverage
# ──────────────────────────────────────────────────────────────────────────


def test_element_transform_rank3():
    A = einsums.create_random_tensor("A", [2, 3, 4])
    before = np.asarray(A).copy()
    einsums.linalg.element_transform(A, lambda x: -x)
    np.testing.assert_allclose(np.asarray(A), -before, rtol=1e-12)


def test_element_transform_rank1():
    A = einsums.create_random_tensor("A", [7])
    before = np.asarray(A).copy()
    einsums.linalg.element_transform(A, lambda x: x + 10.0)
    np.testing.assert_allclose(np.asarray(A), before + 10.0, rtol=1e-12)
