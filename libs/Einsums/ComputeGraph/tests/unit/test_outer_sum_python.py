# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for cg::outer_sum.

``einsums.linalg.outer_sum(result, vectors, coefficients)`` fills a rank-N
tensor with the outer sum ``Σ_k c_k * v_k(i_k)``. Canonical use is the
MP2/CC denominator:

    Δ(i,j,a,b) = ε_i + ε_j − ε_a − ε_b
      ↪ outer_sum(Δ, [eps_o, eps_o, eps_v, eps_v], [+1, +1, -1, -1])

Pass an empty coefficients list to default to all +1.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES


# ──────────────────────────────────────────────────────────────────────────
# Eager mode: rank-2 (simplest)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_outer_sum_rank2_eager(dtype):
    a = einsums.create_random_tensor("a", [3], dtype=dtype)
    b = einsums.create_random_tensor("b", [4], dtype=dtype)
    r = einsums.create_zero_tensor("r", [3, 4], dtype=dtype)

    einsums.linalg.outer_sum(r, [a, b], [+1.0, +1.0])

    a_np = np.asarray(a)
    b_np = np.asarray(b)
    expected = a_np[:, None] + b_np[None, :]
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-5)


def test_outer_sum_empty_coefficients_defaults_to_plus_one():
    a = einsums.create_random_tensor("a", [3])
    b = einsums.create_random_tensor("b", [4])
    r = einsums.create_zero_tensor("r", [3, 4])

    einsums.linalg.outer_sum(r, [a, b], [])

    expected = np.asarray(a)[:, None] + np.asarray(b)[None, :]
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_outer_sum_with_negative_signs(dtype):
    a = einsums.create_random_tensor("a", [3], dtype=dtype)
    b = einsums.create_random_tensor("b", [4], dtype=dtype)
    r = einsums.create_zero_tensor("r", [3, 4], dtype=dtype)

    einsums.linalg.outer_sum(r, [a, b], [+1.0, -1.0])

    expected = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# MP2 denominator pattern: rank-4 with mixed signs
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_outer_sum_mp2_denominator(dtype):
    nocc = 4
    nvirt = 6
    eps_o = einsums.create_random_tensor("eps_o", [nocc], dtype=dtype)
    eps_v = einsums.create_random_tensor("eps_v", [nvirt], dtype=dtype)
    delta = einsums.create_zero_tensor("delta", [nocc, nocc, nvirt, nvirt], dtype=dtype)

    einsums.linalg.outer_sum(delta, [eps_o, eps_o, eps_v, eps_v], [+1.0, +1.0, -1.0, -1.0])

    eo = np.asarray(eps_o)
    ev = np.asarray(eps_v)
    expected = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    np.testing.assert_allclose(np.asarray(delta), expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_outer_sum_mp2_denominator_in_capture(dtype):
    nocc = 3
    nvirt = 4
    eps_o = einsums.create_random_tensor("eps_o", [nocc], dtype=dtype)
    eps_v = einsums.create_random_tensor("eps_v", [nvirt], dtype=dtype)
    delta = einsums.create_zero_tensor("delta", [nocc, nocc, nvirt, nvirt], dtype=dtype)

    g = cg.Graph("denom")
    with cg.capture(g):
        einsums.linalg.outer_sum(delta, [eps_o, eps_o, eps_v, eps_v], [+1.0, +1.0, -1.0, -1.0])

    # 3 unique vector slots (eps_o, eps_v are the same tensor twice each) + delta = 3 tensors.
    assert g.num_tensors() == 3
    assert g.num_nodes() == 1

    # Not executed yet — delta stays zero.
    np.testing.assert_array_equal(np.asarray(delta), 0.0)

    g.execute()

    eo = np.asarray(eps_o)
    ev = np.asarray(eps_v)
    expected = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    np.testing.assert_allclose(np.asarray(delta), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Composed: build denominator then take reciprocal → MP2 energy weight
# ──────────────────────────────────────────────────────────────────────────


def test_outer_sum_and_element_transform_for_mp2():
    """Δ then 1/Δ — the full MP2-weight pattern, graph-only."""
    nocc, nvirt = 3, 4
    eps_o = einsums.create_random_tensor("eps_o", [nocc])
    eps_v = einsums.create_random_tensor("eps_v", [nvirt])
    # Force eigenvalues to be ordered so virtual > occupied and Δ < 0.
    np.asarray(eps_o)[:] = -np.abs(np.asarray(eps_o)) - 0.1
    np.asarray(eps_v)[:] = +np.abs(np.asarray(eps_v)) + 0.1

    inv_delta = einsums.create_zero_tensor("inv_delta", [nocc, nocc, nvirt, nvirt])

    g = cg.Graph("mp2-weight")
    with cg.capture(g):
        einsums.linalg.outer_sum(inv_delta, [eps_o, eps_o, eps_v, eps_v], [+1.0, +1.0, -1.0, -1.0])
        einsums.linalg.element_transform(inv_delta, lambda x: 1.0 / x)
    g.execute()

    eo = np.asarray(eps_o)
    ev = np.asarray(eps_v)
    expected_delta = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    expected = 1.0 / expected_delta
    np.testing.assert_allclose(np.asarray(inv_delta), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────


def test_outer_sum_rank_mismatch_raises():
    a = einsums.create_random_tensor("a", [3])
    b = einsums.create_random_tensor("b", [4])
    r = einsums.create_zero_tensor("r", [3, 4, 5])  # rank-3 but only 2 vectors
    with pytest.raises(Exception):
        einsums.linalg.outer_sum(r, [a, b], [])


def test_outer_sum_dim_mismatch_raises():
    a = einsums.create_random_tensor("a", [3])
    b = einsums.create_random_tensor("b", [4])
    r = einsums.create_zero_tensor("r", [3, 5])  # second axis 5 != b.dim(0)=4
    with pytest.raises(Exception):
        einsums.linalg.outer_sum(r, [a, b], [])


def test_outer_sum_coefficient_length_mismatch_raises():
    a = einsums.create_random_tensor("a", [3])
    b = einsums.create_random_tensor("b", [4])
    r = einsums.create_zero_tensor("r", [3, 4])
    with pytest.raises(Exception):
        einsums.linalg.outer_sum(r, [a, b], [1.0])  # only 1 coefficient


def test_outer_sum_vector_must_be_rank1():
    a = einsums.create_random_tensor("a", [3])
    b = einsums.create_random_tensor("b", [4, 4])  # not rank-1
    r = einsums.create_zero_tensor("r", [3, 4])
    with pytest.raises(Exception):
        einsums.linalg.outer_sum(r, [a, b], [])


def test_outer_sum_no_vectors_raises():
    r = einsums.create_zero_tensor("r", [3])
    with pytest.raises(Exception):
        einsums.linalg.outer_sum(r, [], [])
