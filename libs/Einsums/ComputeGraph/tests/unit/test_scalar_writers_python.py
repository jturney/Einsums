# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for the pointer-writing graph-aware forms of dot, norm, trace.

The returning forms (``einsums.linalg.dot(A, B) -> scalar``) throw inside
``with cg.capture():``. The pointer-writing forms accept a 1-element tensor
as the destination, so the result becomes a graph-tracked slot and the call
can be recorded. Together they unblock graph-only SCF patterns:

    e        = ½ Σ D · (H+F)      → dot(e, D, sum_HF) then scale(0.5, e)
    rms(ΔD)  = ||D − D_old||_F     → axpby + axpy + norm(rms, FROBENIUS, ΔD)
    tr(A)    = Σ A_ii              → trace(t, A)

The result tensor is rank-1 with one element so it composes with downstream
ops; subsequent scale/axpy operate on it as a normal tensor.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES, COMPLEX_DTYPES, assert_close


# Map each input dtype to the dtype of a Frobenius-norm scalar result.
_NORM_RESULT_DTYPE = {
    "float32": "float32",
    "float64": "float64",
    "complex64": "float32",
    "complex128": "float64",
}


# ──────────────────────────────────────────────────────────────────────────
# dot(result, A, B): writes into result[0]
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_dot_writer_eager_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [6], dtype=dtype)
    B = einsums.create_random_tensor("B", [6], dtype=dtype)
    r = einsums.create_zero_tensor("r", [1], dtype=dtype)

    einsums.linalg.dot(r, A, B)

    expected = np.dot(np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_dot_writer_captured_records_node(dtype):
    A = einsums.create_random_tensor("A", [4], dtype=dtype)
    B = einsums.create_random_tensor("B", [4], dtype=dtype)
    r = einsums.create_zero_tensor("r", [1], dtype=dtype)

    g = cg.Graph("dot-writer")
    with cg.capture(g):
        einsums.linalg.dot(r, A, B)

    # 3 tensor slots (A, B, r) + 1 node.
    assert g.num_nodes() == 1
    assert g.num_tensors() == 3


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_dot_writer_captured_matches_eager(dtype):
    A = einsums.create_random_tensor("A", [5], dtype=dtype)
    B = einsums.create_random_tensor("B", [5], dtype=dtype)

    eager = einsums.create_zero_tensor("eager", [1], dtype=dtype)
    capt = einsums.create_zero_tensor("capt", [1], dtype=dtype)

    einsums.linalg.dot(eager, A, B)

    g = cg.Graph("dot-vs-eager")
    with cg.capture(g):
        einsums.linalg.dot(capt, A, B)
    g.execute()

    np.testing.assert_allclose(np.asarray(capt), np.asarray(eager), rtol=1e-5)


def test_dot_writer_zero_size_result_raises():
    A = einsums.create_random_tensor("A", [3])
    B = einsums.create_random_tensor("B", [3])
    r = einsums.create_zero_tensor("r", [0])  # empty
    with pytest.raises(Exception):
        einsums.linalg.dot(r, A, B)


# ──────────────────────────────────────────────────────────────────────────
# norm(result, norm_type, A): writes the (real) norm into result[0]
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_writer_frobenius_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    r_dtype = _NORM_RESULT_DTYPE[dtype]
    r = einsums.create_zero_tensor("r", [1], dtype=r_dtype)

    einsums.linalg.norm(r, einsums.linalg.Norm.FROBENIUS, A)

    expected = np.linalg.norm(np.asarray(A), ord="fro")
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_norm_writer_captured_matches_eager(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)

    eager = einsums.create_zero_tensor("eager", [1], dtype=dtype)
    capt = einsums.create_zero_tensor("capt", [1], dtype=dtype)

    einsums.linalg.norm(eager, einsums.linalg.Norm.FROBENIUS, A)

    g = cg.Graph("norm-vs-eager")
    with cg.capture(g):
        einsums.linalg.norm(capt, einsums.linalg.Norm.FROBENIUS, A)
    g.execute()

    np.testing.assert_allclose(np.asarray(capt), np.asarray(eager), rtol=1e-5)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_norm_writer_complex_returns_real_dtype(dtype):
    """For complex inputs, the norm is real-valued — result tensor must be real-dtype."""
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    r_dtype = _NORM_RESULT_DTYPE[dtype]
    r = einsums.create_zero_tensor("r", [1], dtype=r_dtype)

    einsums.linalg.norm(r, einsums.linalg.Norm.FROBENIUS, A)

    expected = np.linalg.norm(np.asarray(A), ord="fro")
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# trace(result, A): writes Σ A_ii into result[0]
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_trace_writer_eager_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    r = einsums.create_zero_tensor("r", [1], dtype=dtype)

    einsums.linalg.trace(r, A)

    expected = np.trace(np.asarray(A))
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_trace_writer_captured_matches_eager(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)

    eager = einsums.create_zero_tensor("eager", [1], dtype=dtype)
    capt = einsums.create_zero_tensor("capt", [1], dtype=dtype)

    einsums.linalg.trace(eager, A)

    g = cg.Graph("trace-vs-eager")
    with cg.capture(g):
        einsums.linalg.trace(capt, A)
    g.execute()

    np.testing.assert_allclose(np.asarray(capt), np.asarray(eager), rtol=1e-5)


def test_trace_writer_non_square_raises():
    A = einsums.create_random_tensor("A", [3, 4])
    r = einsums.create_zero_tensor("r", [1])
    with pytest.raises(Exception):
        einsums.linalg.trace(r, A)


def test_trace_writer_rank3_raises():
    A = einsums.create_random_tensor("A", [3, 3, 3])
    r = einsums.create_zero_tensor("r", [1])
    with pytest.raises(Exception):
        einsums.linalg.trace(r, A)


# ──────────────────────────────────────────────────────────────────────────
# Composed SCF energy pattern: e = ½ Σ D·(H+F)
# ──────────────────────────────────────────────────────────────────────────


def test_scf_energy_pattern_captured():
    n = 5
    D = einsums.create_random_tensor("D", [n, n])
    H = einsums.create_random_tensor("H", [n, n])
    F = einsums.create_random_tensor("F", [n, n])
    sum_HF = einsums.create_zero_tensor("HF", [n, n])
    e = einsums.create_zero_tensor("e", [1])

    g = cg.Graph("scf-energy")
    with cg.capture(g):
        einsums.linalg.axpby(1.0, H, 0.0, sum_HF)  # sum_HF = H
        einsums.linalg.axpy(1.0, F, sum_HF)         # sum_HF += F
        einsums.linalg.dot(e, D, sum_HF)            # e = D · (H + F)
        einsums.linalg.scale(0.5, e)                # e = 0.5 * (D · (H + F))
    g.execute()

    expected = 0.5 * np.sum(np.asarray(D) * (np.asarray(H) + np.asarray(F)))
    np.testing.assert_allclose(np.asarray(e)[0], expected, rtol=1e-5)
