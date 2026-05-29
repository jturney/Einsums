# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for the cg::* LAPACK bindings.

Mirrors RuntimeTensorLAPACK.cpp + the LAPACK slices of MixedOps.cpp:

  * In-place ops (syev, heev, gesv, invert) — tested in eager mode and
    captured-then-executed mode.
  * Returning-form ops (det, svd, qr, syev_eig, trace) — tested eager
    AND verified to raise inside ``with cg.capture(g):`` since they
    return values by-construction and have no destination slot.

Numerical checks cross-reference numpy.linalg as the source of truth.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES


# ──────────────────────────────────────────────────────────────────────────
# trace — returning scalar
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_trace_eager_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    expected = np.trace(np.asarray(A))
    np.testing.assert_allclose(einsums.linalg.trace(A), expected, rtol=1e-5)


def test_trace_throws_during_capture():
    A = einsums.create_random_tensor("A", [3, 3])
    g = cg.Graph("trace")
    with cg.capture(g):
        with pytest.raises((RuntimeError, ValueError, Exception)):
            einsums.linalg.trace(A)


def test_trace_non_square_raises():
    """trace requires a square matrix; non-square inputs raise."""
    A = einsums.create_random_tensor("A", [3, 4])
    with pytest.raises(Exception):
        einsums.linalg.trace(A)


# ──────────────────────────────────────────────────────────────────────────
# syev: real symmetric eigendecomposition (in-place)
# ──────────────────────────────────────────────────────────────────────────


def _build_symmetric(n, dtype="float64"):
    A = einsums.create_random_tensor("A", [n, n], dtype=dtype)
    arr = np.asarray(A)
    arr[:] = (arr + arr.T) / 2.0
    return A


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_syev_eager_eigenvalues(dtype):
    n = 4
    A = _build_symmetric(n, dtype=dtype)
    A_orig = np.asarray(A).copy()
    W = einsums.create_zero_tensor("W", [n], dtype=dtype)

    einsums.linalg.syev(A, W, compute_eigenvectors=True)

    expected_w = np.linalg.eigvalsh(A_orig)
    np.testing.assert_allclose(np.sort(np.asarray(W)), np.sort(expected_w), rtol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_syev_in_graph_capture(dtype):
    n = 3
    A = _build_symmetric(n, dtype=dtype)
    A_orig = np.asarray(A).copy()
    W = einsums.create_zero_tensor("W", [n], dtype=dtype)

    g = cg.Graph("syev")
    with cg.capture(g):
        einsums.linalg.syev(A, W, compute_eigenvectors=True)
    g.execute()

    expected_w = np.linalg.eigvalsh(A_orig)
    np.testing.assert_allclose(np.sort(np.asarray(W)), np.sort(expected_w), rtol=1e-4)


# ──────────────────────────────────────────────────────────────────────────
# syev_eig: returning form — must raise during capture
# ──────────────────────────────────────────────────────────────────────────


def test_syev_eig_eager_returns_tuple():
    A = _build_symmetric(3)
    A_orig = np.asarray(A).copy()
    result = einsums.linalg.syev_eig(A, compute_eigenvectors=True)
    # syev_eig returns (eigenvectors, eigenvalues) per the docstring.
    assert isinstance(result, tuple)
    assert len(result) == 2
    _, w = result
    expected_w = np.linalg.eigvalsh(A_orig)
    np.testing.assert_allclose(np.sort(np.asarray(w)), np.sort(expected_w), rtol=1e-10)


def test_syev_eig_throws_during_capture():
    A = _build_symmetric(3)
    g = cg.Graph("syev_eig")
    with cg.capture(g):
        with pytest.raises(Exception):
            einsums.linalg.syev_eig(A, compute_eigenvectors=True)


# ──────────────────────────────────────────────────────────────────────────
# heev: Hermitian eigendecomposition
# ──────────────────────────────────────────────────────────────────────────


def _build_hermitian(n, dtype):
    A = einsums.create_random_tensor("A", [n, n], dtype=dtype)
    arr = np.asarray(A)
    arr[:] = (arr + np.conj(arr.T)) / 2.0
    return A


@pytest.mark.parametrize(
    ("complex_dtype", "real_dtype"),
    [("complex64", "float32"), ("complex128", "float64")],
)
def test_heev_eager_eigenvalues(complex_dtype, real_dtype):
    n = 3
    A = _build_hermitian(n, complex_dtype)
    A_orig = np.asarray(A).copy()
    W = einsums.create_zero_tensor("W", [n], dtype=real_dtype)

    einsums.linalg.heev(A, W, compute_eigenvectors=True)

    expected_w = np.linalg.eigvalsh(A_orig)
    np.testing.assert_allclose(np.sort(np.asarray(W)), np.sort(expected_w), rtol=1e-4)


# ──────────────────────────────────────────────────────────────────────────
# gesv: solve A * X = B
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_gesv_eager_2d_rhs(dtype):
    n = 3
    A = einsums.create_random_tensor("A", [n, n], dtype=dtype)
    B = einsums.create_random_tensor("B", [n, 2], dtype=dtype)
    A_np = np.asarray(A).copy()
    B_np = np.asarray(B).copy()
    expected_x = np.linalg.solve(A_np, B_np)

    info = einsums.linalg.gesv(A, B)

    assert info == 0
    np.testing.assert_allclose(np.asarray(B), expected_x, rtol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_gesv_in_graph_capture(dtype):
    n = 3
    A = einsums.create_random_tensor("A", [n, n], dtype=dtype)
    B = einsums.create_random_tensor("B", [n, 1], dtype=dtype)
    A_np = np.asarray(A).copy()
    B_np = np.asarray(B).copy()
    expected_x = np.linalg.solve(A_np, B_np)

    g = cg.Graph("gesv")
    with cg.capture(g):
        einsums.linalg.gesv(A, B)
    g.execute()

    np.testing.assert_allclose(np.asarray(B), expected_x, rtol=1e-4)


# ──────────────────────────────────────────────────────────────────────────
# invert: in-place matrix inverse
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_invert_eager(dtype):
    """A * A^-1 = I after invert(A)."""
    n = 4
    A = einsums.create_random_tensor("A", [n, n], dtype=dtype)
    # create_random_tensor draws from a uniform distribution, so the
    # resulting matrix can land arbitrarily close to singular by chance
    # — observed on Linux clang where getrf hit a zero pivot at (3, 3)
    # for a specific seed. Make A diagonally dominant by adding n*I so
    # invert always succeeds.
    A_np = np.asarray(A)
    A_np[:] = A_np + n * np.eye(n, dtype=A_np.dtype)
    A_orig = A_np.copy()
    einsums.linalg.invert(A)
    product = A_orig @ np.asarray(A)
    np.testing.assert_allclose(product, np.eye(n), atol=1e-3)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_invert_in_graph_capture(dtype):
    n = 3
    A = einsums.create_random_tensor("A", [n, n], dtype=dtype)
    # Same diagonal-dominant shim as test_invert_eager — uniform-random
    # 3×3 matrices are even more likely to be ill-conditioned.
    A_np = np.asarray(A)
    A_np[:] = A_np + n * np.eye(n, dtype=A_np.dtype)
    A_orig = A_np.copy()

    g = cg.Graph("invert")
    with cg.capture(g):
        einsums.linalg.invert(A)
    g.execute()

    product = A_orig @ np.asarray(A)
    np.testing.assert_allclose(product, np.eye(n), atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────
# det — returning scalar; must raise during capture
# ──────────────────────────────────────────────────────────────────────────


def test_det_eager_matches_numpy():
    A = einsums.create_random_tensor("A", [4, 4])
    np.testing.assert_allclose(
        einsums.linalg.det(A), np.linalg.det(np.asarray(A)), rtol=1e-8,
    )


def test_det_throws_during_capture():
    A = einsums.create_random_tensor("A", [3, 3])
    g = cg.Graph("det")
    with cg.capture(g):
        with pytest.raises(Exception):
            einsums.linalg.det(A)


# ──────────────────────────────────────────────────────────────────────────
# svd — returning tuple; must raise during capture
# ──────────────────────────────────────────────────────────────────────────


def test_svd_eager_singular_values_match_numpy():
    A = einsums.create_random_tensor("A", [4, 3])
    A_np = np.asarray(A).copy()
    result = einsums.linalg.svd(A)
    # svd returns (U, S, Vt).
    assert isinstance(result, tuple)
    assert len(result) == 3
    _, S, _ = result
    expected_s = np.linalg.svd(A_np, compute_uv=False)
    np.testing.assert_allclose(np.sort(np.asarray(S))[::-1], expected_s, rtol=1e-4)


def test_svd_throws_during_capture():
    A = einsums.create_random_tensor("A", [3, 2])
    g = cg.Graph("svd")
    with cg.capture(g):
        with pytest.raises(Exception):
            einsums.linalg.svd(A)


# ──────────────────────────────────────────────────────────────────────────
# qr — returning tuple; must raise during capture
# ──────────────────────────────────────────────────────────────────────────


def test_qr_eager_q_r_reconstructs_input():
    n, m = 4, 3
    A = einsums.create_random_tensor("A", [n, m])
    A_np = np.asarray(A).copy()
    Q, R = einsums.linalg.qr(A)
    reconstructed = np.asarray(Q) @ np.asarray(R)
    np.testing.assert_allclose(reconstructed, A_np, rtol=1e-5)


def test_qr_throws_during_capture():
    A = einsums.create_random_tensor("A", [3, 3])
    g = cg.Graph("qr")
    with cg.capture(g):
        with pytest.raises(Exception):
            einsums.linalg.qr(A)
