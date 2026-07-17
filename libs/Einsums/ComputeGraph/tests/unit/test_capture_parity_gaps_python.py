# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

# ----------------------------------------------------------------------------------------------
#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Captured-node parity for the non-einsum coverage-audit gaps.

Each test drives an operation through graph capture with the dtype/view/
dimension class that was previously tested only eagerly, comparing against a
numpy oracle (or eager behavior for the degenerate-dimension parity cases).
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, COMPLEX_DTYPES

_ctr = iter(range(100000))


def _nm():
    return f"cpg{next(_ctr)}"


def _rtol(dtype):
    return 2e-4 if dtype in ("float32", "complex64") else 1e-9


def _mk(arr, dtype, name=None):
    t = einsums.create_zero_tensor(name or _nm(), list(arr.shape), dtype=dtype)
    if arr.size:
        np.asarray(t)[...] = arr
    return t


def _rand(shape, dtype, rng):
    a = rng.standard_normal(shape)
    if dtype.startswith("complex"):
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


# ── Gap 1: heev through capture (plus strided-view input) ────────────────────


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_heev_in_graph_capture(dtype):
    n = 4
    rng = np.random.default_rng(11)
    H0 = _rand((n, n), dtype, rng)
    H0 = ((H0 + np.conj(H0.T)) / 2.0).astype(dtype)
    real_dtype = "float32" if dtype == "complex64" else "float64"

    A = _mk(H0, dtype)
    W = einsums.create_zero_tensor(_nm(), [n], dtype=real_dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.heev(A, W, compute_eigenvectors=True)
    g.execute()

    expected = np.linalg.eigvalsh(H0)
    np.testing.assert_allclose(np.sort(np.asarray(W)), np.sort(expected), rtol=5e-4 if dtype == "complex64" else 1e-9)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_heev_view_operand_is_not_bound(dtype):
    # FINDING (audit follow-up): the heev python binding has no
    # RuntimeTensorView overload, so a Hermitian sub-block cannot be
    # captured directly - unlike the C++ eager path, which covers strided
    # views (LinearAlgebra/heev.cpp). This assertion pins the current
    # binding surface; when a view overload is added, replace it with the
    # sub-block eigvalsh parity check.
    n, big = 4, 6
    rng = np.random.default_rng(12)
    B0 = _rand((big, big), dtype, rng)
    real_dtype = "float32" if dtype == "complex64" else "float64"

    big_t = _mk(B0, dtype)
    W = einsums.create_zero_tensor(_nm(), [n], dtype=real_dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        Av = cg.view(big_t, [(1, 1 + n), (1, 1 + n)])
        with pytest.raises(TypeError):
            einsums.linalg.heev(Av, W, compute_eigenvectors=True)


# ── Gaps 2 & 8: complex direct_product / direct_division / outer_sum ─────────


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_direct_product_complex_in_capture(dtype):
    rng = np.random.default_rng(13)
    A0, B0, C0 = (_rand((3, 4), dtype, rng) for _ in range(3))
    alpha, beta = dtype_scalar(dtype, 1.5, -0.5), dtype_scalar(dtype, 2.0, 1.0)
    A, B, C = _mk(A0, dtype), _mk(B0, dtype), _mk(C0, dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.direct_product(alpha, A, B, beta, C)
    g.execute()

    np.testing.assert_allclose(np.asarray(C), beta * C0 + alpha * A0 * B0, rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_direct_division_complex_in_capture(dtype):
    rng = np.random.default_rng(14)
    A0 = _rand((3, 4), dtype, rng)
    B0 = _rand((3, 4), dtype, rng) + 3.0  # keep away from zero
    C0 = _rand((3, 4), dtype, rng)
    alpha, beta = dtype_scalar(dtype, 1.0, 1.0), dtype_scalar(dtype, 0.5, 0.0)
    A, B, C = _mk(A0, dtype), _mk(B0, dtype), _mk(C0, dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.direct_division(alpha, A, B, beta, C)
    g.execute()

    np.testing.assert_allclose(np.asarray(C), beta * C0 + alpha * A0 / B0, rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_outer_sum_complex_in_capture(dtype):
    rng = np.random.default_rng(15)
    x0, y0 = _rand((3,), dtype, rng), _rand((4,), dtype, rng)
    x, y = _mk(x0, dtype), _mk(y0, dtype)
    D = einsums.create_zero_tensor(_nm(), [3, 4], dtype=dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.outer_sum(D, [x, y], [1.0, -1.0])
    g.execute()

    np.testing.assert_allclose(np.asarray(D), x0[:, None] - y0[None, :], rtol=_rtol(dtype))


def dtype_scalar(dtype, re, im):
    return complex(re, im) if dtype.startswith("complex") else re


# ── Gap 3: symm_gemm complex (incl. trans_a) through capture ─────────────────


def _symm_ref(A, B, trans_a=False):
    Aop = A.T if trans_a else A
    return B.conj().T @ Aop @ B if np.iscomplexobj(A) else B.T @ Aop @ B


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.parametrize("trans_a", [False, True])
def test_symm_gemm_complex_in_capture(dtype, trans_a):
    rng = np.random.default_rng(16)
    A0 = _rand((4, 4), dtype, rng)
    A0 = ((A0 + np.conj(A0.T)) / 2.0).astype(dtype)  # Hermitian A
    B0 = _rand((4, 3), dtype, rng)
    A, B = _mk(A0, dtype), _mk(B0, dtype)
    C = einsums.create_zero_tensor(_nm(), [3, 3], dtype=dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.symm_gemm(A, B, C, trans_a=trans_a)
    g.execute()

    expected = np.asarray(B0).T @ (A0.T if trans_a else A0) @ np.asarray(B0)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=_rtol(dtype))


# ── Gap 4: captured dot / trace pointer-writers with complex dtypes ──────────


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_dot_writer_complex_in_capture(dtype):
    rng = np.random.default_rng(17)
    A0, B0 = _rand((5,), dtype, rng), _rand((5,), dtype, rng)
    A, B = _mk(A0, dtype), _mk(B0, dtype)
    r = einsums.create_zero_tensor(_nm(), [1], dtype=dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.dot(r, A, B)
    g.execute()

    # dot is the non-conjugating product-sum.
    np.testing.assert_allclose(np.asarray(r)[0], np.sum(A0 * B0), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_trace_writer_complex_in_capture(dtype):
    rng = np.random.default_rng(18)
    A0 = _rand((4, 4), dtype, rng)
    A = _mk(A0, dtype)
    r = einsums.create_zero_tensor(_nm(), [1], dtype=dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.trace(r, A)
    g.execute()

    np.testing.assert_allclose(np.asarray(r)[0], np.trace(A0), rtol=_rtol(dtype))


# ── Gaps 5 & 6: strided views through captured gemv / syev ───────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_gemv_on_permuted_view_in_capture(dtype):
    # A transposed (permute_view) matrix has non-unit stride on the fast
    # axis - the captured path previously saw only contiguous slices.
    rng = np.random.default_rng(19)
    M0 = _rand((4, 6), dtype, rng)
    x0 = _rand((4,), dtype, rng)
    M, x = _mk(M0, dtype), _mk(x0, dtype)
    y = einsums.create_zero_tensor(_nm(), [6], dtype=dtype)

    Mt = M.permute_view([1, 0])  # 6x4 view, strides swapped

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.gemv(1.0, Mt, x, 0.0, y)
    g.execute()

    np.testing.assert_allclose(np.asarray(y), M0.T @ x0, rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_syev_view_operand_is_not_bound(dtype):
    # FINDING (same class as heev): the syev python binding has no
    # RuntimeTensorView overload either, so strided/permuted views cannot
    # reach the captured syev node - unlike the C++ eager path
    # (LinearAlgebra/syev.cpp covers Stride{6,2} views). Pins the current
    # binding surface; replace with the eigvalsh parity check when a view
    # overload lands.
    rng = np.random.default_rng(20)
    S0 = _rand((4, 4), dtype, rng)
    S0 = ((S0 + S0.T) / 2.0).astype(dtype)
    S = _mk(S0, dtype)
    W = einsums.create_zero_tensor(_nm(), [4], dtype=dtype)

    Sv = S.permute_view([1, 0])

    g = cg.Graph(_nm())
    with cg.capture(g):
        with pytest.raises(TypeError):
            einsums.linalg.syev(Sv, W, compute_eigenvectors=True)


# ── Gap 7: complex gesv / invert through capture ─────────────────────────────


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_gesv_complex_in_capture(dtype):
    rng = np.random.default_rng(21)
    A0 = _rand((4, 4), dtype, rng) + 4.0 * np.eye(4, dtype=dtype)  # well-conditioned
    b0 = _rand((4, 1), dtype, rng)
    A, b = _mk(A0, dtype), _mk(b0, dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.gesv(A, b)
    g.execute()

    np.testing.assert_allclose(np.asarray(b), np.linalg.solve(A0, b0), rtol=5e-4 if dtype == "complex64" else 1e-8)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_invert_complex_in_capture(dtype):
    rng = np.random.default_rng(22)
    A0 = _rand((4, 4), dtype, rng) + 4.0 * np.eye(4, dtype=dtype)
    A = _mk(A0, dtype)

    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.invert(A)
    g.execute()

    np.testing.assert_allclose(np.asarray(A), np.linalg.inv(A0), rtol=5e-4 if dtype == "complex64" else 1e-8)


# ── Gap 9: degenerate dimensions through captured LAPACK nodes ───────────────


def test_degenerate_n1_lapack_in_capture():
    # N=1: eigen/solve/invert reduce to scalar arithmetic; previously only
    # driven eagerly (hypothesis @example pins).
    A = _mk(np.array([[3.0]]), "float64")
    W = einsums.create_zero_tensor(_nm(), [1], dtype="float64")
    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.linalg.syev(A, W, compute_eigenvectors=True)
    g.execute()
    np.testing.assert_allclose(np.asarray(W), [3.0])

    A2 = _mk(np.array([[2.0]]), "float64")
    b = _mk(np.array([[8.0]]), "float64")
    g2 = cg.Graph(_nm())
    with cg.capture(g2):
        einsums.linalg.gesv(A2, b)
    g2.execute()
    np.testing.assert_allclose(np.asarray(b), [[4.0]])

    A3 = _mk(np.array([[5.0]]), "float64")
    g3 = cg.Graph(_nm())
    with cg.capture(g3):
        einsums.linalg.invert(A3)
    g3.execute()
    np.testing.assert_allclose(np.asarray(A3), [[0.2]])


def test_degenerate_n0_invert_parity_with_eager():
    # N=0: parity contract - whatever eager does (compute or raise), the
    # captured path must do the same.
    def eager():
        A = einsums.create_zero_tensor(_nm(), [0, 0], dtype="float64")
        einsums.linalg.invert(A)

    def captured():
        A = einsums.create_zero_tensor(_nm(), [0, 0], dtype="float64")
        g = cg.Graph(_nm())
        with cg.capture(g):
            einsums.linalg.invert(A)
        g.execute()

    eager_raised = False
    try:
        eager()
    except Exception:
        eager_raised = True

    if eager_raised:
        with pytest.raises(Exception):
            captured()
    else:
        captured()
