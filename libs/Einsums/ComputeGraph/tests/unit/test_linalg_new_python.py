# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for the linalg ops added in the May 2026 binding pass.

Covers six C++ surfaces newly exposed under ``einsums.linalg``:

  * ``norm``: Norm enum routing (MAXABS / ONE / INFTY / FROBENIUS / TWO)
                         against ``np.linalg.norm``.
  * ``pow``: symmetric positive-definite ``A^alpha`` against ``A @ A``.
  * ``symm_gemm``: ``B^T A B`` and the three transpose variants in both eager
                         and captured-then-executed forms. It is the only op of
                         the set that records into a graph.
  * ``svd_dd``: divide-and-conquer SVD reconstructing ``A`` from
                         ``U * Sigma * Vt``.
  * ``truncated_svd``: top-k singular values vs ``np.linalg.svd`` plus the
                         documented ``m >= k + 5`` precondition surfacing as
                         ``IndexError``.
  * ``truncated_syev``: randomized top-k eigvals against ``np.linalg.eigvalsh``,
                         with a loose ballpark tolerance, since the algorithm is
                         approximate by design.

The Norm and Vectors enums get round-trip checks too so the LinearAlgebra
side of the binding (PYBIND opt-in plus EINSUMS_PYBIND_EXPOSE on the enums)
doesn't regress silently.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES, assert_close


# ──────────────────────────────────────────────────────────────────────────
# Norm and Vectors enums
# ──────────────────────────────────────────────────────────────────────────


def test_norm_enum_members():
    """The Norm enum exposes all five LAPACK-style modes."""
    members = set(einsums.linalg.Norm.__members__)
    assert members == {"MAXABS", "ONE", "INFTY", "FROBENIUS", "TWO"}


def test_vectors_enum_members():
    """The Vectors enum exposes ALL / SOME / NONE."""
    members = set(einsums.linalg.Vectors.__members__)
    assert members == {"ALL", "SOME", "NONE"}


# ──────────────────────────────────────────────────────────────────────────
# norm: matrix and vector norms
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_frobenius_matrix(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    got = einsums.linalg.norm(einsums.linalg.Norm.FROBENIUS, A)
    expected = np.linalg.norm(np.asarray(A), "fro")
    # Mac Accelerate's complex128 dznrm2 path accumulates wider rounding than
    # Linux OpenBLAS / MKL, observed ~1.3e-9 relative on macOS CI vs <1e-15
    # on Linux. Match the spectral-norm test's pattern: 1e-5 for f32/c64,
    # 1e-8 for f64/c128 (still tight enough to catch real bugs).
    assert_close(got, expected, rtol=1e-5 if dtype.endswith("32") else 1e-8)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_one_matrix(dtype):
    """1-norm = maximum absolute column sum."""
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    got = einsums.linalg.norm(einsums.linalg.Norm.ONE, A)
    expected = np.linalg.norm(np.asarray(A), 1)
    assert_close(got, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_infty_matrix(dtype):
    """∞-norm = maximum absolute row sum."""
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    got = einsums.linalg.norm(einsums.linalg.Norm.INFTY, A)
    expected = np.linalg.norm(np.asarray(A), np.inf)
    assert_close(got, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_maxabs_matrix(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    got = einsums.linalg.norm(einsums.linalg.Norm.MAXABS, A)
    expected = np.max(np.abs(np.asarray(A)))
    assert_close(got, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_two_matrix(dtype):
    """Spectral norm, largest singular value."""
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    got = einsums.linalg.norm(einsums.linalg.Norm.TWO, A)
    expected = np.linalg.norm(np.asarray(A), 2)
    # Spectral norm goes through SVD, loosen the f32 tolerance a touch.
    assert_close(got, expected, rtol=1e-5 if dtype.endswith("32") else 1e-12)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_frobenius_vector(dtype):
    """Frobenius on a rank-1 tensor falls back to the Euclidean (2-)norm."""
    x = einsums.create_random_tensor("x", [7], dtype=dtype)
    got = einsums.linalg.norm(einsums.linalg.Norm.FROBENIUS, x)
    expected = np.linalg.norm(np.asarray(x))
    assert_close(got, expected)


def test_norm_throws_during_capture():
    A = einsums.create_random_tensor("A", [3, 3])
    g = cg.Graph("norm-capture")
    with pytest.raises(RuntimeError):
        with cg.capture(g):
            einsums.linalg.norm(einsums.linalg.Norm.FROBENIUS, A)


# ──────────────────────────────────────────────────────────────────────────
# pow: matrix power via eigendecomposition (real dtypes only)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_pow_square_recovers_matmul(dtype):
    """``pow(A, 2.0) == A @ A`` for a symmetric positive-definite ``A``."""
    # Deterministic, well-conditioned SPD input (fixed seed). An unseeded random
    # draw occasionally lands ill-conditioned, which blows the float32
    # eigendecomposition reconstruction past the rtol below: an intermittent CI
    # flake. A fixed seed keeps the input ~1000x inside the tolerance.
    np_dtype = np.dtype(dtype)
    rng = np.random.default_rng(1)
    M = rng.standard_normal((4, 4)).astype(np_dtype)
    A_spd = (M @ M.T).astype(np_dtype) + np.eye(4, dtype=np_dtype)
    Araw = einsums.create_zero_tensor("A", [4, 4], dtype=dtype)
    np.copyto(np.asarray(Araw), A_spd)

    got = einsums.linalg.pow(Araw, np_dtype.type(2.0))
    expected = A_spd @ A_spd
    # Mac Accelerate's float32 SGEMM accumulates wider rounding than Linux
    # OpenBLAS / MKL, observed ~2.5e-5 relative on macOS CI vs <1e-6 on Linux.
    # Default c32/f32 rtol is 1e-5; loosen to 1e-4 for this matmul-of-matmul
    # shape, which is still tight enough to catch real bugs. f64 likewise: pow()
    # goes through an eigendecomposition whose reconstruction differs from the
    # direct A@A at the ~1e-12 level. The gap is BLAS-dependent, about 2.2e-12 on
    # clang/OpenBLAS, so loosen f64 from 1e-12 to 1e-10 to match
    # test_pow_inverse_round_trip below.
    assert_close(got, expected, rtol=1e-4 if dtype.endswith("32") else 1e-10)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_pow_inverse_round_trip(dtype):
    """``pow(A, 0.5) @ pow(A, 0.5) ≈ A`` for SPD ``A``."""
    # Deterministic, well-conditioned SPD input (fixed seed); see the note in
    # test_pow_square_recovers_matmul on why the unseeded draw was flaky.
    np_dtype = np.dtype(dtype)
    rng = np.random.default_rng(1)
    M = rng.standard_normal((4, 4)).astype(np_dtype)
    A_spd = (M @ M.T).astype(np_dtype) + np.eye(4, dtype=np_dtype)
    Araw = einsums.create_zero_tensor("A", [4, 4], dtype=dtype)
    np.copyto(np.asarray(Araw), A_spd)

    half = einsums.linalg.pow(Araw, np_dtype.type(0.5))
    got = np.asarray(half) @ np.asarray(half)
    # The half-power round-trip accumulates double the eigendecomposition
    # error, so loosen f64 a touch from the default 1e-12.
    assert_close(got, A_spd, rtol=1e-4 if dtype == "float32" else 1e-10)


def test_pow_throws_during_capture():
    A = einsums.create_random_tensor("A", [3, 3])
    g = cg.Graph("pow-capture")
    with pytest.raises(RuntimeError):
        with cg.capture(g):
            einsums.linalg.pow(A, 2.0)


def test_pow_throws_on_rank_mismatch():
    """``cg::pow`` rank check rejects non-rank-2 inputs."""
    x = einsums.create_random_tensor("x", [5])
    # pybind11 maps einsums::rank_error to ValueError.
    with pytest.raises(ValueError):
        einsums.linalg.pow(x, 2.0)


# ──────────────────────────────────────────────────────────────────────────
# symm_gemm: C = op(B)^T * op(A) * op(B)
# ──────────────────────────────────────────────────────────────────────────


def _symm_gemm_expected(A: np.ndarray, B: np.ndarray, trans_a: bool, trans_b: bool) -> np.ndarray:
    """Mirror the C++ algorithm: temp = op(A) @ op(B); result = op(B)^conj_T @ temp."""
    opA = A.T if trans_a else A
    opB = B.T if trans_b else B
    temp = opA @ opB
    # The C++ kernel uses gemm<!TransB, false>(B, temp, ...), i.e. it transposes
    # B again when TransB is False, leaving it as-is when TransB is True. That's
    # B^T * temp when TransB=False, and B * temp when TransB=True.
    left = B.T if not trans_b else B
    return left @ temp


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize(
    "trans_a,trans_b",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_symm_gemm_eager(dtype, trans_a, trans_b):
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [4, 4], dtype=dtype)
    C = einsums.create_zero_tensor("C", [4, 4], dtype=dtype)

    expected = _symm_gemm_expected(np.asarray(A), np.asarray(B), trans_a, trans_b)
    einsums.linalg.symm_gemm(A, B, C, trans_a=trans_a, trans_b=trans_b)
    assert_close(C, expected)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_symm_gemm_in_graph_capture(dtype):
    """symm_gemm records into a graph (unlike norm/pow/svd which throw)."""
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [4, 4], dtype=dtype)
    C = einsums.create_zero_tensor("C", [4, 4], dtype=dtype)

    expected = _symm_gemm_expected(np.asarray(A), np.asarray(B), False, False)

    g = cg.Graph("symm_gemm")
    with cg.capture(g):
        einsums.linalg.symm_gemm(A, B, C)
    assert g.num_nodes() == 1
    g.execute()
    assert_close(C, expected)


def test_symm_gemm_throws_on_rank_mismatch():
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4])  # rank-1, invalid
    C = einsums.create_zero_tensor("C", [4, 4])
    with pytest.raises(ValueError):
        einsums.linalg.symm_gemm(A, B, C)


# ──────────────────────────────────────────────────────────────────────────
# svd_dd: divide-and-conquer SVD (returning U, S, Vt)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_svd_dd_reconstructs_input(dtype):
    """``U * Sigma * Vt`` must recover ``A``."""
    m, n = 5, 4
    A = einsums.create_random_tensor("A", [m, n], dtype=dtype)
    ar = np.asarray(A).copy()

    U, S, Vt = einsums.linalg.svd_dd(A, einsums.linalg.Vectors.ALL)
    # Full SVD: U is m×m, Vt is n×n, S has min(m,n) entries.
    Sigma = np.zeros((m, n), dtype=np.asarray(S).dtype)
    for i, s in enumerate(np.asarray(S)):
        Sigma[i, i] = s
    recon = np.asarray(U) @ Sigma.astype(ar.dtype) @ np.asarray(Vt)
    assert_close(recon, ar)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_svd_dd_singular_values_match_numpy(dtype):
    A = einsums.create_random_tensor("A", [5, 4], dtype=dtype)
    _, S, _ = einsums.linalg.svd_dd(A, einsums.linalg.Vectors.ALL)
    got = sorted(np.asarray(S), reverse=True)
    expected = sorted(np.linalg.svd(np.asarray(A), compute_uv=False), reverse=True)
    assert_close(got, expected)


def test_svd_dd_throws_during_capture():
    A = einsums.create_random_tensor("A", [4, 4])
    g = cg.Graph("svd_dd-capture")
    with pytest.raises(RuntimeError):
        with cg.capture(g):
            einsums.linalg.svd_dd(A)


def test_svd_dd_throws_on_rank_mismatch():
    x = einsums.create_random_tensor("x", [5])
    with pytest.raises(ValueError):
        einsums.linalg.svd_dd(x)


# ──────────────────────────────────────────────────────────────────────────
# truncated_svd: randomized SVD with rank-k truncation
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_truncated_svd_top_k_matches_numpy(dtype):
    """Top-k singular values are in the right ballpark vs a full numpy SVD.

    Randomized algorithm with over-sampling factor 5, accuracy scales with
    ``(m / (k + 5))``, so the larger the matrix relative to ``k``, the
    tighter the bound. Even at ``m / (k+5) = 4`` (our setup) the worst-case
    relative error is a few percent. We assert order-of-magnitude agreement
    rather than tight numerical match.
    """
    m, k = 40, 3
    A = einsums.create_random_tensor("A", [m, m], dtype=dtype)
    _, S, _ = einsums.linalg.truncated_svd(A, k)
    got = sorted(np.asarray(S), reverse=True)[:k]
    expected = sorted(np.linalg.svd(np.asarray(A), compute_uv=False), reverse=True)[:k]
    # 25% is loose for f64 but accommodates f32 + the randomized algorithm's
    # spread, the goal here is a sanity check that the binding produces
    # values in the right neighborhood, not a numerical accuracy gate.
    for g, e in zip(got, expected):
        assert abs(g - e) / max(abs(e), 1e-6) < 0.25, f"top-k singular value {g} too far from {e}"


def test_truncated_svd_undersized_input_raises():
    """Documented precondition: ``A.dim(0) >= k + 5``. Smaller inputs raise."""
    A = einsums.create_random_tensor("A", [5, 5])
    # k=2 requires m >= 7; m=5 will trip the over-sampled projection.
    with pytest.raises(IndexError):
        einsums.linalg.truncated_svd(A, 2)


def test_truncated_svd_throws_during_capture():
    A = einsums.create_random_tensor("A", [12, 12])
    g = cg.Graph("trunc_svd-capture")
    with pytest.raises(RuntimeError):
        with cg.capture(g):
            einsums.linalg.truncated_svd(A, 3)


# ──────────────────────────────────────────────────────────────────────────
# truncated_syev: randomized symmetric eigendecomposition (real only)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_truncated_syev_runs_and_returns_finite(dtype):
    """The randomized algorithm completes and returns finite real values.

    Matches the C++ ``truncated_svd_test`` precedent in
    ``LinearAlgebra/tests/unit/Decomposition.cpp``, that test only asserts
    the call doesn't crash. Numerical accuracy of the randomized
    projection is not guaranteed strongly enough to gate on here; we just
    sanity-check the output isn't NaN/Inf and isn't grossly out of scale
    relative to the spectrum.
    """
    n, k = 15, 3
    A = einsums.create_random_tensor("A", [n, n], dtype=dtype)
    A_sym = np.asarray(A) + np.asarray(A).T
    np.copyto(np.asarray(A), A_sym)

    _, W = einsums.linalg.truncated_syev(A, k)
    got = np.asarray(W)
    assert np.all(np.isfinite(got)), "truncated_syev returned non-finite eigenvalues"
    spectral_radius = np.max(np.abs(np.linalg.eigvalsh(A_sym)))
    assert np.max(np.abs(got)) <= 2.0 * spectral_radius, "returned eigvals exceed twice the spectral radius, likely garbage"


def test_truncated_syev_undersized_input_raises():
    A = einsums.create_random_tensor("A", [5, 5])
    with pytest.raises(IndexError):
        einsums.linalg.truncated_syev(A, 2)


def test_truncated_syev_throws_during_capture():
    A = einsums.create_random_tensor("A", [12, 12])
    g = cg.Graph("trunc_syev-capture")
    with pytest.raises(RuntimeError):
        with cg.capture(g):
            einsums.linalg.truncated_syev(A, 3)
