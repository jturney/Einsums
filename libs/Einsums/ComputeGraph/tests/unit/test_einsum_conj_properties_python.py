# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Conjugation property tests (Tier 3): cross-path agreement + metamorphic identities.

Cross-path: the conjugate-transpose product A^H @ B can be expressed four ways:
linalg.gemm(trans_a=Transpose.C), einsum(conj_a=) with transposed indices, the
einsum conj(...) spec, and a materialized A.conj() then plain einsum. All must
agree (with each other and numpy), so a bug local to one path is caught.

Metamorphic: identities that hold regardless of numpy's conventions:
conj(conj(A))==A, (A^H)^H==A, (A@B)^H == B^H@A^H, dotc(a,b)==conj(dotc(b,a)).
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums

_ctr = itertools.count()
_DT = ["float64", "complex64", "complex128"]


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"ecq{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _rnd(shape, dt, rng):
    r = rng.standard_normal(shape)
    if dt.startswith("complex"):
        return r + 1j * rng.standard_normal(shape)
    return r


def _tol(dt):
    return (2e-3, 2e-4) if dt in ("float32", "complex64") else (1e-7, 1e-9)


@given(k=st.integers(1, 6), i=st.integers(1, 6), j=st.integers(1, 6),
       dt=st.sampled_from(_DT), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=400, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_conj_cross_path_agreement(k, i, j, dt, seed):
    """A^H @ B computed four ways must all agree."""
    rng = np.random.default_rng(seed)
    A = _rnd((k, i), dt, rng)  # stored (k, i); A^H is (i, k)
    B = _rnd((k, j), dt, rng)
    oracle = np.conj(A).T @ B
    rtol, atol = _tol(dt)
    T = einsums.linalg.Transpose

    # 1. gemm with conjugate-transpose flag
    C1 = _mk(np.zeros((i, j)), dt)
    einsums.linalg.gemm(1.0, _mk(A, dt), _mk(B, dt), 0.0, C1, trans_a=T.C)
    # 2. einsum, conj_a kwarg + transposed index placement
    C2 = _mk(np.zeros((i, j)), dt)
    einsums.einsum("ij <- ki ; kj", C2, _mk(A, dt), _mk(B, dt), conj_a=True)
    # 3. einsum, conj(...) spec notation
    C3 = _mk(np.zeros((i, j)), dt)
    einsums.einsum("ij <- conj(ki) ; kj", C3, _mk(A, dt), _mk(B, dt))
    # 4. compose: materialize conj(A), then plain einsum
    C4 = _mk(np.zeros((i, j)), dt)
    einsums.einsum("ij <- ki ; kj", C4, _mk(A, dt).conj(), _mk(B, dt))

    for name, C in [("gemm", C1), ("einsum_kwarg", C2), ("einsum_spec", C3), ("compose", C4)]:
        np.testing.assert_allclose(np.asarray(C), oracle, rtol=rtol, atol=atol,
            err_msg=f"path={name} k={k} i={i} j={j} dt={dt} seed={seed}")


@given(n=st.integers(1, 6), m=st.integers(1, 6), dt=st.sampled_from(_DT), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_conj_metamorphic(n, m, dt, seed):
    rng = np.random.default_rng(seed)
    rtol, atol = _tol(dt)
    A = _rnd((n, m), dt, rng)

    # conj(conj(A)) == A  and  (A^H)^H == A
    np.testing.assert_allclose(np.asarray(_mk(A, dt).conj().conj()), A, rtol=rtol, atol=atol,
        err_msg=f"double-conj n={n} m={m} dt={dt} seed={seed}")
    np.testing.assert_allclose(np.asarray(_mk(A, dt).H.H), A, rtol=rtol, atol=atol,
        err_msg=f"double-adjoint n={n} m={m} dt={dt} seed={seed}")

    # (A @ B)^H == B^H @ A^H
    B = _rnd((m, n), dt, rng)  # A:(n,m) B:(m,n) -> AB:(n,n)
    T = einsums.linalg.Transpose
    P = _mk(np.zeros((n, n)), dt)
    einsums.linalg.gemm(1.0, _mk(A, dt), _mk(B, dt), 0.0, P)         # P = A @ B
    lhs = np.asarray(P.H)                                            # (A@B)^H
    R = _mk(np.zeros((n, n)), dt)
    einsums.linalg.gemm(1.0, _mk(B, dt), _mk(A, dt), 0.0, R, trans_a=T.C, trans_b=T.C)  # B^H @ A^H
    np.testing.assert_allclose(lhs, np.asarray(R), rtol=rtol, atol=atol,
        err_msg=f"(AB)^H==B^H A^H n={n} m={m} dt={dt} seed={seed}")
    np.testing.assert_allclose(lhs, np.conj(A @ B).T, rtol=rtol, atol=atol)

    # dotc(a, b) == conj(dotc(b, a))   (Hermitian inner product)
    a = _rnd((max(n, 2),), dt, rng)
    b = _rnd((max(n, 2),), dt, rng)
    rab = _mk(np.zeros((1,)), dt)
    rba = _mk(np.zeros((1,)), dt)
    einsums.linalg.dotc(rab, _mk(a, dt), _mk(b, dt))  # sum conj(a) * b
    einsums.linalg.dotc(rba, _mk(b, dt), _mk(a, dt))  # sum conj(b) * a
    np.testing.assert_allclose(np.asarray(rab).ravel()[0], np.conj(np.asarray(rba).ravel()[0]), rtol=rtol, atol=atol,
        err_msg=f"dotc symmetry n={n} dt={dt} seed={seed}")
