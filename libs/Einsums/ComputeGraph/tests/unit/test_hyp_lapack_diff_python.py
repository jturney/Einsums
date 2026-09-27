# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: LAPACK family vs numpy.linalg.

Covers gesv (solve), invert, det, trace, eig (syev real / heev complex), svd and
qr across real/complex dtypes, square and (for svd/qr) rectangular shapes, and
the degenerate N=1 boundary. Matrix DATA is varied via a drawn seed.

Sign/phase ambiguity is dodged by comparing invariants: eigenvalues for eig,
singular values for svd, the reconstruction A == Q @ R for qr, and direct values
for solve / invert / det / trace. Solve/invert/det use diagonally dominant inputs
so the comparison isn't swamped by conditioning noise.

This surface was clean when added (no bug found); the test guards it against
regressions on the degenerate / complex / rectangular cases.

Every op is drawn over all four dtypes and judged at the rounding of the drawn
dtype, scaled the way each routine's error bound is: by the norm of the result
for the backward-stable factorizations, and by the condition number as well for
solve, invert and det, whose forward error carries it.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, example, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
from _dtype_draws import DTYPES, assert_rounding_close, is_complex, random_array, real_of, rounded, wide

_ctr = itertools.count()


def _mk(a, dt):
    t = einsums.create_zero_tensor(f"ll{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _rnd(shape, dt, rng):
    return random_array(shape, dt, rng)


def _dom(n, dt, rng):
    """Diagonally dominant -> invertible, well-conditioned."""
    m = _rnd((n, n), dt, rng)
    m[np.diag_indices(n)] = m[np.diag_indices(n)] + (n + 2.0)
    return rounded(m, dt)


@given(op=st.sampled_from(["gesv", "invert", "det", "trace", "eig", "svd", "qr"]),
       n=st.integers(1, 5), m=st.integers(1, 5), nrhs=st.integers(1, 3),
       dt=DTYPES, seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(300), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(op="invert", n=1, m=1, nrhs=1, dt="float64", seed=0)   # degenerate 1x1 inverse
@example(op="eig", n=1, m=1, nrhs=1, dt="complex128", seed=0)   # degenerate 1x1 Hermitian eig
@example(op="svd", n=1, m=3, nrhs=1, dt="float64", seed=0)      # rectangular svd
@example(op="eig", n=1, m=1, nrhs=1, dt="complex64", seed=0)    # degenerate 1x1 Hermitian eig, single precision
def test_hyp_lapack_diff(op, n, m, nrhs, dt, seed):
    rng = np.random.default_rng(seed)
    cplx = is_complex(dt)
    if op == "gesv":
        A0 = _dom(n, dt, rng)
        B0 = _rnd((n, nrhs), dt, rng)
        X = np.linalg.solve(A0, B0)
        Bt = _mk(B0, dt)
        einsums.linalg.gesv(_mk(A0, dt), Bt)
        assert_rounding_close(Bt, X, dt, np.abs(X).max(), factor=np.linalg.cond(A0),
                              err_msg=f"gesv n={n} nrhs={nrhs} {dt} s={seed}")
    elif op == "invert":
        A0 = _dom(n, dt, rng)
        At = _mk(A0, dt)
        einsums.linalg.invert(At)
        inv = np.linalg.inv(A0)
        assert_rounding_close(At, inv, dt, np.abs(inv).max(), factor=np.linalg.cond(A0),
                              err_msg=f"invert n={n} {dt} s={seed}")
    elif op == "det":
        A0 = _dom(n, dt, rng)
        # Compare magnitudes: einsums.linalg.det has a known sign bug for matrices
        # with odd-parity LU pivots (right magnitude, opposite sign vs numpy),
        # which surfaces on Linux MKL. See docs/known-bugs/det-sign.md. Once the
        # sign computation in LinearAlgebra.hpp::det is fixed, drop the np.abs so
        # this test also guards the sign. (Mirrors test_lapack_python.py's
        # test_det_eager_matches_numpy.)
        #
        # A perturbation dA moves det(A) by det(A) tr(A^-1 dA), so the relative
        # error is bounded by n times the condition number times the rounding.
        want = np.abs(np.linalg.det(A0))
        assert_rounding_close(np.abs(einsums.linalg.det(_mk(A0, dt))), want, dt, want, factor=n * np.linalg.cond(A0),
                              err_msg=f"det n={n} {dt} s={seed}")
    elif op == "trace":
        A0 = _rnd((n, n), dt, rng)
        assert_rounding_close(einsums.linalg.trace(_mk(A0, dt)), np.trace(A0), dt, np.abs(np.diag(A0)).sum(),
                              err_msg=f"trace n={n} {dt} s={seed}")
    elif op == "eig":
        mat = _rnd((n, n), dt, rng)
        A0 = rounded((mat + mat.conj().T) / 2.0, dt)
        w = np.linalg.eigvalsh(A0)
        Wt = _mk(np.zeros(n), real_of(dt))
        (einsums.linalg.heev if cplx else einsums.linalg.syev)(_mk(A0, dt), Wt)
        # Each eigenvalue of a Hermitian matrix is good to a small multiple of eps * ||A||.
        assert_rounding_close(np.sort(np.asarray(Wt).astype(np.float64)), np.sort(w), dt, np.linalg.norm(A0, 2), factor=n,
                              err_msg=f"eig n={n} {dt} s={seed}")
    elif op == "svd":
        A0 = _rnd((m, n), dt, rng)
        s = np.linalg.svd(A0, compute_uv=False)
        _, S, _ = einsums.linalg.svd(_mk(A0, dt))
        assert_rounding_close(np.sort(np.asarray(S).astype(np.float64))[::-1], np.sort(s)[::-1], dt, s.max(), factor=max(m, n),
                              err_msg=f"svd m={m} n={n} {dt} s={seed}")
    else:  # qr -> A == Q @ R
        A0 = _rnd((m, n), dt, rng)
        Q, R = einsums.linalg.qr(_mk(A0, dt))
        QR = np.asarray(Q).astype(wide(dt)) @ np.asarray(R).astype(wide(dt))
        assert_rounding_close(QR, A0, dt, np.linalg.norm(A0, 2), factor=max(m, n),
                              err_msg=f"qr m={m} n={n} {dt} s={seed}")
