# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: graph-captured linalg gemm/gemv/ger vs numpy.

Exercises the OpKind::Gemm / Gemv / Ger executors with non-contiguous view
operands, degenerate (size-1) dimensions, all four dtypes, prefactors,
and transpose flags, with the default pass manager optionally applied. numpy
is the oracle.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from _dtype_draws import DTYPES, assert_rounding_close, random_array

_ctr = itertools.count()


def _nm() -> str:
    return f"hld{next(_ctr)}"


def _mkv(arr, use_view, dt, rng):
    if not use_view or arr.ndim < 2:
        return einsums.asarray(arr, dtype=dt)
    perm = list(rng.permutation(arr.ndim))
    if perm == list(range(arr.ndim)):
        perm = perm[::-1]
    t = einsums.asarray(np.ascontiguousarray(np.transpose(arr, perm)), dtype=dt)
    return t.permute_view(list(np.argsort(perm)))


def _rnd(shape, dt, rng):
    return random_array(shape, dt, rng)


_D = st.integers(1, 5)
_PF = st.sampled_from([0.0, 1.0, -2.0])
_DT = DTYPES
_BV = st.booleans()


@given(op=st.sampled_from(["gemm", "gemv", "ger"]), m=_D, n=_D, k=_D,
       alpha=st.sampled_from([1.0, -2.0, 0.5]), beta=_PF, dt=_DT,
       ta=_BV, tb=_BV, va=_BV, vb=_BV, passes=_BV)
@settings(max_examples=sanitizer_examples(300), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_linalg_diff(op, m, n, k, alpha, beta, dt, ta, tb, va, vb, passes):
    rng = np.random.default_rng(0)
    g = cg.Graph(_nm())
    if op == "gemm":
        A0 = _rnd((k, m) if ta else (m, k), dt, rng)
        B0 = _rnd((n, k) if tb else (k, n), dt, rng)
        C0 = _rnd((m, n), dt, rng)
        opA = A0.T if ta else A0
        opB = B0.T if tb else B0
        oracle = alpha * (opA @ opB) + beta * C0
        scale = abs(alpha) * (np.abs(opA) @ np.abs(opB)) + abs(beta) * np.abs(C0)
        At, Bt, Ct = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng), einsums.asarray(C0, dtype=dt)
        with cg.capture(g):
            einsums.linalg.gemm(alpha, At, Bt, beta, Ct, trans_a=ta, trans_b=tb)
        out = Ct
    elif op == "gemv":
        A0 = _rnd((k, m) if ta else (m, k), dt, rng)
        z0 = _rnd((k,), dt, rng)
        y0 = _rnd((m,), dt, rng)
        opA = A0.T if ta else A0
        oracle = alpha * (opA @ z0) + beta * y0
        scale = abs(alpha) * (np.abs(opA) @ np.abs(z0)) + abs(beta) * np.abs(y0)
        At, zt, yt = _mkv(A0, va, dt, rng), einsums.asarray(z0, dtype=dt), einsums.asarray(y0, dtype=dt)
        with cg.capture(g):
            einsums.linalg.gemv(alpha, At, zt, beta, yt, trans_a=ta)
        out = yt
    else:  # ger: A += alpha * X * Y^T
        X0 = _rnd((m,), dt, rng)
        Y0 = _rnd((n,), dt, rng)
        A0 = _rnd((m, n), dt, rng)
        oracle = A0 + alpha * np.outer(X0, Y0)
        scale = np.abs(A0) + abs(alpha) * np.outer(np.abs(X0), np.abs(Y0))
        Xt, Yt, At = einsums.asarray(X0, dtype=dt), einsums.asarray(Y0, dtype=dt), _mkv(A0, va, dt, rng)
        with cg.capture(g):
            einsums.linalg.ger(alpha, Xt, Yt, At)
        out = At
    if passes:
        g.apply(cg.default_pass_manager())
    g.execute()
    assert_rounding_close(
        out, oracle, dt, scale,
        err_msg=f"op={op} m={m} n={n} k={k} alpha={alpha} beta={beta} dt={dt} "
                f"ta={ta} tb={tb} va={va} vb={vb} passes={passes}")
