# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: gemv Transpose enum (N/T/C) and gerc, vs numpy.

gemv: y = op(A) @ z, op in {N, T, C (conjugate-transpose)}. gerc: the conjugating
rank-1 update A += alpha * X * Y^H (the Hermitian counterpart of ger's X*Y^T).
Both swept over real/complex (gerc complex-only), owning + view operands, eager +
graph; 'C'/gerc are the new conjugating paths.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()
_T = einsums.linalg.Transpose
_MODES = {"N": _T.N, "T": _T.T, "C": _T.C}


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"gg{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _mkv(arr, use_view, dt, rng):
    if not use_view or arr.ndim < 2:
        return _mk(arr, dt)
    perm = list(rng.permutation(arr.ndim))
    if perm == list(range(arr.ndim)):
        perm = perm[::-1]
    return _mk(np.ascontiguousarray(np.transpose(arr, perm)), dt).permute_view(list(np.argsort(perm)))


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


def _op(X, mode):
    return {"N": X, "T": X.T, "C": X.conj().T}[mode]


_SZ = st.integers(1, 5)


@given(ta=st.sampled_from("NTC"), p=_SZ, q=_SZ, cplx=st.booleans(), va=st.booleans(),
       graph=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(400), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_gemv_transpose(ta, p, q, cplx, va, graph, seed):
    rng = np.random.default_rng(seed)
    dt = "complex128" if cplx else "float64"
    A0 = _rnd((p, q), cplx, rng)
    opA = _op(A0, ta)
    z0 = _rnd((opA.shape[1],), cplx, rng)
    oracle = opA @ z0
    At = _mkv(A0, va, dt, rng)
    zt = _mk(z0, dt)
    y = _mk(np.zeros((opA.shape[0],)), dt)
    if graph:
        g = cg.Graph(f"gg{next(_ctr)}")
        with cg.capture(g):
            einsums.linalg.gemv(1.0, At, zt, 0.0, y, trans_a=_MODES[ta])
        g.execute()
    else:
        einsums.linalg.gemv(1.0, At, zt, 0.0, y, trans_a=_MODES[ta])
    np.testing.assert_allclose(np.asarray(y), oracle, rtol=1e-6, atol=1e-7,
        err_msg=f"ta={ta} p={p} q={q} cplx={cplx} va={va} graph={graph} s={seed}")


@given(m=_SZ, n=_SZ, alpha_re=st.sampled_from([1.0, -2.0, 0.5]), graph=st.booleans(),
       seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(250), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_gerc(m, n, alpha_re, graph, seed):
    rng = np.random.default_rng(seed)
    alpha = alpha_re + 0.5j
    X0 = _rnd((m,), True, rng)
    Y0 = _rnd((n,), True, rng)
    A0 = _rnd((m, n), True, rng)
    oracle = A0 + alpha * np.outer(X0, Y0.conj())
    Xt, Yt, At = _mk(X0, "complex128"), _mk(Y0, "complex128"), _mk(A0, "complex128")
    if graph:
        g = cg.Graph(f"gg{next(_ctr)}")
        with cg.capture(g):
            einsums.linalg.gerc(alpha, Xt, Yt, At)
        g.execute()
    else:
        einsums.linalg.gerc(alpha, Xt, Yt, At)
    np.testing.assert_allclose(np.asarray(At), oracle, rtol=1e-7, atol=1e-9,
        err_msg=f"m={m} n={n} alpha={alpha} graph={graph} s={seed}")
