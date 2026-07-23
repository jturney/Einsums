# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: gemm with the runtime Transpose enum (N / T / C).

C = op(A) @ op(B), op in {N: X, T: X^T, C: X^H (conjugate-transpose)}. Sweeps all
9 (trans_a, trans_b) combinations, real/complex dtypes, owning + view operands,
and eager + graph execution, against numpy. The 'C' modes (conjugate-transpose,
new for complex via BLAS 'c') are the focus; 'T' must NOT conjugate.
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
    t = einsums.create_zero_tensor(f"gt{next(_ctr)}", list(a.shape), dtype=dt)
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


@given(ta=st.sampled_from("NTC"), tb=st.sampled_from("NTC"), m=_SZ, k=_SZ, n=_SZ,
       cplx=st.booleans(), va=st.booleans(), vb=st.booleans(), graph=st.booleans(),
       seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(500), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_gemm_transpose(ta, tb, m, k, n, cplx, va, vb, graph, seed):
    rng = np.random.default_rng(seed)
    dt = "complex128" if cplx else "float64"
    # op(A) is m x k, op(B) is k x n -> raw shapes depend on the mode.
    A0 = _rnd((k, m) if ta in "TC" else (m, k), cplx, rng)
    B0 = _rnd((n, k) if tb in "TC" else (k, n), cplx, rng)
    oracle = _op(A0, ta) @ _op(B0, tb)
    At = _mkv(A0, va, dt, rng)
    Bt = _mkv(B0, vb, dt, rng)
    C = _mk(np.zeros((m, n)), dt)
    if graph:
        g = cg.Graph(f"gt{next(_ctr)}")
        with cg.capture(g):
            einsums.linalg.gemm(1.0, At, Bt, 0.0, C, trans_a=_MODES[ta], trans_b=_MODES[tb])
        g.execute()
    else:
        einsums.linalg.gemm(1.0, At, Bt, 0.0, C, trans_a=_MODES[ta], trans_b=_MODES[tb])
    np.testing.assert_allclose(np.asarray(C), oracle, rtol=1e-6, atol=1e-7,
        err_msg=f"ta={ta} tb={tb} m={m} k={k} n={n} cplx={cplx} va={va} vb={vb} graph={graph} s={seed}")
