# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: symm_gemm (C = op(B)^T op(A) op(B)) vs numpy.

op(A) = A or A^T (square m x m); op(B) = B or B^T (m x q); C is q x q. Sweeps the
trans_a/trans_b flags, view operands for A/B/C, degenerate (size-1) and larger
sizes, and real/complex dtypes. The product uses a plain transpose (B^T, not B^H)
for complex, matching the op() convention. numpy is the oracle.

Complements the fixed-case test_symm_gemm_views_python.py with randomized
coverage; this surface was clean when mined (1500 examples, 0 failures).
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums

_ctr = itertools.count()


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"sg{next(_ctr)}", list(a.shape), dtype=dt)
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


_SZ = st.sampled_from([1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24])
_DT = st.sampled_from(["float64", "complex128"])


@given(m=_SZ, q=_SZ, ta=st.booleans(), tb=st.booleans(), dt=_DT, conj=st.booleans(),
       va=st.booleans(), vb=st.booleans(), vc=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(350), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_symmgemm_diff(m, q, ta, tb, dt, conj, va, vb, vc, seed):
    rng = np.random.default_rng(seed)
    cplx = (dt == "complex128")
    A0 = _rnd((m, m), cplx, rng)
    B0 = _rnd((q, m) if tb else (m, q), cplx, rng)  # op(B) is m x q
    opA = A0.T if ta else A0
    opB = B0.T if tb else B0
    # conjugate=True is the Hermitian congruence op(B)^H op(A) op(B); default is op(B)^T ...
    outer = opB.conj().T if conj else opB.T
    oracle = outer @ opA @ opB
    At = _mkv(A0, va, dt, rng)
    Bt = _mkv(B0, vb, dt, rng)
    Ct = _mkv(np.zeros((q, q)), vc, dt, rng)
    einsums.linalg.symm_gemm(At, Bt, Ct, trans_a=ta, trans_b=tb, conjugate=conj)
    np.testing.assert_allclose(np.asarray(Ct), oracle, rtol=1e-6, atol=1e-7,
        err_msg=f"m={m} q={q} ta={ta} tb={tb} dt={dt} conj={conj} va={va} vb={vb} vc={vc} s={seed}")
