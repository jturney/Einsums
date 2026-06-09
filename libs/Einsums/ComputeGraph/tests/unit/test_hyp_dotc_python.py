# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: dotc (Hermitian inner product) vs numpy.vdot.

dotc(A, B) = sum_i conj(A_i) * B_i -- the conjugating counterpart of dot
(sum_i A_i B_i). numpy.vdot flattens and conjugates its first argument, so it is
the oracle. Sweeps eager and graph execution, owning + permuted-view operands
(the view path exercises the conjugating stride-mismatched reduction), degenerate
extents, and real/complex dtypes.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"dc{next(_ctr)}", list(a.shape), dtype=dt)
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


@given(shape=st.lists(st.integers(1, 4), min_size=1, max_size=3), cplx=st.booleans(),
       va=st.booleans(), vb=st.booleans(), graph=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=400, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_dotc(shape, cplx, va, vb, graph, seed):
    rng = np.random.default_rng(seed)
    shape = tuple(shape)
    dt = "complex128" if cplx else "float64"
    rdt = "complex128" if cplx else "float64"
    A0 = _rnd(shape, cplx, rng)
    B0 = _rnd(shape, cplx, rng)
    oracle = np.vdot(A0, B0)  # conj(A) . B
    At = _mkv(A0, va, dt, rng)
    Bt = _mkv(B0, vb, dt, rng)
    res = einsums.create_zero_tensor(f"dc{next(_ctr)}", [1], dtype=rdt)
    if graph:
        g = cg.Graph(f"dc{next(_ctr)}")
        with cg.capture(g):
            einsums.linalg.dotc(res, At, Bt)
        g.execute()
    else:
        einsums.linalg.dotc(res, At, Bt)
    got = np.asarray(res).ravel()[0]
    np.testing.assert_allclose(got, oracle, rtol=1e-7, atol=1e-9,
        err_msg=f"shape={shape} cplx={cplx} va={va} vb={vb} graph={graph} s={seed}")
