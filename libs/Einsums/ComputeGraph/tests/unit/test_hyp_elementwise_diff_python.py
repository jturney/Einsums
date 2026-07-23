# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: graph-captured element-wise family vs numpy.

Covers scale / axpy / axpby / direct_product / direct_division with
non-contiguous view operands, degenerate (size-1) dims, real/complex dtypes,
prefactors (incl. accumulation), and the optional default pass manager.

This is the TensorImpl vectorization surface (get_incx / is_totally_vectorable /
query_vectorable_params + the per-op layout guards). The ``@example`` entries
pin the reducers that were once miscompiled:
  * axpy into a (1,2,1) view (strides 2,1,2): a size-1 boundary dim's inflated
    stride was taken as the increment, so the op under-processed.
  * direct_product into a transposed (3,3,1) C view: the coarse is_column_major
    flag matched while the stride order didn't, pairing mismatched elements.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, example, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()


def _nm() -> str:
    return f"hew{next(_ctr)}"


def _mk(arr, dt):
    t = einsums.create_zero_tensor(_nm(), list(arr.shape), dtype=dt)
    if arr.size:
        np.asarray(t)[...] = arr
    return t


def _mkv(arr, use_view, dt, rng):
    if not use_view or arr.ndim < 2:
        return _mk(arr, dt)
    perm = list(rng.permutation(arr.ndim))
    if perm == list(range(arr.ndim)):
        perm = perm[::-1]
    t = _mk(np.ascontiguousarray(np.transpose(arr, perm)), dt)
    return t.permute_view(list(np.argsort(perm)))


def _rnd(shape, dt, rng):
    if dt == "complex128":
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


@st.composite
def _shape(draw):
    r = draw(st.integers(1, 3))
    return tuple(draw(st.integers(1, 4)) for _ in range(r))


@given(op=st.sampled_from(["scale", "axpy", "axpby", "direct_product", "direct_division"]),
       shape=_shape(), alpha=st.sampled_from([1.0, -2.0, 0.5]), beta=st.sampled_from([0.0, 1.0, -1.5]),
       dt=st.sampled_from(["float64", "complex128"]),
       va=st.booleans(), vb=st.booleans(), vc=st.booleans(), passes=st.booleans())
@settings(max_examples=sanitizer_examples(300), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(op="axpy", shape=(1, 2, 1), alpha=1.0, beta=0.0, dt="float64", va=False, vb=True, vc=False, passes=False)
@example(op="direct_product", shape=(3, 3, 1), alpha=1.0, beta=0.0, dt="complex128", va=False, vb=False, vc=True, passes=False)
def test_hyp_elementwise_diff(op, shape, alpha, beta, dt, va, vb, vc, passes):
    rng = np.random.default_rng(0)
    g = cg.Graph(_nm())
    if op == "scale":
        A0 = _rnd(shape, dt, rng)
        oracle = alpha * A0
        At = _mkv(A0, va, dt, rng)
        with cg.capture(g):
            einsums.linalg.scale(alpha, At)
        out = At
    elif op == "axpy":
        X0 = _rnd(shape, dt, rng)
        Y0 = _rnd(shape, dt, rng)
        oracle = Y0 + alpha * X0
        Xt, Yt = _mkv(X0, va, dt, rng), _mkv(Y0, vb, dt, rng)
        with cg.capture(g):
            einsums.linalg.axpy(alpha, Xt, Yt)
        out = Yt
    elif op == "axpby":
        X0 = _rnd(shape, dt, rng)
        Y0 = _rnd(shape, dt, rng)
        oracle = alpha * X0 + beta * Y0
        Xt, Yt = _mkv(X0, va, dt, rng), _mkv(Y0, vb, dt, rng)
        with cg.capture(g):
            einsums.linalg.axpby(alpha, Xt, beta, Yt)
        out = Yt
    elif op == "direct_product":
        A0 = _rnd(shape, dt, rng)
        B0 = _rnd(shape, dt, rng)
        C0 = _rnd(shape, dt, rng)
        oracle = alpha * A0 * B0 + beta * C0
        At, Bt, Ct = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng), _mkv(C0, vc, dt, rng)
        with cg.capture(g):
            einsums.linalg.direct_product(alpha, At, Bt, beta, Ct)
        out = Ct
    else:  # direct_division -- keep |B| >= ~1 to avoid blow-up
        A0 = _rnd(shape, dt, rng)
        B0 = _rnd(shape, dt, rng)
        B0 = B0 + 2 * np.sign(B0.real if dt == "complex128" else B0)
        C0 = _rnd(shape, dt, rng)
        oracle = alpha * A0 / B0 + beta * C0
        At, Bt, Ct = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng), _mkv(C0, vc, dt, rng)
        with cg.capture(g):
            einsums.linalg.direct_division(alpha, At, Bt, beta, Ct)
        out = Ct
    if passes:
        g.apply(cg.default_pass_manager())
    g.execute()
    np.testing.assert_allclose(
        np.asarray(out), oracle, rtol=1e-9, atol=1e-9,
        err_msg=f"op={op} shape={shape} alpha={alpha} beta={beta} dt={dt} "
                f"va={va} vb={vb} vc={vc} passes={passes}")
