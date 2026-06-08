# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""numpy-style conjugation tensor methods: .conj() / .real / .imag / .H / abs().

These are Python ergonomic wrappers over the linalg conj/real/imag/abs primitives.
.conj() returns a fresh tensor (original unchanged); .real/.imag/.H are properties;
abs() is __abs__. Each is graph-aware (works inside cg.capture). Verified vs numpy
over real/complex dtypes, owning + view operands, and eager + captured execution.
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
    t = einsums.create_zero_tensor(f"tm{next(_ctr)}", list(a.shape), dtype=dt)
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


def _run(graph, fn):
    """Run fn() eagerly or inside a capture+execute, returning its tensor result."""
    if not graph:
        return fn()
    g = cg.Graph(f"tm{next(_ctr)}")
    with cg.capture(g):
        out = fn()
    g.execute()
    return out


@given(method=st.sampled_from(["conj", "real", "imag", "H", "abs"]),
       shape=st.lists(st.integers(1, 4), min_size=1, max_size=3), cplx=st.booleans(),
       view=st.booleans(), graph=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=400, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_tensor_conj_methods(method, shape, cplx, view, graph, seed):
    rng = np.random.default_rng(seed)
    shape = tuple(shape)
    dt = "complex128" if cplx else "float64"
    A0 = _rnd(shape, cplx, rng)

    if method == "conj":
        At = _mkv(A0, view, dt, rng)
        out = _run(graph, lambda: At.conj())
        np.testing.assert_allclose(np.asarray(out), A0.conj(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(At), A0, rtol=1e-12, atol=1e-14)  # original unchanged
    elif method == "real":
        At = _mkv(A0, view, dt, rng)
        out = _run(graph, lambda: At.real)
        np.testing.assert_allclose(np.asarray(out), A0.real, rtol=1e-7, atol=1e-9)
    elif method == "imag":
        At = _mkv(A0, view, dt, rng)
        out = _run(graph, lambda: At.imag)
        np.testing.assert_allclose(np.asarray(out), A0.imag, rtol=1e-7, atol=1e-9)
    elif method == "abs":
        At = _mkv(A0, view, dt, rng)
        out = _run(graph, lambda: abs(At))
        np.testing.assert_allclose(np.asarray(out), np.abs(A0), rtol=1e-7, atol=1e-9)
    else:  # H -> conjugate-transpose (reverse all axes)
        At = _mkv(A0, view, dt, rng)
        out = _run(graph, lambda: At.H)
        np.testing.assert_allclose(np.asarray(out), A0.conj().T, rtol=1e-12, atol=1e-14)
