# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: conj / real / imag / abs vs numpy.

conj is in-place (no-op on real dtypes, like numpy.conj); real/imag extract parts
of a complex tensor into a real result; abs is the magnitude (real or complex ->
real). Sweeps eager and graph-captured execution, owning + permuted-view operands,
degenerate (size-1) extents, and real/complex dtypes.
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
    t = einsums.create_zero_tensor(f"cj{next(_ctr)}", list(a.shape), dtype=dt)
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


_SHAPE = st.lists(st.integers(1, 4), min_size=1, max_size=3)


@given(op=st.sampled_from(["conj", "real", "imag", "abs"]),
       shape=_SHAPE, cplx=st.booleans(), view=st.booleans(), graph=st.booleans(),
       seed=st.integers(0, 2**31 - 1))
@settings(max_examples=400, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_conjugation(op, shape, cplx, view, graph, seed):
    rng = np.random.default_rng(seed)
    shape = tuple(shape)
    # real/imag require a complex input; conj/abs accept either.
    if op in ("real", "imag"):
        cplx = True
    cdt = "complex128" if cplx else "float64"
    A0 = _rnd(shape, cplx, rng)

    if op == "conj":
        oracle = A0.conj()
        At = _mkv(A0, view, cdt, rng)
        if graph:
            g = cg.Graph(f"cj{next(_ctr)}")
            with cg.capture(g):
                einsums.linalg.conj(At)
            g.execute()
        else:
            einsums.linalg.conj(At)
        np.testing.assert_allclose(np.asarray(At), oracle, rtol=1e-12, atol=1e-14,
            err_msg=f"conj shape={shape} cplx={cplx} view={view} graph={graph} s={seed}")
        return

    # real / imag / abs: out-of-place, real result
    if op == "real":
        oracle = A0.real
    elif op == "imag":
        oracle = A0.imag
    else:
        oracle = np.abs(A0)
    At = _mkv(A0, view, cdt, rng)
    out = _mk(np.zeros(shape), "float64")
    fn = {"real": einsums.linalg.real, "imag": einsums.linalg.imag, "abs": einsums.linalg.abs}[op]
    if graph:
        g = cg.Graph(f"cj{next(_ctr)}")
        with cg.capture(g):
            fn(out, At)
        g.execute()
    else:
        fn(out, At)
    np.testing.assert_allclose(np.asarray(out), oracle, rtol=1e-6, atol=1e-8,
        err_msg=f"{op} shape={shape} cplx={cplx} view={view} graph={graph} s={seed}")
