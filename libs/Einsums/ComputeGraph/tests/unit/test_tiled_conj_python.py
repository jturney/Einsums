# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Conjugation primitives on TiledRuntimeTensor: conj / real / imag / abs.

conj is in-place per stored tile (no-op for real); real/imag/abs map a complex
tiled tensor to a matching-grid real tiled tensor (abs also real->real). Verified
vs numpy via dense reconstruction, eager + graph, over random tile grids.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()
_TRT = {np.float64: einsums.TiledRuntimeTensorD, np.complex128: einsums.TiledRuntimeTensorZ}


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


def _make(dtype, grid, ref):
    t = _TRT[np.dtype(dtype).type](f"t{next(_ctr)}", grid)
    off, sz = t.tile_offsets(), t.tile_sizes()
    for i in range(len(sz[0])):
        for j in range(len(sz[1])):
            t.add_tile([i, j])
    t.materialize()
    for i in range(len(sz[0])):
        for j in range(len(sz[1])):
            a = np.asarray(t.tile_view([i, j]))
            r0, c0 = off[0][i], off[1][j]
            a[...] = ref[r0:r0 + sz[0][i], c0:c0 + sz[1][j]]
    return t


def _gather(t, R, C, dt):
    off, sz = t.tile_offsets(), t.tile_sizes()
    M = np.zeros((R, C), dtype=dt)
    for i in range(len(sz[0])):
        for j in range(len(sz[1])):
            if t.has_tile([i, j]):
                r0, c0 = off[0][i], off[1][j]
                M[r0:r0 + sz[0][i], c0:c0 + sz[1][j]] = np.asarray(t.tile_view([i, j]))
    return M


@st.composite
def _axis(draw):
    return [draw(st.integers(1, 3)) for _ in range(draw(st.integers(1, 3)))]


@given(op=st.sampled_from(["conj", "real", "imag", "abs"]), rows=_axis(), cols=_axis(),
       cplx=st.booleans(), graph=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_tiled_conj(op, rows, cols, cplx, graph, seed):
    rng = np.random.default_rng(seed)
    grid = [rows, cols]
    R, C = sum(rows), sum(cols)
    # Tiled real/imag are bound complex-input-only (extracting parts of a complex
    # tiled tensor); for a real tiled tensor real()==copy / imag()==0 is trivial.
    # conj/abs accept real or complex input.
    if op in ("real", "imag"):
        cplx = True
    dt = np.complex128 if cplx else np.float64
    ref = _rnd((R, C), cplx, rng)

    if op == "conj":
        A = _make(dt, grid, ref)
        if graph:
            g = cg.Graph(f"t{next(_ctr)}")
            with cg.capture(g):
                einsums.linalg.conj(A)
            g.execute()
        else:
            einsums.linalg.conj(A)
        np.testing.assert_allclose(_gather(A, R, C, dt), ref.conj(), rtol=1e-12, atol=1e-14)
        return

    # real/imag/abs: complex tiled in -> real tiled out (matching grid)
    A = _make(dt, grid, ref)
    out = _TRT[np.float64](f"t{next(_ctr)}", grid)
    fn = {"real": einsums.linalg.real, "imag": einsums.linalg.imag, "abs": einsums.linalg.abs}[op]
    oracle = {"real": ref.real, "imag": ref.imag, "abs": np.abs(ref)}[op]
    if graph:
        g = cg.Graph(f"t{next(_ctr)}")
        with cg.capture(g):
            fn(out, A)
        g.execute()
    else:
        fn(out, A)
    np.testing.assert_allclose(_gather(out, R, C, np.float64), oracle, rtol=1e-6, atol=1e-8)
