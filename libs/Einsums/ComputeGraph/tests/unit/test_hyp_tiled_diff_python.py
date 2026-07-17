# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: TiledRuntimeTensor ops vs numpy (dense reconstruction).

Covers scale / axpy / dot / norm / trace / einsum-gemm over tiled operands with
random tile grids (including degenerate size-1 tiles and single-tile axes), real
and complex dtypes, and -- via the ``sparse`` flag -- randomly ABSENT tiles
(an absent tile == zeros). The dense oracle reconstructs the tiled tensor (absent
-> 0) and compares to numpy. Plus an explicit check that axpy into a tiled tensor
missing a tile auto-materializes it from the source.

This surface was clean when mined (3500 examples, 0 failures); the test guards it.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums

_TRT = {np.float64: einsums.TiledRuntimeTensorD, np.complex128: einsums.TiledRuntimeTensorZ,
        np.float32: einsums.TiledRuntimeTensorF, np.complex64: einsums.TiledRuntimeTensorC}
_RT = {np.float64: einsums.RuntimeTensorD, np.complex128: einsums.RuntimeTensorZ,
       np.float32: einsums.RuntimeTensorF, np.complex64: einsums.RuntimeTensorC}
_REAL_RT = {np.float64: einsums.RuntimeTensorD, np.complex128: einsums.RuntimeTensorD,
            np.float32: einsums.RuntimeTensorF, np.complex64: einsums.RuntimeTensorF}
_DT32 = (np.float32, np.complex64)
_ctr = itertools.count()


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


def _offsets(grid):
    return [list(np.cumsum([0] + g)[:-1]) for g in grid]


def _zero_absent(ref, grid, present):
    off = _offsets(grid)
    out = np.zeros_like(ref)
    for ti in range(len(grid[0])):
        for tj in range(len(grid[1])):
            if (ti, tj) in present:
                r0, c0 = off[0][ti], off[1][tj]
                out[r0:r0 + grid[0][ti], c0:c0 + grid[1][tj]] = ref[r0:r0 + grid[0][ti], c0:c0 + grid[1][tj]]
    return out


def _make(dtype, grid, ref, present):
    t = _TRT[np.dtype(dtype).type](f"t{next(_ctr)}", grid)
    off, sz = t.tile_offsets(), t.tile_sizes()
    for (ti, tj) in present:
        t.add_tile([ti, tj])
    t.materialize()
    for (ti, tj) in present:
        a = np.asarray(t.tile_view([ti, tj]))
        r0, c0 = off[0][ti], off[1][tj]
        a[...] = ref[r0:r0 + sz[0][ti], c0:c0 + sz[1][tj]]
    return t


def _gather(t, R, C):
    off, sz = t.tile_offsets(), t.tile_sizes()
    dt = None
    for ti in range(len(sz[0])):
        for tj in range(len(sz[1])):
            if t.has_tile([ti, tj]):
                dt = np.asarray(t.tile_view([ti, tj])).dtype
                break
        if dt is not None:
            break
    M = np.zeros((R, C), dtype=dt or np.float64)
    for ti in range(len(sz[0])):
        for tj in range(len(sz[1])):
            if t.has_tile([ti, tj]):
                r0, c0 = off[0][ti], off[1][tj]
                M[r0:r0 + sz[0][ti], c0:c0 + sz[1][tj]] = np.asarray(t.tile_view([ti, tj]))
    return M


@st.composite
def _axis(draw):
    return [draw(st.integers(1, 3)) for _ in range(draw(st.integers(1, 3)))]


def _mask(grid, sparse, rng):
    full = {(ti, tj) for ti in range(len(grid[0])) for tj in range(len(grid[1]))}
    if not sparse:
        return full
    p = {tj for tj in full if rng.random() < 0.65}
    p.add((0, 0))
    return p


@given(op=st.sampled_from(["scale", "axpy", "dot", "norm", "trace", "einsum", "einsum_diag"]),
       rows=_axis(), cols=_axis(), kk=_axis(),
       dtype=st.sampled_from([np.float64, np.complex128, np.float32, np.complex64]),
       sparse=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=350, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_tiled_diff(op, rows, cols, kk, dtype, sparse, seed):
    rng = np.random.default_rng(seed)
    dt = dtype
    cplx = dtype in (np.complex128, np.complex64)
    # 32-bit dtypes carry looser tolerances.
    rt = 2e-3 if dtype in _DT32 else 1e-6
    at = 1e-4 if dtype in _DT32 else 1e-8
    R, C, K = sum(rows), sum(cols), sum(kk)
    if op == "scale":
        g = [rows, cols]; pm = _mask(g, sparse, rng); ref = _zero_absent(_rnd((R, C), cplx, rng), g, pm)
        A = _make(dt, g, ref, pm); einsums.linalg.scale(2.0, A)
        np.testing.assert_allclose(_gather(A, R, C), 2.0 * ref, rtol=rt, atol=at)
    elif op == "axpy":
        g = [rows, cols]; pm = _mask(g, sparse, rng)
        xr = _zero_absent(_rnd((R, C), cplx, rng), g, pm); yr = _zero_absent(_rnd((R, C), cplx, rng), g, pm)
        X = _make(dt, g, xr, pm); Y = _make(dt, g, yr, pm); einsums.linalg.axpy(1.5, X, Y)
        np.testing.assert_allclose(_gather(Y, R, C), yr + 1.5 * xr, rtol=rt, atol=at)
    elif op == "dot":
        g = [rows, cols]; pmx = _mask(g, sparse, rng); pmy = _mask(g, sparse, rng)
        xr = _zero_absent(_rnd((R, C), cplx, rng), g, pmx); yr = _zero_absent(_rnd((R, C), cplx, rng), g, pmy)
        X = _make(dt, g, xr, pmx); Y = _make(dt, g, yr, pmy); res = _RT[dt]("r", [1]); einsums.linalg.dot(res, X, Y)
        np.testing.assert_allclose(np.asarray(res).ravel()[0], np.sum(xr * yr), rtol=rt, atol=at)
    elif op == "norm":
        g = [rows, cols]; pm = _mask(g, sparse, rng); xr = _zero_absent(_rnd((R, C), cplx, rng), g, pm)
        X = _make(dt, g, xr, pm); res = _REAL_RT[dt]("r", [1]); einsums.linalg.norm(res, einsums.linalg.Norm.FROBENIUS, X)
        np.testing.assert_allclose(np.asarray(res).ravel()[0], np.linalg.norm(xr.ravel()), rtol=rt, atol=at)
    elif op == "trace":
        g = [rows, rows]; pm = _mask(g, sparse, rng); xr = _zero_absent(_rnd((R, R), cplx, rng), g, pm)
        X = _make(dt, g, xr, pm); res = _RT[dt]("r", [1]); einsums.linalg.trace(res, X)
        np.testing.assert_allclose(np.asarray(res).ravel()[0], np.trace(xr), rtol=rt, atol=at)
    elif op == "einsum":  # einsum gemm with (possibly) absent input tiles
        ga = [rows, kk]; gb = [kk, cols]; pma = _mask(ga, sparse, rng); pmb = _mask(gb, sparse, rng)
        ar = _zero_absent(_rnd((R, K), cplx, rng), ga, pma).astype(dt)
        br = _zero_absent(_rnd((K, C), cplx, rng), gb, pmb).astype(dt)
        A = _make(dt, ga, ar, pma); B = _make(dt, gb, br, pmb)
        Cc = _TRT[np.dtype(dt).type]("C", [rows, cols]); einsums.einsum("ij <- ik ; kj", Cc, A, B)
        np.testing.assert_allclose(_gather(Cc, R, C), ar @ br, rtol=rt, atol=at)
    else:  # einsum_diag: repeated-letter diagonal outer, C(i,j) = A(i,i)*B(j,j)
        # Diagonal elements live in the diagonal TILES; sparse masks exercise
        # absent-tile handling of a repeated-letter (bug-1023 class) spec on
        # the tiled dispatch path.
        ga = [rows, rows]; gb = [cols, cols]; pma = _mask(ga, sparse, rng); pmb = _mask(gb, sparse, rng)
        ar = _zero_absent(_rnd((R, R), cplx, rng), ga, pma).astype(dt)
        br = _zero_absent(_rnd((C, C), cplx, rng), gb, pmb).astype(dt)
        A = _make(dt, ga, ar, pma); B = _make(dt, gb, br, pmb)
        Cc = _TRT[np.dtype(dt).type]("C", [rows, cols]); einsums.einsum("ij <- ii ; jj", Cc, A, B)
        np.testing.assert_allclose(_gather(Cc, R, C), np.diag(ar)[:, None] * np.diag(br)[None, :], rtol=rt, atol=at)


def test_tiled_axpy_materializes_missing_tile():
    """axpy into a tiled tensor missing a tile creates it from the source."""
    g = [[2, 2], [2, 2]]
    rng = np.random.default_rng(0)
    xr = rng.standard_normal((4, 4))
    yr = rng.standard_normal((4, 4)); yr[0:2, 2:4] = 0.0  # Y absent at tile (0,1)
    X = _make(np.float64, g, xr, {(0, 0), (0, 1), (1, 0), (1, 1)})
    Y = _make(np.float64, g, yr, {(0, 0), (1, 0), (1, 1)})
    einsums.linalg.axpy(1.0, X, Y)
    np.testing.assert_allclose(_gather(Y, 4, 4), yr + xr, rtol=1e-9, atol=1e-12)
