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

``test_hyp_tiled_across_a_loop`` captures the same kinds of op into a graph with a
loop, drawing a producer and a consumer of one tiled tensor on either side of the
loop boundary, and runs the result through TiledExpansion in the orders that meet
its cross-graph cases: alone, before Reorder or LoopInvariantHoisting, and inside
the default pipeline once or twice.
"""
from __future__ import annotations

import itertools
import os

# The graph draws below are meant to run under the per-pass graph check, which
# ctest sets for every test; a manual run gets it too.
os.environ.setdefault("EINSUMS_PASS_VERIFY", "1")

import numpy as np
from hypothesis import HealthCheck, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

from _fuzz_diff_common import _apply_one_pass

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
@settings(max_examples=sanitizer_examples(350), deadline=None,
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
        # absent-tile handling of a repeated-letter (diagonal) spec on
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


#: One-element tiles, 17 to an axis: a contraction over it is 17**3 tile combinations,
#: over TiledExpansion's default budget, so it stays the opaque tiled op while an
#: elementwise op over the same tensor still expands.
_OVER_BUDGET_AXIS = [1] * 17

_LOOP_ORDERS = ([], ["TiledExpansion"], ["TiledExpansion", "Reorder"], ["LoopInvariantHoisting", "TiledExpansion"],
                ["TiledExpansion", "LoopInvariantHoisting", "TiledExpansion"], ["default"], ["default", "default"])


@given(prod_side=st.sampled_from(["before", "body"]), cons_side=st.sampled_from(["body", "after"]),
       prod_op=st.sampled_from(["einsum", "axpy", "scale"]), cons_op=st.sampled_from(["einsum", "axpy", "permute"]),
       axis=_axis(), big=st.booleans(), deferred=st.booleans(), iters=st.integers(1, 3),
       order=st.sampled_from(_LOOP_ORDERS), dtype=st.sampled_from([np.float64, np.complex128, np.float32, np.complex64]),
       sparse=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(120), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_tiled_across_a_loop(prod_side, cons_side, prod_op, cons_op, axis, big, deferred, iters, order, dtype, sparse,
                                 seed):
    """A tiled tensor X written on one side of a loop boundary and read on the other.

    The producer of X runs before the loop or in its body, the consumer in the
    body or after the loop, and ``big`` puts every operand on a grid whose
    contractions are over TiledExpansion's node budget, so they stay opaque while
    the elementwise ops beside them expand. A deferred X starts with no tiles at
    pass time, which is what makes a consumer's tile prediction depend on seeing
    the producer. The body also scales an unrelated tensor, so the loop is never
    empty.
    """
    rng = np.random.default_rng(seed)
    cplx = dtype in (np.complex128, np.complex64)
    rt = 2e-3 if dtype in _DT32 else 1e-9
    at = 1e-3 if dtype in _DT32 else 1e-9
    ax = _OVER_BUDGET_AXIS if big else axis
    grid = [ax, ax]
    n = sum(ax)
    full = {(ti, tj) for ti in range(len(ax)) for tj in range(len(ax))}

    def dense(masked):
        ref = _rnd((n, n), cplx, rng)
        return _zero_absent(ref, grid, _mask(grid, sparse, rng) if masked else full).astype(dtype)

    a, f, y0, w0 = dense(True), dense(True), dense(False), dense(False)
    x0 = np.zeros((n, n), dtype=dtype) if deferred else dense(False)
    s_prod, s_cons = float(rng.choice([-1.5, 0.5, 2.0])), float(rng.choice([-1.0, 0.5, 1.5]))
    c_pf = 0.0 if deferred else float(rng.choice([0.0, 1.0, 0.5]))

    A = _make(dtype, grid, a, full)
    F = _make(dtype, grid, f, full)
    Y = _make(dtype, grid, y0, full)
    W = _make(dtype, grid, w0, full)
    g = cg.Graph(f"across_{next(_ctr)}")
    name = np.dtype(dtype).name
    if deferred:
        X = g.declare_zero_tiled_tensor(f"x{next(_ctr)}", grid, intermediate=True, dtype=name)
    else:
        X = _make(dtype, grid, x0, full)

    def produce():
        if prod_op == "einsum":
            einsums.einsum("ij <- ik ; kj", X, A, F, c_pf=c_pf, ab_pf=s_prod)
        elif prod_op == "axpy":
            einsums.linalg.axpy(s_prod, A, X)
        else:
            einsums.linalg.scale(s_prod, X)

    def consume():
        if cons_op == "einsum":
            einsums.einsum("ij <- ik ; kj", Y, X, F, c_pf=1.0, ab_pf=s_cons)
        elif cons_op == "axpy":
            einsums.linalg.axpy(s_cons, X, Y)
        else:
            einsums.permute("j,i <- i,j", Y, X, c_pf=0.5, a_pf=s_cons)

    def produce_np(x):
        if prod_op == "einsum":
            return c_pf * x + s_prod * (a @ f)
        if prod_op == "axpy":
            return x + s_prod * a
        return s_prod * x

    def consume_np(y, x):
        if cons_op == "einsum":
            return y + s_cons * (x @ f)
        if cons_op == "axpy":
            return y + s_cons * x
        return 0.5 * y + s_cons * x.T

    if prod_side == "before":
        with cg.capture(g):
            produce()
    body = g.add_loop("l", iters, lambda it: it < iters - 1)
    with cg.capture(body):
        if prod_side == "body":
            produce()
        if cons_side == "body":
            consume()
        einsums.linalg.scale(0.5, W)
    if cons_side == "after":
        with cg.capture(g):
            consume()

    x, y, w = x0.astype(np.complex128 if cplx else np.float64), y0.astype(np.complex128 if cplx else np.float64), w0
    if prod_side == "before":
        x = produce_np(x)
    for _ in range(iters):
        if prod_side == "body":
            x = produce_np(x)
        if cons_side == "body":
            y = consume_np(y, x)
        w = 0.5 * w
    if cons_side == "after":
        y = consume_np(y, x)

    for step in order:
        if step == "default":
            g.apply(cg.default_pass_manager())
        else:
            _apply_one_pass(g, step)
    _apply_one_pass(g, "Materialization")
    g.execute()

    scale = max(1.0, float(np.max(np.abs(y))))
    np.testing.assert_allclose(_gather(Y, n, n), y, rtol=rt, atol=at * scale,
                               err_msg=f"order={order} producer={prod_op}@{prod_side} consumer={cons_op}@{cons_side}")
    np.testing.assert_allclose(_gather(W, n, n), w, rtol=rt, atol=at)
