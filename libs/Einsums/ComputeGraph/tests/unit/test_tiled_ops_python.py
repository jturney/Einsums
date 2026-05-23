# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for ComputeGraph ops over TiledRuntimeTensor operands.

The tiled type is filled per tile (tile_view -> numpy); these tests then run the
graph ops (scale / axpy / einsum) over the whole tiled tensor and compare a
gathered dense view against a numpy reference.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import einsums
from einsums.testing import ALL_DTYPES, assert_close

DTYPE_TO_TRT = {
    np.float32: einsums.TiledRuntimeTensorF,
    np.float64: einsums.TiledRuntimeTensorD,
    np.complex64: einsums.TiledRuntimeTensorC,
    np.complex128: einsums.TiledRuntimeTensorZ,
}

# 1-element dense result tensors for scalar reductions: same dtype for
# dot/trace, the real type for norm.
_RT = {
    np.float32: einsums.RuntimeTensorF,
    np.float64: einsums.RuntimeTensorD,
    np.complex64: einsums.RuntimeTensorC,
    np.complex128: einsums.RuntimeTensorZ,
}
_REAL_RT = {
    np.float32: einsums.RuntimeTensorF,
    np.float64: einsums.RuntimeTensorD,
    np.complex64: einsums.RuntimeTensorF,
    np.complex128: einsums.RuntimeTensorD,
}


def _make(dtype, name, grid, fill=None):
    """Build a tiled tensor over ``grid`` (list per axis), populate every tile,
    and optionally fill from a global (row, col) -> value function."""
    t = DTYPE_TO_TRT[np.dtype(dtype).type](name, grid)
    off, sz = t.tile_offsets(), t.tile_sizes()
    for ti in range(len(sz[0])):
        for tj in range(len(sz[1])):
            t.add_tile([ti, tj])
    t.materialize()
    if fill is not None:
        for ti in range(len(sz[0])):
            for tj in range(len(sz[1])):
                a = np.asarray(t.tile_view([ti, tj]))
                for lr in range(sz[0][ti]):
                    for lc in range(sz[1][tj]):
                        a[lr, lc] = fill(off[0][ti] + lr, off[1][tj] + lc)
    return t


def _gather(t, R, C):
    """Reconstruct a dense R x C array from a 2-D tiled tensor (absent -> 0)."""
    off, sz = t.tile_offsets(), t.tile_sizes()
    M = np.zeros((R, C), dtype=np.asarray(t.tile_view([0, 0])).dtype)
    for ti in range(len(sz[0])):
        for tj in range(len(sz[1])):
            if t.has_tile([ti, tj]):
                r0, c0 = off[0][ti], off[1][tj]
                M[r0 : r0 + sz[0][ti], c0 : c0 + sz[1][tj]] = np.asarray(t.tile_view([ti, tj]))
    return M


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_scale(dtype):
    ref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    t = _make(dtype, "A", [[2, 3], [4, 5]], fill=lambda r, c: ref[r, c])
    einsums.linalg.scale(2.0, t)
    assert_close(_gather(t, 5, 9), 2.0 * ref)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_axpy(dtype):
    xref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    yref = (3.0 - np.arange(45, dtype=dtype)).reshape(5, 9)
    X = _make(dtype, "X", [[2, 3], [4, 5]], fill=lambda r, c: xref[r, c])
    Y = _make(dtype, "Y", [[2, 3], [4, 5]], fill=lambda r, c: yref[r, c])
    einsums.linalg.axpy(1.5, X, Y)
    assert_close(_gather(Y, 5, 9), yref + 1.5 * xref)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_einsum_gemm(dtype):
    # C[i,j] = sum_k A[i,k] B[k,j]; contracted k partition {4,5} matches in A and B.
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    bref = (2.0 - np.arange(63, dtype=dtype)).reshape(9, 7)
    A = _make(dtype, "A", [[2, 3], [4, 5]], fill=lambda r, c: aref[r, c])
    B = _make(dtype, "B", [[4, 5], [3, 4]], fill=lambda r, c: bref[r, c])
    C = DTYPE_TO_TRT[np.dtype(dtype).type]("C", [[2, 3], [3, 4]])  # empty: infer-and-create

    einsums.einsum("ij <- ik ; kj", C, A, B)

    assert C.num_filled_tiles() == 4
    assert_close(_gather(C, 5, 7), aref @ bref)


# ── Rank-3 / rank-4 validation (the engine is rank-generic; CC contractions
#    are rank-3/4-dominated) ─────────────────────────────────────────────────


def _make_nd(dtype, name, grid, ref=None):
    """Build an N-D tiled tensor over ``grid`` (one tile-size list per axis),
    populate every tile, and optionally fill from a dense reference array."""
    t = DTYPE_TO_TRT[np.dtype(dtype).type](name, grid)
    sizes, offs = t.tile_sizes(), t.tile_offsets()
    counts = [range(len(s)) for s in sizes]
    for coord in itertools.product(*counts):
        t.add_tile(list(coord))
    t.materialize()
    if ref is not None:
        for coord in itertools.product(*counts):
            slc = tuple(slice(offs[ax][coord[ax]], offs[ax][coord[ax]] + sizes[ax][coord[ax]]) for ax in range(len(sizes)))
            np.asarray(t.tile_view(list(coord)))[...] = ref[slc]
    return t


def _gather_nd(t, shape, dtype):
    sizes, offs = t.tile_sizes(), t.tile_offsets()
    M = np.zeros(shape, dtype=dtype)
    for coord in itertools.product(*[range(len(s)) for s in sizes]):
        if t.has_tile(list(coord)):
            slc = tuple(slice(offs[ax][coord[ax]], offs[ax][coord[ax]] + sizes[ax][coord[ax]]) for ax in range(len(sizes)))
            M[slc] = np.asarray(t.tile_view(list(coord)))
    return M


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_einsum_rank3(dtype):
    # C[i,j,k] = sum_l A[i,j,l] B[l,k]; contracted l partition {3,4} aligns (A axis 2, B axis 0).
    ipart, jpart, lpart, kpart = [2, 3], [2], [3, 4], [2, 3]
    aref = (1.0 + np.arange(5 * 2 * 7, dtype=dtype)).reshape(5, 2, 7)
    bref = (2.0 - np.arange(7 * 5, dtype=dtype)).reshape(7, 5)
    A = _make_nd(dtype, "A", [ipart, jpart, lpart], aref)
    B = _make_nd(dtype, "B", [lpart, kpart], bref)
    C = DTYPE_TO_TRT[np.dtype(dtype).type]("C", [ipart, jpart, kpart])  # empty: infer-and-create

    einsums.einsum("ijk <- ijl ; lk", C, A, B)

    assert_close(_gather_nd(C, (5, 2, 5), dtype), np.einsum("ijl,lk->ijk", aref, bref))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_einsum_rank4_two_contractions(dtype):
    # C[i,j,a,b] = sum_{c,d} A[i,j,c,d] B[c,d,a,b]  (CCSD-like; contract c,d).
    ipart, jpart, cpart, dpart, apart, bpart = [2, 1], [2], [2, 1], [3], [1, 2], [2]
    aref = (1.0 + np.arange(3 * 2 * 3 * 3, dtype=dtype)).reshape(3, 2, 3, 3)
    bref = (0.5 - np.arange(3 * 3 * 3 * 2, dtype=dtype)).reshape(3, 3, 3, 2)
    A = _make_nd(dtype, "A", [ipart, jpart, cpart, dpart], aref)
    B = _make_nd(dtype, "B", [cpart, dpart, apart, bpart], bref)
    C = DTYPE_TO_TRT[np.dtype(dtype).type]("C", [ipart, jpart, apart, bpart])

    einsums.einsum("ijab <- ijcd ; cdab", C, A, B)

    assert_close(_gather_nd(C, (3, 2, 3, 2), dtype), np.einsum("ijcd,cdab->ijab", aref, bref))


# ── Scalar reductions (dot / norm / trace) ──────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_dot(dtype):
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    bref = (2.0 - np.arange(45, dtype=dtype)).reshape(5, 9)
    A = _make_nd(dtype, "A", [[2, 3], [4, 5]], aref)
    B = _make_nd(dtype, "B", [[2, 3], [4, 5]], bref)
    r = _RT[np.dtype(dtype).type]("r", [1])
    einsums.linalg.dot(r, A, B)
    # cg::dot is non-conjugated (sum A*B), matching np.sum(A*B).
    assert_close(np.asarray(r)[0], np.sum(aref * bref))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_norm_frobenius(dtype):
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    A = _make_nd(dtype, "A", [[2, 3], [4, 5]], aref)
    r = _REAL_RT[np.dtype(dtype).type]("r", [1])
    einsums.linalg.norm(r, einsums.linalg.Norm.FROBENIUS, A)
    assert_close(np.asarray(r)[0], np.linalg.norm(aref.ravel()))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_trace(dtype):
    # Square, with matching row/col tile partition so diagonal tiles are square.
    sref = (1.0 + np.arange(25, dtype=dtype)).reshape(5, 5)
    S = _make_nd(dtype, "S", [[2, 3], [2, 3]], sref)
    r = _RT[np.dtype(dtype).type]("r", [1])
    einsums.linalg.trace(r, S)
    assert_close(np.asarray(r)[0], np.trace(sref))
