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
