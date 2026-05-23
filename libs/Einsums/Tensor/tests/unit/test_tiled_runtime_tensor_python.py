# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for TiledRuntimeTensor bindings produced by einsums-pybind.

A tiled tensor has no single contiguous buffer, so it does NOT expose the numpy
buffer protocol. Python instead fills/reads it per tile via ``tile_view()``,
which returns a numpy-backed ``RuntimeTensorView`` over one tile. These tests
cover construction, the grid/tile API, the per-tile numpy round-trip (including
that ``tile_view`` exposes live storage), and lifecycle (materialize/zero).
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums

DTYPE_TO_TRT = {
    np.float32: einsums.TiledRuntimeTensorF,
    np.float64: einsums.TiledRuntimeTensorD,
    np.complex64: einsums.TiledRuntimeTensorC,
    np.complex128: einsums.TiledRuntimeTensorZ,
}


@pytest.mark.parametrize("np_dtype, TRT", list(DTYPE_TO_TRT.items()))
def test_construct_and_metadata(np_dtype, TRT):
    # axis 0 tiled {2, 3} -> 5; axis 1 tiled {4, 5} -> 9.
    t = TRT("A", [[2, 3], [4, 5]])
    assert t.rank() == 2
    assert list(t.dims()) == [5, 9]
    assert t.grid_size() == 4
    assert t.num_filled_tiles() == 0
    assert t.name == "A"
    t.name = "renamed"
    assert t.name == "renamed"
    assert t.tile_sizes() == [[2, 3], [4, 5]]


@pytest.mark.parametrize("np_dtype, TRT", list(DTYPE_TO_TRT.items()))
def test_add_tiles_and_membership(np_dtype, TRT):
    t = TRT("A", [[2, 3], [4, 5]])
    t.add_tile([0, 0])  # 2 x 4 (diagonal)
    t.add_tile([0, 1])  # 2 x 5 (off-diagonal, rectangular -> non-symmetric case)
    t.add_tile([1, 1])  # 3 x 5
    assert t.num_filled_tiles() == 3
    assert t.has_tile([0, 0])
    assert t.has_tile([0, 1])
    assert not t.has_tile([1, 0])


@pytest.mark.parametrize("np_dtype, TRT", list(DTYPE_TO_TRT.items()))
def test_tile_view_numpy_round_trip(np_dtype, TRT):
    t = TRT("A", [[2, 3], [4, 5]])

    # Off-diagonal rectangular tile: shape comes from (axis0 tile 0)=2 x (axis1 tile 1)=5.
    v = t.tile_view([0, 1])
    arr = np.asarray(v)
    assert arr.shape == (2, 5)
    assert arr.dtype == np_dtype

    seed = np.arange(10, dtype=np_dtype).reshape(2, 5)
    arr[...] = seed
    # Re-fetching the view must see the write — tile_view exposes live storage.
    arr2 = np.asarray(t.tile_view([0, 1]))
    assert np.array_equal(arr2, seed)


@pytest.mark.parametrize("np_dtype, TRT", list(DTYPE_TO_TRT.items()))
def test_materialize_and_zero(np_dtype, TRT):
    t = TRT("A", [[2, 3], [4, 5]])
    t.add_tile([0, 0])
    t.add_tile([1, 1])
    t.materialize()

    np.asarray(t.tile_view([0, 0]))[...] = 7
    np.asarray(t.tile_view([1, 1]))[...] = -2
    t.zero()
    assert np.asarray(t.tile_view([0, 0])).sum() == 0
    assert np.asarray(t.tile_view([1, 1])).sum() == 0
