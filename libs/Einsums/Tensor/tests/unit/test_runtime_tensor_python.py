# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for RuntimeTensor bindings produced by einsums-pybind.

Exercises the protocols emitted from the Plan-C codegen path:
* numpy buffer protocol (zero-copy round-trip via np.asarray)
* iterator protocol (__iter__)
* index protocol (__getitem__ / __setitem__ with int, slice, ellipsis, tuple)
* dim / stride / size accessors
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums

DTYPE_TO_RTT = {
    np.float32: einsums.RuntimeTensorF,
    np.float64: einsums.RuntimeTensorD,
    np.complex64: einsums.RuntimeTensorC,
    np.complex128: einsums.RuntimeTensorZ,
}


@pytest.mark.parametrize("np_dtype, RTT", list(DTYPE_TO_RTT.items()))
def test_buffer_protocol_round_trip(np_dtype, RTT):
    t = RTT("buf", [3, 4])
    arr = np.asarray(t)
    assert arr.shape == (3, 4)
    assert arr.dtype == np_dtype

    seed = np.arange(12, dtype=np_dtype).reshape(3, 4)
    arr[...] = seed
    arr2 = np.asarray(t)
    assert np.array_equal(arr2, seed), "buffer protocol must expose live storage"


def test_iter_yields_each_element():
    # 1D tensor avoids column-vs-row-major ambiguity.
    t = einsums.RuntimeTensorD("iter", [6])
    np.asarray(t)[...] = np.arange(6, dtype=np.float64)
    assert list(t) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_getitem_int_returns_scalar():
    t = einsums.RuntimeTensorD("g", [4])
    np.asarray(t)[...] = [10.0, 20.0, 30.0, 40.0]
    assert t[0] == 10.0
    assert t[3] == 40.0


def test_getitem_slice_returns_view():
    t = einsums.RuntimeTensorD("g", [4, 4])
    np.asarray(t)[...] = np.arange(16, dtype=np.float64).reshape(4, 4)
    sub = t[1:3, :]
    assert np.array_equal(np.asarray(sub), np.arange(16).reshape(4, 4)[1:3, :])


def test_getitem_ellipsis_returns_full_view():
    t = einsums.RuntimeTensorD("g", [2, 2])
    np.asarray(t)[...] = [[1.0, 2.0], [3.0, 4.0]]
    full = t[...]
    assert np.array_equal(np.asarray(full), [[1.0, 2.0], [3.0, 4.0]])


def test_setitem_int_writes_scalar():
    t = einsums.RuntimeTensorD("s", [3])
    np.asarray(t)[...] = 0.0
    t[1] = 7.5
    assert t[1] == 7.5


def test_dim_size_rank_match_numpy():
    t = einsums.RuntimeTensorD("a", [5, 3, 2])
    arr = np.asarray(t)
    assert t.rank() == arr.ndim == 3
    assert [t.dim(i) for i in range(3)] == list(arr.shape)
    assert t.size == arr.size
