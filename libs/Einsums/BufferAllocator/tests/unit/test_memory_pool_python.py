# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for ``einsums.MemoryPool``.

The pool is where Python's per-tensor allocation cost is supposed to collapse:
one pybind call carves from pre-reserved memory instead of visiting the system
allocator. These tests check the surface and the semantics that make that safe -
that a carved tensor behaves like any other tensor, that dropping it returns its
bytes, and that an epoch refuses to close under a tensor Python still holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
from einsums.testing import ALL_DTYPES, assert_close

MIB = 1024 * 1024


def test_pool_reports_its_reservation():
    pool = einsums.MemoryPool(32 * MIB, "py")

    assert pool.name == "py"
    assert pool.bytes_reserved >= 32 * MIB
    assert pool.bytes_used == 0
    assert pool.high_water == 0
    assert pool.live_borrows == 0
    assert pool.arenas == 1
    assert pool.epoch_depth == 0
    assert pool.warn_bytes == 0

    pool.warn_bytes = 4 * MIB
    assert pool.warn_bytes == 4 * MIB


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_pool_empty_is_a_usable_tensor(dtype):
    pool = einsums.MemoryPool(32 * MIB, "empty")

    t = pool.empty((6, 5), dtype=dtype, name="A")

    assert t.shape == (6, 5)
    assert t.ndim == 2
    assert np.dtype(t.dtype) == np.dtype(dtype)
    assert pool.live_borrows == 1
    assert pool.bytes_used >= 30 * np.dtype(dtype).itemsize

    src = np.arange(30, dtype=dtype).reshape(6, 5)
    t[:, :] = src
    assert_close(t, src, dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_pool_zeros_is_zero(dtype):
    pool = einsums.MemoryPool(32 * MIB, "zeros")

    # Dirty the arena so a zeroed carve cannot pass by accident.
    dirty = pool.empty((64, 64), dtype=dtype)
    dirty[:, :] = np.full((64, 64), 3.5, dtype=dtype)
    del dirty

    z = pool.zeros((64, 64), dtype=dtype)
    assert_close(z, np.zeros((64, 64), dtype=dtype), dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_pooled_tensors_compute_like_owned_ones(dtype):
    pool = einsums.MemoryPool(32 * MIB, "compute")

    a = np.arange(12, dtype=dtype).reshape(3, 4) + 1
    b = np.arange(20, dtype=dtype).reshape(4, 5) + 1

    A = pool.empty((3, 4), dtype=dtype)
    B = pool.empty((4, 5), dtype=dtype)
    A[:, :] = a
    B[:, :] = b

    assert_close(A @ B, a @ b, dtype=dtype)


def test_dropping_a_pooled_tensor_returns_its_bytes():
    pool = einsums.MemoryPool(48 * MIB, "reclaim")

    for _ in range(24):
        t = pool.empty((1024, 1024), dtype="float64")  # 8 MiB
        t[0, 0] = 1.0
        del t

    assert pool.arenas == 1
    assert pool.live_borrows == 0
    assert pool.bytes_used == 0
    assert pool.high_water >= 8 * MIB


def test_reset_refuses_to_run_under_a_live_tensor():
    pool = einsums.MemoryPool(32 * MIB, "reset")

    t = pool.empty((128, 128), dtype="float64")
    with pytest.raises(RuntimeError):
        pool.reset()

    del t
    pool.reset()
    assert pool.bytes_used == 0
    assert pool.empty((16, 16), dtype="float64") is not None


def test_reserve_grows_capacity():
    pool = einsums.MemoryPool(MIB, "reserve")
    before = pool.bytes_reserved

    pool.reserve(before + 64 * MIB)
    assert pool.bytes_reserved >= before + 64 * MIB
    assert pool.arenas == 2


def test_epoch_frees_its_cohort():
    pool = einsums.MemoryPool(64 * MIB, "epoch")

    keep = pool.empty((256,), dtype="float64")
    outside = pool.bytes_used

    with pool.epoch():
        inner = pool.empty((1024, 256), dtype="float64")
        assert pool.epoch_depth == 1
        assert pool.bytes_used > outside
        del inner

    assert pool.epoch_depth == 0
    assert pool.bytes_used == outside
    assert keep is not None


def test_nested_epochs():
    pool = einsums.MemoryPool(64 * MIB, "nested")

    with pool.epoch():
        assert pool.epoch_depth == 1
        with pool.epoch():
            assert pool.epoch_depth == 2
        assert pool.epoch_depth == 1
    assert pool.epoch_depth == 0


def test_epoch_refuses_to_close_under_a_live_tensor():
    pool = einsums.MemoryPool(32 * MIB, "epoch-live")

    scope = pool.epoch()
    held = pool.empty((256, 256), dtype="float64")

    with pytest.raises(RuntimeError):
        scope.close()
    assert scope.open

    del held
    scope.close()
    assert not scope.open
    assert pool.epoch_depth == 0


def test_pooled_tensor_outlives_the_pool():
    pool = einsums.MemoryPool(32 * MIB, "outlive")
    t = pool.zeros((512, 64), dtype="float64")
    t[0, 0] = 7.0

    del pool

    # The keepalive token, not the pool object, is what holds the arena open.
    assert t[0, 0] == 7.0
    assert_close(np.asarray(t)[1:, :], np.zeros((511, 64)), dtype="float64")
