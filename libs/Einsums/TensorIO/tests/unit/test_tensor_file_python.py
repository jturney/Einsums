#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Python unit tests for the TensorFile bindings exposed under einsums.io.

The C++ tests in TensorFileBasic.cpp / TensorFileSliceWrite.cpp cover the
wire format. These tests guard the Python surface, that the bindings
load, dispatch on dtype, round-trip via numpy, and the slice overloads
are wired up.
"""

import os
import tempfile
import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close


def _temp_path(name):
    return os.path.join(tempfile.gettempdir(), f"einsums_pyio_{os.getpid()}_{name}.etn")


@pytest.fixture
def etn_path(request):
    path = _temp_path(request.node.name.replace("[", "_").replace("]", ""))
    yield path
    if os.path.exists(path):
        os.remove(path)


def _filled(name, shape, dtype, values):
    """A zero tensor of ``dtype`` with ``values`` copied in (complex dtypes get an imaginary part)."""
    rt = einsums.create_zero_tensor(name, list(shape), dtype=dtype)
    values = np.asarray(values, dtype=np.float64)
    if np.dtype(dtype).kind == "c":
        values = values - 0.5j * values
    np.asarray(rt, copy=False)[:] = values.astype(dtype)
    return rt


def test_tensor_file_module_layout():
    assert einsums.io.TensorFile is not None
    assert einsums.io.Mode.Read != einsums.io.Mode.Write
    assert hasattr(einsums.io.TensorFile, "read")
    assert hasattr(einsums.io.TensorFile, "write")
    assert hasattr(einsums.io.TensorFile, "read_slice")
    assert hasattr(einsums.io.TensorFile, "write_slice")


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_full_roundtrip(etn_path, dtype):
    rt = _filled("A", (3, 4), dtype, np.arange(12).reshape(3, 4))

    f = einsums.io.TensorFile(etn_path, einsums.io.Mode.Write)
    f.write("A", rt)
    del f

    g = einsums.io.TensorFile(etn_path, einsums.io.Mode.Read)
    assert g.tensor_names() == ["A"]
    assert g.dims("A") == [3, 4]

    rt2 = einsums.create_zero_tensor("back", [3, 4], dtype=dtype)
    g.read("A", rt2)
    assert np.asarray(rt2, copy=False).dtype == np.dtype(dtype)
    assert_close(np.asarray(rt2, copy=False), np.asarray(rt, copy=False), dtype=dtype)
    # Every handle has to be released before the cleanup: Windows refuses to
    # remove a file that is still open, where POSIX is happy to unlink it.
    # Same reason the writer above is dropped explicitly.
    del g


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_module_level_roundtrip(etn_path, dtype):
    """``einsums.io.write`` / ``einsums.io.read`` open the file per call."""
    src = einsums.create_random_tensor("X", [2, 3], dtype=dtype)
    einsums.io.write(etn_path, "X", src)

    back = einsums.create_zero_tensor("Y", [2, 3], dtype=dtype)
    einsums.io.read(etn_path, "X", back)
    assert_close(np.asarray(back, copy=False), np.asarray(src, copy=False), dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_full_roundtrip_resizes_from_zero_dim_preserves_layout(etn_path, dtype):
    """Regression: GeneralRuntimeTensor::resize() used to propagate
    is_row_major() (which collapses to True for rank ≤ 1), so growing a
    [0]-rank placeholder up to e.g. [3, 4] silently flipped the new
    tensor to row-major. Fixed by switching resize to stored_row_major().
    """
    rt = _filled("A", (3, 4), dtype, np.arange(12).reshape(3, 4))
    arr = np.asarray(rt, copy=False)
    assert arr.flags.f_contiguous, "source tensor must be column-major"

    f = einsums.io.TensorFile(etn_path, einsums.io.Mode.Write)
    f.write("A", rt)
    del f

    # Resize-from-[0]: read() grows rt2 from rank-1 dim [0] to dim [3, 4].
    rt2 = einsums.create_zero_tensor("back", [0], dtype=dtype)
    g = einsums.io.TensorFile(etn_path, einsums.io.Mode.Read)
    g.read("A", rt2)
    arr2 = np.asarray(rt2, copy=False)

    assert arr2.flags.f_contiguous, "resize must preserve column-major layout"
    assert_close(arr2, arr, dtype=dtype)
    del g


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_slice_write_then_full_read(etn_path, dtype):
    h = einsums.io.TensorFile(etn_path, einsums.io.Mode.ReadWrite)
    rt = einsums.create_zero_tensor("A", [4, 4], dtype=dtype)
    h.write("A", rt)

    patch = _filled("p", (2, 2), dtype, np.full((2, 2), 7.0))
    h.write_slice("A", patch, [(1, 3), (1, 3)])
    del h

    g = einsums.io.TensorFile(etn_path, einsums.io.Mode.Read)
    rt2 = einsums.create_zero_tensor("back", [4, 4], dtype=dtype)
    g.read("A", rt2)

    expected = np.zeros((4, 4), dtype=dtype)
    expected[1:3, 1:3] = np.asarray(patch, copy=False)
    assert_close(np.asarray(rt2, copy=False), expected, dtype=dtype)
    del g


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_slice_read_returns_pre_sized_slab(etn_path, dtype):
    values = np.arange(16).reshape(4, 4)
    h = einsums.io.TensorFile(etn_path, einsums.io.Mode.ReadWrite)
    rt = _filled("A", (4, 4), dtype, values)
    h.write("A", rt)
    del h

    g = einsums.io.TensorFile(etn_path, einsums.io.Mode.Read)
    slab = einsums.create_zero_tensor("s", [2, 3], dtype=dtype)
    g.read_slice("A", slab, [(1, 3), (0, 3)])
    assert_close(np.asarray(slab, copy=False), np.asarray(rt, copy=False)[1:3, 0:3], dtype=dtype)
    del g


def test_write_slice_rejects_dim_mismatch(etn_path):
    h = einsums.io.TensorFile(etn_path, einsums.io.Mode.ReadWrite)
    rt = einsums.RuntimeTensorD("A", [4, 4])
    np.asarray(rt, copy=False)[:] = 0.0
    h.write("A", rt)

    wrong = einsums.RuntimeTensorD("w", [3, 2])  # doesn't fit (1:3,1:3)
    with pytest.raises(RuntimeError):
        h.write_slice("A", wrong, [(1, 3), (1, 3)])
    del h


# Every read into (and slice write from) a tensor checks that the stored element type matches the
# tensor's. Without the check a complex128 entry read into a float32 tensor wrote four times the
# destination's size, and same-width pairs (float64 and complex64) reinterpreted the bits.
_MISMATCHED = [(stored, requested) for stored in ALL_DTYPES for requested in ALL_DTYPES if stored != requested]


@pytest.mark.parametrize("stored,requested", _MISMATCHED)
def test_read_rejects_dtype_mismatch(etn_path, stored, requested):
    src = einsums.create_random_tensor("X", [2, 3], dtype=stored)
    einsums.io.write(etn_path, "X", src)

    dst = einsums.create_zero_tensor("Y", [2, 3], dtype=requested)
    with pytest.raises(ValueError, match=rf"'X'.*holds {stored} data.*the tensor holds {requested}"):
        einsums.io.read(etn_path, "X", dst)

    f = einsums.io.TensorFile(etn_path, einsums.io.Mode.Read)
    with pytest.raises(ValueError):
        f.read("X", dst)
    with pytest.raises(ValueError):
        f.read_slice("X", dst, [(0, 2), (0, 3)])
    del f

    slab = einsums.io.Slab([(0, 2), (0, 3)])
    with pytest.raises(ValueError):
        einsums.io.read_slice(etn_path, "X", slab, dst)

    # Nothing was written into the destination.
    arr = np.asarray(dst, copy=False)
    assert arr.shape == (2, 3)
    assert not np.any(arr)


@pytest.mark.parametrize("stored,requested", _MISMATCHED)
def test_write_slice_rejects_dtype_mismatch(etn_path, stored, requested):
    src = einsums.create_random_tensor("X", [2, 3], dtype=stored)
    einsums.io.write(etn_path, "X", src)

    patch = einsums.create_random_tensor("P", [2, 2], dtype=requested)
    h = einsums.io.TensorFile(etn_path, einsums.io.Mode.ReadWrite)
    with pytest.raises(ValueError):
        h.write_slice("X", patch, [(0, 2), (0, 2)])
    del h

    slab = einsums.io.Slab([(0, 2), (0, 2)])
    with pytest.raises(ValueError):
        einsums.io.write_slice(etn_path, "X", slab, patch)

    back = einsums.create_zero_tensor("B", [2, 3], dtype=stored)
    einsums.io.read(etn_path, "X", back)
    np.testing.assert_array_equal(np.asarray(back, copy=False), np.asarray(src, copy=False))


def test_captured_read_rejects_dtype_mismatch(etn_path):
    einsums.io.write(etn_path, "X", einsums.create_random_tensor("X", [2, 3], dtype="complex128"))

    dst = einsums.create_zero_tensor("Y", [2, 3], dtype="float32")
    g = cg.Graph("dtype-mismatch")
    with cg.capture(g):
        einsums.io.read(etn_path, "X", dst)
    with pytest.raises(ValueError, match="holds complex128 data"):
        g.execute()
    assert not np.any(np.asarray(dst, copy=False))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_slice_ranges_must_lie_inside_the_entry(etn_path, dtype):
    einsums.io.write(etn_path, "X", einsums.create_random_tensor("X", [2, 3], dtype=dtype))

    f = einsums.io.TensorFile(etn_path, einsums.io.Mode.Read)
    slab = einsums.create_zero_tensor("s", [1], dtype=dtype)
    with pytest.raises(IndexError):
        f.read_slice("X", slab, [(0, 3), (0, 3)])
    with pytest.raises(ValueError):
        f.read_slice("X", slab, [(0, 2)])
    with pytest.raises(ValueError):
        f.read_slice("X", slab, [(2, 1), (0, 3)])
    del f
