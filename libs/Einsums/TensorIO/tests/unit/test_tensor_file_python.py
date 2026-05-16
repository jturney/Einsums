#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Python unit tests for the TensorFile bindings exposed under einsums.io.

The C++ tests in TensorFileBasic.cpp / TensorFileSliceWrite.cpp cover the
wire format. These tests guard the Python surface — that the bindings
load, dispatch on dtype, round-trip via numpy, and the slice overloads
are wired up.
"""

import os
import tempfile
import numpy as np
import pytest

import einsums


def _temp_path(name):
    return os.path.join(tempfile.gettempdir(), f"einsums_pyio_{os.getpid()}_{name}.etn")


def test_tensor_file_module_layout():
    assert einsums.io.TensorFile is not None
    assert einsums.io.Mode.Read != einsums.io.Mode.Write
    assert hasattr(einsums.io.TensorFile, "read")
    assert hasattr(einsums.io.TensorFile, "write")
    assert hasattr(einsums.io.TensorFile, "read_slice")
    assert hasattr(einsums.io.TensorFile, "write_slice")


def test_full_roundtrip_double():
    path = _temp_path("full_double")
    try:
        rt = einsums.RuntimeTensorD("A", [3, 4])
        arr = np.asarray(rt, copy=False)
        arr[:] = np.arange(12, dtype=np.float64).reshape(3, 4)

        f = einsums.io.TensorFile(path, einsums.io.Mode.Write)
        f.write("A", rt)
        del f

        g = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        assert g.tensor_names() == ["A"]
        assert g.dims("A") == [3, 4]

        rt2 = einsums.RuntimeTensorD("back", [3, 4])
        g.read("A", rt2)
        arr2 = np.asarray(rt2, copy=False)
        assert np.allclose(arr, arr2)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_full_roundtrip_resizes_from_zero_dim_preserves_layout():
    """Regression: GeneralRuntimeTensor::resize() used to propagate
    is_row_major() (which collapses to True for rank ≤ 1), so growing a
    [0]-rank placeholder up to e.g. [3, 4] silently flipped the new
    tensor to row-major. Fixed by switching resize to stored_row_major().
    """
    path = _temp_path("resize_layout")
    try:
        rt = einsums.RuntimeTensorD("A", [3, 4])
        arr = np.asarray(rt, copy=False)
        arr[:] = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert arr.flags.f_contiguous, "source tensor must be column-major"

        f = einsums.io.TensorFile(path, einsums.io.Mode.Write)
        f.write("A", rt)
        del f

        # Resize-from-[0]: read() grows rt2 from rank-1 dim [0] to dim [3, 4].
        rt2 = einsums.RuntimeTensorD("back", [0])
        g = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        g.read("A", rt2)
        arr2 = np.asarray(rt2, copy=False)

        assert arr2.flags.f_contiguous, "resize must preserve column-major layout"
        assert np.allclose(arr, arr2)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_slice_write_then_full_read():
    path = _temp_path("slab_write")
    try:
        h = einsums.io.TensorFile(path, einsums.io.Mode.ReadWrite)
        rt = einsums.RuntimeTensorD("A", [4, 4])
        np.asarray(rt, copy=False)[:] = 0.0
        h.write("A", rt)

        patch = einsums.RuntimeTensorD("p", [2, 2])
        np.asarray(patch, copy=False)[:] = 7.0
        h.write_slice("A", patch, [(1, 3), (1, 3)])
        del h

        g = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        rt2 = einsums.RuntimeTensorD("back", [4, 4])
        g.read("A", rt2)
        arr2 = np.asarray(rt2, copy=False)

        expected = np.zeros((4, 4))
        expected[1:3, 1:3] = 7.0
        assert np.allclose(arr2, expected)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_slice_read_returns_pre_sized_slab():
    path = _temp_path("slab_read")
    try:
        h = einsums.io.TensorFile(path, einsums.io.Mode.ReadWrite)
        rt = einsums.RuntimeTensorD("A", [4, 4])
        np.asarray(rt, copy=False)[:] = 0.0
        h.write("A", rt)

        patch = einsums.RuntimeTensorD("p", [2, 2])
        np.asarray(patch, copy=False)[:] = 9.0
        h.write_slice("A", patch, [(1, 3), (1, 3)])
        del h

        g = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        slab = einsums.RuntimeTensorD("s", [2, 2])
        g.read_slice("A", slab, [(1, 3), (1, 3)])
        sa = np.asarray(slab, copy=False)
        assert np.allclose(sa, 9.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.parametrize("ctor,dtype", [
    (einsums.RuntimeTensorF, np.float32),
    (einsums.RuntimeTensorD, np.float64),
])
def test_dtype_dispatch_real(ctor, dtype):
    path = _temp_path(f"dtype_{dtype.__name__}")
    try:
        rt = ctor("A", [2, 3])
        np.asarray(rt, copy=False)[:] = np.arange(6, dtype=dtype).reshape(2, 3)

        f = einsums.io.TensorFile(path, einsums.io.Mode.Write)
        f.write("A", rt)
        del f

        g = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        rt2 = ctor("back", [2, 3])
        g.read("A", rt2)
        assert np.allclose(np.asarray(rt2, copy=False), np.asarray(rt, copy=False))
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_write_slice_rejects_dim_mismatch():
    path = _temp_path("mismatch")
    try:
        h = einsums.io.TensorFile(path, einsums.io.Mode.ReadWrite)
        rt = einsums.RuntimeTensorD("A", [4, 4])
        np.asarray(rt, copy=False)[:] = 0.0
        h.write("A", rt)

        wrong = einsums.RuntimeTensorD("w", [3, 2])  # doesn't fit (1:3,1:3)
        with pytest.raises(RuntimeError):
            h.write_slice("A", wrong, [(1, 3), (1, 3)])
    finally:
        if os.path.exists(path):
            os.remove(path)
