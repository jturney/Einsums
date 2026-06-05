# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Numpy ↔ RuntimeTensor interop coverage.

Goes deeper than test_runtime_tensor_python.py's smoke checks:
  * np.asarray returns a live view — mutations on the numpy side are
    visible through the tensor object (and to subsequent C++ ops).
  * np.array() makes a copy — mutations to that copy do NOT alter the
    original tensor.
  * Mutating a tensor through ``einsums.linalg.scale`` is visible to
    a previously-created np.asarray view.
  * Shape / dtype / size / rank match across multiple ranks (1D, 2D,
    3D, 4D).
  * Default layout (column-major in Einsums) is preserved through the
    round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums


DTYPE_TO_RTT = [
    pytest.param(np.float32, einsums.RuntimeTensorF, id="float32"),
    pytest.param(np.float64, einsums.RuntimeTensorD, id="float64"),
    pytest.param(np.complex64, einsums.RuntimeTensorC, id="complex64"),
    pytest.param(np.complex128, einsums.RuntimeTensorZ, id="complex128"),
]


# ──────────────────────────────────────────────────────────────────────────
# np.asarray = view (zero-copy, mutations propagate)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("np_dtype, RTT", DTYPE_TO_RTT)
def test_asarray_is_a_view_mutations_propagate(np_dtype, RTT):
    """Writing through np.asarray must update the underlying tensor."""
    t = RTT("v", [4, 5])
    view = np.asarray(t)
    view[...] = np.full((4, 5), 7, dtype=np_dtype)

    again = np.asarray(t)
    assert np.array_equal(again, np.full((4, 5), 7, dtype=np_dtype))


@pytest.mark.parametrize("np_dtype, RTT", DTYPE_TO_RTT)
def test_two_views_share_storage(np_dtype, RTT):
    """Two ``np.asarray`` calls return aliases of the same buffer."""
    t = RTT("v", [3, 3])
    a = np.asarray(t)
    b = np.asarray(t)
    a[...] = np.eye(3, dtype=np_dtype)
    assert np.array_equal(b, np.eye(3, dtype=np_dtype))


# ──────────────────────────────────────────────────────────────────────────
# np.array() = copy (mutations are isolated)
# ──────────────────────────────────────────────────────────────────────────


def test_np_array_creates_a_copy():
    """``np.array(t)`` decouples the resulting array from the tensor."""
    t = einsums.RuntimeTensorD("c", [3])
    np.asarray(t)[...] = [1.0, 2.0, 3.0]
    snapshot = np.array(t)
    snapshot[0] = 99.0

    assert np.asarray(t)[0] == 1.0
    assert snapshot[0] == 99.0


# ──────────────────────────────────────────────────────────────────────────
# C++ ops mutate the same storage that views see
# ──────────────────────────────────────────────────────────────────────────


def test_scale_mutation_visible_via_existing_view():
    """A view created BEFORE a C++ in-place op reflects the mutation."""
    t = einsums.RuntimeTensorD("m", [3, 3])
    view = np.asarray(t)
    view[...] = np.full((3, 3), 2.0)

    # Hand the same tensor to a C++ in-place op; the existing view must
    # see the result without a refresh.
    einsums.linalg.scale(0.5, t)

    assert np.allclose(view, 1.0), "view did not pick up in-place scale"


def test_axpy_mutation_visible_via_view():
    X = einsums.RuntimeTensorD("x", [4])
    Y = einsums.RuntimeTensorD("y", [4])
    np.asarray(X)[...] = [1.0, 2.0, 3.0, 4.0]
    np.asarray(Y)[...] = [10.0, 10.0, 10.0, 10.0]

    Y_view = np.asarray(Y)  # view captured before the op
    einsums.linalg.axpy(2.0, X, Y)
    np.testing.assert_allclose(Y_view, [12.0, 14.0, 16.0, 18.0])


# ──────────────────────────────────────────────────────────────────────────
# Shape / dtype / rank fidelity across ranks
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dims", [[5], [4, 6], [2, 3, 4], [2, 2, 2, 2]])
def test_shape_round_trip(dims):
    t = einsums.RuntimeTensorD("r", dims)
    arr = np.asarray(t)
    assert arr.shape == tuple(dims)
    assert arr.size == int(np.prod(dims))
    assert t.rank() == len(dims)
    assert [t.dim(i) for i in range(len(dims))] == list(dims)


@pytest.mark.parametrize("np_dtype, RTT", DTYPE_TO_RTT)
def test_dtype_round_trip(np_dtype, RTT):
    t = RTT("d", [4])
    arr = np.asarray(t)
    assert arr.dtype == np_dtype


# ──────────────────────────────────────────────────────────────────────────
# Default layout: Einsums tensors are column-major.
# ──────────────────────────────────────────────────────────────────────────


def test_default_layout_is_column_major():
    """A 2D RuntimeTensor exposes a column-major (Fortran-order) buffer."""
    t = einsums.RuntimeTensorD("layout", [3, 4])
    arr = np.asarray(t)
    assert arr.flags["F_CONTIGUOUS"], (
        "Einsums tensors are column-major by default; np.asarray's view "
        "should report F_CONTIGUOUS"
    )


def test_strides_match_column_major_for_2d():
    """Column-major: stride[0] = itemsize, stride[1] = itemsize * dim(0)."""
    t = einsums.RuntimeTensorD("s", [3, 4])
    arr = np.asarray(t)
    # numpy strides are bytes — compare against the Einsums-side getter.
    itemsize = arr.itemsize
    assert arr.strides[0] == itemsize
    assert arr.strides[1] == itemsize * 3


# ──────────────────────────────────────────────────────────────────────────
# Initial state: freshly-allocated tensors come up zeroed via
# create_zero_tensor; create_tensor doesn't promise zero.
# ──────────────────────────────────────────────────────────────────────────


def test_create_zero_tensor_is_actually_zero():
    t = einsums.create_zero_tensor("z", [4, 4])
    assert np.all(np.asarray(t) == 0)


def test_create_random_tensor_is_not_all_zero():
    """Sanity check that random data fills the buffer (1 in 2^512 chance of
    a flake on a 4×4 float64, so functionally never)."""
    t = einsums.create_random_tensor("r", [4, 4])
    assert not np.all(np.asarray(t) == 0)


# ──────────────────────────────────────────────────────────────────────────
# Round-tripping data through einsum
# ──────────────────────────────────────────────────────────────────────────


def test_einsum_result_visible_through_view():
    """An einsum into C is visible through a view captured BEFORE the call."""
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [4, 5])
    C = einsums.create_zero_tensor("C", [3, 5])
    C_view = np.asarray(C)  # captured early

    expected = np.asarray(A) @ np.asarray(B)
    einsums.einsum("ij <- ik ; kj", C, A, B)

    np.testing.assert_allclose(C_view, expected, rtol=1e-5)
