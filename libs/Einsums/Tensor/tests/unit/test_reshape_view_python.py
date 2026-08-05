#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""``reshape_view``: reinterpreting a tensor's shape without moving anything.

The distinction that matters is against ``linalg.reshape``, which copies. A
reshape is free exactly when each new axis lands inside a run of old axes that
already abut in memory; when it does not, this throws rather than quietly
copying, so a caller that wanted a view finds out.
"""

import numpy as np
import pytest

import einsums
from einsums.testing import ALL_DTYPES, assert_close


def _filled(name, shape, dtype, rng):
    t = einsums.create_zero_tensor(name, list(shape), dtype=dtype)
    a = rng.random(shape)
    if np.iscomplexobj(np.zeros(1, dtype=dtype)):
        a = a + 1j * rng.random(shape)
    np.asarray(t)[...] = a.astype(dtype)
    return t


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_merge_trailing_axes(dtype):
    rng = np.random.default_rng(3)
    A = _filled("A", (9, 9, 10), dtype, rng)
    # Column major, so numpy's Fortran-order reshape is the oracle.
    assert_close(np.asarray(A.reshape_view([9, 90])),
                 np.asarray(A).reshape(9, 90, order="F"))


def test_it_is_a_view():
    A = einsums.create_zero_tensor("A", [4, 5], dtype="float64")
    V = A.reshape_view([20])
    np.asarray(V)[7] = 3.5
    assert np.asarray(A)[3, 1] == 3.5


def test_split_inverts_merge():
    rng = np.random.default_rng(4)
    A = _filled("A", (9, 9, 10), "float64", rng)
    flat = A.reshape_view([9, 90])
    assert_close(np.asarray(flat.reshape_view([9, 9, 10])), np.asarray(A))


def test_full_flatten_and_back():
    rng = np.random.default_rng(5)
    A = _filled("A", (3, 4, 5), "float64", rng)
    flat = A.reshape_view([60])
    assert_close(np.asarray(flat), np.asarray(A).reshape(60, order="F"))
    assert_close(np.asarray(flat.reshape_view([3, 4, 5])), np.asarray(A))


def test_extent_one_axes():
    """Length-1 axes carry no data, so they neither block nor need a real stride."""
    rng = np.random.default_rng(6)
    A = _filled("A", (9, 9, 10), "float64", rng)
    V = A.reshape_view([9, 1, 9, 10, 1])
    assert_close(np.squeeze(np.asarray(V)), np.asarray(A))


def test_slice_of_a_complete_axis_still_merges():
    """``A[:, 0:4, :]`` keeps axis 0 whole, so its first 36 elements ARE the slice."""
    rng = np.random.default_rng(7)
    A = _filled("A", (9, 9, 10), "float64", rng)
    S = A[:, 0:4, :]
    assert S.reshapable_as_view([36, 10]) is True
    assert_close(np.asarray(S.reshape_view([36, 10])),
                 np.asarray(S).reshape(36, 10, order="F"))


def test_gap_on_the_fastest_axis_is_rejected():
    """Slicing the fastest axis leaves a gap between consecutive outer elements."""
    A = einsums.create_zero_tensor("A", [9, 9, 10], dtype="float64")
    S = A[0:4, :, :]
    assert S.reshapable_as_view([36, 10]) is False
    with pytest.raises(Exception):
        S.reshape_view([36, 10])
    # A merge that does not cross the gap is still fine.
    assert S.reshapable_as_view([4, 90]) is True


def test_wrong_element_count_is_rejected():
    A = einsums.create_zero_tensor("A", [9, 9, 10], dtype="float64")
    assert A.reshapable_as_view([9, 91]) is False
    with pytest.raises(Exception):
        A.reshape_view([9, 91])


def test_view_of_a_view():
    rng = np.random.default_rng(8)
    A = _filled("A", (4, 6), "float64", rng)
    assert_close(np.asarray(A.reshape_view([24]).reshape_view([2, 12])),
                 np.asarray(A).reshape(24, order="F").reshape(2, 12, order="F"))


def test_zero_size_tensor():
    A = einsums.create_zero_tensor("A", [0, 5], dtype="float64")
    assert np.asarray(A.reshape_view([5, 0])).shape == (5, 0)


def test_row_major_view_merges_from_the_other_end():
    """A transposed view is row major, where the LAST axis is the fast one.

    The storage order comes from the strides rather than from the tensor's
    row-major flag, which reports true for any rank below 2 and so cannot be
    trusted after a flatten.
    """
    rng = np.random.default_rng(9)
    A = _filled("A", (3, 4, 5), "float64", rng)
    T = A.T  # dims (5, 4, 3), row major
    assert_close(np.asarray(T.reshape_view([5, 12])),
                 np.asarray(T).reshape(5, 12))
