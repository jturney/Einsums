#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""``linalg.gather``: index-list extraction, with numpy's ``np.ix_`` as oracle."""

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
from einsums.testing import ALL_DTYPES, assert_close


def _filled(name, shape, dtype, rng):
    t = einsums.create_zero_tensor(name, list(shape), dtype=dtype)
    a = rng.random(shape)
    if np.iscomplexobj(np.zeros(1, dtype=dtype)):
        a = a + 1j * rng.random(shape)
    np.asarray(t)[...] = a.astype(dtype)
    return t


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_gather_matches_ix_(dtype):
    rng = np.random.default_rng(0)
    A = _filled("A", (7, 5), dtype, rng)
    # out of order and repeated, which np.ix_ allows and so must gather
    rows, cols = [5, 1, 1, 3], [4, 0]
    out = einsums.create_zero_tensor("out", [len(rows), len(cols)], dtype=dtype)
    la.gather(out, A, [rows, cols])
    assert_close(np.asarray(out), np.asarray(A)[np.ix_(rows, cols)], dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_gather_rank3_subset_of_axes(dtype):
    """The DLPNO ``(Q|iu)`` shape: two axes restricted, one kept whole."""
    rng = np.random.default_rng(1)
    B = _filled("B", (5, 3, 7), dtype, rng)
    q, u = [3, 0], [6, 2, 4]
    out = einsums.create_zero_tensor("out", [len(q), 3, len(u)], dtype=dtype)
    la.gather(out, B, [q, list(range(3)), u])
    assert_close(np.asarray(out), np.asarray(B)[np.ix_(q, range(3), u)], dtype=dtype)


def test_gather_empty_selection_is_not_a_wildcard():
    """An empty index list selects nothing; it must not mean "whole axis".

    The callers are domain-restricted methods where an empty domain is a
    legitimate input, and silently promoting it to the full axis would turn a
    screened-out domain into a full-rank one with no error.
    """
    A = einsums.create_zero_tensor("A", [4, 6], dtype="float64")
    np.asarray(A)[...] = np.arange(24).reshape(4, 6)
    out = einsums.create_zero_tensor("out", [0, 6], dtype="float64")
    la.gather(out, A, [[], list(range(6))])
    assert np.asarray(out).shape == (0, 6)


def test_gather_captures_and_rereads_source_on_replay():
    A = einsums.create_zero_tensor("A", [6, 6], dtype="float64")
    np.asarray(A)[...] = np.arange(36).reshape(6, 6)
    dom = [4, 1, 0]
    out = einsums.create_zero_tensor("out", [len(dom), len(dom)], dtype="float64")

    g = cg.Graph("gather")
    with cg.capture(g):
        la.gather(out, A, [dom, dom])
    g.execute()
    assert_close(np.asarray(out), np.asarray(A)[np.ix_(dom, dom)])

    # capture-once, replay-many: the second run must see the new values
    np.asarray(A)[...] += 100.0
    g.execute()
    assert_close(np.asarray(out), np.asarray(A)[np.ix_(dom, dom)])


def test_gather_rejects_bad_shapes_and_indices():
    A = einsums.create_zero_tensor("A", [4, 4], dtype="float64")
    whole = list(range(4))

    out = einsums.create_zero_tensor("wrong", [3, 4], dtype="float64")
    with pytest.raises(Exception):
        la.gather(out, A, [[0, 1], whole])

    out = einsums.create_zero_tensor("oob", [2, 4], dtype="float64")
    with pytest.raises(Exception):
        la.gather(out, A, [[0, 9], whole])


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_scatter_matches_numpy_ix_assignment(dtype):
    rng = np.random.default_rng(2)
    dst = einsums.create_zero_tensor("dst", [5, 6], dtype=dtype)
    src = _filled("src", (3, 2), dtype, rng)
    rows, cols = [4, 0, 2], [5, 1]
    la.scatter(dst, src, [rows, cols])

    want = np.zeros((5, 6), dtype=dtype)
    want[np.ix_(rows, cols)] = np.asarray(src)
    assert_close(np.asarray(dst), want, dtype=dtype)


def test_scatter_leaves_the_rest_untouched():
    """A scatter is a placement, not an assignment of the whole tensor."""
    dst = einsums.create_zero_tensor("dst", [4, 4], dtype="float64")
    np.asarray(dst)[...] = 7.0
    src = einsums.create_zero_tensor("src", [2, 2], dtype="float64")
    np.asarray(src)[...] = 1.0
    la.scatter(dst, src, [[0, 3], [0, 3]])

    got = np.asarray(dst)
    assert got[0, 0] == 1.0 and got[3, 3] == 1.0
    assert got[1, 1] == 7.0 and got[2, 0] == 7.0


def test_scatter_round_trips_with_gather():
    A = einsums.create_zero_tensor("A", [6, 6], dtype="float64")
    np.asarray(A)[...] = np.arange(36).reshape(6, 6)
    dom = [4, 1, 0]

    block = einsums.create_zero_tensor("block", [len(dom), len(dom)], dtype="float64")
    la.gather(block, A, [dom, dom])
    back = einsums.create_zero_tensor("back", [6, 6], dtype="float64")
    la.scatter(back, block, [dom, dom])

    assert_close(np.asarray(back)[np.ix_(dom, dom)], np.asarray(A)[np.ix_(dom, dom)])


def test_scatter_rejects_repeated_indices():
    """Two writes to one element would be order-dependent, so it is an error.

    gather allows repeats - reading the same element twice is harmless - but
    the asymmetry is deliberate.
    """
    dst = einsums.create_zero_tensor("dst", [4, 4], dtype="float64")
    src = einsums.create_zero_tensor("src", [3, 4], dtype="float64")
    with pytest.raises(Exception):
        la.scatter(dst, src, [[1, 2, 1], list(range(4))])


# ── axis-permuting gather ────────────────────────────────────────────────────
#
# ``axes[k]`` is the destination axis that source axis ``k`` lands on. The point
# is to avoid a separate permute pass: selecting and reordering are both full
# passes over the result, and doing them together is one.


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_gather_axes_permutes_on_the_way_out(dtype):
    rng = np.random.default_rng(11)
    src = _filled("src", (4, 3, 7), dtype, rng)
    perm = [5, 0, 3, 6, 1, 4, 2]

    # (a, b, n) -> (a, n, b): source axis 2 lands on destination axis 1.
    dst = einsums.create_zero_tensor("dst", [4, 7, 3], dtype=dtype)
    la.gather(dst, src, [list(range(4)), list(range(3)), perm], [0, 2, 1])

    want = np.asarray(src)[:, :, perm].transpose(0, 2, 1)
    assert_close(np.asarray(dst), want)


def test_gather_axes_default_is_the_identity():
    rng = np.random.default_rng(12)
    src = _filled("src", (4, 3, 7), "float64", rng)
    perm = [5, 0, 3, 6, 1, 4, 2]
    idx = [list(range(4)), list(range(3)), perm]

    a = einsums.create_zero_tensor("a", [4, 3, 7], dtype="float64")
    b = einsums.create_zero_tensor("b", [4, 3, 7], dtype="float64")
    la.gather(a, src, idx)
    la.gather(b, src, idx, [0, 1, 2])

    assert_close(np.asarray(a), np.asarray(src)[:, :, perm])
    assert_close(np.asarray(b), np.asarray(a))


def test_gather_axes_composes_with_selection():
    """Selecting a subset and reordering axes at once."""
    rng = np.random.default_rng(13)
    src = _filled("src", (4, 3, 7), "float64", rng)

    dst = einsums.create_zero_tensor("dst", [4, 2, 2], dtype="float64")
    la.gather(dst, src, [list(range(4)), [0, 2], [1, 5]], [0, 2, 1])

    want = np.asarray(src)[:, [0, 2], :][:, :, [1, 5]].transpose(0, 2, 1)
    assert_close(np.asarray(dst), want)


@pytest.mark.parametrize("axes", [[0, 1], [0, 1, 1], [0, 1, 3]])
def test_gather_rejects_bad_axes(axes):
    """Wrong length, a repeat, and out of range - none is a permutation."""
    src = einsums.create_zero_tensor("src", [4, 3, 7], dtype="float64")
    dst = einsums.create_zero_tensor("dst", [4, 7, 3], dtype="float64")
    with pytest.raises(Exception):
        la.gather(dst, src, [list(range(4)), list(range(3)), list(range(7))], axes)


def test_gather_axes_checks_the_permuted_extent():
    """dst's extent is checked on the axis the source axis actually lands on."""
    src = einsums.create_zero_tensor("src", [4, 3, 7], dtype="float64")
    unpermuted = einsums.create_zero_tensor("dst", [4, 3, 7], dtype="float64")
    with pytest.raises(Exception):
        la.gather(unpermuted, src, [list(range(4)), list(range(3)), list(range(7))], [0, 2, 1])


def test_gather_axes_captures():
    """The permutation survives capture and replay, like any other gather."""
    rng = np.random.default_rng(14)
    src = _filled("src", (4, 3, 7), "float64", rng)
    dst = einsums.create_zero_tensor("dst", [4, 7, 3], dtype="float64")
    perm = [5, 0, 3, 6, 1, 4, 2]

    g = cg.Graph("permuting gather")
    with cg.capture(g):
        la.gather(dst, src, [list(range(4)), list(range(3)), perm], [0, 2, 1])
    assert np.asarray(dst).any() == False  # capture does not run it
    g.execute()

    assert_close(np.asarray(dst), np.asarray(src)[:, :, perm].transpose(0, 2, 1))
