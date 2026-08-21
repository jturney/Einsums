#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""``linalg.grouped_gather_rotate``: the q-tiled gather-and-rotate run.

The oracle is numpy: an ``np.ix_`` selection of the parent, then the same
two-sided rotation as one ``np.einsum``. That is the emission the node replaces,
written where nothing it shares code with can hide a matching mistake.
"""

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
from einsums.testing import REAL_DTYPES, assert_close


def _filled(name, shape, dtype, rng):
    t = einsums.create_zero_tensor(name, list(shape), dtype=dtype)
    np.asarray(t)[...] = rng.random(shape).astype(dtype)
    return t


def _oracle(src, qs, us, X):
    """``sum_uv src[Q[q], U[u], U[v]] X[u, a] X[v, b]``, through numpy."""
    block = np.asarray(src)[np.ix_(qs, us, us)]
    return np.einsum("quv,ua,vb->qab", block, np.asarray(X), np.asarray(X))


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_matches_numpy_over_mixed_members(dtype):
    rng = np.random.default_rng(0)
    src = _filled("(Q|u v)", (23, 9, 9), dtype, rng)

    # An ascending run, an out-of-order selection with a repeat, and a member
    # whose auxiliary list spans several q tiles.
    members = [
        ([0, 1, 2, 3, 4, 5], [2, 3, 4], 2),
        ([7, 0, 3, 3, 11], [8, 1, 5, 0], 3),
        (list(range(23)), [0, 2, 4, 6], 4),
    ]

    c_list, x_list, q_list, u_list, expected = [], [], [], [], []
    for idx, (qs, us, nt) in enumerate(members):
        X = _filled(f"X{idx}", (len(us), nt), dtype, rng)
        # create_zero_tensor rather than an uninitialized store, so that a
        # kernel that failed to write would show as zeros rather than garbage.
        C = einsums.create_zero_tensor(f"C{idx}", [len(qs), nt, nt], dtype=dtype)
        c_list.append(C)
        x_list.append(X)
        q_list.append(qs)
        u_list.append(us)
        expected.append(_oracle(src, qs, us, X))

    la.grouped_gather_rotate(c_list, src, q_list, u_list, x_list)

    for C, want in zip(c_list, expected):
        assert_close(np.asarray(C), want, dtype=dtype)


def test_zero_extent_members_are_quick_returns():
    """A screened-out domain is a legitimate member, not an error.

    An empty auxiliary list has nothing to write; an empty PAO list has an empty
    sum, and the operation assigns, so its destination must come back zero
    whatever it held before.
    """
    rng = np.random.default_rng(1)
    src = _filled("(Q|u v)", (6, 4, 4), "float64", rng)

    empty_q = einsums.create_zero_tensor("empty q", [0, 2, 2], dtype="float64")
    empty_u = einsums.create_zero_tensor("empty u", [3, 2, 2], dtype="float64")
    np.asarray(empty_u)[...] = 7.0
    x_empty_q = _filled("X", (3, 2), "float64", rng)
    x_empty_u = einsums.create_zero_tensor("X (no rows)", [0, 2], dtype="float64")

    la.grouped_gather_rotate([empty_q, empty_u], src,
                             [[], [0, 2, 5]], [[1, 2, 3], []],
                             [x_empty_q, x_empty_u])

    assert np.asarray(empty_q).size == 0
    assert_close(np.asarray(empty_u), np.zeros((3, 2, 2)), dtype="float64")


def test_capture_records_one_node_and_replays_identically():
    rng = np.random.default_rng(2)
    src = _filled("(Q|u v)", (31, 7, 7), "float64", rng)
    X0 = _filled("X0", (4, 3), "float64", rng)
    X1 = _filled("X1", (5, 2), "float64", rng)
    qs0, us0 = list(range(20)), [0, 3, 6, 1]
    qs1, us1 = [30, 20, 10, 0], [1, 2, 3, 4, 5]

    C0 = einsums.create_zero_tensor("C0", [len(qs0), 3, 3], dtype="float64")
    C1 = einsums.create_zero_tensor("C1", [len(qs1), 2, 2], dtype="float64")

    g = cg.Graph("gather rotate")
    with cg.capture(g):
        la.grouped_gather_rotate([C0, C1], src, [qs0, qs1], [us0, us1], [X0, X1])
    assert g.num_nodes() == 1

    g.execute()
    first = (np.asarray(C0).copy(), np.asarray(C1).copy())
    assert_close(first[0], _oracle(src, qs0, us0, X0), dtype="float64")
    assert_close(first[1], _oracle(src, qs1, us1, X1), dtype="float64")

    # The node assigns rather than accumulates, and the tiled axis indexes no
    # sum, so a replay from anywhere lands on the same bits.
    np.asarray(C0)[...] = np.nan
    np.asarray(C1)[...] = np.nan
    g.execute()
    assert np.array_equal(np.asarray(C0), first[0])
    assert np.array_equal(np.asarray(C1), first[1])
    g.execute()
    assert np.array_equal(np.asarray(C0), first[0])
    assert np.array_equal(np.asarray(C1), first[1])


def test_rejects_a_shared_destination():
    rng = np.random.default_rng(3)
    src = _filled("(Q|u v)", (8, 4, 4), "float64", rng)
    X = _filled("X", (2, 2), "float64", rng)
    C = einsums.create_zero_tensor("C", [3, 2, 2], dtype="float64")
    with pytest.raises(Exception):
        la.grouped_gather_rotate([C, C], src, [[0, 1, 2], [0, 1, 2]],
                                 [[0, 1], [0, 1]], [X, X])


def test_rejects_a_destination_the_lists_do_not_fit():
    rng = np.random.default_rng(4)
    src = _filled("(Q|u v)", (8, 4, 4), "float64", rng)
    X = _filled("X", (2, 2), "float64", rng)
    C = einsums.create_zero_tensor("C", [3, 2, 2], dtype="float64")
    with pytest.raises(Exception):
        la.grouped_gather_rotate([C], src, [[0, 1]], [[0, 1]], [X])
    with pytest.raises(Exception):
        la.grouped_gather_rotate([C], src, [[0, 1, 8]], [[0, 1]], [X])
