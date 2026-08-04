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
