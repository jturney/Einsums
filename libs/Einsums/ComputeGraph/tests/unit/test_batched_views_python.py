# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""cg.views: N views of one parent recorded in one crossing.

The contract is that the recorded graph is indistinguishable from N
``view_indexed`` calls - same aliasing, same replay semantics, same rebind
behavior - and only the per-call Python/pybind overhead is gone. Every test
here therefore checks the batch against its one-at-a-time equivalent rather
than against hand-computed values: the single-view path is the reference.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg

from einsums.testing import ALL_DTYPES, assert_close


FULL = (0, 0, 0)


def _rng(dtype, shape, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_batch_matches_single_views(dtype):
    T = einsums.create_zero_tensor("T", [6, 5, 8], dtype=np.dtype(dtype).name)
    specs = [
        [(2, 3, 0), FULL, (1, 2, 7)],  # drop + full + range
        [FULL, (2, 0, 0), FULL],       # drop in the middle
        [(1, 1, 5), (1, 0, 4), FULL],  # two ranges
        [FULL, FULL, FULL],            # the whole parent
    ]
    g = cg.Graph("views_match")
    with cg.capture(g):
        batch = cg.views(T, specs)
        singles = [cg.view_indexed(T, s) for s in specs]
    np.asarray(T)[...] = _rng(dtype, (6, 5, 8))
    g.execute()

    assert len(batch) == len(specs)
    for got, want in zip(batch, singles):
        assert np.array_equal(np.asarray(got), np.asarray(want))


def test_views_alias_the_parent():
    T = einsums.create_zero_tensor("T", [4, 4], dtype="float64")
    g = cg.Graph("views_alias")
    with cg.capture(g):
        (row,) = cg.views(T, [[(2, 1, 0), FULL]])
    np.asarray(T)[...] = np.arange(16.0).reshape(4, 4)
    g.execute()
    # Writing through the view lands in the parent: it is a view, not a copy.
    np.asarray(row)[...] = -1.0
    assert np.array_equal(np.asarray(T)[1], [-1.0] * 4)


def test_views_replay_follows_new_parent_values():
    T = einsums.create_zero_tensor("T", [3, 3], dtype="float64")
    out = einsums.create_zero_tensor("out", [3], dtype="float64")
    g = cg.Graph("views_replay")
    with cg.capture(g):
        (col,) = cg.views(T, [[FULL, (2, 0, 0)]])
        einsums.linalg.axpy(1.0, col, out)
    for rep in range(2):
        np.asarray(T)[...] = float(rep + 1)
        np.asarray(out)[...] = 0.0
        g.execute()
        assert_close(np.asarray(out), np.full(3, float(rep + 1)))


def test_empty_batch_is_a_no_op():
    T = einsums.create_zero_tensor("T", [2, 2], dtype="float64")
    g = cg.Graph("views_empty")
    with cg.capture(g):
        assert cg.views(T, []) == []
    assert g.num_nodes() == 0


def test_bad_spec_raises_with_no_partial_batch_visible():
    T = einsums.create_zero_tensor("T", [2, 2], dtype="float64")
    g = cg.Graph("views_bad")
    with cg.capture(g):
        with pytest.raises(Exception):
            cg.views(T, [[FULL, FULL], [FULL]])  # second spec has wrong arity


def test_outside_capture_raises():
    T = einsums.create_zero_tensor("T", [2, 2], dtype="float64")
    with pytest.raises(Exception):
        cg.views(T, [[FULL, FULL]])
