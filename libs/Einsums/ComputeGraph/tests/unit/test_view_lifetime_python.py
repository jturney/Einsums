# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Regression tests for view operands in dot / graph capture.

Two issues found by the dot/scalar-output Hypothesis mining harness:

* A ``RuntimeTensorView`` captured as a temporary and garbage-collected before
  ``Graph.execute()`` used to be a heap-use-after-free in the executor's
  validator (which dereferenced the freed view to read its name). Views now
  carry a ``liveness_token`` like owning tensors, so a destroyed captured view
  is reported cleanly instead of corrupting memory.

* The scalar-returning 2-arg ``dot(A, B)`` had no overloads accepting view
  operands (only the 3-arg ``dot(result, A, B)`` did), so ``dot(tensor, view)``
  raised ``TypeError`` despite the template handling any tensor type.
"""
from __future__ import annotations

import gc
import itertools

import numpy as np
import pytest

import einsums
import einsums.graph as cg

_ctr = itertools.count()


def _mk(arr):
    t = einsums.create_zero_tensor(f"vl{next(_ctr)}", list(arr.shape), dtype="float64")
    if arr.size:
        np.asarray(t)[...] = arr
    return t


def _view(arr):
    # Non-contiguous view whose logical data equals ``arr`` (parent kept alive
    # by the storage tensor the view is taken from inside this helper only when
    # the caller keeps the return value).
    return _mk(np.ascontiguousarray(arr.T)).permute_view([1, 0])


def test_2arg_dot_accepts_views():
    rng = np.random.default_rng(0)
    A0 = rng.standard_normal((2, 3))
    B0 = rng.standard_normal((2, 3))
    oracle = float(np.sum(A0 * B0))
    A, B = _mk(A0), _mk(B0)
    Av, Bv = _view(A0), _view(B0)
    assert np.isclose(einsums.linalg.dot(A, Bv), oracle)
    assert np.isclose(einsums.linalg.dot(Av, B), oracle)
    assert np.isclose(einsums.linalg.dot(Av, Bv), oracle)


def test_captured_view_kept_alive_executes():
    """A view operand kept alive across execute() computes correctly."""
    rng = np.random.default_rng(1)
    A0 = rng.standard_normal((2, 3))
    B0 = rng.standard_normal((2, 3))
    oracle = float(np.sum(A0 * B0))
    res = _mk(np.zeros(1))
    Av, Bv = _view(A0), _view(B0)  # kept alive in locals
    g = cg.Graph(f"vl{next(_ctr)}")
    with cg.capture(g):
        einsums.linalg.dot(res, Av, Bv)
    g.execute()
    assert np.isclose(np.asarray(res).ravel()[0], oracle)


def test_captured_view_temp_reports_cleanly():
    """A view operand captured as a temporary then dropped before execute() is
    reported as destroyed (clean RuntimeError) -- previously a use-after-free."""
    rng = np.random.default_rng(2)
    A = _mk(rng.standard_normal((2, 3)))
    B = _mk(rng.standard_normal((2, 3)))
    res = _mk(np.zeros(1))
    g = cg.Graph(f"vl{next(_ctr)}")
    with cg.capture(g):
        # permute_view temporaries: no Python reference survives the statement.
        einsums.linalg.dot(res, A.permute_view([1, 0]), B.permute_view([1, 0]))
    gc.collect()
    with pytest.raises(RuntimeError):
        g.execute()
