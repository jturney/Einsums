# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Storage-order assignment, from Python.

A capability that is not reachable from Python is not shipped, and this module's
history is four separate misses of exactly that kind - each a piece of C++ that
worked and had no spelling anyone writing an example could use. The C++ cases
prove the arithmetic; these prove a person can get at it, which means
constructing the pass, running it, reading its counters, and getting the same
numbers out the other side.

The chain is the one the C++ file describes:

    W[i,j,x] = A[i,k]   B[k,x,j]
    R[i,x,y] = W[i,j,x] D[j,y]

captured with W's contracted letter between its free ones, which costs a copy of
W at the consumer and a copy of B at the producer. Storing W as (i,x,j) removes
both.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close

I, J, X, K, Y = 6, 5, 7, 3, 4


def _tensor(name, array, dtype):
    t = einsums.create_zero_tensor(name, list(array.shape), dtype=dtype)
    np.asarray(t)[...] = array.astype(dtype)
    return t


def _operands(dtype, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((I, K)), rng.standard_normal((K, X, J)), rng.standard_normal((J, Y)))


def _build(graph, a, b, d, dtype, w_letters, w_dims, w_user=False):
    A = _tensor("A", a, dtype)
    B = _tensor("B", b, dtype)
    D = _tensor("D", d, dtype)
    R = _tensor("R", np.zeros((I, X, Y)), dtype)
    if w_user:
        W = _tensor("W", np.zeros(w_dims), dtype)
    else:
        W = graph.declare_tensor("W", list(w_dims), intermediate=True, dtype=dtype)
    with cg.capture(graph):
        einsums.einsum(f"{w_letters} <- i,k ; k,x,j", W, A, B)
        einsums.einsum(f"i,x,y <- {w_letters} ; j,y", R, W, D)
    # Held so the pool outlives the graph; a capture adopts its operands but the
    # locals are what keep the fixture readable.
    return R, (A, B, D, W)


def _expected(a, b, d):
    w = np.einsum("ik,kxj->ijx", a, b)
    return np.einsum("ijx,jy->ixy", w, d)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_layout_assignment_keeps_the_answer(dtype):
    a, b, d = _operands(dtype)

    graph = cg.Graph("layout")
    R, _pool = _build(graph, a, b, d, dtype, "i,j,x", (I, J, X))

    pass_ = cg.LayoutAssignment()
    pm = cg.PassManager()
    pm.add(pass_)
    pm.populate_tuning()  # Materialization has no Python spelling of its own; the phase does.
    assert pm.run(graph), "the pass reported no change on a chain it should have re-laid out"
    graph.execute()

    assert pass_.num_relaid_out == 1
    # Both copies, not one. A pass that only looked at the consumer would find the
    # same order and report half the saving.
    assert pass_.num_copies_removed == 2
    assert pass_.estimated_saving_us > 0.0

    assert_close(np.asarray(R), _expected(a, b, d).astype(dtype), dtype=dtype)


def test_the_report_says_what_moved():
    a, b, d = _operands("float64")

    graph = cg.Graph("layout-report")
    _R, _pool = _build(graph, a, b, d, "float64", "i,j,x", (I, J, X))

    pm = cg.PassManager()
    pm.add(cg.LayoutAssignment())
    pm.run(graph)

    report = pm.explain()
    assert "LayoutAssignment" in report
    assert "structural-algebraic" in report


def test_a_caller_owned_tensor_is_never_moved():
    a, b, d = _operands("float64")

    graph = cg.Graph("layout-user")
    _R, _pool = _build(graph, a, b, d, "float64", "i,j,x", (I, J, X), w_user=True)

    pass_ = cg.LayoutAssignment()
    pm = cg.PassManager()
    pm.add(pass_)
    pm.run(graph)

    # W's axis order is part of what the caller asked for, so the copies stay.
    assert pass_.num_relaid_out == 0


def test_the_order_it_picks_is_the_one_a_hand_capture_would_use():
    a, b, d = _operands("float64")

    # Captured directly in the order the pass chooses. If the two disagree the
    # model is describing something other than the arithmetic it claims to.
    graph = cg.Graph("layout-by-hand")
    R, _pool = _build(graph, a, b, d, "float64", "i,x,j", (I, X, J))

    pass_ = cg.LayoutAssignment()
    pm = cg.PassManager()
    pm.add(pass_)
    pm.populate_tuning()
    pm.run(graph)
    graph.execute()

    assert pass_.num_relaid_out == 0, "the hand-written order was already the one it wants"
    assert_close(np.asarray(R), _expected(a, b, d), dtype="float64")
