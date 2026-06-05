# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for ``cg::Graph::add_loop`` bindings.

Mirrors the loop slices of ControlFlow.cpp on the Python side: exercises
both the Graph-returning and body-fn forms of ``add_loop``, verifies the
condition lambda receives the (0-based) iteration index, and confirms
``max_iterations`` acts as a safety cap when the condition never returns
False.

Loop semantics from Graph.cpp::

    for (size_t iter = 0; iter < max_iterations; iter++) {
        body->execute();
        if (condition && !condition(iter)) break;
    }

i.e. body always runs first, then the condition is evaluated with the
just-completed iteration index. ``False`` from the condition breaks
immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


# ──────────────────────────────────────────────────────────────────────────
# API shape
# ──────────────────────────────────────────────────────────────────────────


def test_add_loop_two_arg_returns_body_graph():
    """The three-arg form returns a reference to the loop body subgraph."""
    g = cg.Graph("api")
    body = g.add_loop("noop", 1, lambda it: False)
    assert isinstance(body, cg.Graph)


def test_add_loop_four_arg_returns_none():
    """The four-arg (body-fn) form has no useful return value."""
    g = cg.Graph("api")
    value = einsums.create_random_tensor("v", [1])
    ret = g.add_loop("noop", 1, lambda it: False,
                     lambda: einsums.linalg.scale(1.0, value))
    assert ret is None


# ──────────────────────────────────────────────────────────────────────────
# Execution semantics
# ──────────────────────────────────────────────────────────────────────────


def test_loop_body_captured_via_with_converges():
    """``with cg.capture(body):`` records ops that run each iteration."""
    g = cg.Graph("halve")
    value = einsums.create_random_tensor("value", [1])
    np.asarray(value)[0] = 100.0

    body = g.add_loop("halving", 1000,
                      lambda it: float(np.asarray(value)[0]) >= 1.0)
    with cg.capture(body):
        einsums.linalg.scale(0.5, value)

    g.execute()
    # iter 0..6 run the body (7 halvings); cond(6) sees 0.78125 < 1.0 → break.
    assert float(np.asarray(value)[0]) == pytest.approx(100.0 * 0.5 ** 7, rel=1e-12)


def test_loop_body_fn_variant_converges():
    """The body-fn overload captures via the passed lambda — no explicit ``with``."""
    g = cg.Graph("halve-fn")
    value = einsums.create_random_tensor("value", [1])
    np.asarray(value)[0] = 16.0

    g.add_loop("halving", 100,
               lambda it: float(np.asarray(value)[0]) >= 1.0,
               lambda: einsums.linalg.scale(0.5, value))

    g.execute()
    # 16 → 8 → 4 → 2 → 1 (≥ 1, continue) → 0.5 (< 1, stop). Five body runs.
    assert float(np.asarray(value)[0]) == pytest.approx(0.5, rel=1e-12)


def test_max_iterations_caps_runaway_condition():
    """``max_iterations`` bounds execution even when the condition stays True."""
    g = cg.Graph("capped")
    value = einsums.create_random_tensor("value", [1])
    np.asarray(value)[0] = 100.0

    body = g.add_loop("never_stops", 10, lambda it: True)
    with cg.capture(body):
        einsums.linalg.scale(0.9, value)

    g.execute()
    assert float(np.asarray(value)[0]) == pytest.approx(100.0 * 0.9 ** 10, rel=1e-10)


def test_max_iterations_zero_runs_no_body():
    """``max_iterations=0`` means the body executes zero times."""
    g = cg.Graph("zero")
    value = einsums.create_random_tensor("value", [1])
    before = float(np.asarray(value)[0])

    seen = []
    body = g.add_loop("none", 0, lambda it: seen.append(it) or True)
    with cg.capture(body):
        einsums.linalg.scale(0.0, value)  # would zero the tensor if it ran

    g.execute()
    assert seen == []
    assert float(np.asarray(value)[0]) == pytest.approx(before, rel=1e-12)


def test_condition_receives_zero_based_iteration_index():
    """The condition is called once per iteration with the just-completed index."""
    g = cg.Graph("counter")
    value = einsums.create_random_tensor("value", [1])

    seen: list[int] = []

    def cond(it: int) -> bool:
        seen.append(it)
        return it < 2

    body = g.add_loop("count", 5, cond)
    with cg.capture(body):
        einsums.linalg.scale(1.0, value)

    g.execute()
    # iter 0: body, cond(0)→True; iter 1: body, cond(1)→True; iter 2: body, cond(2)→False.
    assert seen == [0, 1, 2]


def test_condition_false_on_first_call_runs_body_once():
    """Returning False from cond(0) ends after exactly one body run."""
    g = cg.Graph("once")
    value = einsums.create_random_tensor("value", [1])
    np.asarray(value)[0] = 10.0

    body = g.add_loop("once", 100, lambda it: False)
    with cg.capture(body):
        einsums.linalg.scale(0.5, value)

    g.execute()
    assert float(np.asarray(value)[0]) == pytest.approx(5.0, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# Composition: loops alongside other captured ops
# ──────────────────────────────────────────────────────────────────────────


def test_loop_with_pre_and_post_operations():
    """Pre-loop captures run first, then the loop, then post-loop captures."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    A_np = np.asarray(A).copy()
    B_np = np.asarray(B).copy()

    g = cg.Graph("pre_loop_post")

    # Pre-loop: C = A @ B.
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    # Loop: halve C three times (cond it<2 → 3 body runs).
    body = g.add_loop("halve", 10, lambda it: it < 2)
    with cg.capture(body):
        einsums.linalg.scale(0.5, C)

    # Post-loop: scale by 8 (0.5^3 * 8 = 1.0 → restores A@B).
    with cg.capture(g):
        einsums.linalg.scale(8.0, C)

    g.execute()
    assert_close(C, A_np @ B_np)


def test_two_independent_loops_in_same_graph():
    """Sequential loops each iterate to their own condition."""
    g = cg.Graph("two-loops")
    a = einsums.create_random_tensor("a", [1])
    b = einsums.create_random_tensor("b", [1])
    np.asarray(a)[0] = 100.0
    np.asarray(b)[0] = 100.0

    body_a = g.add_loop("halve_a", 50,
                        lambda it: float(np.asarray(a)[0]) >= 1.0)
    with cg.capture(body_a):
        einsums.linalg.scale(0.5, a)

    body_b = g.add_loop("halve_b", 50,
                        lambda it: float(np.asarray(b)[0]) >= 10.0)
    with cg.capture(body_b):
        einsums.linalg.scale(0.5, b)

    g.execute()
    # a halves until < 1.0  → 7 runs → 100 * 0.5^7 ≈ 0.78125
    # b halves until < 10.0 → 4 runs → 100 * 0.5^4 = 6.25
    assert float(np.asarray(a)[0]) == pytest.approx(100.0 * 0.5 ** 7, rel=1e-12)
    assert float(np.asarray(b)[0]) == pytest.approx(100.0 * 0.5 ** 4, rel=1e-12)


def test_loop_graph_can_be_re_executed():
    """A captured loop graph replays cleanly when execute() is called multiple times."""
    g = cg.Graph("replay")
    value = einsums.create_random_tensor("value", [1])

    body = g.add_loop("halve", 50,
                      lambda it: float(np.asarray(value)[0]) >= 1.0)
    with cg.capture(body):
        einsums.linalg.scale(0.5, value)

    expected = 100.0 * 0.5 ** 7
    for _ in range(3):
        np.asarray(value)[0] = 100.0
        g.execute()
        assert float(np.asarray(value)[0]) == pytest.approx(expected, rel=1e-12)
