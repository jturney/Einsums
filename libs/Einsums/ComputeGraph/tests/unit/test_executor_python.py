# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for the bound executor backends.

Mirrors the relevant slices of Executor.cpp on the Python side. Every
test runs the same captured graph through each of the three bound
executors (Sequential, OpenMP, Dataflow) and verifies they produce the
same numerical result. The cross-backend check is what catches
threading/dependency bugs that a single-backend test would miss.

Execution edge cases covered:
  * Empty graph — all backends should accept and run cleanly.
  * Single-node graph — minimal viable execution.
  * Diamond DAG — two parallel branches feeding a common consumer.
  * Wide fan-out — one input feeding many independent consumers.
  * Replay — re-executing the same graph picks up updated tensor data.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


EXECUTORS = [
    pytest.param(cg.SequentialExecutor, id="sequential"),
    pytest.param(cg.OpenMPExecutor, id="openmp"),
    pytest.param(cg.DataflowExecutor, id="dataflow"),
]


# ──────────────────────────────────────────────────────────────────────────
# Empty graph: all three backends accept it without error.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_empty_graph_executes(ExecCls):
    g = cg.Graph(f"empty-{ExecCls.__name__}")
    assert g.num_nodes() == 0
    g.execute(ExecCls())
    assert g.num_nodes() == 0


# ──────────────────────────────────────────────────────────────────────────
# Single-node graph: minimal viable graph for each backend.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_single_node_scale(ExecCls):
    A = einsums.create_random_tensor("A", [4, 4])
    expected = 2.5 * np.asarray(A).copy()

    g = cg.Graph(f"single-{ExecCls.__name__}")
    with cg.capture(g):
        einsums.linalg.scale(2.5, A)

    assert g.num_nodes() == 1
    g.execute(ExecCls())
    np.testing.assert_allclose(np.asarray(A), expected, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# Diamond DAG:
#       A
#      / \
#     B   C       (independent, run in parallel under OpenMP/Dataflow)
#      \ /
#       D
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_diamond_dag(ExecCls):
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_zero_tensor("B", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])
    D = einsums.create_zero_tensor("D", [4, 4])

    A_np = np.asarray(A).copy()
    expected_b = 2.0 * A_np
    expected_c = 0.5 * A_np
    expected_d = expected_b + expected_c   # B + C, both written before D's axpy

    g = cg.Graph(f"diamond-{ExecCls.__name__}")
    with cg.capture(g):
        # B = 2 * A   (axpby into zero-initialized B, beta=0)
        einsums.linalg.axpby(2.0, A, 0.0, B)
        # C = 0.5 * A
        einsums.linalg.axpby(0.5, A, 0.0, C)
        # D = B + 0 (scale to put B's data in D)
        einsums.linalg.axpby(1.0, B, 0.0, D)
        einsums.linalg.axpy(1.0, C, D)

    g.execute(ExecCls())
    np.testing.assert_allclose(np.asarray(B), expected_b, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(C), expected_c, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(D), expected_d, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Wide fan-out: one input feeds K independent consumers.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_wide_fan_out(ExecCls):
    """One source tensor → 5 independent destinations.

    Under the parallel backends these consumers can run concurrently;
    the dependency tracker must not order them spuriously.
    """
    K = 5
    A = einsums.create_random_tensor("A", [3, 3])
    Bs = [einsums.create_zero_tensor(f"B{k}", [3, 3]) for k in range(K)]
    factors = [0.5 + 0.5 * k for k in range(K)]

    g = cg.Graph(f"fanout-{ExecCls.__name__}")
    with cg.capture(g):
        for B, factor in zip(Bs, factors):
            einsums.linalg.axpby(factor, A, 0.0, B)

    assert g.num_nodes() == K
    g.execute(ExecCls())

    A_np = np.asarray(A).copy()
    for B, factor in zip(Bs, factors):
        np.testing.assert_allclose(np.asarray(B), factor * A_np, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# Replay: re-executing the same graph re-runs the work against current
# tensor data. Update inputs, re-execute, get new results.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_replay_picks_up_updated_inputs(ExecCls):
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph(f"replay-{ExecCls.__name__}")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    # First execution: C should equal A @ B.
    g.execute(ExecCls())
    expected_first = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected_first, rtol=1e-5)

    # Mutate B in place and replay; C should now equal A @ B (new values).
    np.asarray(B)[:] = np.random.default_rng(seed=123).standard_normal((3, 3)).astype(np.asarray(B).dtype)
    expected_second = np.asarray(A) @ np.asarray(B)

    g.execute(ExecCls())
    np.testing.assert_allclose(np.asarray(C), expected_second, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Cross-backend agreement: same graph, three executors, identical results.
# ──────────────────────────────────────────────────────────────────────────


def test_three_backends_agree_on_chain():
    """Sequential, OpenMP, Dataflow must all produce the same answer."""
    rng = np.random.default_rng(seed=42)
    A_data = rng.standard_normal((4, 4))
    B_data = rng.standard_normal((4, 4))

    results = []
    for ExecCls in [cg.SequentialExecutor, cg.OpenMPExecutor, cg.DataflowExecutor]:
        A = einsums.create_zero_tensor("A", [4, 4])
        B = einsums.create_zero_tensor("B", [4, 4])
        C = einsums.create_zero_tensor("C", [4, 4])
        D = einsums.create_zero_tensor("D", [4, 4])
        np.asarray(A)[:] = A_data
        np.asarray(B)[:] = B_data

        g = cg.Graph(f"agree-{ExecCls.__name__}")
        with cg.capture(g):
            einsums.linalg.gemm(1.0, A, B, 0.0, C)
            einsums.linalg.scale(0.5, C)
            einsums.linalg.axpby(1.0, C, 0.0, D)

        g.execute(ExecCls())
        results.append(np.asarray(D).copy())

    np.testing.assert_allclose(results[0], results[1], rtol=1e-5)
    np.testing.assert_allclose(results[0], results[2], rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Dataflow memory budget — Phase C.19 binding.
# ──────────────────────────────────────────────────────────────────────────


def test_dataflow_memory_budget_setter_and_getter():
    """``DataflowExecutor.set_memory_budget(N)`` round-trips through the binding."""
    df = cg.DataflowExecutor()
    df.set_memory_budget(1 << 30)  # 1 GiB
    assert df.memory_budget() == (1 << 30)
    df.set_memory_budget(0)
    assert df.memory_budget() == 0


def test_dataflow_with_budget_runs_successfully():
    """The budget shouldn't cap correctness — only scheduling."""
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])

    g = cg.Graph("budget")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    df = cg.DataflowExecutor()
    df.set_memory_budget(1 << 28)  # 256 MiB
    g.execute(df)

    expected = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Python callbacks under parallel executors — GIL regression.
#
# element_transform invokes a Python callable per element. On a parallel
# executor that callback runs on a worker thread and re-acquires the GIL. If
# Graph.execute() holds the GIL while waiting for its workers, the worker can
# never take it → deadlock. execute() must release the GIL (it does, via
# py::call_guard<py::gil_scoped_release>). This test would hang forever on the
# OpenMP/Dataflow backends before that fix.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_python_callback_under_executor_does_not_deadlock(ExecCls):
    A = einsums.create_random_tensor("A", [4, 4])
    before = np.asarray(A).copy()

    g = cg.Graph(f"pycb-{ExecCls.__name__}")
    with cg.capture(g):
        # A Python lambda runs per element on whatever thread the executor uses.
        einsums.linalg.element_transform(A, lambda x: 0.5 * x + 1.0)

    g.execute(ExecCls())
    np.testing.assert_allclose(np.asarray(A), 0.5 * before + 1.0, rtol=1e-5)


def test_python_callback_among_parallel_nodes():
    """A Python-callback node sharing a dependency level with BLAS nodes — the
    exact shape (independent element_transform + gemm) that wedged the parallel
    executors before the GIL release was added to execute()."""
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])
    D = einsums.create_random_tensor("D", [4, 4])

    a0 = np.asarray(A).copy()
    b0 = np.asarray(B).copy()
    d0 = np.asarray(D).copy()

    for ExecCls in (cg.OpenMPExecutor, cg.DataflowExecutor):
        np.asarray(A)[:] = a0
        np.asarray(B)[:] = b0
        np.asarray(D)[:] = d0
        np.asarray(C)[:] = 0.0

        g = cg.Graph(f"mixed-{ExecCls.__name__}")
        with cg.capture(g):
            einsums.linalg.gemm(1.0, A, B, 0.0, C)        # BLAS node
            einsums.linalg.element_transform(D, lambda x: -x)  # Python-callback node (independent)

        g.execute(ExecCls())
        np.testing.assert_allclose(np.asarray(C), a0 @ b0, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(D), -d0, rtol=1e-5)
