# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""End-to-end coverage for the einsums.graph Python bindings.

Exercises the capture → optimize → execute workflow that Phases C.17–C.19
opened up: graph construction via ``with cg.capture(g):``, optimization via
``cg.default_pass_manager()``, and execution against each of the bound
backends (Sequential, OpenMP, Dataflow). Each test cross-checks numerical
results against numpy as the source of truth.

This file doubles as the canonical worked example for the Python compute-
graph API. New users opening this file should be able to copy-paste an
individual test and have a working starting point.
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
# Smoke: module-level objects exist
# ──────────────────────────────────────────────────────────────────────────


def test_module_surface():
    """Every type promised by the public surface is reachable."""
    assert callable(cg.capture)
    assert callable(cg.default_pass_manager)
    for cls_name in (
        "Graph",
        "CaptureContext",
        "PassManager",
        "Pipeline",
        "Executor",
        "SequentialExecutor",
        "OpenMPExecutor",
        "DataflowExecutor",
        "OpKind",
    ):
        assert hasattr(cg, cls_name), f"einsums.graph.{cls_name} missing"


def test_opkind_enum():
    """OpKind is a real Python enum populated from the C++ scoped enum."""
    assert cg.OpKind.Gemm.name == "Gemm"
    assert cg.OpKind.Einsum.name == "Einsum"
    # All 54 op kinds defined in C++ should round-trip through Python.
    assert len(cg.OpKind.__members__) >= 50


# ──────────────────────────────────────────────────────────────────────────
# Capture: the with-block actually records work into the graph
# ──────────────────────────────────────────────────────────────────────────


def test_capture_records_one_node():
    g = cg.Graph("oneshot")
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [4, 5])
    C = einsums.create_zero_tensor("C", [3, 5])

    assert g.num_nodes() == 0

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    assert g.num_nodes() == 1
    # 3 inputs → 3 registered tensors (A, B, C share the slot table).
    assert g.num_tensors() == 3


def test_capture_records_multiple_nodes():
    g = cg.Graph("multi")
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)   # C = A @ B
        einsums.linalg.gemm(1.0, C, B, 0.0, D)   # D = C @ B

    assert g.num_nodes() == 2
    assert g.num_tensors() == 4


def test_capture_context_is_capturing_flag():
    """CaptureContext.is_capturing() flips inside the with-block."""
    ctx = cg.CaptureContext.current()
    assert ctx.is_capturing() is False

    g = cg.Graph("flag")
    with cg.capture(g):
        assert ctx.is_capturing() is True

    assert ctx.is_capturing() is False


def test_capture_cleanup_on_exception():
    """Even if the user raises inside the with-block, capture must end."""
    g = cg.Graph("bad")
    ctx = cg.CaptureContext.current()

    with pytest.raises(RuntimeError, match="boom"):
        with cg.capture(g):
            raise RuntimeError("boom")

    assert ctx.is_capturing() is False


# ──────────────────────────────────────────────────────────────────────────
# Execute: each backend produces correct results
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_execute_gemm_matches_numpy(ExecCls):
    g = cg.Graph(f"gemm-{ExecCls.__name__}")
    A = einsums.create_random_tensor("A", [4, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    C = einsums.create_zero_tensor("C", [4, 3])

    Anp = np.array(A, copy=True)
    Bnp = np.array(B, copy=True)
    expected = Anp @ Bnp

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    g.execute(ExecCls())
    assert np.allclose(np.array(C), expected, atol=1e-5)


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_execute_chained_ops(ExecCls):
    """C = A @ B,  D = scale(C, 2.0)  in one captured graph."""
    g = cg.Graph(f"chain-{ExecCls.__name__}")
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    Anp = np.array(A, copy=True)
    Bnp = np.array(B, copy=True)
    expected = 2.0 * (Anp @ Bnp)

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)
        einsums.linalg.scale(2.0, C)

    g.execute(ExecCls())
    assert np.allclose(np.array(C), expected, atol=1e-5)


def test_execute_no_arg_overload():
    """g.execute() without an executor uses the default sequential path."""
    g = cg.Graph("default-exec")
    A = einsums.create_random_tensor("A", [2, 2])
    B = einsums.create_random_tensor("B", [2, 2])
    C = einsums.create_zero_tensor("C", [2, 2])

    expected = np.array(A) @ np.array(B)

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    g.execute()
    assert np.allclose(np.array(C), expected, atol=1e-5)


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_replay_same_graph(ExecCls):
    """A captured graph can be replayed multiple times."""
    g = cg.Graph("replay")
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    expected = np.array(A) @ np.array(B)

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    e = ExecCls()
    for _ in range(3):
        np.array(C, copy=False)[:] = 0.0
        g.execute(e)
        assert np.allclose(np.array(C), expected, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Optimize: pass manager round-trip
# ──────────────────────────────────────────────────────────────────────────


def test_default_pass_manager_loads_passes():
    pm = cg.default_pass_manager()
    # Phase C.17 ships ~27 passes in the canonical pipeline; allow some slack
    # so this doesn't break every time someone adds a pass.
    assert pm.size >= 20


def test_apply_pass_manager_preserves_correctness():
    """Optimization must not change the numerical result."""
    g = cg.Graph("optimized")
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])

    expected = np.array(A) @ np.array(B)

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    pm = cg.default_pass_manager()
    g.apply(pm)  # may or may not modify; should not break
    g.execute()
    assert np.allclose(np.array(C), expected, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline: multi-stage workflow
# ──────────────────────────────────────────────────────────────────────────


def test_pipeline_two_stages_share_tensor():
    """Two stages, each capturing into its own Graph, executed via Pipeline."""
    p = cg.Pipeline("two-stage")

    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    Anp = np.array(A, copy=True)
    Bnp = np.array(B, copy=True)
    expected_C = Anp @ Bnp
    expected_D = 2.0 * expected_C

    s1 = p.add_stage("matmul")
    with cg.capture(s1):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    s2 = p.add_stage("scale")
    with cg.capture(s2):
        einsums.linalg.scale(2.0, D)
        einsums.linalg.axpby(1.0, C, 0.0, D)  # D := C, then scale below
        einsums.linalg.scale(2.0, D)

    p.execute()
    assert np.allclose(np.array(C), expected_C, atol=1e-5)
    assert np.allclose(np.array(D), expected_D, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# DataflowExecutor: memory budget knob is reachable
# ──────────────────────────────────────────────────────────────────────────


def test_dataflow_memory_budget_roundtrip():
    df = cg.DataflowExecutor()
    assert df.memory_budget() == 0  # 0 ⇒ unlimited
    df.set_memory_budget(1 << 20)
    assert df.memory_budget() == 1 << 20


def test_executor_names():
    assert cg.SequentialExecutor().name() == "Sequential"
    assert cg.OpenMPExecutor().name() == "OpenMP"
    assert cg.DataflowExecutor().name() == "Dataflow"


# ──────────────────────────────────────────────────────────────────────────
# Workspace: cross-pipeline scratch container
# ──────────────────────────────────────────────────────────────────────────


def test_workspace_lifecycle():
    """Workspace exposes its lifecycle hooks; empty workspace materialize is a no-op."""
    ws = cg.Workspace("scratch")
    assert ws.name == "scratch"
    assert ws.size == 0
    ws.materialize_all()
    assert ws.size == 0


@pytest.mark.parametrize(
    "dtype,expected_class",
    [
        ("float32", "RuntimeTensorF"),
        ("float64", "RuntimeTensorD"),
        ("complex64", "RuntimeTensorC"),
        ("complex128", "RuntimeTensorZ"),
    ],
)
def test_workspace_declare_tensor_dispatches_on_dtype(dtype, expected_class):
    """Member-template dtype dispatcher (C.21) — same Python name, dtype kwarg."""
    ws = cg.Workspace("scratch")
    t = ws.declare_tensor("t", [3, 4], dtype=dtype)
    assert type(t).__name__ == expected_class
    ws.materialize_all()
    assert t.size == 12


def test_workspace_declare_zero_tensor():
    ws = cg.Workspace("zw")
    t = ws.declare_zero_tensor("t", [4, 4], dtype="float64")
    ws.materialize_all()
    assert type(t).__name__ == "RuntimeTensorD"
    assert np.allclose(np.array(t), 0.0)


def test_workspace_declare_random_tensor_default_dtype():
    """Default dtype is float64 (numpy convention)."""
    ws = cg.Workspace("rw")
    t = ws.declare_random_tensor("t", [4, 4])
    ws.materialize_all()
    assert type(t).__name__ == "RuntimeTensorD"
    # random fills the tensor; just check it's not all zero
    assert np.any(np.array(t) != 0.0)


def test_graph_create_tensor_dispatches_on_dtype():
    g = cg.Graph("g")
    A = g.create_tensor("A", [3, 3], dtype="float64")
    B = g.create_zero_tensor("B", [3, 3], dtype="complex128")
    assert type(A).__name__ == "RuntimeTensorD"
    assert type(B).__name__ == "RuntimeTensorZ"
    assert g.num_tensors() == 2


def test_graph_add_loop_invokes_condition_per_iteration():
    """add_loop wires Python callbacks via std::function. The condition
    callback is invoked after each iteration; loop stops when it returns False
    or when max_iterations is reached.
    """
    g = cg.Graph("loop")

    seen = []

    def cond(i):
        seen.append(i)
        return i < 3  # stop once condition() sees iteration 3

    def body():
        # Body callback runs once at construction to record ops into the
        # loop's body subgraph. Empty body = a graph that does nothing per
        # iteration but still drives the iteration counter.
        pass

    g.add_loop("lp", 10, cond, body)
    g.execute()

    assert seen == [0, 1, 2, 3]


def test_graph_add_loop_respects_max_iterations():
    """max_iterations caps the loop even if the condition keeps returning True."""
    g = cg.Graph("loop")

    seen = []

    def cond(i):
        seen.append(i)
        return True  # condition never stops; max_iterations must

    g.add_loop("lp", 4, cond, lambda: None)
    g.execute()

    assert len(seen) == 4

