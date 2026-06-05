# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Mixed-operation graph coverage on the Python side.

Mirrors the C++ ``OperationsCoverage.cpp`` chain tests — multiple
operations recorded into one graph, executed once, and compared against
a reference computed with the same ops applied eagerly. Verifies that
data dependencies between captured nodes resolve correctly when
optimization passes do (or don't) reorder them.
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


@pytest.mark.parametrize("ExecCls", EXECUTORS)
def test_chain_gemm_scale_axpy(ExecCls):
    """``D = A + 0.5 * (A @ B)`` via gemm + scale + axpy in one graph.

    Eager-mode reference uses numpy; captured-then-executed result must
    match across all three bound executor backends.
    """
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])
    D = einsums.create_zero_tensor("D", [4, 4])

    # Reference: applied eagerly with numpy semantics.
    A_np = np.asarray(A).copy()
    B_np = np.asarray(B).copy()
    expected = 0.5 * (A_np @ B_np) + A_np

    g = cg.Graph("chain")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)   # C = A @ B
        einsums.linalg.scale(0.5, C)              # C *= 0.5
        einsums.linalg.axpy(1.0, C, D)            # D += C   (D started zero)
        einsums.linalg.axpy(1.0, A, D)            # D += A

    assert g.num_nodes() == 4

    g.execute(ExecCls())

    np.testing.assert_allclose(np.asarray(D), expected, rtol=1e-5)


def test_chain_ger_then_gemm():
    """``C = (x outer y) @ B`` — rank-1 update feeding a gemm."""
    x = einsums.create_random_tensor("x", [4])
    y = einsums.create_random_tensor("y", [4])
    A = einsums.create_zero_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])

    expected = np.outer(np.asarray(x), np.asarray(y)) @ np.asarray(B)

    g = cg.Graph("ger_gemm")
    with cg.capture(g):
        einsums.linalg.ger(1.0, x, y, A)        # A = x ⊗ y
        einsums.linalg.gemm(1.0, A, B, 0.0, C)  # C = A @ B
    g.execute()

    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_chain_axpby_then_gemv():
    """``y = (alpha*x + beta*y_old) doesn't apply — axpby into Y, then gemv reads Y``."""
    A = einsums.create_random_tensor("A", [3, 4])
    x = einsums.create_random_tensor("x", [4])
    y_init = einsums.create_random_tensor("y_init", [4])
    y_target = einsums.create_zero_tensor("y_target", [3])

    A_np = np.asarray(A).copy()
    x_np = np.asarray(x).copy()
    y_np = np.asarray(y_init).copy()

    # axpby: y_init = 2*x + 0.5*y_init  → updated y_init becomes the
    # gemv input vector.
    expected_y_init = 2.0 * x_np + 0.5 * y_np
    expected_y_target = A_np @ expected_y_init

    g = cg.Graph("axpby_gemv")
    with cg.capture(g):
        einsums.linalg.axpby(2.0, x, 0.5, y_init)
        einsums.linalg.gemv(1.0, A, y_init, 0.0, y_target)
    g.execute()

    np.testing.assert_allclose(np.asarray(y_init), expected_y_init, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(y_target), expected_y_target, rtol=1e-5)


def test_chain_passes_through_default_pass_manager():
    """The default pass manager preserves correctness on a multi-op chain."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    A_np = np.asarray(A).copy()
    B_np = np.asarray(B).copy()
    expected = A_np @ B_np + A_np

    g = cg.Graph("passes")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)
        einsums.linalg.axpy(1.0, A, C)

    pm = cg.default_pass_manager()
    g.apply(pm)
    g.execute()

    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_replay_chain_produces_same_result_each_run():
    """Re-executing the same captured chain re-runs the work."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    A_np = np.asarray(A).copy()
    B_np = np.asarray(B).copy()
    expected_one_run = A_np @ B_np

    g = cg.Graph("replay")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    g.execute()
    np.testing.assert_allclose(np.asarray(C), expected_one_run, rtol=1e-5)

    # Reset C and re-execute the same graph — should land on the same result.
    einsums.linalg.scale(0.0, C)
    assert np.all(np.asarray(C) == 0)

    g.execute()
    np.testing.assert_allclose(np.asarray(C), expected_one_run, rtol=1e-5)
