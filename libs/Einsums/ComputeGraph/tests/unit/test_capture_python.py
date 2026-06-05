# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Capture-context corner cases on the Python side.

Mirrors the relevant slices of GraphCapture.cpp / GraphFeatures.cpp:
  * Capture begins/ends correctly (``is_capturing`` flag).
  * Outside-capture calls go eager rather than recording.
  * Re-using the same ``Graph`` across multiple capture sessions.
  * Capture ends cleanly when the body raises.
  * Mixing different operation kinds inside one capture.
  * Empty captures are valid and produce zero-node graphs.
  * ``num_tensors`` counts unique tensors registered into a graph.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# Capture flag and eager fallthrough
# ──────────────────────────────────────────────────────────────────────────


def test_is_capturing_flag_toggles():
    ctx = cg.CaptureContext.current()
    assert not ctx.is_capturing()
    g = cg.Graph("flag")
    with cg.capture(g):
        assert ctx.is_capturing()
    assert not ctx.is_capturing()


def test_outside_capture_is_eager():
    """Calls outside ``with cg.capture(g):`` execute immediately."""
    A = einsums.create_random_tensor("A", [3, 3])
    before = np.asarray(A).copy()
    einsums.linalg.scale(2.0, A)
    np.testing.assert_allclose(np.asarray(A), 2.0 * before, rtol=1e-12)


def test_no_graph_capture_means_no_nodes_recorded():
    """An untouched Graph has zero nodes/tensors."""
    g = cg.Graph("untouched")
    assert g.num_nodes() == 0
    assert g.num_tensors() == 0


# ──────────────────────────────────────────────────────────────────────────
# Multiple captures into the same graph
# ──────────────────────────────────────────────────────────────────────────


def test_two_captures_into_same_graph_concatenate_nodes():
    """Re-entering ``with cg.capture(g):`` appends to the existing graph."""
    g = cg.Graph("multi-capture")
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    with cg.capture(g):
        einsums.linalg.scale(2.0, A)

    assert g.num_nodes() == 1

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    assert g.num_nodes() == 2


def test_two_captures_into_separate_graphs_dont_cross_contaminate():
    """Each Graph collects only its own captures."""
    g1 = cg.Graph("g1")
    g2 = cg.Graph("g2")
    A = einsums.create_random_tensor("A", [3, 3])

    with cg.capture(g1):
        einsums.linalg.scale(2.0, A)

    with cg.capture(g2):
        einsums.linalg.scale(0.5, A)

    assert g1.num_nodes() == 1
    assert g2.num_nodes() == 1


# ──────────────────────────────────────────────────────────────────────────
# Exception handling: capture ends cleanly on raise
# ──────────────────────────────────────────────────────────────────────────


class _Sentinel(Exception):
    """Distinguished exception type so the test can assert it propagated."""


def test_capture_ends_when_body_raises():
    """The contextmanager's __exit__ runs end_capture even on exception."""
    g = cg.Graph("error-path")
    A = einsums.create_random_tensor("A", [3, 3])
    ctx = cg.CaptureContext.current()

    with pytest.raises(_Sentinel):
        with cg.capture(g):
            einsums.linalg.scale(2.0, A)
            raise _Sentinel("boom")

    # After the with-block (even on raise), capture must be off.
    assert not ctx.is_capturing()
    # The graph still has the work that was recorded before the raise.
    assert g.num_nodes() == 1


# ──────────────────────────────────────────────────────────────────────────
# Tensor accounting
# ──────────────────────────────────────────────────────────────────────────


def test_num_tensors_counts_distinct_inputs():
    g = cg.Graph("count")
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)

    # Three distinct tensors participate in the gemm.
    assert g.num_tensors() == 3


def test_num_tensors_dedupes_repeated_inputs():
    """Using the same tensor in two argument slots should register it once."""
    g = cg.Graph("dedupe")
    x = einsums.create_random_tensor("x", [4])
    A = einsums.create_zero_tensor("A", [4, 4])

    with cg.capture(g):
        # ger writes A += alpha * x * y^T; reuse x as both X and Y so the
        # capture should register two distinct tensor slots (x and A),
        # not three.
        einsums.linalg.ger(1.0, x, x, A)

    assert g.num_tensors() == 2


# ──────────────────────────────────────────────────────────────────────────
# Mixing operation kinds in one capture
# ──────────────────────────────────────────────────────────────────────────


def test_mixed_kinds_in_one_capture():
    """A single capture session can record any combination of bound ops."""
    g = cg.Graph("mixed")
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)
        einsums.linalg.scale(0.5, C)
        einsums.linalg.axpy(1.0, A, D)         # D += A
        einsums.permute("ij <- ji", D, A)      # D += A.T (accumulating into prev D)

    assert g.num_nodes() == 4

    expected_c = 0.5 * (np.asarray(A) @ np.asarray(B))
    # D started at zero, then += A, then permute("ij <- ji", D, A) writes
    # D = c_pf * D + a_pf * permute(A) with default c_pf=0, a_pf=1 → D = A.T.
    expected_d = np.asarray(A).T

    g.execute()
    np.testing.assert_allclose(np.asarray(C), expected_c, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(D), expected_d, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Capture-then-execute consistency
# ──────────────────────────────────────────────────────────────────────────


def test_captured_result_matches_eager():
    """Same operation, captured-then-executed vs eager — bit-for-bit (within tolerance)."""
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])

    # Eager
    C_eager = einsums.create_zero_tensor("C", [4, 4])
    einsums.linalg.gemm(1.0, A, B, 0.0, C_eager)

    # Captured
    C_capt = einsums.create_zero_tensor("C", [4, 4])
    g = cg.Graph("vs-eager")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C_capt)
    g.execute()

    np.testing.assert_allclose(np.asarray(C_eager), np.asarray(C_capt), rtol=1e-5)
