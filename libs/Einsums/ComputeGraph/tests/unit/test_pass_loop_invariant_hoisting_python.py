# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_LoopInvariantHoisting.cpp."""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_lih_empty_loop_body():
    g = cg.Graph("lih_empty")
    g.add_loop("loop", 3, lambda it: it < 2)

    pass_inst = cg.LoopInvariantHoisting()
    assert not _run(pass_inst, g)
    assert pass_inst.num_hoisted == 0


def test_lih_nothing_to_hoist():
    value = einsums.create_zero_tensor("value", [1])

    g = cg.Graph("no_hoist")
    body = g.add_loop("loop", 5, lambda it: it < 4)
    with cg.capture(body):
        einsums.linalg.scale(0.5, value)

    pass_inst = cg.LoopInvariantHoisting()
    assert not _run(pass_inst, g)


def test_lih_hoists_invariant_node():
    # C = A·B is invariant and C's only writer, so it hoists; the body then
    # accumulates the invariant C into acc each iteration.
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    acc = einsums.create_zero_tensor("acc", [3, 3])

    g = cg.Graph("hoist_test")
    body = g.add_loop("loop", 5, lambda it: it < 4)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", C, A, B)  # invariant, single-writer of C
        einsums.linalg.axpy(1.0, C, acc)           # acc += C (reads C)

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 1


def test_lih_does_not_hoist_overwritten_output():
    # C = A·B then C *= 0.9 in place every iteration. Hoisting the einsum
    # would drop the per-iteration reset and the scale would compound, so
    # the single-writer guard must refuse the hoist.
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("no_hoist_overwrite")
    body = g.add_loop("loop", 5, lambda it: it + 1 < 5)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", C, A, B)  # C = A·B
        einsums.linalg.scale(0.9, C)               # second writer of C

    pass_inst = cg.LoopInvariantHoisting()
    modified = _run(pass_inst, g)
    assert not modified
    assert pass_inst.num_hoisted == 0


def test_lih_dependency_chain_partially_hoists():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_random_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    g = cg.Graph("lih_dep_chain")
    body = g.add_loop("loop", 3, lambda it: it < 2)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", D, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.linalg.scale(0.5, C)

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 1


def test_lih_all_nodes_invariant():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    g = cg.Graph("lih_all_invariant")
    body = g.add_loop("loop", 3, lambda it: it < 2)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", D, A, B, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 2


def test_lih_rank3_batched_gemm_hoists():
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])
    C = einsums.create_zero_tensor("C", [3, 6, 4])
    D = einsums.create_random_tensor("D", [3, 6, 4])

    g = cg.Graph("lih_rank3")
    body = g.add_loop("loop", 4, lambda it: it < 3)
    with cg.capture(body):
        einsums.einsum("ijb <- ikb ; kjb", C, A, B)  # invariant
        einsums.linalg.scale(0.9, D)  # not invariant, writes D

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 1


# ──────────────────────────────────────────────────────────────────────────
# Execution after LIH, verifies that the rewritten graph still runs and
# produces correct values. The previous suite only checked num_hoisted;
# without executing the graph the tensor-id rewrite and loop-input wiring
# couldn't be exercised.
# ──────────────────────────────────────────────────────────────────────────


def test_lih_does_not_corrupt_body_when_nothing_is_invariant():
    """Regression: an early bug moved every body node into a temporary
    ``remaining`` vector, then short-circuited on ``hoisted.empty()``
    without putting them back, turning the loop body into a no-op."""
    value = einsums.create_zero_tensor("value", [1])
    np.asarray(value)[0] = 1.0

    g = cg.Graph("preserve-body")
    body = g.add_loop("loop", 5, lambda it: it < 2)
    with cg.capture(body):
        einsums.linalg.scale(2.0, value)   # self-modifying, never invariant

    pass_inst = cg.LoopInvariantHoisting()
    _run(pass_inst, g)
    assert pass_inst.num_hoisted == 0

    g.execute()
    # cond returns True for it=0, True for it=1, False for it=2 →
    # body runs 3 times → value = 1 * 2^3 = 8.
    assert float(np.asarray(value)[0]) == pytest.approx(8.0)


def test_lih_executes_correctly_after_hoist():
    """Hoist a real invariant gemm and confirm the rewritten graph executes
    end-to-end. Exercises the tensor-id remap from body to parent graph
    and the explicit loop-input edge that orders the hoisted node first."""
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [4, 5])
    C = einsums.create_zero_tensor("C", [3, 5])
    accum = einsums.create_zero_tensor("accum", [3, 5])

    g = cg.Graph("hoist-exec")
    body = g.add_loop("loop", 3, lambda it: it < 2)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", C, A, B)   # invariant, A, B never change
        einsums.linalg.axpy(1.0, C, accum)         # self-modifying, stays in body

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 1

    g.execute()
    expected_C = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected_C, rtol=1e-5)
    # cond returns True for it=0, True for it=1, False for it=2 →
    # body runs 3 times → accum = 3 * C.
    np.testing.assert_allclose(np.asarray(accum), 3.0 * expected_C, rtol=1e-5)
