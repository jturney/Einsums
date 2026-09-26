# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_ConstantFolding.cpp."""

from __future__ import annotations

import json

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def _count_kind(g, kind):
    return sum(1 for n in json.loads(g.to_json()).get("nodes", []) if n.get("kind") == kind)


def test_constant_folding_user_owned_not_assumed_constant():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("cf_user_owned")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.ConstantFolding()
    assert not _run(pass_inst, g)
    assert pass_inst.num_folded == 0


def test_constant_folding_written_intermediate_is_not_constant():
    g = cg.Graph("cf_intermediate")
    T = g.create_zero_tensor("T", [3, 3], dtype="float64")
    np.asarray(T)[np.arange(3), np.arange(3)] = 1.0

    with cg.capture(g):
        einsums.linalg.scale(2.0, T)

    pass_inst = cg.ConstantFolding()
    assert not _run(pass_inst, g)


def test_constant_folding_empty_graph():
    g = cg.Graph("cf_empty")
    pass_inst = cg.ConstantFolding()
    assert not _run(pass_inst, g)
    assert pass_inst.num_folded == 0


def test_constant_folding_skips_control_flow_nodes():
    A = einsums.create_random_tensor("A", [3, 3])

    g = cg.Graph("cf_loop")
    body = g.add_loop("loop", 3, lambda it: it < 2)
    with cg.capture(body):
        einsums.linalg.scale(0.5, A)

    pass_inst = cg.ConstantFolding()
    _run(pass_inst, g)
    assert pass_inst.num_folded == 0


def test_constant_folding_safe_with_pipeline_loop_body():
    """ConstantFolding (via default PassManager) must be safe across a Pipeline loop body."""
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])

    # Reference: 3 iterations of (C := A@B then scale 0.9). Condition is iter<2 so
    # the body runs at iter 0, 1, 2 = 3 times.
    C_ref = np.zeros_like(np.asarray(C))
    for _ in range(3):
        C_ref = np.asarray(A) @ np.asarray(B)
        C_ref *= 0.9

    pipeline = cg.Pipeline("cf_pipeline")
    loop = pipeline.add_loop("iter", 3, lambda it: it < 2)
    with cg.capture(loop):
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.linalg.scale(0.9, C)

    pm = cg.default_pass_manager()
    pipeline.apply(pm)
    pipeline.execute()

    assert_close(C, C_ref)


def test_constant_folding_rank3_user_owned_tensors_are_not_folded():
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])
    C = einsums.create_zero_tensor("C", [3, 6, 4])

    g = cg.Graph("cf_rank3")
    with cg.capture(g):
        einsums.einsum("ijb <- ikb ; kjb", C, A, B)

    assert _count_kind(g, "Einsum") == 1

    pass_inst = cg.ConstantFolding()
    assert not _run(pass_inst, g)
    assert pass_inst.num_folded == 0


# ──────────────────────────────────────────────────────────────────────────
# Folding something
#
# Until 2026-09-01 every assertion in this file, and every num_folded assertion
# anywhere in the tree, checked for ZERO. The pass could not fire: it counted a
# tensor's own Alloc node as a writer, so an eagerly created graph-owned tensor
# was never constant, and a deferred one is not materialized, which the other
# guard rejects. Between them no tensor could ever qualify.
#
# These are the cases that would have caught that.
# ──────────────────────────────────────────────────────────────────────────


def test_constant_folding_folds_a_contraction_over_constants():
    """Operands the graph owns and no node writes are constant, so the
    contraction over them is evaluated once at pass time."""
    rng = np.random.default_rng(20260901)
    k = rng.standard_normal((3, 3))

    out = einsums.create_zero_tensor("out", [3, 3])

    g = cg.Graph("cf_folds")
    konst = g.create_zero_tensor("konst", [3, 3], intermediate=True, dtype="float64")
    folded = g.create_zero_tensor("folded", [3, 3], intermediate=True, dtype="float64")
    # Filled outside capture: a node writing it would make it non-constant, and
    # leaving it zero would fold an all-zero contraction and prove nothing.
    np.asarray(konst)[...] = k

    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", folded, konst, konst)
        einsums.linalg.axpy(1.0, folded, out)

    pass_inst = cg.ConstantFolding()
    assert _run(pass_inst, g)
    assert pass_inst.num_folded == 1

    g.execute()
    assert_close(out, k @ k)


def test_constant_folding_replays_the_baked_value():
    """A folded node is a no-op on replay and its value still stands.

    The point of folding is that the second execute does not recompute, so this
    is what would break if the no-op were installed without the value having
    been evaluated first.
    """
    rng = np.random.default_rng(20260902)
    k = rng.standard_normal((3, 3))

    out = einsums.create_zero_tensor("out", [3, 3])

    g = cg.Graph("cf_replay")
    konst = g.create_zero_tensor("konst", [3, 3], intermediate=True, dtype="float64")
    folded = g.create_zero_tensor("folded", [3, 3], intermediate=True, dtype="float64")
    np.asarray(konst)[...] = k

    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", folded, konst, konst)
        einsums.linalg.axpy(1.0, folded, out)

    assert _run(cg.ConstantFolding(), g)

    g.execute()
    first = np.asarray(out).copy()
    np.asarray(out)[...] = 0.0
    g.execute()

    assert_close(out, k @ k)
    assert np.array_equal(first, np.asarray(out))


def test_constant_folding_leaves_a_written_operand_alone():
    """The guard that makes the above safe: an operand some node writes is not
    constant, however graph-owned it is."""
    A = einsums.create_random_tensor("A", [3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])

    g = cg.Graph("cf_written")
    scratch = g.create_zero_tensor("scratch", [3, 3], intermediate=True, dtype="float64")

    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", scratch, A, A)   # scratch is WRITTEN here
        einsums.einsum("ij <- ik ; kj", out, scratch, scratch)

    pass_inst = cg.ConstantFolding()
    assert not _run(pass_inst, g)
    assert pass_inst.num_folded == 0


@pytest.mark.parametrize("container", ["loop", "conditional"])
def test_constant_folding_sees_a_write_inside_a_body(container):
    """A tensor a loop body or branch writes is not constant, though the parent lists no writer.

    A Loop or Conditional node lists none of its body's writes, so the pass took T for constant
    and folded the gemm after the loop with T still zero.
    """
    S = einsums.create_zero_tensor("S", [3, 3])
    np.asarray(S)[...] = 1.0
    out = einsums.create_zero_tensor("out", [3, 3])

    g = cg.Graph("cf_body_write")
    T = g.create_zero_tensor("T", [3, 3], intermediate=True, dtype="float64")
    if container == "loop":
        body = g.add_loop("loop", 1, lambda it: False)
    else:
        body, _ = g.add_conditional("cond", lambda: True)
    with cg.capture(body):
        einsums.linalg.axpy(1.0, S, T)
    with cg.capture(g):
        einsums.linalg.gemm(1.0, T, T, 0.0, out)

    g.apply(cg.default_pass_manager())
    g.execute()
    assert_close(out, np.ones((3, 3)) @ np.ones((3, 3)))


def test_constant_folding_leaves_an_output_an_earlier_node_reads():
    """A fold writes its output at pass time, before any node runs.

    `out` is read by the axpy before the folded axpby's turn, so folding the axpby let that read
    see K instead of 10 * out. A caller-owned output is also one the caller may change between
    replays, which a node that never runs again would not follow.
    """
    out = einsums.create_zero_tensor("out", [2, 2])
    np.asarray(out)[...] = 1.0
    other = einsums.create_zero_tensor("other", [2, 2])

    g = cg.Graph("cf_early_read")
    K = g.create_zero_tensor("K", [2, 2], intermediate=True, dtype="float64")
    np.asarray(K)[...] = np.eye(2)
    with cg.capture(g):
        einsums.linalg.scale(10.0, out)
        einsums.linalg.axpy(1.0, out, other)
        einsums.linalg.axpby(1.0, K, 0.0, out)

    g.apply(cg.default_pass_manager())
    g.execute()
    assert_close(other, np.full((2, 2), 10.0))
    assert_close(out, np.eye(2))
