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

    assert _count_kind(g, "BatchedGemm") == 1

    pass_inst = cg.ConstantFolding()
    assert not _run(pass_inst, g)
    assert pass_inst.num_folded == 0
