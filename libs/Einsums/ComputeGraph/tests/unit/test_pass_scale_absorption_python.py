# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_ScaleAbsorption.cpp."""

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


def test_scale_absorption_absorbs_into_einsum():
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_random_tensor("C", [4, 5])

    # einsum has c_pf=0, ab_pf=1 → C = 0*C + 1*A@B = A@B (scale is overwritten).
    C_ref = np.asarray(A) @ np.asarray(B)

    g = cg.Graph("absorb_einsum")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)
    assert g.num_nodes() == 2

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(C, C_ref)


def test_scale_absorption_absorbs_into_permute():
    A = einsums.create_random_tensor("A", [4, 6])
    C = einsums.create_random_tensor("C", [6, 4])

    C_ref = np.asarray(A).T  # 0.0 * (5*C) + 1.0 * permute(A)

    g = cg.Graph("absorb_permute")
    with cg.capture(g):
        einsums.linalg.scale(5.0, C)
        einsums.permute("ji <- ij", C, A, c_pf=0.0, a_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1

    g.execute()
    assert_close(C, C_ref)


def test_scale_absorption_no_absorption_when_beta_not_zero():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_random_tensor("C", [3, 3])

    g = cg.Graph("no_absorb")
    with cg.capture(g):
        einsums.linalg.scale(2.0, C)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=1.0, ab_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)


def test_scale_absorption_no_fusion_when_different_tensors():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_random_tensor("D", [3, 3])

    g = cg.Graph("different_tensors")
    with cg.capture(g):
        einsums.linalg.scale(2.0, D)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)
    assert g.num_nodes() == 2


def test_scale_absorption_does_not_absorb_when_intervening_reader():
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_random_tensor("C", [4, 5])
    D = einsums.create_zero_tensor("D", [4, 5])

    g = cg.Graph("sa_intervening")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", D, A, C, c_pf=0.0, ab_pf=1.0)  # D reads scaled C
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)


def test_scale_absorption_empty_graph():
    g = cg.Graph("sa_empty")
    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)
    assert pass_inst.num_absorbed == 0


def test_scale_absorption_single_node():
    A = einsums.create_random_tensor("A", [3, 3])
    g = cg.Graph("sa_single")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)


def test_scale_absorption_in_pipeline_loop():
    """ScaleAbsorption must fuse correctly inside a Pipeline loop body."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    # Reference: 3 iterations of (scale 0.5; einsum c_pf=0,ab_pf=1) → C = A@B.
    C_ref = np.zeros_like(np.asarray(C))
    for _ in range(3):
        C_ref *= 0.5
        C_ref = np.asarray(A) @ np.asarray(B)

    pipeline = cg.Pipeline("fuse_loop")
    loop_body = pipeline.add_loop("iter", 3, lambda it: it < 2)
    with cg.capture(loop_body):
        einsums.linalg.scale(0.5, C)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    pm = cg.PassManager()
    pm.add(cg.ScaleAbsorption())
    pipeline.apply(pm)
    pipeline.execute()

    assert_close(C, C_ref)


def test_scale_absorption_skips_rank3_batched_gemm():
    """Documented gap: BatchedGemm-captured rank-3 einsum is not handled by ScaleAbsorption."""
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])
    C = einsums.create_random_tensor("C", [3, 6, 4])

    g = cg.Graph("sa_rank3_batched")
    with cg.capture(g):
        einsums.linalg.scale(2.5, C)
        einsums.einsum("ijb <- ikb ; kjb", C, A, B, c_pf=0.0, ab_pf=1.0)

    assert g.num_nodes() == 2
    assert _count_kind(g, "BatchedGemm") == 1

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)
    assert pass_inst.num_absorbed == 0
    assert g.num_nodes() == 2


def test_scale_absorption_rank4_scale_into_permute():
    A = einsums.create_random_tensor("A", [3, 4, 5, 6])
    C = einsums.create_random_tensor("C", [6, 5, 4, 3])

    C_ref = np.transpose(np.asarray(A), (3, 2, 1, 0))  # 1.5 * permute, absorbed

    g = cg.Graph("sa_rank4_permute")
    with cg.capture(g):
        einsums.linalg.scale(1.5, C)
        einsums.permute("lkji <- ijkl", C, A, c_pf=0.0, a_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1

    g.execute()
    assert_close(C, C_ref)
