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


def test_scale_absorption_no_fold_into_accumulating_permute():
    """An accumulating consumer folds only when it exposes live shared params.

    permute bakes its prefactors into the executor closure, so the scale stays.
    """
    A = einsums.create_random_tensor("A", [3, 3])
    C = einsums.create_random_tensor("C", [3, 3])

    A_np = np.asarray(A).copy()
    expected = 2.0 * np.asarray(C).copy() + A_np.T

    g = cg.Graph("no_absorb_accum_permute")
    with cg.capture(g):
        einsums.linalg.scale(2.0, C)
        einsums.permute("j,i <- i,j", C, A, c_pf=1.0, a_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)

    g.execute()
    assert_close(C, expected)


def test_scale_absorption_folds_into_accumulating_einsum():
    """scale(a, C) then C = c_pf*C + ... folds a into the accumulate prefactor."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_random_tensor("C", [3, 3])

    expected = 3.0 * np.asarray(C).copy() + np.asarray(A) @ np.asarray(B)

    g = cg.Graph("fold_accumulator")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=1.0, ab_pf=1.0)

    assert _run(cg.ScaleAbsorption(), g)

    g.execute()
    assert_close(C, expected)


def test_scale_absorption_folds_into_axpby_source():
    """axpby is linear in X, so scaling X equals scaling alpha."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    X = einsums.create_random_tensor("X", [4, 5])
    Y = einsums.create_zero_tensor("Y", [4, 5])

    expected = 2.0 * 3.0 * np.asarray(X).copy()

    g = cg.Graph("fold_axpby_operand")
    with cg.capture(g):
        einsums.linalg.scale(3.0, X)
        einsums.linalg.axpby(2.0, X, 0.0, Y)
        einsums.einsum("ij <- ik ; kj", X, A, B, c_pf=0.0, ab_pf=1.0)

    assert _run(cg.ScaleAbsorption(), g)

    g.execute()
    assert_close(Y, expected)


def test_scale_absorption_folds_into_every_reader():
    """Two readers before the overwrite: the factor goes into both."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_random_tensor("C", [4, 5])
    E1 = einsums.create_random_tensor("E1", [4, 4])
    E2 = einsums.create_random_tensor("E2", [4, 4])
    D1 = einsums.create_zero_tensor("D1", [4, 5])
    D2 = einsums.create_zero_tensor("D2", [4, 5])

    C_np = np.asarray(C).copy()
    d1_expected = 3.0 * (np.asarray(E1) @ C_np)
    d2_expected = 3.0 * (np.asarray(E2) @ C_np)

    g = cg.Graph("fold_all_readers")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", D1, E1, C, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", D2, E2, C, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    assert _run(cg.ScaleAbsorption(), g)

    g.execute()
    assert_close(D1, d1_expected)
    assert_close(D2, d2_expected)


def test_scale_absorption_loop_body_reader_keeps_scale():
    """A loop body reading the scaled tensor is a reader the parent scan must see."""
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    C = einsums.create_random_tensor("C", [4, 4])
    out = einsums.create_zero_tensor("out", [4, 4])

    expected = 3.0 * np.asarray(C).copy()

    g = cg.Graph("sa_loop_body_reader")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
    body = g.add_loop("once", 1, lambda it: it < 1)
    with cg.capture(body):
        einsums.linalg.axpby(1.0, C, 0.0, out)
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    assert not _run(cg.ScaleAbsorption(), g)

    g.execute()
    assert_close(out, expected)


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


def test_scale_absorption_folds_into_sole_einsum_operand():
    # scale(3, C) read by one einsum as an operand, then C overwritten: fold 3
    # into that einsum's ab_prefactor. Runs through pm.run so the program-order
    # validator is active - the fold must declare its compensated read.
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_random_tensor("C", [4, 5])
    D = einsums.create_zero_tensor("D", [4, 5])
    E = einsums.create_random_tensor("E", [4, 4])

    D_ref = 3.0 * (np.asarray(E) @ np.asarray(C))  # evaluated now, before execute overwrites C

    g = cg.Graph("sa_fold_operand")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", D, E, C, c_pf=0.0, ab_pf=1.0)  # sole reader of scaled C
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)  # C overwritten

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1
    g.execute()
    assert_close(D, D_ref)


def test_scale_absorption_keeps_scale_when_result_is_live():
    # scale(3, C) read by an einsum but NOT overwritten afterward: C's scaled
    # value is still observable (in-place scale), so the scale must be kept.
    C = einsums.create_random_tensor("C", [4, 5])
    E = einsums.create_random_tensor("E", [4, 4])
    D = einsums.create_zero_tensor("D", [4, 5])

    g = cg.Graph("sa_live")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", D, E, C, c_pf=0.0, ab_pf=1.0)  # C not overwritten after

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


def test_scale_absorption_rank3_batched_gemm():
    """BatchedGemm with beta=0 overwrites C, so the preceding scale is dead and removed."""
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])
    C = einsums.create_random_tensor("C", [3, 6, 4])

    C_ref = np.einsum("ikb,kjb->ijb", np.asarray(A), np.asarray(B))

    g = cg.Graph("sa_rank3_batched")
    with cg.capture(g):
        einsums.linalg.scale(2.5, C)
        einsums.einsum("ijb <- ikb ; kjb", C, A, B, c_pf=0.0, ab_pf=1.0)

    assert g.num_nodes() == 2
    assert _count_kind(g, "BatchedGemm") == 1

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(C, C_ref)


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
