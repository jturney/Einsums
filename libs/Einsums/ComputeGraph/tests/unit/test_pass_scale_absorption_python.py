# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_ScaleAbsorption.cpp, over every dtype.

The pass moves a REAL scale factor onto the prefactors of the nodes that read
the scaled tensor, so on a complex dtype the factor stays real while the data,
and the prefactors it is multiplied into, are complex. A complex factor is left
alone by design; the complex-only tests at the end pin that.
"""

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


def _pf(dtype, re, im):
    """A prefactor for ``dtype``: both parts for a complex dtype, the real part otherwise."""
    return complex(re, im) if np.dtype(dtype).kind == "c" else re


def _count_kind(g, kind):
    return sum(1 for n in json.loads(g.to_json()).get("nodes", []) if n.get("kind") == kind)


def test_scale_absorption_absorbs_into_einsum(dtype):
    A = einsums.create_random_tensor("A", [4, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 5], dtype=dtype)
    C = einsums.create_random_tensor("C", [4, 5], dtype=dtype)

    # einsum has c_pf=0 → C = 0*C + ab*A@B (scale is overwritten).
    ab = _pf(dtype, 1.0, 0.5)
    C_ref = ab * (np.asarray(A) @ np.asarray(B))

    g = cg.Graph("absorb_einsum")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=ab)
    assert g.num_nodes() == 2

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(C, C_ref)


def test_scale_absorption_absorbs_into_permute(dtype):
    A = einsums.create_random_tensor("A", [4, 6], dtype=dtype)
    C = einsums.create_random_tensor("C", [6, 4], dtype=dtype)

    a_pf = _pf(dtype, 1.0, -0.5)
    C_ref = a_pf * np.asarray(A).T  # 0.0 * (5*C) + a_pf * permute(A)

    g = cg.Graph("absorb_permute")
    with cg.capture(g):
        einsums.linalg.scale(5.0, C)
        einsums.permute("ji <- ij", C, A, c_pf=0.0, a_pf=a_pf)

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1

    g.execute()
    assert_close(C, C_ref)


def test_scale_absorption_no_fold_into_accumulating_permute(dtype):
    """An accumulating consumer folds only when it exposes live shared params.

    permute bakes its prefactors into the executor closure, so the scale stays.
    """
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    C = einsums.create_random_tensor("C", [3, 3], dtype=dtype)

    c_pf, a_pf = _pf(dtype, 1.0, 0.25), _pf(dtype, 1.0, -0.5)
    A_np = np.asarray(A).copy()
    expected = c_pf * 2.0 * np.asarray(C).copy() + a_pf * A_np.T

    g = cg.Graph("no_absorb_accum_permute")
    with cg.capture(g):
        einsums.linalg.scale(2.0, C)
        einsums.permute("j,i <- i,j", C, A, c_pf=c_pf, a_pf=a_pf)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)

    g.execute()
    assert_close(C, expected)


def test_scale_absorption_folds_into_accumulating_einsum(dtype):
    """scale(a, C) then C = c_pf*C + ... folds a into the accumulate prefactor."""
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 3], dtype=dtype)
    C = einsums.create_random_tensor("C", [3, 3], dtype=dtype)

    c_pf, ab = _pf(dtype, 1.0, 0.25), _pf(dtype, 1.0, -0.5)
    expected = c_pf * 3.0 * np.asarray(C).copy() + ab * (np.asarray(A) @ np.asarray(B))

    g = cg.Graph("fold_accumulator")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=c_pf, ab_pf=ab)

    assert _run(cg.ScaleAbsorption(), g)

    g.execute()
    assert_close(C, expected)


def test_scale_absorption_folds_into_axpby_source(dtype):
    """axpby is linear in X, so scaling X equals scaling alpha."""
    A = einsums.create_random_tensor("A", [4, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 5], dtype=dtype)
    X = einsums.create_random_tensor("X", [4, 5], dtype=dtype)
    Y = einsums.create_zero_tensor("Y", [4, 5], dtype=dtype)

    alpha = _pf(dtype, 2.0, 0.5)
    expected = alpha * 3.0 * np.asarray(X).copy()

    g = cg.Graph("fold_axpby_operand")
    with cg.capture(g):
        einsums.linalg.scale(3.0, X)
        einsums.linalg.axpby(alpha, X, 0.0, Y)
        einsums.einsum("ij <- ik ; kj", X, A, B, c_pf=0.0, ab_pf=1.0)

    assert _run(cg.ScaleAbsorption(), g)

    g.execute()
    assert_close(Y, expected)


def test_scale_absorption_folds_into_every_reader(dtype):
    """Two readers before the overwrite: the factor goes into both."""
    A = einsums.create_random_tensor("A", [4, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 5], dtype=dtype)
    C = einsums.create_random_tensor("C", [4, 5], dtype=dtype)
    E1 = einsums.create_random_tensor("E1", [4, 4], dtype=dtype)
    E2 = einsums.create_random_tensor("E2", [4, 4], dtype=dtype)
    D1 = einsums.create_zero_tensor("D1", [4, 5], dtype=dtype)
    D2 = einsums.create_zero_tensor("D2", [4, 5], dtype=dtype)

    ab1, ab2 = _pf(dtype, 1.0, 0.5), _pf(dtype, -0.5, 1.0)
    C_np = np.asarray(C).copy()
    d1_expected = 3.0 * ab1 * (np.asarray(E1) @ C_np)
    d2_expected = 3.0 * ab2 * (np.asarray(E2) @ C_np)

    g = cg.Graph("fold_all_readers")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", D1, E1, C, c_pf=0.0, ab_pf=ab1)
        einsums.einsum("ij <- ik ; kj", D2, E2, C, c_pf=0.0, ab_pf=ab2)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    assert _run(cg.ScaleAbsorption(), g)

    g.execute()
    assert_close(D1, d1_expected)
    assert_close(D2, d2_expected)


def test_scale_absorption_loop_body_reader_keeps_scale(dtype):
    """A loop body reading the scaled tensor is a reader the parent scan must see."""
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [4, 4], dtype=dtype)
    C = einsums.create_random_tensor("C", [4, 4], dtype=dtype)
    out = einsums.create_zero_tensor("out", [4, 4], dtype=dtype)

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


def test_scale_absorption_no_fusion_when_different_tensors(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 3], dtype=dtype)
    C = einsums.create_zero_tensor("C", [3, 3], dtype=dtype)
    D = einsums.create_random_tensor("D", [3, 3], dtype=dtype)

    g = cg.Graph("different_tensors")
    with cg.capture(g):
        einsums.linalg.scale(2.0, D)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)
    assert g.num_nodes() == 2


def test_scale_absorption_folds_into_sole_einsum_operand(dtype):
    # scale(3, C) read by one einsum as an operand, then C overwritten: fold 3
    # into that einsum's ab_prefactor. Runs through pm.run so the program-order
    # validator is active - the fold must declare its compensated read.
    A = einsums.create_random_tensor("A", [4, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 5], dtype=dtype)
    C = einsums.create_random_tensor("C", [4, 5], dtype=dtype)
    D = einsums.create_zero_tensor("D", [4, 5], dtype=dtype)
    E = einsums.create_random_tensor("E", [4, 4], dtype=dtype)

    ab = _pf(dtype, 1.0, -0.75)
    D_ref = 3.0 * ab * (np.asarray(E) @ np.asarray(C))  # evaluated now, before execute overwrites C

    g = cg.Graph("sa_fold_operand")
    with cg.capture(g):
        einsums.linalg.scale(3.0, C)
        einsums.einsum("ij <- ik ; kj", D, E, C, c_pf=0.0, ab_pf=ab)  # sole reader of scaled C
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)  # C overwritten

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1
    g.execute()
    assert_close(D, D_ref)


def test_scale_absorption_keeps_scale_when_result_is_live(dtype):
    # scale(3, C) read by an einsum but NOT overwritten afterward: C's scaled
    # value is still observable (in-place scale), so the scale must be kept.
    C = einsums.create_random_tensor("C", [4, 5], dtype=dtype)
    E = einsums.create_random_tensor("E", [4, 4], dtype=dtype)
    D = einsums.create_zero_tensor("D", [4, 5], dtype=dtype)

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


def test_scale_absorption_single_node(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    g = cg.Graph("sa_single")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)


def test_scale_absorption_in_pipeline_loop(dtype):
    """ScaleAbsorption must fuse correctly inside a Pipeline loop body."""
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 3], dtype=dtype)
    C = einsums.create_zero_tensor("C", [3, 3], dtype=dtype)

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


def test_scale_absorption_rank3_batched_gemm(dtype):
    """A batched einsum with beta=0 overwrites C, so the preceding scale is dead and removed."""
    A = einsums.create_random_tensor("A", [3, 5, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [5, 6, 4], dtype=dtype)
    C = einsums.create_random_tensor("C", [3, 6, 4], dtype=dtype)

    ab = _pf(dtype, 1.0, 0.5)
    C_ref = ab * np.einsum("ikb,kjb->ijb", np.asarray(A), np.asarray(B))

    g = cg.Graph("sa_rank3_batched")
    with cg.capture(g):
        einsums.linalg.scale(2.5, C)
        einsums.einsum("ijb <- ikb ; kjb", C, A, B, c_pf=0.0, ab_pf=ab)

    assert g.num_nodes() == 2
    assert _count_kind(g, "Einsum") == 1

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(C, C_ref)


def test_scale_absorption_rank4_scale_into_permute(dtype):
    A = einsums.create_random_tensor("A", [3, 4, 5, 6], dtype=dtype)
    C = einsums.create_random_tensor("C", [6, 5, 4, 3], dtype=dtype)

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


# ──────────────────────────────────────────────────────────────────────────
# Complex scale factors
#
# The fold multiplies a real factor into the readers' prefactors. A complex
# factor is declined outright rather than projected onto its real part, which
# is what the descriptor once did without a word.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_scale_absorption_complex_factor_is_not_folded(dtype):
    """A complex factor on the sole reader's operand stays a Scale node, and the values hold."""
    A = einsums.create_random_tensor("A", [4, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 5], dtype=dtype)
    C = einsums.create_random_tensor("C", [4, 5], dtype=dtype)
    D = einsums.create_zero_tensor("D", [4, 5], dtype=dtype)
    E = einsums.create_random_tensor("E", [4, 4], dtype=dtype)

    factor = complex(3.0, -1.0)
    D_ref = factor * (np.asarray(E) @ np.asarray(C))

    g = cg.Graph("sa_complex_fold")
    with cg.capture(g):
        einsums.linalg.scale(factor, C)
        einsums.einsum("ij <- ik ; kj", D, E, C, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert not _run(pass_inst, g)
    assert pass_inst.num_absorbed == 0
    assert g.num_nodes() == 3

    g.execute()
    assert_close(D, D_ref)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_scale_absorption_removes_a_dead_complex_scale(dtype):
    """A dead scale by a complex factor is removed like a real one.

    Nothing reads C between the scale and the overwrite, so removing the scale
    needs no factor to move anywhere and its value does not matter. The pass
    used to check for a real factor before it told the dead case from the
    fold, and so kept this scale.
    """
    A = einsums.create_random_tensor("A", [4, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 5], dtype=dtype)
    C = einsums.create_random_tensor("C", [4, 5], dtype=dtype)

    C_ref = np.asarray(A) @ np.asarray(B)

    g = cg.Graph("sa_dead_complex")
    with cg.capture(g):
        einsums.linalg.scale(complex(3.0, -1.0), C)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.ScaleAbsorption()
    assert _run(pass_inst, g)
    assert pass_inst.num_absorbed == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(C, C_ref)
