# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Tests for LinearCombinationContractionFolding (the CCSD '2J-K' fold).

Folds transpose-paired contractions that reuse the SAME operand tensor with
permuted index patterns into one contraction against L = sum_k a_k * P_k(B).
"""

from __future__ import annotations

import json

import numpy as np

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _run(g):
    pm = cg.PassManager()
    pm.add(cg.LinearCombinationContractionFolding())
    return pm.run(g)


def _count_kind(g, kind):
    return sum(1 for n in json.loads(g.to_json()).get("nodes", []) if n.get("kind") == kind)


def test_folds_transpose_pair():
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4, 3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])

    a = np.asarray(A); b = np.asarray(B)
    ref = 2.0 * np.einsum("k,kij->ij", a, b) - np.einsum("k,kji->ij", a, b)

    g = cg.Graph("fold_pair")
    with cg.capture(g):
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- k ; k,j,i", out, A, B, c_pf=1.0, ab_pf=-1.0)
    assert _count_kind(g, "Einsum") == 2

    assert _run(g)
    # both contractions fold into one Custom node (L/T scratch get one-time Allocs).
    assert _count_kind(g, "Einsum") == 0
    assert _count_kind(g, "Custom") == 1

    g.execute()
    assert_close(out, ref)


def test_no_fold_for_different_tensors():
    # Same index pattern but DIFFERENT operand tensors is DistributiveFactoring's
    # job, not ours — we require the same tensor read with permuted indices.
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4, 3, 3])
    C = einsums.create_random_tensor("C", [4, 3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])

    g = cg.Graph("no_fold_diff_tensor")
    with cg.capture(g):
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- k ; k,i,j", out, A, C, c_pf=1.0, ab_pf=-1.0)

    assert not _run(g)
    assert g.num_nodes() == 2


def test_no_fold_for_identical_specs():
    # Same tensor, same index order on both = pure duplicate, no permutation.
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4, 3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])

    g = cg.Graph("no_fold_identical")
    with cg.capture(g):
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=1.0, ab_pf=-1.0)

    assert not _run(g)
    assert g.num_nodes() == 2


def test_interference_guard_blocks_fold():
    # An intervening op that reads the partial-sum output must block the fold.
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4, 3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])
    snap = einsums.create_zero_tensor("snap", [3, 3])

    g = cg.Graph("fold_interference")
    with cg.capture(g):
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.linalg.axpby(1.0, out, 0.0, snap)  # reads the partial sum
        einsums.einsum("i,j <- k ; k,j,i", out, A, B, c_pf=1.0, ab_pf=-1.0)

    assert not _run(g)
    assert g.num_nodes() == 3


def test_fold_three_terms():
    # 2*B[kij] - B[kji] + 0.5*B[kij] folds via one canonical + permuted terms.
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4, 3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])

    a = np.asarray(A); b = np.asarray(B)
    ref = (2.0 * np.einsum("k,kij->ij", a, b)
           - 1.0 * np.einsum("k,kji->ij", a, b)
           + 0.5 * np.einsum("k,kij->ij", a, b))

    g = cg.Graph("fold_three")
    with cg.capture(g):
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- k ; k,j,i", out, A, B, c_pf=1.0, ab_pf=-1.0)
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=1.0, ab_pf=0.5)
    assert _count_kind(g, "Einsum") == 3

    assert _run(g)
    assert _count_kind(g, "Einsum") == 0
    assert _count_kind(g, "Custom") == 1

    g.execute()
    assert_close(out, ref)


def test_fold_inside_loop_body():
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4, 3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])

    a = np.asarray(A); b = np.asarray(B)
    one_iter = 2.0 * np.einsum("k,kij->ij", a, b) - np.einsum("k,kji->ij", a, b)

    pipeline = cg.Pipeline("fold_loop")
    body = pipeline.add_loop("iter", 3, lambda it: it < 2)
    with cg.capture(body):
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- k ; k,j,i", out, A, B, c_pf=1.0, ab_pf=-1.0)

    pm = cg.PassManager()
    pm.add(cg.LinearCombinationContractionFolding())
    assert pipeline.apply(pm)
    pipeline.execute()

    # out is overwritten each iteration (first term c_pf=0), so last iter wins.
    assert_close(out, one_iter)
