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
    # The pair folds to ONE contraction. That contraction is a real Einsum node
    # (Graph.make_einsum_node) so the descriptor-reading passes can still see it;
    # the separate Custom node builds L. L/T scratch get one-time Allocs.
    assert _count_kind(g, "Einsum") == 1
    assert _count_kind(g, "Custom") == 1
    assert _count_kind(g, "Custom") == 1

    g.execute()
    assert_close(out, ref)


def test_no_fold_for_different_tensors():
    # Same index pattern but DIFFERENT operand tensors is DistributiveFactoring's
    # job, not ours, we require the same tensor read with permuted indices.
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
    # One fused contraction (a real Einsum node) plus the Custom L builder.
    assert _count_kind(g, "Einsum") == 1
    assert _count_kind(g, "Custom") == 1

    g.execute()
    assert_close(out, ref)


def test_verbosity_setting():
    # set_verbosity propagates from PassManager to passes and narrates folds to
    # stderr without affecting correctness.
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4, 3, 3])
    out = einsums.create_zero_tensor("out", [3, 3])
    a = np.asarray(A); b = np.asarray(B)
    ref = 2.0 * np.einsum("k,kij->ij", a, b) - np.einsum("k,kji->ij", a, b)

    g = cg.Graph("verbose_fold")
    with cg.capture(g):
        einsums.einsum("i,j <- k ; k,i,j", out, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- k ; k,j,i", out, A, B, c_pf=1.0, ab_pf=-1.0)

    p = cg.LinearCombinationContractionFolding()
    assert p.verbosity == 0  # silent by default
    pm = cg.PassManager()
    pm.add(p)
    pm.set_verbosity(2)  # propagates to p
    assert p.verbosity == 2
    assert pm.run(g)

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


def _kinds(graph):
    return [n["kind"] for n in json.loads(graph.to_json())["nodes"]]


def test_default_pipeline_folds_and_hoists_the_l_builder():
    """LCCF is in populate_default, ordered before LoopInvariantHoisting.

    The two together are the point: LCCF emits the L construction as its own
    node whose only input is the paired operand, so when that operand is
    loop-invariant -- an integral block from one-time setup, the common case --
    LIH lifts the builder out of the loop and L is built once rather than
    rebuilt on every replay. Before the split, L lived inside the fused
    contraction's executor and nothing could see that half was invariant.
    """
    o, v, niters = 2, 3, 4
    rng = np.random.default_rng(7)
    g_np = rng.standard_normal((o, v, v, v))   # loop-invariant integral
    d_np = rng.standard_normal((o, v))         # per-iteration t1 increment

    integral = einsums.asarray(np.ascontiguousarray(g_np), name="g")
    incr = einsums.asarray(np.ascontiguousarray(d_np), name="d")
    t1 = einsums.create_zero_tensor("t1", [o, v], dtype="float64")
    Fae = einsums.create_zero_tensor("Fae", [v, v], dtype="float64")

    # Oracle: t1 gains d AFTER each iteration's contraction.
    t1_ref, ref = np.zeros((o, v)), np.zeros((v, v))
    for _ in range(niters):
        ref = ref + 2.0 * np.einsum("mf,mafe->ae", t1_ref, g_np)
        ref = ref - 1.0 * np.einsum("mf,maef->ae", t1_ref, g_np)
        t1_ref = t1_ref + d_np

    g = cg.Graph("ccsd_spin_adaptation")
    body = g.add_loop("iter", niters, lambda it, N=niters: it < N - 1)
    with cg.capture(body):
        einsums.einsum("a,e <- m,f ; m,a,f,e", Fae, t1, integral, c_pf=1.0, ab_pf=2.0)
        einsums.einsum("a,e <- m,f ; m,a,e,f", Fae, t1, integral, c_pf=1.0, ab_pf=-1.0)
        einsums.linalg.axpby(1.0, incr, 1.0, t1)

    assert _kinds(body).count("Einsum") == 2, _kinds(body)

    g.apply(cg.default_pass_manager())

    body_kinds, parent_kinds = _kinds(body), _kinds(g)
    # The pair folded to a single contraction, and it is a REAL Einsum node --
    # built via Graph.make_einsum_node -- not an opaque Custom blob, so the
    # descriptor-reading passes can still see it.
    assert body_kinds.count("Einsum") == 1, body_kinds
    assert body_kinds.count("Custom") == 0, body_kinds
    # The L builder is the Custom node, now sitting in the parent before the loop.
    assert parent_kinds.count("Custom") == 1, parent_kinds

    g.execute()
    assert_close(Fae, ref)
