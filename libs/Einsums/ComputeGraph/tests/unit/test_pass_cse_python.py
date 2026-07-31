# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for the CSE (Common Subexpression Elimination) pass.

One-to-one mirror of ``Pass_CSE.cpp``. Every ``TEST_CASE`` in the C++
file has a corresponding ``test_*`` function here that builds the same
graph from Python, runs ``cg.CSE()`` via ``PassManager``, and asserts
the same pre/post invariants.

The C++ tests call ``graph.apply<cg::passes::CSE>()`` which returns a
``(modified, pass)`` pair. The Python equivalent is ``g.apply(pm)``
returning just ``modified``, we build a one-pass PassManager per test
since composing passes via ``pm.add(cg.SomePass())`` is the natural
Python idiom.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _one_pass(pass_obj) -> cg.PassManager:
    """Convenience: build a PassManager containing a single pass."""
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm


def _count_kind(g: cg.Graph, kind: str) -> int:
    """Count nodes of a given ``OpKind`` (string) via the JSON dump.

    ``Graph.nodes()`` isn't exposed to Python (would require binding
    ``Node`` + ``OpKind`` + the variant op_data); ``to_json()`` already
    serializes the node kind as a string, so we use that.
    """
    return sum(1 for n in json.loads(g.to_json()).get("nodes", []) if n.get("kind") == kind)


# ──────────────────────────────────────────────────────────────────────────
# Basic emptiness / single-node cases
# ──────────────────────────────────────────────────────────────────────────


def test_cse_empty_graph():
    """CSE on an empty graph reports no modification."""
    g = cg.Graph("cse_empty")
    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified


def test_cse_single_node_graph():
    """CSE on a single-node graph reports no modification and keeps the node."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("cse_single")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified
    assert g.num_nodes() == 1


# ──────────────────────────────────────────────────────────────────────────
# Elimination of true duplicates
# ──────────────────────────────────────────────────────────────────────────


def test_cse_eliminates_duplicate_einsum():
    """Two einsums with identical inputs + spec -> CSE collapses to one node
    (the duplicate's output is graph-owned scratch; user-visible outputs are
    never elided - see test_cse_keeps_user_visible_duplicates)."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])

    g = cg.Graph("cse_test")
    D = g.create_zero_tensor("D", [4, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)
    nodes_before = g.num_nodes()

    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert g.num_nodes() < nodes_before

    g.execute()

    expected = np.asarray(A) @ np.asarray(B)
    assert_close(C, expected)


def test_cse_keeps_user_visible_duplicates():
    """A duplicate writing a USER tensor keeps its producer: the user reads
    that tensor directly, so eliding the write would break the capture
    contract. Both tensors must hold the result."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])
    D = einsums.create_zero_tensor("D", [4, 5])

    g = cg.Graph("cse_user_visible")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)

    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified
    assert g.num_nodes() == 2

    g.execute()
    expected = np.asarray(A) @ np.asarray(B)
    assert_close(C, expected)
    assert_close(D, expected)


def test_cse_three_identical_einsums_reduce_to_one():
    """Three identical einsums collapse to a single node."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])

    g = cg.Graph("cse_triple")
    D = g.create_zero_tensor("D", [4, 5], dtype="float64")
    E = g.create_zero_tensor("E", [4, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)
        einsums.einsum("ij <- ik ; kj", E, A, B)
    nodes_before = g.num_nodes()

    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert g.num_nodes() < nodes_before


# ──────────────────────────────────────────────────────────────────────────
# Non-equivalence: different prefactors / different inputs / different ops
# ──────────────────────────────────────────────────────────────────────────


def test_cse_does_not_eliminate_different_prefactors():
    """Same spec + same inputs but different alpha → not equivalent, no merge."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    g = cg.Graph("cse_no_match")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", D, A, B, c_pf=0.0, ab_pf=2.0)

    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified
    assert g.num_nodes() == 2


def test_cse_does_not_eliminate_different_inputs():
    """Same spec but swapped inputs → not equivalent, no merge."""
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    g = cg.Graph("cse_diff_inputs")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, B, A)  # swapped

    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified
    assert g.num_nodes() == 2


def test_cse_merges_a_duplicate_inside_a_loop_body():
    """CSE descends the tree itself, so bodies are optimized too.

    Iterative workloads capture everything into a loop body, so until it did
    the pass had no effect on them at all.
    """
    n = 8
    A = einsums.create_random_tensor("A", [n, n])
    B = einsums.create_random_tensor("B", [n, n])
    out = einsums.create_zero_tensor("out", [n, n])

    g = cg.Graph("cse_in_body")
    P = g.create_zero_tensor("P", [n, n], dtype="float64")
    Q = g.create_zero_tensor("Q", [n, n], dtype="float64")
    body = g.add_loop("once", 1, lambda it: it < 1)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", P, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", Q, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ij ; ij", out, P, Q, c_pf=0.0, ab_pf=1.0)

    before = body.num_nodes()
    assert g.apply(_one_pass(cg.CSE()))
    assert body.num_nodes() == before - 1

    g.execute()
    ref = np.asarray(A) @ np.asarray(B)
    assert_close(out, ref * ref)


def test_cse_keeps_a_body_duplicate_the_parent_reads():
    """The slot redirect reaches only the graph it runs on, so a parent reader
    of a body-local duplicate blocks the merge."""
    n = 8
    A = einsums.create_random_tensor("A", [n, n])
    B = einsums.create_random_tensor("B", [n, n])
    seen = einsums.create_zero_tensor("seen", [n, n])

    g = cg.Graph("cse_body_escapes")
    P = g.create_zero_tensor("P", [n, n], dtype="float64")
    Q = g.create_zero_tensor("Q", [n, n], dtype="float64")
    body = g.add_loop("once", 1, lambda it: it < 1)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", P, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", Q, A, B, c_pf=0.0, ab_pf=1.0)
    with cg.capture(g):
        einsums.linalg.axpby(1.0, Q, 0.0, seen)

    assert not g.apply(_one_pass(cg.CSE()))

    g.execute()
    assert_close(seen, np.asarray(A) @ np.asarray(B))


def test_cse_keeps_a_duplicate_a_loop_body_reads():
    """A loop body reading the duplicate's output blocks the merge.

    Regression: a control-flow node's inputs do not list what its body reads,
    and the slot redirect reaches only the parent graph, so the merge left the
    body reading a never-written buffer.
    """
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])
    out = einsums.create_zero_tensor("out", [4, 5])

    g = cg.Graph("cse_loop_body_reader")
    D = g.create_zero_tensor("D", [4, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)
    body = g.add_loop("once", 1, lambda it: it < 1)
    with cg.capture(body):
        einsums.linalg.axpby(1.0, D, 0.0, out)

    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified

    g.execute()
    assert_close(out, np.asarray(A) @ np.asarray(B))


def test_cse_does_not_merge_permutes_with_different_index_orders():
    """Two transposes of one source differ in their index orders alone.

    Regression: the descriptor comparison looked only at alpha and beta, so
    these merged and P2 was left holding P1's values.
    """
    A = einsums.create_random_tensor("A", [3, 3, 3])

    g = cg.Graph("cse_permute_orders")
    P1 = g.create_zero_tensor("P1", [3, 3, 3], dtype="float64")
    P2 = g.create_zero_tensor("P2", [3, 3, 3], dtype="float64")
    with cg.capture(g):
        einsums.permute("j,i,k <- i,j,k", P1, A, c_pf=0.0, a_pf=1.0)
        einsums.permute("i,k,j <- i,j,k", P2, A, c_pf=0.0, a_pf=1.0)

    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified

    g.execute()

    A_np = np.asarray(A)
    assert_close(np.asarray(P1), A_np.transpose(1, 0, 2))
    assert_close(np.asarray(P2), A_np.transpose(0, 2, 1))


# ──────────────────────────────────────────────────────────────────────────
# Proportional duplicates
# ──────────────────────────────────────────────────────────────────────────


def test_cse_merges_a_proportional_duplicate():
    """Q = 0.5 * P is eliminated and the 0.5 moves onto Q's reader."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    F = einsums.create_random_tensor("F", [5, 2])
    out1 = einsums.create_zero_tensor("out1", [4, 2])
    out2 = einsums.create_zero_tensor("out2", [4, 2])

    g = cg.Graph("cse_proportional")
    P = g.create_zero_tensor("P", [4, 5], dtype="float64")
    Q = g.create_zero_tensor("Q", [4, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", P, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", Q, A, B, c_pf=0.0, ab_pf=0.5)
        einsums.einsum("ij <- ik ; kj", out1, P, F, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", out2, Q, F, c_pf=0.0, ab_pf=1.0)

    n_before = g.num_nodes()
    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert g.num_nodes() == n_before - 1

    g.execute()

    ref = (np.asarray(A) @ np.asarray(B)) @ np.asarray(F)
    assert_close(out1, ref)
    assert_close(out2, 0.5 * ref)


def test_cse_declines_a_ratio_that_is_not_a_power_of_two():
    """3x is exactly representable but is not folded, so the result never
    depends on which of two proportional nodes the pass kept."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    F = einsums.create_random_tensor("F", [5, 2])
    out2 = einsums.create_zero_tensor("out2", [4, 2])

    g = cg.Graph("cse_odd_ratio")
    P = g.create_zero_tensor("P", [4, 5], dtype="float64")
    Q = g.create_zero_tensor("Q", [4, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", P, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", Q, A, B, c_pf=0.0, ab_pf=3.0)
        einsums.einsum("ij <- ik ; kj", out2, Q, F, c_pf=0.0, ab_pf=1.0)

    n_before = g.num_nodes()
    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified
    assert g.num_nodes() == n_before


def test_cse_merges_proportional_axpby_copies():
    """Axpby had no descriptor arm at all, so even identical copies never
    merged; `Y = alpha*X` is linear in alpha, so the ratio path applies."""
    X = einsums.create_random_tensor("X", [4, 5])
    G = einsums.create_random_tensor("G", [5, 2])
    out1 = einsums.create_zero_tensor("out1", [4, 2])
    out2 = einsums.create_zero_tensor("out2", [4, 2])

    g = cg.Graph("cse_axpby_proportional")
    Y1 = g.create_zero_tensor("Y1", [4, 5], dtype="float64")
    Y2 = g.create_zero_tensor("Y2", [4, 5], dtype="float64")
    with cg.capture(g):
        einsums.linalg.axpby(2.0, X, 0.0, Y1)
        einsums.linalg.axpby(1.0, X, 0.0, Y2)
        einsums.einsum("ij <- ik ; kj", out1, Y1, G, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", out2, Y2, G, c_pf=0.0, ab_pf=1.0)

    n_before = g.num_nodes()
    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert g.num_nodes() == n_before - 1

    g.execute()

    ref = np.asarray(X) @ np.asarray(G)
    assert_close(out1, 2.0 * ref)
    assert_close(out2, ref)


def test_cse_does_not_merge_scale_with_different_factors():
    """scale(2.0, A) and scale(3.0, A) have different OpData → no merge."""
    A = einsums.create_random_tensor("A", [3, 3])

    g = cg.Graph("cse_diff_scale")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)
        einsums.linalg.scale(3.0, A)

    modified = g.apply(_one_pass(cg.CSE()))
    assert not modified
    assert g.num_nodes() == 2


# ──────────────────────────────────────────────────────────────────────────
# Composition with downstream passes
# ──────────────────────────────────────────────────────────────────────────


def test_cse_then_dead_node_elimination_composition():
    """Run CSE, verify shrink; then run DNE to confirm composition works.
    The duplicate writes a graph-owned intermediate: CSE never elides writes
    to user-visible tensors."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])

    g = cg.Graph("cse_dne")
    D = g.create_zero_tensor("D", [4, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)

    n_before = g.num_nodes()
    assert n_before >= 2

    g.apply(_one_pass(cg.CSE()))
    assert g.num_nodes() < n_before

    # DNE may or may not find further dead nodes; just verify it runs cleanly.
    g.apply(_one_pass(cg.DeadNodeElimination()))


# ──────────────────────────────────────────────────────────────────────────
# Rank-3 BatchedGemm de-duplication
# ──────────────────────────────────────────────────────────────────────────


def test_cse_deduplicates_rank3_batched_gemm_col_major():
    """Two identical rank-3 batched contractions (col-major) → one BatchedGemm."""
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])
    C = einsums.create_zero_tensor("C", [3, 6, 4])

    g = cg.Graph("cse_rank3_col")
    # Graph-owned duplicate: CSE never elides writes to user-visible tensors.
    D = g.create_zero_tensor("D", [3, 6, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ijb <- ikb ; kjb", C, A, B)
        einsums.einsum("ijb <- ikb ; kjb", D, A, B)

    assert _count_kind(g, "BatchedGemm") == 2
    n_before = g.num_nodes()

    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert g.num_nodes() < n_before
    assert _count_kind(g, "BatchedGemm") == 1


@pytest.mark.skip(
    reason="Row-major tensor creation isn't exposed to Python yet. "
           "The C++ counterpart uses `create_random_tensor<T>(/*row_major=*/true, ...)`; "
           "until that flag is bound, batch-prefix patterns over Python-created "
           "(col-major) tensors don't capture as BatchedGemm."
)
def test_cse_deduplicates_rank3_batched_gemm_row_major():
    """Two identical rank-3 batched contractions (row-major batch-prefix) → one BatchedGemm.

    Exercises the row_mode branch of CSE's BatchedGemm descriptor equality.
    """
    A = einsums.create_random_tensor("A", [4, 3, 5])
    B = einsums.create_random_tensor("B", [4, 5, 6])
    C = einsums.create_zero_tensor("C", [4, 3, 6])
    D = einsums.create_zero_tensor("D", [4, 3, 6])

    g = cg.Graph("cse_rank3_row")
    with cg.capture(g):
        einsums.einsum("bij <- bik ; bkj", C, A, B)
        einsums.einsum("bij <- bik ; bkj", D, A, B)

    assert _count_kind(g, "BatchedGemm") == 2

    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert _count_kind(g, "BatchedGemm") == 1
