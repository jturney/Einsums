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
    """Two einsums with identical inputs + spec → CSE collapses to one node."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])
    D = einsums.create_zero_tensor("D", [4, 5])

    g = cg.Graph("cse_test")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)
    assert g.num_nodes() == 2

    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert g.num_nodes() == 1

    g.execute()

    expected = np.asarray(A) @ np.asarray(B)
    assert_close(C, expected)


def test_cse_three_identical_einsums_reduce_to_one():
    """Three identical einsums collapse to a single node."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])
    D = einsums.create_zero_tensor("D", [4, 5])
    E = einsums.create_zero_tensor("E", [4, 5])

    g = cg.Graph("cse_triple")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)
        einsums.einsum("ij <- ik ; kj", E, A, B)
    assert g.num_nodes() == 3

    modified = g.apply(_one_pass(cg.CSE()))
    assert modified
    assert g.num_nodes() == 1


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
    """Run CSE, verify shrink; then run DNE to confirm composition works."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [3, 5])
    C = einsums.create_zero_tensor("C", [4, 5])
    D = einsums.create_zero_tensor("D", [4, 5])

    g = cg.Graph("cse_dne")
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
    D = einsums.create_zero_tensor("D", [3, 6, 4])

    g = cg.Graph("cse_rank3_col")
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
