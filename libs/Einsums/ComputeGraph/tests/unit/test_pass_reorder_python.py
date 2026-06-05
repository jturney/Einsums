# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_Reorder.cpp."""

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


def test_reorder_empty_graph():
    g = cg.Graph("reorder_empty")
    pass_inst = cg.Reorder()
    assert not _run(pass_inst, g)


def test_reorder_single_node():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("reorder_single")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.Reorder()
    assert not _run(pass_inst, g)


def test_reorder_produces_valid_topological_order():
    A = einsums.create_random_tensor("A", [5, 5])
    B = einsums.create_random_tensor("B", [5, 5])
    C = einsums.create_zero_tensor("C", [5, 5])
    D = einsums.create_zero_tensor("D", [5, 5])

    C_ref = np.asarray(A) @ np.asarray(B)
    D_ref = np.asarray(A) @ np.asarray(B)

    g = cg.Graph("reorder_indep")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, A, B)

    _run(cg.Reorder(), g)
    g.execute()

    assert_close(C, C_ref)
    assert_close(D, D_ref)


def test_reorder_preserves_data_dependencies():
    A = einsums.create_random_tensor("A", [5, 5])
    B = einsums.create_random_tensor("B", [5, 5])
    C = einsums.create_zero_tensor("C", [5, 5])
    D = einsums.create_zero_tensor("D", [5, 5])

    C_ref = np.asarray(A) @ np.asarray(B)
    D_ref = C_ref @ np.asarray(B)

    g = cg.Graph("reorder_chain")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ij <- ik ; kj", D, C, B)

    _run(cg.Reorder(), g)
    g.execute()

    assert_close(C, C_ref)
    assert_close(D, D_ref)


def test_reorder_memory_aware_frees_large_tensor_early():
    """Smoke test: just verify the pass runs and execute() succeeds."""
    A = einsums.create_random_tensor("A", [128, 128])
    B = einsums.create_random_tensor("B", [128, 128])
    C = einsums.create_zero_tensor("C", [128, 128])
    D = einsums.create_random_tensor("D", [4, 4])

    g = cg.Graph("reorder_memory")
    with cg.capture(g):
        einsums.linalg.scale(2.0, D)
        einsums.einsum("ij <- ik ; kj", C, A, B)

    _run(cg.Reorder(), g)
    g.execute()


@pytest.mark.skip(
    reason="Row-major tensor creation isn't exposed to Python yet. "
           "The C++ counterpart uses `create_random_tensor<T>(true, ...)`."
)
def test_reorder_preserves_rank3_batched_gemm_chain_row_major():
    """C++ test uses row-major + batch-prefix to hit row_mode fast path."""


def test_reorder_preserves_rank3_batched_gemm_dependency_chain():
    """Col-major batch-suffix → each stage is a BatchedGemm; Reorder must preserve the chain."""
    A = einsums.create_random_tensor("A", [3, 3, 4])
    B = einsums.create_random_tensor("B", [3, 3, 4])
    C = einsums.create_zero_tensor("C", [3, 3, 4])
    D = einsums.create_zero_tensor("D", [3, 3, 4])

    g = cg.Graph("reorder_rank3")
    with cg.capture(g):
        einsums.einsum("ijb <- ikb ; kjb", C, A, B)
        einsums.einsum("ijb <- ikb ; kjb", D, C, B)

    assert _count_kind(g, "BatchedGemm") == 2

    _run(cg.Reorder(), g)
    g.execute()

    # Verify C and D have the expected per-batch contraction.
    A_np = np.asarray(A)
    B_np = np.asarray(B)
    C_np = np.asarray(C)
    D_np = np.asarray(D)
    for b in range(4):
        assert_close(C_np[..., b], A_np[..., b] @ B_np[..., b])
        assert_close(D_np[..., b], C_np[..., b] @ B_np[..., b])
