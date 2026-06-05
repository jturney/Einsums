# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_DeadNodeElimination.cpp."""

from __future__ import annotations

import json

import einsums
import einsums.graph as cg


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def _count_kind(g, kind):
    return sum(1 for n in json.loads(g.to_json()).get("nodes", []) if n.get("kind") == kind)


def test_dne_empty_graph():
    g = cg.Graph("dne_empty")
    pass_inst = cg.DeadNodeElimination()
    assert not _run(pass_inst, g)
    assert pass_inst.num_eliminated == 0


def test_dne_keeps_nodes_with_user_owned_outputs():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("dne_user_owned")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.DeadNodeElimination()
    assert not _run(pass_inst, g)
    assert pass_inst.num_eliminated == 0


def test_dne_eliminates_intermediate_with_no_reader():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])

    g = cg.Graph("dne_intermediate")
    T = g.create_zero_tensor("T", [3, 3], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, B)

    n_before = g.num_nodes()
    assert n_before >= 1

    pass_inst = cg.DeadNodeElimination()
    assert _run(pass_inst, g)
    assert pass_inst.num_eliminated >= 1
    assert g.num_nodes() < n_before


def test_dne_keeps_intermediate_with_reader():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("dne_live_intermediate")
    T = g.create_zero_tensor("T", [3, 3], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, B)
        einsums.einsum("ij <- ik ; kj", C, T, A)

    pass_inst = cg.DeadNodeElimination()
    assert not _run(pass_inst, g)
    assert pass_inst.num_eliminated == 0


def test_dne_rank3_batched_gemm_intermediate_eliminated():
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])

    g = cg.Graph("dne_rank3")
    T = g.create_zero_tensor("T", [3, 6, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ijb <- ikb ; kjb", T, A, B)

    assert _count_kind(g, "BatchedGemm") >= 1
    n_before = g.num_nodes()

    pass_inst = cg.DeadNodeElimination()
    assert _run(pass_inst, g)
    assert pass_inst.num_eliminated >= 1
    assert g.num_nodes() < n_before
