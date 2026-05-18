# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_LoopInvariantHoisting.cpp."""

from __future__ import annotations

import einsums
import einsums.graph as cg


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_lih_empty_loop_body():
    g = cg.Graph("lih_empty")
    g.add_loop("loop", 3, lambda it: it < 2)

    pass_inst = cg.LoopInvariantHoisting()
    assert not _run(pass_inst, g)
    assert pass_inst.num_hoisted == 0


def test_lih_nothing_to_hoist():
    value = einsums.create_zero_tensor("value", [1])

    g = cg.Graph("no_hoist")
    body = g.add_loop("loop", 5, lambda it: it < 4)
    with cg.capture(body):
        einsums.linalg.scale(0.5, value)

    pass_inst = cg.LoopInvariantHoisting()
    assert not _run(pass_inst, g)


def test_lih_hoists_invariant_node():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("hoist_test")
    body = g.add_loop("loop", 5, lambda it: it < 4)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.linalg.scale(0.9, C)

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 1


def test_lih_dependency_chain_partially_hoists():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_random_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    g = cg.Graph("lih_dep_chain")
    body = g.add_loop("loop", 3, lambda it: it < 2)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", D, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.linalg.scale(0.5, C)

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 1


def test_lih_all_nodes_invariant():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    D = einsums.create_zero_tensor("D", [3, 3])

    g = cg.Graph("lih_all_invariant")
    body = g.add_loop("loop", 3, lambda it: it < 2)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ik ; kj", D, A, B, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 2


def test_lih_rank3_batched_gemm_hoists():
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])
    C = einsums.create_zero_tensor("C", [3, 6, 4])
    D = einsums.create_random_tensor("D", [3, 6, 4])

    g = cg.Graph("lih_rank3")
    body = g.add_loop("loop", 4, lambda it: it < 3)
    with cg.capture(body):
        einsums.einsum("ijb <- ikb ; kjb", C, A, B)  # invariant
        einsums.linalg.scale(0.9, D)  # not invariant — writes D

    pass_inst = cg.LoopInvariantHoisting()
    assert _run(pass_inst, g)
    assert pass_inst.num_hoisted == 1
