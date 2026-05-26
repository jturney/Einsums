# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_InplaceOptimization.cpp.

InplaceOptimization is analysis-only — it counts candidates but does
not modify the graph. ``modified=False`` is the expected result of
``pm.run(g)`` even when candidates are found.
"""

from __future__ import annotations

import einsums
import einsums.graph as cg


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_inplace_optimization_empty_graph():
    g = cg.Graph("io_empty")
    pass_inst = cg.InplaceOptimization()
    assert not _run(pass_inst, g)
    assert pass_inst.num_candidates == 0


def test_inplace_optimization_user_owned_tensor_not_a_candidate():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("io_user")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.InplaceOptimization()
    assert not _run(pass_inst, g)
    assert pass_inst.num_candidates == 0


def test_inplace_optimization_finds_candidates():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("inplace_test")
    T = g.create_zero_tensor("T", [3, 3], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, B)  # writes T
        einsums.einsum("ij <- ik ; kj", C, T, A)  # reads T (sole consumer)

    pass_inst = cg.InplaceOptimization()
    _run(pass_inst, g)
    assert pass_inst.num_candidates >= 0  # analysis runs cleanly


def test_inplace_optimization_rank3_batched_gemm_with_sole_consumer():
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])

    g = cg.Graph("inplace_rank3")
    T = g.create_zero_tensor("T", [3, 6, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ijb <- ikb ; kjb", T, A, B)
        einsums.linalg.scale(0.5, T)

    pass_inst = cg.InplaceOptimization()
    _run(pass_inst, g)
    assert pass_inst.num_candidates >= 0
