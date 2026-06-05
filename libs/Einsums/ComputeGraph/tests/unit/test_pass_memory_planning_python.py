# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_MemoryPlanning.cpp."""

from __future__ import annotations

import einsums
import einsums.graph as cg


_DOUBLE_SIZE = 8


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_memory_planning_empty_graph():
    g = cg.Graph("mp_empty")
    pass_inst = cg.MemoryPlanning()
    assert not _run(pass_inst, g)
    assert pass_inst.total_memory == 0
    assert pass_inst.peak_memory == 0


def test_memory_planning_basic_analysis():
    A = einsums.create_random_tensor("A", [10, 10])
    B = einsums.create_random_tensor("B", [10, 10])
    C = einsums.create_zero_tensor("C", [10, 10])

    g = cg.Graph("memory_test")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.MemoryPlanning()
    _run(pass_inst, g)

    expected = 3 * 10 * 10 * _DOUBLE_SIZE
    assert pass_inst.total_memory == expected
    assert pass_inst.peak_memory == expected


def test_memory_planning_chain_shows_lower_peak_than_total():
    A = einsums.create_random_tensor("A", [10, 10])
    B = einsums.create_random_tensor("B", [10, 10])
    T1 = einsums.create_zero_tensor("T1", [10, 10])
    T2 = einsums.create_zero_tensor("T2", [10, 10])

    g = cg.Graph("chain_memory")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T1, A, B)
        einsums.einsum("ij <- ik ; kj", T2, T1, A)

    pass_inst = cg.MemoryPlanning()
    _run(pass_inst, g)

    assert pass_inst.total_memory == 4 * 10 * 10 * _DOUBLE_SIZE
    assert pass_inst.peak_memory < pass_inst.total_memory


def test_memory_planning_rank3_batched_gemm_tensor_liveness():
    """Sizes: A(3,5,4), B(5,6,4), C(3,6,4), D(3,6,4)."""
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])
    C = einsums.create_zero_tensor("C", [3, 6, 4])
    D = einsums.create_zero_tensor("D", [3, 6, 4])

    g = cg.Graph("mp_rank3")
    with cg.capture(g):
        einsums.einsum("ijb <- ikb ; kjb", C, A, B)
        einsums.einsum("ijb <- ikb ; kjb", D, A, B)

    pass_inst = cg.MemoryPlanning()
    _run(pass_inst, g)

    expected_total = (3 * 5 * 4 + 5 * 6 * 4 + 2 * 3 * 6 * 4) * _DOUBLE_SIZE
    assert pass_inst.total_memory == expected_total
    assert pass_inst.peak_memory > 0
