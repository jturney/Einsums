# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_ChainParenthesization.cpp.

This pass reasons about matrix chain multiplication, analysis only, no
graph modification. The pass is not part of the default pipeline (per
Optimizer.hpp's docstring); tests verify it's wired up correctly when
explicitly added to a PassManager.
"""

from __future__ import annotations

import einsums
import einsums.graph as cg


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_chain_parenthesization_empty_graph():
    g = cg.Graph("cp_empty")
    pass_inst = cg.ChainParenthesization()
    assert not _run(pass_inst, g)
    assert pass_inst.original_flops == 0
    assert pass_inst.optimal_flops == 0


def test_chain_parenthesization_non_gemm_operations():
    A = einsums.create_random_tensor("A", [4, 4])
    g = cg.Graph("cp_non_gemm")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)
        einsums.linalg.scale(3.0, A)

    pass_inst = cg.ChainParenthesization()
    assert not _run(pass_inst, g)
    assert pass_inst.original_flops == 0


def test_chain_parenthesization_detects_chain_and_computes_savings():
    """Chain A(100x1) * B(1x100) * C(100x1):

    Left-to-right (A*B)*C has 100*1*100 + 100*100*1 = 30000 mults.
    Optimal A*(B*C) has 1*100*1 + 100*1*1 = 200 mults.
    """
    A = einsums.create_random_tensor("A", [100, 1])
    B = einsums.create_random_tensor("B", [1, 100])
    C = einsums.create_random_tensor("C", [100, 1])
    T1 = einsums.create_zero_tensor("T1", [100, 100])
    T2 = einsums.create_zero_tensor("T2", [100, 1])

    g = cg.Graph("chain_test")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T1, A, B)
        einsums.einsum("ij <- ik ; kj", T2, T1, C)

    pass_inst = cg.ChainParenthesization()
    _run(pass_inst, g)

    assert pass_inst.original_flops > 0
    assert pass_inst.optimal_flops > 0
    assert pass_inst.optimal_flops < pass_inst.original_flops


def test_chain_parenthesization_no_chain_for_single_gemm():
    A = einsums.create_random_tensor("A", [10, 5])
    B = einsums.create_random_tensor("B", [5, 8])
    C = einsums.create_zero_tensor("C", [10, 8])

    g = cg.Graph("single_gemm")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.ChainParenthesization()
    _run(pass_inst, g)

    assert pass_inst.original_flops == 0
    assert pass_inst.optimal_flops == 0
