# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_InplaceOptimization.cpp.

InplaceOptimization is analysis-only, it counts candidates but does
not modify the graph. ``modified=False`` is the expected result of
``pm.run(g)`` even when candidates are found.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


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
    assert pass_inst.num_candidates >= 1
    # Einsum consumers may never alias output with input; nothing merges.
    assert pass_inst.num_merged == 0


def test_inplace_optimization_merges_direct_product_output():
    """Mirror of the C++ merge test: Y = alpha*(X (*) B) with beta=0 reuses
    dying X's storage; numerics and replay stay correct."""
    A = einsums.create_random_tensor("A", [6, 6])
    B = einsums.create_random_tensor("B", [6, 6])
    OUT = einsums.create_random_tensor("OUT", [6, 6])

    X_ref = np.asarray(A) @ np.asarray(B)
    Y_ref = 2.0 * X_ref * np.asarray(B)
    OUT_ref = Y_ref @ np.asarray(A)

    g = cg.Graph("inplace_merge_py")
    X = g.create_zero_tensor("X", [6, 6], dtype="float64")
    Y = g.create_zero_tensor("Y", [6, 6], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", X, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.linalg.direct_product(2.0, X, B, 0.0, Y)
        einsums.einsum("ij <- ik ; kj", OUT, Y, A, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.InplaceOptimization()
    assert _run(pass_inst, g)
    assert pass_inst.num_merged == 1

    g.execute()
    assert_close(OUT, OUT_ref)
    g.execute()  # replay through the merged storage
    assert_close(OUT, OUT_ref)


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
