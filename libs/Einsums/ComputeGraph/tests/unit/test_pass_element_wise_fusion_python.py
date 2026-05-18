# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_ElementWiseFusion.cpp."""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_ewf_empty_graph():
    g = cg.Graph("ewf_empty")
    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)
    assert pass_inst.num_fused == 0


def test_ewf_single_node():
    A = einsums.create_random_tensor("A", [3, 3])
    g = cg.Graph("ewf_single")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)

    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)


def test_ewf_fuses_consecutive_scales():
    A = einsums.create_random_tensor("A", [3, 3])
    A_ref = 2.0 * 3.0 * np.asarray(A).copy()

    g = cg.Graph("ewf_test")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)
        einsums.linalg.scale(3.0, A)
    assert g.num_nodes() == 2

    pass_inst = cg.ElementWiseFusion()
    assert _run(pass_inst, g)
    assert pass_inst.num_fused == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(A, A_ref)


def test_ewf_three_consecutive_scales_fuse_to_one():
    A = einsums.create_random_tensor("A", [3, 3])
    A_ref = 2.0 * 3.0 * 4.0 * np.asarray(A).copy()

    g = cg.Graph("ewf_triple")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)
        einsums.linalg.scale(3.0, A)
        einsums.linalg.scale(4.0, A)
    assert g.num_nodes() == 3

    pass_inst = cg.ElementWiseFusion()
    assert _run(pass_inst, g)
    assert pass_inst.num_fused == 2
    assert g.num_nodes() == 1

    g.execute()
    assert_close(A, A_ref)


def test_ewf_no_fusion_for_different_tensors():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])

    g = cg.Graph("ewf_no_fuse")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)
        einsums.linalg.scale(3.0, B)

    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)


def test_ewf_scale_separated_by_einsum_does_not_fuse():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])

    g = cg.Graph("ewf_barrier")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)
        einsums.einsum("ij <- ik ; kj", A, A, B)
        einsums.linalg.scale(3.0, A)

    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)
    assert pass_inst.num_fused == 0
    assert g.num_nodes() == 3


def test_ewf_fuses_consecutive_rank3_scales():
    A = einsums.create_random_tensor("A", [4, 3, 5])
    A_ref = 2.0 * 3.0 * np.asarray(A).copy()

    g = cg.Graph("ewf_rank3")
    with cg.capture(g):
        einsums.linalg.scale(2.0, A)
        einsums.linalg.scale(3.0, A)
    assert g.num_nodes() == 2

    pass_inst = cg.ElementWiseFusion()
    assert _run(pass_inst, g)
    assert pass_inst.num_fused == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(A, A_ref)
