# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_ElementWiseFusion.cpp, over every dtype.

A complex dtype takes complex factors, so a fused factor that dropped an
imaginary part, or conjugated one, lands on the wrong value.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _pf(dtype, re, im):
    """A prefactor for ``dtype``: both parts for a complex dtype, the real part otherwise."""
    return complex(re, im) if np.dtype(dtype).kind == "c" else re


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_ewf_empty_graph():
    g = cg.Graph("ewf_empty")
    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)
    assert pass_inst.num_fused == 0


def test_ewf_single_node(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    g = cg.Graph("ewf_single")
    with cg.capture(g):
        einsums.linalg.scale(_pf(dtype, 2.0, 0.5), A)

    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)


def test_ewf_fuses_consecutive_scales(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    a, b = _pf(dtype, 2.0, 0.5), _pf(dtype, 3.0, -1.5)
    A_ref = b * (a * np.asarray(A).copy())

    g = cg.Graph("ewf_test")
    with cg.capture(g):
        einsums.linalg.scale(a, A)
        einsums.linalg.scale(b, A)
    assert g.num_nodes() == 2

    pass_inst = cg.ElementWiseFusion()
    assert _run(pass_inst, g)
    assert pass_inst.num_fused == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(A, A_ref)


def test_ewf_three_consecutive_scales_fuse_to_one(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    a, b, c = _pf(dtype, 2.0, 0.5), _pf(dtype, 3.0, -1.5), _pf(dtype, 4.0, 0.25)
    A_ref = c * (b * (a * np.asarray(A).copy()))

    g = cg.Graph("ewf_triple")
    with cg.capture(g):
        einsums.linalg.scale(a, A)
        einsums.linalg.scale(b, A)
        einsums.linalg.scale(c, A)
    assert g.num_nodes() == 3

    pass_inst = cg.ElementWiseFusion()
    assert _run(pass_inst, g)
    assert pass_inst.num_fused == 2
    assert g.num_nodes() == 1

    g.execute()
    assert_close(A, A_ref)


def test_ewf_no_fusion_for_different_tensors(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 3], dtype=dtype)

    g = cg.Graph("ewf_no_fuse")
    with cg.capture(g):
        einsums.linalg.scale(_pf(dtype, 2.0, 0.5), A)
        einsums.linalg.scale(_pf(dtype, 3.0, -1.5), B)

    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)


def test_ewf_scale_separated_by_einsum_does_not_fuse(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 3], dtype=dtype)

    g = cg.Graph("ewf_barrier")
    with cg.capture(g):
        einsums.linalg.scale(_pf(dtype, 2.0, 0.5), A)
        einsums.einsum("ij <- ik ; kj", A, A, B)
        einsums.linalg.scale(_pf(dtype, 3.0, -1.5), A)

    pass_inst = cg.ElementWiseFusion()
    assert not _run(pass_inst, g)
    assert pass_inst.num_fused == 0
    assert g.num_nodes() == 3


def test_ewf_fuses_consecutive_rank3_scales(dtype):
    A = einsums.create_random_tensor("A", [4, 3, 5], dtype=dtype)
    a, b = _pf(dtype, 2.0, 0.5), _pf(dtype, 3.0, -1.5)
    A_ref = b * (a * np.asarray(A).copy())

    g = cg.Graph("ewf_rank3")
    with cg.capture(g):
        einsums.linalg.scale(a, A)
        einsums.linalg.scale(b, A)
    assert g.num_nodes() == 2

    pass_inst = cg.ElementWiseFusion()
    assert _run(pass_inst, g)
    assert pass_inst.num_fused == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(A, A_ref)


# ──────────────────────────────────────────────────────────────────────────
# axpby chains
# ──────────────────────────────────────────────────────────────────────────


def test_ewf_fuses_consecutive_axpby(dtype):
    """Y = a1*X + b1*Y then Y = a2*X + b2*Y composes into one axpby."""
    X = einsums.create_random_tensor("X", [4, 5], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [4, 5], dtype=dtype)

    X_np = np.asarray(X).copy()
    Y_np = np.asarray(Y).copy()
    a1, b1 = _pf(dtype, 2.0, 0.5), _pf(dtype, 3.0, -1.5)
    a2, b2 = _pf(dtype, 5.0, -0.75), _pf(dtype, 7.0, 1.25)
    expected = a2 * X_np + b2 * (a1 * X_np + b1 * Y_np)

    g = cg.Graph("ewf_axpby")
    with cg.capture(g):
        einsums.linalg.axpby(a1, X, b1, Y)
        einsums.linalg.axpby(a2, X, b2, Y)
    assert g.num_nodes() == 2

    pass_inst = cg.ElementWiseFusion()
    assert _run(pass_inst, g)
    assert pass_inst.num_fused == 1
    assert g.num_nodes() == 1

    g.execute()
    assert_close(Y, expected)


def test_ewf_axpby_on_a_different_source_does_not_fuse(dtype):
    """Two different sources are a three-operand update; no axpby expresses it."""
    X1 = einsums.create_random_tensor("X1", [4, 5], dtype=dtype)
    X2 = einsums.create_random_tensor("X2", [4, 5], dtype=dtype)
    Y = einsums.create_random_tensor("Y", [4, 5], dtype=dtype)

    x1 = np.asarray(X1).copy()
    x2 = np.asarray(X2).copy()
    a1, b1 = _pf(dtype, 2.0, 0.5), _pf(dtype, 3.0, -1.5)
    a2, b2 = _pf(dtype, 5.0, -0.75), _pf(dtype, 7.0, 1.25)
    expected = a2 * x2 + b2 * (a1 * x1 + b1 * np.asarray(Y).copy())

    g = cg.Graph("ewf_axpby_diff_src")
    with cg.capture(g):
        einsums.linalg.axpby(a1, X1, b1, Y)
        einsums.linalg.axpby(a2, X2, b2, Y)

    assert not _run(cg.ElementWiseFusion(), g)
    assert g.num_nodes() == 2

    g.execute()
    assert_close(Y, expected)
