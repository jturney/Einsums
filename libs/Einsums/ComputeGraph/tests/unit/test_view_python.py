# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for RuntimeTensorView as a first-class capture citizen.

``einsums.graph.view(parent, ranges)`` returns a non-owning slice that the
graph treats as aliasing its parent — writes through the view are visible
on the parent, and the optimization passes know about the dependency via
``TensorHandle::aliases``.

The ``ranges`` argument is a list of ``(lo, hi)`` integer pairs, one per
parent axis. A pair where both values are negative (e.g. ``(-1, -1)``)
means "full axis".

Only the canonical same-type ops (scale, axpy, axpby, element_transform)
and block_copy currently accept views as arguments. gemm/einsum need a
deeper template refactor to accept mixed view/owning operands; for those
patterns, extract the slab into a fresh owning tensor via block_copy first.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES


# ──────────────────────────────────────────────────────────────────────────
# View tensor protocol (dim/stride/size/rank)
# ──────────────────────────────────────────────────────────────────────────


def test_view_exposes_dim_stride_size_rank():
    """Slicing via Python __getitem__ yields a View with the full tensor protocol."""
    C = einsums.create_random_tensor("C", [5, 6])
    sub = C[:, :3]

    assert sub.rank() == 2
    assert sub.dim(0) == 5
    assert sub.dim(1) == 3
    assert sub.size == 15
    # Stride values aren't fixed by the API but they must be callable.
    assert sub.stride(0) >= 1
    assert sub.stride(1) >= 1


# ──────────────────────────────────────────────────────────────────────────
# cg.view outside capture: must raise
# ──────────────────────────────────────────────────────────────────────────


def test_cg_view_outside_capture_raises():
    C = einsums.create_random_tensor("C", [5, 5])
    with pytest.raises(Exception):
        cg.view(C, [(-1, -1), (0, 3)])


def test_cg_view_wrong_ranges_length_raises():
    C = einsums.create_random_tensor("C", [5, 5])
    g = cg.Graph("bad-ranges")
    with pytest.raises(Exception):
        with cg.capture(g):
            cg.view(C, [(-1, -1)])  # only 1 entry for a rank-2 parent


# ──────────────────────────────────────────────────────────────────────────
# Scale through a view → parent reflects the change
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_scale_through_view_modifies_parent(dtype):
    C = einsums.create_zero_tensor("C", [4, 6], dtype=dtype)
    np.asarray(C)[...] = 1.0

    g = cg.Graph("vscale")
    with cg.capture(g):
        sub = cg.view(C, [(-1, -1), (0, 3)])  # C[:, :3]
        einsums.linalg.scale(2.0, sub)
    g.execute()

    expected = np.ones((4, 6))
    expected[:, :3] = 2.0
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# axpy/axpby through views
# ──────────────────────────────────────────────────────────────────────────


def test_axpy_through_two_views_on_disjoint_columns():
    """X view = C[:, 0:3], Y view = D[:, 0:3]; Y += 1.5 * X writes only into D's first 3 cols."""
    C = einsums.create_random_tensor("C", [4, 6])
    D = einsums.create_zero_tensor("D", [4, 6])
    np.asarray(D)[...] = 10.0

    C_np_before = np.asarray(C).copy()

    g = cg.Graph("vaxpy")
    with cg.capture(g):
        Cv = cg.view(C, [(-1, -1), (0, 3)])
        Dv = cg.view(D, [(-1, -1), (0, 3)])
        einsums.linalg.axpy(1.5, Cv, Dv)
    g.execute()

    expected = np.full((4, 6), 10.0)
    expected[:, :3] = 10.0 + 1.5 * C_np_before[:, :3]
    np.testing.assert_allclose(np.asarray(D), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# block_copy between an owning tensor and a view (mixed type)
# ──────────────────────────────────────────────────────────────────────────


def test_block_copy_view_dst_owning_src_writes_into_subregion():
    """Write a small owning tensor into a view slab of a larger one."""
    big = einsums.create_zero_tensor("big", [5, 6])
    np.asarray(big)[...] = 9.0
    small = einsums.create_random_tensor("small", [3, 2])

    g = cg.Graph("write-slab")
    with cg.capture(g):
        slab = cg.view(big, [(1, 4), (2, 4)])  # big[1:4, 2:4]
        # block_copy into the view: copy small[0:3, 0:2] into slab[0:3, 0:2].
        einsums.linalg.block_copy(slab, small, [0, 0], [0, 0], [3, 2])
    g.execute()

    expected = np.full((5, 6), 9.0)
    expected[1:4, 2:4] = np.asarray(small)
    np.testing.assert_array_equal(np.asarray(big), expected)


def test_block_copy_owning_dst_view_src_extracts_slab():
    """Extract a slab through a view directly (zero-copy alternative to writing offsets)."""
    src = einsums.create_random_tensor("src", [6, 6])
    dst = einsums.create_zero_tensor("dst", [3, 3])

    g = cg.Graph("extract-via-view")
    with cg.capture(g):
        slab = cg.view(src, [(1, 4), (2, 5)])  # src[1:4, 2:5]
        einsums.linalg.block_copy(dst, slab, [0, 0], [0, 0], [3, 3])
    g.execute()

    np.testing.assert_array_equal(np.asarray(dst), np.asarray(src)[1:4, 2:5])


# ──────────────────────────────────────────────────────────────────────────
# element_transform through a view
# ──────────────────────────────────────────────────────────────────────────


def test_element_transform_through_view_only_touches_subregion():
    C = einsums.create_zero_tensor("C", [4, 6])
    np.asarray(C)[...] = 2.0

    g = cg.Graph("vet")
    with cg.capture(g):
        sub = cg.view(C, [(-1, -1), (0, 3)])
        einsums.linalg.element_transform(sub, lambda x: 1.0 / x)
    g.execute()

    expected = np.full((4, 6), 2.0)
    expected[:, :3] = 0.5
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# View aliasing is visible to the graph: the view node + downstream op
# both end up in the graph with correct dependency structure.
# ──────────────────────────────────────────────────────────────────────────


def test_view_records_view_node_with_aliasing():
    C = einsums.create_random_tensor("C", [4, 4])

    g = cg.Graph("alias")
    with cg.capture(g):
        sub = cg.view(C, [(-1, -1), (0, 2)])
        einsums.linalg.scale(2.0, sub)

    # Two nodes (view + scale), two tensors (C parent + view child).
    assert g.num_nodes() == 2
    assert g.num_tensors() == 2


# ──────────────────────────────────────────────────────────────────────────
# Composed: SCF density build via view-then-block_copy.
# (Direct gemm-on-view is not yet supported; need to copy to an owning
# tensor first. This test documents the current workflow.)
# ──────────────────────────────────────────────────────────────────────────


def test_scf_density_via_view_and_block_copy_then_gemm():
    nbf, nocc = 5, 3
    C = einsums.create_random_tensor("C", [nbf, nbf])
    C_occ = einsums.create_zero_tensor("C_occ", [nbf, nocc])
    D = einsums.create_zero_tensor("D", [nbf, nbf])

    g = cg.Graph("density-via-view")
    with cg.capture(g):
        slab = cg.view(C, [(-1, -1), (0, nocc)])
        # block_copy from the view to an owning tensor, then gemm on the owning copy.
        einsums.linalg.block_copy(C_occ, slab, [0, 0], [0, 0], [nbf, nocc])
        einsums.linalg.gemm(2.0, C_occ, C_occ, 0.0, D, trans_b=True)
    g.execute()

    C_np = np.asarray(C)
    expected = 2.0 * C_np[:, :nocc] @ C_np[:, :nocc].T
    np.testing.assert_allclose(np.asarray(D), expected, rtol=1e-5)
