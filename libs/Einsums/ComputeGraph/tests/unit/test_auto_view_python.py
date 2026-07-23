# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python __getitem__ on RuntimeTensor routes to cg.view inside a capture.

Outside capture the existing behavior is preserved: ``t[1:3, :]`` returns a
non-graph RuntimeTensorView. Inside ``with cg.capture(g):`` the same syntax
auto-dispatches to ``cg.view(t, [(1, 3), (-1, -1)])`` so the returned view
is graph-aware and aliases the parent.

Supported keys (inside capture):
  - Pure slices, any combination of ``slice(start, stop)`` and ``slice(None)``.
  - ``Ellipsis`` to fill remaining axes with full slices.

Unsupported (raises IndexError with a clear message):
  - Integer indices (rank-reducing, cg.view doesn't support Drop axis yet).
  - Strided slices (step != 1).
  - Index arity that doesn't match the parent's rank after Ellipsis expansion.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# Eager: behavior unchanged
# ──────────────────────────────────────────────────────────────────────────


def test_eager_slice_returns_non_graph_view():
    C = einsums.create_random_tensor("C", [5, 6])
    sub = C[:, :3]
    assert sub.rank() == 2
    assert sub.dim(0) == 5
    assert sub.dim(1) == 3


def test_eager_element_access_returns_scalar():
    t = einsums.RuntimeTensorD("t", [4])
    np.asarray(t)[...] = [10.0, 20.0, 30.0, 40.0]
    assert t[2] == 30.0


# ──────────────────────────────────────────────────────────────────────────
# Captured: slice routes through cg.view; view aliases parent
# ──────────────────────────────────────────────────────────────────────────


def test_captured_slice_alias_round_trip_via_scale():
    big = einsums.create_zero_tensor("big", [4, 6])
    np.asarray(big)[...] = 1.0

    g = cg.Graph("auto-slice-scale")
    with cg.capture(g):
        sub = big[:, :3]  # auto cg.view
        einsums.linalg.scale(2.0, sub)
    g.execute()

    expected = np.ones((4, 6))
    expected[:, :3] = 2.0
    np.testing.assert_allclose(np.asarray(big), expected, rtol=1e-5)


def test_captured_slice_in_gemm_inputs():
    """Both A and B sliced from larger parents; C is owning."""
    A = einsums.create_random_tensor("A", [5, 5])
    B = einsums.create_random_tensor("B", [5, 5])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("auto-slice-gemm")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A[:3, :], B[:, :4], 0.0, C)
    g.execute()

    expected = np.asarray(A)[:3, :] @ np.asarray(B)[:, :4]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_captured_slice_in_gemm_output():
    """C is sliced from a larger parent, gemm writes through the view."""
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [4, 3])
    big_C = einsums.create_zero_tensor("big_C", [5, 5])

    g = cg.Graph("auto-slice-output")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, big_C[:3, :3])
    g.execute()

    expected = np.zeros((5, 5))
    expected[:3, :3] = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_captured_full_axis_via_colon():
    """``t[:, :3]``, the first axis is a bare colon (slice(None))."""
    big = einsums.create_random_tensor("big", [4, 6])
    out = einsums.create_zero_tensor("out", [4, 3])

    g = cg.Graph("auto-full")
    with cg.capture(g):
        einsums.linalg.axpy(1.0, big[:, :3], out)
    g.execute()

    expected = np.asarray(big)[:, :3]
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5)


def test_captured_ellipsis_expands_to_full_axes():
    """``t[..., :3]`` on a rank-3 tensor expands to ``t[:, :, :3]``."""
    big = einsums.create_random_tensor("big", [2, 3, 6])
    out = einsums.create_zero_tensor("out", [2, 3, 4])
    np.asarray(out)[...] = 1.0

    g = cg.Graph("auto-ellipsis")
    with cg.capture(g):
        einsums.linalg.axpy(2.0, big[..., :4], out)
    g.execute()

    expected = np.ones((2, 3, 4)) + 2.0 * np.asarray(big)[..., :4]
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5)


def test_captured_mp2_iajb_via_slicing():
    """MP2 (ia|jb) block extraction with auto cg.view from slicing."""
    nocc, nvirt = 3, 4
    nbf = nocc + nvirt
    eri_mo = einsums.create_random_tensor("eri_mo", [nbf, nbf, nbf, nbf])
    e = einsums.create_zero_tensor("e", [])  # rank-0 scalar

    g = cg.Graph("mp2-auto-slice")
    with cg.capture(g):
        iajb = eri_mo[:nocc, nocc:, :nocc, nocc:]
        einsums.einsum(" <- ijab ; ijab", e, iajb, iajb)
    g.execute()

    sliced = np.asarray(eri_mo)[:nocc, nocc:, :nocc, nocc:]
    expected = float(np.sum(sliced * sliced))
    np.testing.assert_allclose(float(np.asarray(e)), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Captured: slicing a VIEW (view-of-a-view) stays graph-aware
# ──────────────────────────────────────────────────────────────────────────


def test_captured_slice_of_view_aliases_parent_chain():
    """Slicing a view (``(A.T)[1:]``) must route through cg.view too, so the
    sub-view aliases the parent chain. Regression: the slice used to take a
    raw eager path with ``aliases == 0``, so a reduction over it had no
    dependency on an in-place ``Scale`` of the owning tensor and the
    optimizer's Reorder pass floated the read ahead of the write, reading
    unscaled data."""
    A = einsums.create_random_tensor("A", [2, 3])
    ref = np.asarray(A).copy()
    factor = -1.84

    g = cg.Graph("slice-of-view")
    out = einsums.create_zero_tensor("out", [2, 2])
    with cg.capture(g):
        einsums.linalg.scale(factor, A)   # in-place scale of the owner
        sub = A.T[1:]                      # (A.T)[1:], a view of a view
        einsums.linalg.axpy(1.0, sub, out)
    # The full default pipeline (incl. Reorder) must preserve scale-before-read.
    g.apply(cg.default_pass_manager())
    g.execute()

    expected = (ref * factor).T[1:]
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5)


def test_captured_slice_of_view_is_graph_registered():
    """The view-of-a-view is a graph-registered RuntimeTensorView, not the
    raw eager kind, i.e. the capture-aware __getitem__ fires on view
    classes, mirroring how it already fires on owning tensors."""
    A = einsums.create_random_tensor("A", [4, 6])
    g = cg.Graph("slice-of-view-reg")
    with cg.capture(g):
        sub = A.T[1:]          # rank-2 view-of-view
        assert sub.rank() == 2
        assert sub.dim(0) == 5   # (6 - 1) after dropping row 0 of the transpose
        assert sub.dim(1) == 4
    g.execute()
    np.testing.assert_allclose(np.asarray(sub), np.asarray(A).T[1:], rtol=1e-5)


def test_captured_assign_into_slice_of_view_writes_parent():
    """Write-side twin: ``(A.T)[1:] = X`` inside capture must route the target
    through cg.view too (the capture-aware __setitem__ is installed on the
    view classes), so the write lands in A's storage and any later read of an
    aliasing slice is correctly ordered after it. Without it the slice-assign
    target is a raw eager view with aliases == 0."""
    A = einsums.create_zero_tensor("A", [2, 3])
    rhs = einsums.create_random_tensor("rhs", [2, 2])

    g = cg.Graph("assign-into-slice-of-view")
    with cg.capture(g):
        A.T[1:] = rhs          # write through a view-of-a-view into A's storage
    g.execute()

    # A.T has shape (3,2); [1:] is rows 1..2 of A.T == columns 1..2 of A.
    expected = np.zeros((2, 3))
    expected.T[1:] = np.asarray(rhs)
    np.testing.assert_allclose(np.asarray(A), expected, rtol=1e-5)


def test_captured_assign_then_read_slice_of_view_ordering():
    """A write through a view-of-a-view followed by a read of an aliasing
    slice must stay ordered under the full optimizer pipeline (the dependency
    is only visible if the slice-assign target aliases the parent chain)."""
    A = einsums.create_zero_tensor("A", [2, 3])
    rhs = einsums.create_random_tensor("rhs", [2, 2])
    out = einsums.create_zero_tensor("out", [2, 2])

    g = cg.Graph("assign-read-slice-of-view")
    with cg.capture(g):
        A.T[1:] = rhs          # write through view-of-view
        sub = A.T[1:]          # read the same region back
        einsums.linalg.axpy(1.0, sub, out)
    g.apply(cg.default_pass_manager())
    g.execute()

    np.testing.assert_allclose(np.asarray(out), np.asarray(rhs), rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Captured: unsupported keys raise IndexError with a clear message
# ──────────────────────────────────────────────────────────────────────────


def test_captured_int_index_rank_reduces():
    """Integer indices inside capture now rank-reduce via cg.view_indexed
    (the Drop axis), matching eager numpy semantics."""
    big = einsums.create_random_tensor("big", [4, 6])
    ref = np.asarray(big).copy()
    g = cg.Graph("int-index")
    with cg.capture(g):
        sub = big[0, :3]  # drop axis 0, range axis 1 -> (3,)
    g.execute()
    np.testing.assert_allclose(np.asarray(sub), ref[0, :3], rtol=1e-5)


def test_captured_strided_slice_raises():
    big = einsums.create_random_tensor("big", [4, 6])
    g = cg.Graph("err-step")
    with pytest.raises(IndexError, match="step"):
        with cg.capture(g):
            _ = big[::2, :3]


def test_captured_short_key_pads_trailing_axes():
    """A key shorter than the rank gets implicit trailing ':' (numpy)."""
    big = einsums.create_random_tensor("big", [4, 6])
    ref = np.asarray(big).copy()
    g = cg.Graph("short-key")
    with cg.capture(g):
        sub = big[:3]  # == big[:3, :] -> (3, 6)
    g.execute()
    np.testing.assert_allclose(np.asarray(sub), ref[:3], rtol=1e-5)


def test_captured_too_many_indices_raises():
    big = einsums.create_random_tensor("big", [4, 6])
    g = cg.Graph("err-arity")
    with pytest.raises(IndexError, match="too many indices"):
        with cg.capture(g):
            _ = big[1, 2, 3]  # 3 indices for a rank-2 tensor


def test_captured_double_ellipsis_raises():
    big = einsums.create_random_tensor("big", [2, 3, 4])
    g = cg.Graph("err-ellipsis2")
    with pytest.raises(IndexError, match="Ellipsis"):
        with cg.capture(g):
            _ = big[..., ..., :2]


# ──────────────────────────────────────────────────────────────────────────
# Captured: scalar element access via [0] on rank-0 still works for
# reading-the-result patterns (not view-producing, but it's not invoked
# inside capture in practice, this just confirms we route correctly when
# the user actually wants element access).
#
# Wait: __getitem__ inside capture with a single int on a rank-1 would
# attempt to auto-slice and raise. That's the right behavior, since element
# access inside capture is unusual.
# ──────────────────────────────────────────────────────────────────────────


def test_eager_element_access_after_capture():
    """Confirm that after a capture ends, element access on the result
    (now eager) works as before."""
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4])
    e = einsums.create_zero_tensor("e", [1])

    g = cg.Graph("scalar-result")
    with cg.capture(g):
        einsums.linalg.dot(e, A, B)
    g.execute()

    # Outside capture again, element read works.
    val = e[0]
    expected = float(np.dot(np.asarray(A), np.asarray(B)))
    np.testing.assert_allclose(val, expected, rtol=1e-5)
