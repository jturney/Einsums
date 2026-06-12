# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::einsum.

einsum dispatches contractions described by a string spec:
    "ij <- ik ; kj"      matrix multiply
    "ij <- ji"           transpose  (single operand, but this is permute,
                          not einsum; einsum is always 2-operand)
    "i <- ij ; j"        matrix-vector
    " <- i ; i"          dot (scalar result on rank-0 tensor)

The einsum_python wrapper already had independent AType, BType, CType
template parameters, so all 2**3 = 8 cells of the (C, A, B) x (owning,
view) matrix are bound per dtype.

Cell labels: O = owning, V = view, in (C, A, B) order. Each test runs a
matmul where the parent tensors are oversized and slices select the
relevant region.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# Matmul "ij <- ik ; kj" through all 8 cells
# ──────────────────────────────────────────────────────────────────────────


def test_einsum_OOO_baseline_matmul():
    A = einsums.create_random_tensor("A", [4, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("es-OOO")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
    g.execute()

    expected = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_einsum_OOV_B_is_view():
    A = einsums.create_random_tensor("A", [4, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 8])
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("es-OOV")
    with cg.capture(g):
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        einsums.einsum("ij <- ik ; kj", C, A, Bv)
    g.execute()

    expected = np.asarray(A) @ np.asarray(big_B)[:, :3]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_einsum_OVO_A_is_view():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("es-OVO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        einsums.einsum("ij <- ik ; kj", C, Av, B)
    g.execute()

    expected = np.asarray(big_A)[1:5, :] @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_einsum_OVV_A_and_B_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 8])
    C = einsums.create_zero_tensor("C", [4, 3])

    g = cg.Graph("es-OVV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        einsums.einsum("ij <- ik ; kj", C, Av, Bv)
    g.execute()

    expected = np.asarray(big_A)[1:5, :] @ np.asarray(big_B)[:, :3]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_einsum_VOO_C_is_view():
    A = einsums.create_random_tensor("A", [4, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("es-VOO")
    with cg.capture(g):
        Cv = cg.view(big_C, [(1, 5), (2, 5)])
        einsums.einsum("ij <- ik ; kj", Cv, A, B)
    g.execute()

    expected = np.zeros((6, 6))
    expected[1:5, 2:5] = np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_einsum_VOV_C_and_B_are_views():
    A = einsums.create_random_tensor("A", [4, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 8])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("es-VOV")
    with cg.capture(g):
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        Cv = cg.view(big_C, [(1, 5), (2, 5)])
        einsums.einsum("ij <- ik ; kj", Cv, A, Bv)
    g.execute()

    expected = np.zeros((6, 6))
    expected[1:5, 2:5] = np.asarray(A) @ np.asarray(big_B)[:, :3]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_einsum_VVO_C_and_A_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    B = einsums.create_random_tensor("B", [5, 3])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("es-VVO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        Cv = cg.view(big_C, [(1, 5), (2, 5)])
        einsums.einsum("ij <- ik ; kj", Cv, Av, B)
    g.execute()

    expected = np.zeros((6, 6))
    expected[1:5, 2:5] = np.asarray(big_A)[1:5, :] @ np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_einsum_VVV_all_three_views():
    big_A = einsums.create_random_tensor("big_A", [6, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 8])
    big_C = einsums.create_zero_tensor("big_C", [6, 6])

    g = cg.Graph("es-VVV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 5), (-1, -1)])
        Bv = cg.view(big_B, [(-1, -1), (0, 3)])
        Cv = cg.view(big_C, [(1, 5), (2, 5)])
        einsums.einsum("ij <- ik ; kj", Cv, Av, Bv)
    g.execute()

    expected = np.zeros((6, 6))
    expected[1:5, 2:5] = np.asarray(big_A)[1:5, :] @ np.asarray(big_B)[:, :3]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Higher-rank einsum patterns with views, exercise spec parsing + view aliasing
# ──────────────────────────────────────────────────────────────────────────


def test_einsum_view_batched_matmul():
    """bij <- bik ; bkj with a view of a larger batched tensor."""
    big_A = einsums.create_random_tensor("big_A", [4, 3, 5])  # batch dim 4
    big_B = einsums.create_random_tensor("big_B", [4, 5, 6])
    C = einsums.create_zero_tensor("C", [3, 3, 4])  # 3 batches in result

    g = cg.Graph("es-batched-view")
    with cg.capture(g):
        Av = cg.view(big_A, [(0, 3), (-1, -1), (-1, -1)])  # first 3 batches
        Bv = cg.view(big_B, [(0, 3), (-1, -1), (0, 4)])  # first 3 batches, first 4 cols
        einsums.einsum("bij <- bik ; bkj", C, Av, Bv)
    g.execute()

    expected = np.einsum("bik,bkj->bij", np.asarray(big_A)[:3, :, :], np.asarray(big_B)[:3, :, :4])
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_einsum_view_outer_product():
    """ij <- i ; j with rank-1 views."""
    big_a = einsums.create_random_tensor("big_a", [10])
    big_b = einsums.create_random_tensor("big_b", [10])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("es-outer-view")
    with cg.capture(g):
        av = cg.view(big_a, [(2, 5)])
        bv = cg.view(big_b, [(1, 5)])
        einsums.einsum("ij <- i ; j", C, av, bv)
    g.execute()

    expected = np.outer(np.asarray(big_a)[2:5], np.asarray(big_b)[1:5])
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_einsum_view_full_reduction_scalar():
    """ <- ij ; ij with views, writing to a rank-0 scalar slot."""
    big_A = einsums.create_random_tensor("big_A", [5, 5])
    big_B = einsums.create_random_tensor("big_B", [5, 5])
    e = einsums.create_zero_tensor("e", [])  # rank-0 scalar

    g = cg.Graph("es-scalar-view")
    with cg.capture(g):
        Av = cg.view(big_A, [(0, 3), (0, 4)])
        Bv = cg.view(big_B, [(0, 3), (0, 4)])
        einsums.einsum(" <- ij ; ij", e, Av, Bv)
    g.execute()

    expected = np.sum(np.asarray(big_A)[:3, :4] * np.asarray(big_B)[:3, :4])
    np.testing.assert_allclose(np.asarray(e), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# MP2 (ia|jb) extraction via einsum on a view (the canonical motivation)
# ──────────────────────────────────────────────────────────────────────────


def test_einsum_mp2_iajb_block_via_view():
    """Build the (ia|jb) block from eri_mo via a view, then contract with itself."""
    nocc, nvirt = 3, 4
    nbf = nocc + nvirt
    eri_mo = einsums.create_random_tensor("eri_mo", [nbf, nbf, nbf, nbf])
    # MP2 energy contribution: sum_{ijab} (ia|jb)^2 , toy contraction with itself
    e = einsums.create_zero_tensor("e", [])

    g = cg.Graph("mp2-iajb-view")
    with cg.capture(g):
        iajb = cg.view(eri_mo, [(0, nocc), (nocc, nbf), (0, nocc), (nocc, nbf)])
        einsums.einsum(" <- ijab ; ijab", e, iajb, iajb)
    g.execute()

    sliced = np.asarray(eri_mo)[:nocc, nocc:, :nocc, nocc:]
    expected = float(np.sum(sliced * sliced))
    np.testing.assert_allclose(float(np.asarray(e)), expected, rtol=1e-5)
