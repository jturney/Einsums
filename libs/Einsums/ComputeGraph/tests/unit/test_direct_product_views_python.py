# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Exhaustive view/owning combination coverage for cg::direct_product.

direct_product computes C = alpha * (A ⊙ B) + beta * C, element-wise
multiplication of A and B accumulated into C with prefactors. All three
tensors must have the same shape and dtype.

direct_product had independent template parameters for A, B, C from the
start, so all 2**3 = 8 cells of the (A, B, C) x (owning, view) matrix are
bound per dtype.

Cell labels: O = owning, V = view, in (A, B, C) order.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES


def test_direct_product_OOO_baseline():
    A = einsums.create_random_tensor("A", [4, 5])
    B = einsums.create_random_tensor("B", [4, 5])
    C = einsums.create_zero_tensor("C", [4, 5])

    g = cg.Graph("dp-OOO")
    with cg.capture(g):
        einsums.linalg.direct_product(1.5, A, B, 0.0, C)
    g.execute()

    expected = 1.5 * np.asarray(A) * np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_OOV_C_is_view():
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [3, 4])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])

    g = cg.Graph("dp-OOV")
    with cg.capture(g):
        Cv = cg.view(big_C, [(1, 4), (2, 6)])
        einsums.linalg.direct_product(2.0, A, B, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 8))
    expected[1:4, 2:6] = 2.0 * np.asarray(A) * np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_direct_product_OVO_B_is_view():
    A = einsums.create_random_tensor("A", [3, 4])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("dp-OVO")
    with cg.capture(g):
        Bv = cg.view(big_B, [(0, 3), (0, 4)])
        einsums.linalg.direct_product(1.0, A, Bv, 0.0, C)
    g.execute()

    expected = np.asarray(A) * np.asarray(big_B)[:3, :4]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_OVV_B_and_C_are_views():
    A = einsums.create_random_tensor("A", [3, 4])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])

    g = cg.Graph("dp-OVV")
    with cg.capture(g):
        Bv = cg.view(big_B, [(0, 3), (0, 4)])
        Cv = cg.view(big_C, [(2, 5), (3, 7)])
        einsums.linalg.direct_product(1.0, A, Bv, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 8))
    expected[2:5, 3:7] = np.asarray(A) * np.asarray(big_B)[:3, :4]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_direct_product_VOO_A_is_view():
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    B = einsums.create_random_tensor("B", [3, 4])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("dp-VOO")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 4), (2, 6)])
        einsums.linalg.direct_product(1.0, Av, B, 0.0, C)
    g.execute()

    expected = np.asarray(big_A)[1:4, 2:6] * np.asarray(B)
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_VOV_A_and_C_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    B = einsums.create_random_tensor("B", [3, 4])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])

    g = cg.Graph("dp-VOV")
    with cg.capture(g):
        Av = cg.view(big_A, [(1, 4), (2, 6)])
        Cv = cg.view(big_C, [(2, 5), (3, 7)])
        einsums.linalg.direct_product(1.0, Av, B, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 8))
    expected[2:5, 3:7] = np.asarray(big_A)[1:4, 2:6] * np.asarray(B)
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


def test_direct_product_VVO_A_and_B_are_views():
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    C = einsums.create_zero_tensor("C", [3, 4])

    g = cg.Graph("dp-VVO")
    with cg.capture(g):
        Av = cg.view(big_A, [(0, 3), (0, 4)])
        Bv = cg.view(big_B, [(1, 4), (2, 6)])
        einsums.linalg.direct_product(1.0, Av, Bv, 0.0, C)
    g.execute()

    expected = np.asarray(big_A)[:3, :4] * np.asarray(big_B)[1:4, 2:6]
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5)


def test_direct_product_VVV_all_three_views_with_accumulation():
    """All three views + nonzero beta to confirm read-and-write of C-view."""
    big_A = einsums.create_random_tensor("big_A", [6, 8])
    big_B = einsums.create_random_tensor("big_B", [6, 8])
    big_C = einsums.create_zero_tensor("big_C", [6, 8])
    np.asarray(big_C)[...] = 1.0
    big_C_before = np.asarray(big_C).copy()

    g = cg.Graph("dp-VVV")
    with cg.capture(g):
        Av = cg.view(big_A, [(0, 3), (0, 4)])
        Bv = cg.view(big_B, [(1, 4), (2, 6)])
        Cv = cg.view(big_C, [(2, 5), (3, 7)])
        einsums.linalg.direct_product(1.0, Av, Bv, 0.5, Cv)
    g.execute()

    expected = big_C_before.copy()
    expected[2:5, 3:7] = np.asarray(big_A)[:3, :4] * np.asarray(big_B)[1:4, 2:6] + 0.5 * big_C_before[2:5, 3:7]
    np.testing.assert_allclose(np.asarray(big_C), expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# A (or B) and C are overlapping views of ONE parent
# ──────────────────────────────────────────────────────────────────────────
#
# The vendor path scales C by beta and then multiplies into it, so an input
# that shares elements with C without being C read values already scaled or
# already written. Only an input with C's exact base pointer was guarded. The
# expected value reads every input from the parent as it was before the call.

_SHIFTS = {
    "row-shifted": ((0, 3), (0, 3), (1, 4), (0, 3)),
    "column-shifted": ((0, 3), (0, 3), (0, 3), (1, 4)),
    "diagonal-shifted": ((1, 4), (1, 4), (0, 3), (0, 3)),
}


def _view(P, rows, cols, mode):
    """A view of @p P: a slice eagerly (``cg.view`` is capture-only), ``cg.view`` in a graph."""
    return P[slice(*rows), slice(*cols)] if mode == "eager" else cg.view(P, [rows, cols])


def _parent(dtype):
    """A 4x4 parent of small exact values, genuinely complex on a complex dtype."""
    vals = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    if np.dtype(dtype).kind == "c":
        vals = vals + 1j * (np.arange(16).reshape(4, 4) % 3 - 1)
    P = einsums.create_zero_tensor("P", [4, 4], dtype=dtype)
    np.asarray(P)[...] = vals
    return P


@pytest.mark.parametrize("mode", ["eager", "graph", "graph+passes"])
@pytest.mark.parametrize("shift", list(_SHIFTS))
@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("beta", [0.0, 0.5])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_direct_product_input_overlapping_output(dtype, beta, operand, shift, mode):
    (ar, ac, cr, cc) = _SHIFTS[shift]
    alpha = (0.5 - 0.75j) if np.dtype(dtype).kind == "c" else 2.0
    P = _parent(dtype)
    other = einsums.create_zero_tensor("other", [3, 3], dtype=dtype)
    np.asarray(other)[...] = np.arange(3, 12).reshape(3, 3) % 4 - 1.5
    before = np.asarray(P).copy()
    other_vals = np.asarray(other).copy()

    def body():
        shared = _view(P, ar, ac, mode)
        A, B = (shared, other) if operand == "A" else (other, shared)
        einsums.linalg.direct_product(alpha, A, B, beta, _view(P, cr, cc, mode))

    if mode == "eager":
        body()
    else:
        g = cg.Graph("dp-overlap")
        with cg.capture(g):
            body()
        if mode == "graph+passes":
            g.apply(cg.default_pass_manager())
        g.execute()

    expected = before.copy()
    expected[slice(*cr), slice(*cc)] = (alpha * before[slice(*ar), slice(*ac)] * other_vals
                                        + beta * before[slice(*cr), slice(*cc)])
    np.testing.assert_array_equal(np.asarray(P), expected)
