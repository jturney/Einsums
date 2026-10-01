# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Coverage for cg::direct_division, element-wise (Hadamard) quotient.

direct_division computes C = alpha * (A ./ B) + beta * C, the counterpart to
direct_product. It is the fast path for amplitude denominators (1/Δ) that
previously required a per-element Python `element_transform` callback. All three
tensors share shape and dtype; A/B/C may each be owning or a view.
"""

from __future__ import annotations

import numpy as np

import einsums
from einsums.testing import assert_close


def _pf(dtype, re, im):
    """A prefactor for ``dtype``: complex(re, im) for a complex dtype, re otherwise."""
    return complex(re, im) if np.dtype(dtype).kind == "c" else re
import einsums.graph as cg


def _nonzero(name, shape, dtype="float64"):
    """Random tensor shifted away from zero (safe denominator)."""
    t = einsums.create_random_tensor(name, list(shape), dtype=dtype)
    np.asarray(t)[...] += 2.0
    return t


def test_direct_division_eager(dtype):
    pf_1_5 = _pf(dtype, 1.5, 0.75)
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    B = _nonzero("B", [4, 5], dtype)
    C = einsums.create_zero_tensor("C", [4, 5], dtype=dtype)

    einsums.linalg.direct_division(pf_1_5, A, B, 0.0, C)  # no capture -> eager

    assert_close(np.asarray(C), pf_1_5 * np.asarray(A) / np.asarray(B))


def test_direct_division_capture_baseline(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    B = _nonzero("B", [4, 5], dtype)
    C = einsums.create_zero_tensor("C", [4, 5], dtype=dtype)

    g = cg.Graph("dd-OOO")
    with cg.capture(g):
        einsums.linalg.direct_division(1.0, A, B, 0.0, C)
    g.execute()

    assert_close(np.asarray(C), np.asarray(A) / np.asarray(B))


def test_direct_division_beta_accumulate(dtype):
    pf_3_0 = _pf(dtype, 3.0, 0.75)
    pf_2_0 = _pf(dtype, 2.0, 0.75)
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = _nonzero("B", [3, 4], dtype)
    C = einsums.create_random_tensor("C", [3, 4], dtype=dtype)
    c0 = np.asarray(C).copy()

    g = cg.Graph("dd-beta")
    with cg.capture(g):
        einsums.linalg.direct_division(pf_3_0, A, B, pf_2_0, C)
    g.execute()

    assert_close(np.asarray(C), pf_2_0 * c0 + pf_3_0 * np.asarray(A) / np.asarray(B))


def test_direct_division_C_is_view(dtype):
    pf_2_0 = _pf(dtype, 2.0, 0.75)
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = _nonzero("B", [3, 4], dtype)
    big_C = einsums.create_zero_tensor("big_C", [6, 8], dtype=dtype)

    g = cg.Graph("dd-OOV")
    with cg.capture(g):
        Cv = cg.view(big_C, [(1, 4), (2, 6)])
        einsums.linalg.direct_division(pf_2_0, A, B, 0.0, Cv)
    g.execute()

    expected = np.zeros((6, 8), dtype=dtype)
    expected[1:4, 2:6] = pf_2_0 * np.asarray(A) / np.asarray(B)
    assert_close(np.asarray(big_C), expected)


def test_direct_division_B_is_view(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    big_B = _nonzero("big_B", [6, 8], dtype)
    C = einsums.create_zero_tensor("C", [3, 4], dtype=dtype)

    g = cg.Graph("dd-OVO")
    with cg.capture(g):
        Bv = cg.view(big_B, [(0, 3), (0, 4)])
        einsums.linalg.direct_division(1.0, A, Bv, 0.0, C)
    g.execute()

    assert_close(np.asarray(C), np.asarray(A) / np.asarray(big_B)[:3, :4])


def test_direct_division_reciprocal_denominator(dtype):
    """The motivating use: amplitude T = K / D where D is an MP2/CC-style
    denominator built with outer_sum. Exercises divide inside an optimized graph."""
    no, nv = 3, 5
    eo = np.array([-0.7, -0.5, -0.3])
    ev = np.array([0.2, 0.4, 0.6, 0.9, 1.3])

    K = einsums.create_random_tensor("K", [nv, nv], dtype=dtype)
    evt = einsums.create_zero_tensor("ev", [nv], dtype=dtype)
    np.asarray(evt)[:] = ev
    D = einsums.create_zero_tensor("D", [nv, nv], dtype=dtype)
    T = einsums.create_zero_tensor("T", [nv, nv], dtype=dtype)

    i, j = 0, 1  # a fixed occupied pair: D_ab = e_i + e_j - e_a - e_b
    g = cg.Graph("dd-denom")
    with cg.capture(g):
        einsums.linalg.outer_sum(D, [evt, evt], [-1.0, -1.0])      # -e_a - e_b
        ones = einsums.create_zero_tensor("ones", [nv, nv], dtype=dtype)
        einsums.linalg.element_transform(ones, lambda _: 1.0)
        einsums.linalg.axpby(eo[i] + eo[j], ones, 1.0, D)          # += e_i + e_j
        einsums.linalg.direct_division(1.0, K, D, 0.0, T)          # T = K / D
    pm = cg.PassManager()
    pm.populate_default()
    g.apply(pm)
    g.execute()

    Dref = eo[i] + eo[j] - ev[:, None] - ev[None, :]
    assert_close(np.asarray(T), np.asarray(K) / Dref)
