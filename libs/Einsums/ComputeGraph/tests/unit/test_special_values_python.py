# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Special-value (NaN / inf) handling vs numpy.

Element-wise ops, gemm and einsum do IEEE pass-through (a NaN/inf operand flows
through exactly as numpy computes it). The ``max`` reduction must propagate NaN
like numpy.max -- a comparison-based reduction silently drops NaN (``x > acc`` is
false for NaN) and an all-NaN reduction used to leak the ``lowest()`` seed
(-DBL_MAX); both are regression-guarded here.
"""
from __future__ import annotations

import itertools

import numpy as np

import einsums
import einsums.graph as cg

_ctr = itertools.count()
INF = np.inf
NAN = np.nan


def _mk(a):
    a = np.asarray(a, dtype=float)
    t = einsums.create_zero_tensor(f"sv{next(_ctr)}", list(a.shape), dtype="float64")
    if a.size:
        np.asarray(t)[...] = a
    return t


def _scalar():
    return einsums.create_zero_tensor(f"sv{next(_ctr)}", [], dtype="float64")


def _emax(v):
    r = _scalar()
    einsums.linalg.max(r, _mk(v))
    return float(np.asarray(r))


def test_max_propagates_nan():
    # Any NaN -> NaN (matches numpy.max, not nanmax), regardless of position.
    for v in ([1.0, NAN, 2.0], [NAN, 1.0, 2.0], [1.0, 2.0, NAN], [NAN, NAN]):
        assert np.isnan(_emax(v)), f"max({v}) should be NaN"
    # No NaN: ordinary maximum, inf handled.
    assert _emax([3.0, 1.0, 2.0]) == 3.0
    assert _emax([1.0, INF, 2.0]) == INF
    assert _emax([1.0, -INF]) == 1.0


def test_elementwise_ieee_passthrough():
    A = np.array([[1.0, INF], [NAN, -INF]])
    B = np.full((2, 2), 2.0)
    # scale
    t = _mk(A)
    einsums.linalg.scale(3.0, t)
    np.testing.assert_allclose(np.asarray(t), 3.0 * A, equal_nan=True)
    # axpy
    y = _mk(A.copy())
    x = _mk(B)
    einsums.linalg.axpy(1.0, x, y)
    np.testing.assert_allclose(np.asarray(y), A + B, equal_nan=True)
    # direct_product
    a, b, c = _mk(A), _mk(B), _mk(np.zeros((2, 2)))
    einsums.linalg.direct_product(1.0, a, b, 0.0, c)
    np.testing.assert_allclose(np.asarray(c), A * B, equal_nan=True)
    # direct_division through a zero -> inf
    Bz = np.array([[0.0, 1.0], [2.0, 0.0]])
    num, den, out = _mk(np.ones((2, 2))), _mk(Bz), _mk(np.zeros((2, 2)))
    einsums.linalg.direct_division(1.0, num, den, 0.0, out)
    np.testing.assert_allclose(np.asarray(out), np.ones((2, 2)) / Bz, equal_nan=True)


def test_gemm_einsum_ieee_passthrough():
    A = np.array([[1.0, INF], [1.0, 1.0]])
    B = np.ones((2, 2))
    # gemm (graph) -- operands kept alive across execute
    At, Bt, Ct = _mk(A), _mk(B), _mk(np.zeros((2, 2)))
    g = cg.Graph(f"sv{next(_ctr)}")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, At, Bt, 0.0, Ct)
    g.execute()
    np.testing.assert_allclose(np.asarray(Ct), A @ B, equal_nan=True)
    # einsum (graph)
    An = np.array([[NAN, 1.0], [1.0, 1.0]])
    Ae, Be, Ce = _mk(An), _mk(B), _mk(np.zeros((2, 2)))
    g2 = cg.Graph(f"sv{next(_ctr)}")
    with cg.capture(g2):
        einsums.einsum("ij <- ik ; kj", Ce, Ae, Be)
    g2.execute()
    np.testing.assert_allclose(np.asarray(Ce), An @ B, equal_nan=True)
