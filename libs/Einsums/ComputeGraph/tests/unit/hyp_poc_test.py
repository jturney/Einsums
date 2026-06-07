# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Proof-of-concept: Hypothesis-driven differential testing of einsums.

Mirrors a slice of the hand-rolled ComputeGraph fuzzer, but lets Hypothesis
generate the shapes/ops and (on failure) auto-shrink to a minimal reproducer.

Key point vs the hand-rolled fuzzer: dimensions are drawn from
`st.integers(1, 4)` — Hypothesis includes the boundary value 1 by default and
biases toward it, so the degenerate-dim class is covered without anyone
hardcoding it. The `*_empty` test pushes further (dim 0) into a vector the
hand-rolled fuzzer never explored.

Run:  pytest hyp_poc_test.py -q            (under the ASan DYLD setup)
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

DIM = st.integers(min_value=1, max_value=4)
SCALAR = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False).map(lambda x: round(x, 4))
_ctr = itertools.count()


def nm() -> str:
    return f"h{next(_ctr)}"


def mk(arr):
    t = einsums.create_zero_tensor(nm(), list(arr.shape), dtype="float64")
    if arr.size:
        np.asarray(t)[...] = arr
    return t


_POOL = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


# ── regression guard for fix #1: axpy into a (possibly 1-row) view ──────────
@given(R=DIM, C=DIM, a=SCALAR, data=st.data())
@_POOL
def test_hyp_view_axpy(R, C, a, data):
    r0 = data.draw(st.integers(0, R - 1));  r1 = data.draw(st.integers(r0 + 1, R))
    c0 = data.draw(st.integers(0, C - 1));  c1 = data.draw(st.integers(c0 + 1, C))
    rng = np.random.default_rng(0)
    M0 = rng.standard_normal((R, C))
    S0 = rng.standard_normal((r1 - r0, c1 - c0))
    oracle = M0.copy()
    oracle[r0:r1, c0:c1] += a * S0
    Mt, St = mk(M0), mk(S0)
    g = cg.Graph(nm())
    with cg.capture(g):
        einsums.linalg.axpy(a, St, cg.view(Mt, [(r0, r1), (c0, c1)]))
    g.execute()
    np.testing.assert_allclose(np.asarray(Mt), oracle, rtol=1e-10, atol=1e-10)


# ── gemm with degenerate dims ───────────────────────────────────────────────
@given(m=DIM, k=DIM, n=DIM, a=SCALAR, b=st.sampled_from([0.0, 1.0]))
@_POOL
def test_hyp_gemm(m, k, n, a, b):
    rng = np.random.default_rng(0)
    A0, B0, C0 = rng.standard_normal((m, k)), rng.standard_normal((k, n)), rng.standard_normal((m, n))
    oracle = a * (A0 @ B0) + b * C0
    At, Bt, Ct = mk(A0), mk(B0), mk(C0)
    g = cg.Graph(nm())
    with cg.capture(g):
        einsums.linalg.gemm(a, At, Bt, b, Ct)
    g.execute()
    np.testing.assert_allclose(np.asarray(Ct), oracle, rtol=1e-9, atol=1e-9)


# ── regression guard for fix #2: batched einsum (A^T@B per batch) ────────────
@given(i=DIM, k=DIM, j=DIM, batch=st.integers(1, 3), a=SCALAR)
@_POOL
def test_hyp_beinsum_kib(i, k, j, batch, a):
    rng = np.random.default_rng(0)
    A0 = rng.standard_normal((k, i, batch))
    B0 = rng.standard_normal((k, j, batch))
    C0 = rng.standard_normal((i, j, batch))
    oracle = a * np.einsum("kib,kjb->ijb", A0, B0) + C0
    At, Bt, Ct = mk(A0), mk(B0), mk(C0)
    g = cg.Graph(nm())
    with cg.capture(g):
        einsums.einsum("ijb <- kib ; kjb", Ct, At, Bt, c_pf=1.0, ab_pf=a)
    g.execute()
    np.testing.assert_allclose(np.asarray(Ct), oracle, rtol=1e-9, atol=1e-9)


# ── NEW vector: empty (size-0) extents — never tried by the hand-rolled fuzzer
@given(m=st.integers(0, 3), k=st.integers(0, 3), n=st.integers(0, 3), a=SCALAR)
@_POOL
def test_hyp_gemm_empty(m, k, n, a):
    assume(m == 0 or k == 0 or n == 0)  # only the empty cases here
    rng = np.random.default_rng(0)
    A0, B0, C0 = rng.standard_normal((m, k)), rng.standard_normal((k, n)), rng.standard_normal((m, n))
    oracle = a * (A0 @ B0) + C0
    At, Bt, Ct = mk(A0), mk(B0), mk(C0)
    g = cg.Graph(nm())
    with cg.capture(g):
        einsums.linalg.gemm(a, At, Bt, 1.0, Ct)
    g.execute()
    np.testing.assert_allclose(np.asarray(Ct), oracle, rtol=1e-9, atol=1e-9)
