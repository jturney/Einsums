# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Proof-of-concept: Hypothesis-driven differential testing of einsums.

Mirrors a slice of the hand-rolled ComputeGraph fuzzer, but lets Hypothesis
generate the shapes/ops and (on failure) auto-shrink to a minimal reproducer.

Key point vs the hand-rolled fuzzer: dimensions are drawn from
`st.integers(1, 4)`. Hypothesis includes the boundary value 1 by default and
biases toward it, so the degenerate-dim class is covered without anyone
hardcoding it. The `*_empty` test pushes further, to dim 0, into a vector the
hand-rolled fuzzer never explored.

Run:  pytest hyp_poc_test.py -q            (under the ASan DYLD setup)
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, example, given, settings
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


# ── NEW vector: empty (size-0) extents, never tried by the hand-rolled fuzzer
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


# ── #1: arbitrary einsum-spec differential vs numpy.einsum ──────────────────
# The hand-rolled fuzzer hard-codes a handful of contraction patterns. Here we
# generate valid two-operand contractions across the whole role model:
#   batch index -> in A, B, and C
#   M index     -> in A and C   (free of B)
#   N index     -> in B and C   (free of A)
#   K index     -> in A and B, summed out (absent from C); K may be empty (outer
#                  product)
# Index order within each operand is permuted to exercise transposes, and each
# extent is drawn 1..3 so size-1 boundaries are covered. The einsums arrow spec
# and the equivalent numpy spec come from the same role assignment, making
# numpy.einsum an exact oracle for the entire contraction surface.
_LETTERS = "ijklmnpqrs"


@st.composite
def _einsum_problem(draw):
    n_batch = draw(st.integers(0, 1))
    n_m = draw(st.integers(1, 2))
    n_n = draw(st.integers(1, 2))
    n_k = draw(st.integers(0, 2))
    total = n_batch + n_m + n_n + n_k
    letters = draw(st.permutations(list(_LETTERS)))[:total]
    pos = 0
    batch = letters[pos:pos + n_batch]; pos += n_batch
    mids = letters[pos:pos + n_m];      pos += n_m
    nids = letters[pos:pos + n_n];      pos += n_n
    kids = letters[pos:pos + n_k];      pos += n_k
    extent = {ix: draw(st.integers(1, 3)) for ix in letters[:total]}
    a_idx = draw(st.permutations(batch + mids + kids))
    b_idx = draw(st.permutations(batch + kids + nids))
    c_idx = draw(st.permutations(batch + mids + nids))
    return a_idx, b_idx, c_idx, extent


@given(prob=_einsum_problem())
@settings(max_examples=500, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prob=(["i", "k"], ["k", "j"], ["i", "j"], {"i": 2, "k": 1, "j": 3}))   # K=1 degenerate matmul
@example(prob=(["i"], ["j"], ["i", "j"], {"i": 1, "j": 2}))                      # outer product, M=1
@example(prob=(["b", "i", "k"], ["b", "k", "j"], ["b", "i", "j"],                # batched matmul
               {"b": 2, "i": 3, "k": 2, "j": 2}))
def test_hyp_einsum_vs_numpy(prob):
    a_idx, b_idx, c_idx, extent = prob
    rng = np.random.default_rng(0)
    A0 = rng.standard_normal([extent[x] for x in a_idx])
    B0 = rng.standard_normal([extent[x] for x in b_idx])
    np_spec = f"{''.join(a_idx)},{''.join(b_idx)}->{''.join(c_idx)}"
    oracle = np.einsum(np_spec, A0, B0)
    es_spec = f"{''.join(c_idx)} <- {''.join(a_idx)} ; {''.join(b_idx)}"
    At, Bt = mk(A0), mk(B0)
    Ct = mk(np.zeros([extent[x] for x in c_idx]))
    g = cg.Graph(nm())
    with cg.capture(g):
        einsums.einsum(es_spec, Ct, At, Bt)
    g.execute()
    np.testing.assert_allclose(
        np.asarray(Ct), oracle, rtol=1e-9, atol=1e-9,
        err_msg=f"einsums '{es_spec}'  numpy '{np_spec}'  extents={extent}")
