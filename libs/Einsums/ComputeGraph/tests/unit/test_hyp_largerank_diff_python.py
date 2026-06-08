# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: larger-RANK einsum vs numpy.einsum.

Pushes more indices per role (batch / M / N / K) than the baseline einsum harness
-- operands up to ~rank 6-7 -- with permuted (transposed) orders, small extents
(1..4, total size bounded), real/complex dtypes and accumulation. Exercises the
multi-K flatten / batched-gather / pack paths at depth. numpy.einsum is the oracle.

Clean when mined (2500 examples 0 failures); guards the deep-rank paths.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()
_LETTERS = "ijklmnpqrs"


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"lr{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


def _mkview(arr, use_view, dt, rng):
    if not use_view or arr.ndim < 2:
        return _mk(arr, dt)
    perm = list(rng.permutation(arr.ndim))
    if perm == list(range(arr.ndim)):
        perm = perm[::-1]
    return _mk(np.ascontiguousarray(np.transpose(arr, perm)), dt).permute_view(list(np.argsort(perm)))


@st.composite
def _prob(draw):
    nb = draw(st.integers(0, 2))
    nm = draw(st.integers(1, 3))
    nn = draw(st.integers(1, 3))
    nk = draw(st.integers(0, 3))
    tot = nb + nm + nn + nk
    assume(tot <= 9)
    L = draw(st.permutations(list(_LETTERS)))[:tot]
    p = 0
    batch = L[p:p + nb]; p += nb
    mids = L[p:p + nm]; p += nm
    nids = L[p:p + nn]; p += nn
    kids = L[p:p + nk]; p += nk
    ext = {ix: draw(st.integers(1, 4)) for ix in L[:tot]}
    a = draw(st.permutations(batch + mids + kids))
    b = draw(st.permutations(batch + kids + nids))
    c = draw(st.permutations(batch + mids + nids))
    return (a, b, c, ext, draw(st.sampled_from(["float64", "complex128"])),
            draw(st.sampled_from([0.0, 1.0])), draw(st.sampled_from([1.0, -2.0])),
            draw(st.booleans()), draw(st.booleans()))


@given(prob=_prob())
@settings(max_examples=350, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_largerank_diff(prob):
    a_idx, b_idx, c_idx, ext, dt, c_pf, ab_pf, va, vb = prob
    cplx = (dt == "complex128")
    rng = np.random.default_rng(0)
    assume(int(np.prod([ext[x] for x in set(a_idx) | set(b_idx) | set(c_idx)])) <= 20000)
    A0 = _rnd([ext[x] for x in a_idx], cplx, rng)
    B0 = _rnd([ext[x] for x in b_idx], cplx, rng)
    C0 = _rnd([ext[x] for x in c_idx], cplx, rng)
    np_spec = f"{''.join(a_idx)},{''.join(b_idx)}->{''.join(c_idx)}"
    oracle = c_pf * C0 + ab_pf * np.einsum(np_spec, A0, B0)
    es = f"{''.join(c_idx)} <- {''.join(a_idx)} ; {''.join(b_idx)}"
    At = _mkview(A0, va, dt, rng)
    Bt = _mkview(B0, vb, dt, rng)
    Ct = _mk(C0, dt)
    g = cg.Graph(f"lr{next(_ctr)}")
    with cg.capture(g):
        einsums.einsum(es, Ct, At, Bt, c_pf=c_pf, ab_pf=ab_pf)
    g.execute()
    np.testing.assert_allclose(np.asarray(Ct), oracle, rtol=1e-8, atol=1e-9,
        err_msg=f"{es} ext={ext} dt={dt} c_pf={c_pf} ab_pf={ab_pf} va={va} vb={vb}")
