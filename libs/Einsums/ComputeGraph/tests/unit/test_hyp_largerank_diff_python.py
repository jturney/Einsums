# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: larger-RANK einsum vs numpy.einsum.

Pushes more indices per role (batch / M / N / K) than the baseline einsum harness
-- operands up to ~rank 6-7 -- with permuted (transposed) orders, small extents
(1..4, total size bounded), real/complex dtypes and accumulation, optionally a
lone reduction index summed in one operand only (weighted trace, empty link).
Exercises the multi-K flatten / batched-gather / pack paths at depth, plus the
lone-summed path at rank. numpy.einsum is the oracle.

Clean when mined (2500 examples 0 failures); guards the deep-rank paths.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from einsums.testing import assert_exact, integer_data

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
    # Lone reduction index ("weighted trace"): a fresh letter in exactly one
    # operand, absent from the other operand and from C - summed with no shared
    # link, the empty-link/summed-index shape the K-only role model cannot emit
    # (see the note in test_hyp_einsum_diff). Exercised here at larger rank.
    lone_pool = [x for x in draw(st.permutations(list(_LETTERS))) if x not in L[:tot]]
    n_lone    = draw(st.integers(0, min(2, len(lone_pool))))
    for li in range(n_lone):
        ln = lone_pool[li]
        ext[ln] = draw(st.integers(1, 4))
        if draw(st.booleans()):
            ins = draw(st.integers(0, len(a)))
            a = a[:ins] + [ln] + a[ins:]
        else:
            ins = draw(st.integers(0, len(b)))
            b = b[:ins] + [ln] + b[ins:]
    return (a, b, c, ext, draw(st.sampled_from(["float64", "complex128"])),
            draw(st.sampled_from([0.0, 1.0])), draw(st.sampled_from([1.0, -2.0])),
            draw(st.booleans()), draw(st.booleans()))


def _run_largerank(prob, exact):
    """Execute one deep-rank contraction and compare to numpy.

    largerank already draws only float64/complex128 with integer prefactors, so
    exact mode simply swaps random floats for integer_data and compares with
    assert_exact (bit-exact) instead of a tolerance.
    """
    a_idx, b_idx, c_idx, ext, dt, c_pf, ab_pf, va, vb = prob
    cplx = (dt == "complex128")
    rng = np.random.default_rng(0)
    assume(int(np.prod([ext[x] for x in set(a_idx) | set(b_idx) | set(c_idx)])) <= 20000)
    gen = (lambda idx: integer_data([ext[x] for x in idx], dt, rng)) if exact \
        else (lambda idx: _rnd([ext[x] for x in idx], cplx, rng))
    A0, B0, C0 = gen(a_idx), gen(b_idx), gen(c_idx)
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
    if exact:
        assert_exact(np.asarray(Ct), oracle)
    else:
        np.testing.assert_allclose(np.asarray(Ct), oracle, rtol=1e-8, atol=1e-9,
            err_msg=f"{es} ext={ext} dt={dt} c_pf={c_pf} ab_pf={ab_pf} va={va} vb={vb}")


@given(prob=_prob())
@settings(max_examples=350, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prob=(["i", "j", "l"], ["m", "i", "l", "k"], ["i", "j", "k"],
               {"i": 2, "j": 3, "k": 2, "l": 4, "m": 5}, "float64", 0.0, 1.0, False, False))  # link (l) + lone summed (m in B only): fast path dropped m before the guard
def test_hyp_largerank_diff(prob):
    _run_largerank(prob, exact=False)


@given(prob=_prob())
@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prob=(["i", "j", "l"], ["m", "i", "l", "k"], ["i", "j", "k"],
               {"i": 2, "j": 3, "k": 2, "l": 4, "m": 5}, "float64", 0.0, 1.0, False, False))  # link + lone, exact
def test_hyp_largerank_diff_exact(prob):
    _run_largerank(prob, exact=True)
