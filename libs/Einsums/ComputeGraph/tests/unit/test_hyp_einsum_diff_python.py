# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: graph-captured einsum vs numpy.einsum.

Generates VALID two-operand contractions across the whole role model
(batch / M / N / K indices), permutes the index order within each operand
to exercise transposes, and draws each extent 1..N so size-1 boundaries are
covered. Operands may be passed as non-contiguous permuted VIEWS, the dtype
may be real or complex, the prefactors exercise accumulation (c_pf != 0), and
the default pass manager may be applied before execution. numpy.einsum is the
oracle.

This is the registered/regression form of the local mining harness; the
``@example`` entries pin the specific reducers that were once miscompiled:
  * "kji <- jli ; lki" with i=j=l=1, k=2 -- the capture-time BatchedGemm
    fast path miscomputed a transposed-output batched contraction (commit
    fixing OpKind::BatchedGemm canonical-order gate).
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()
_LETTERS = "ijklmnpqrs"


def _nm() -> str:
    return f"hed{next(_ctr)}"


def _mk(arr, dt):
    t = einsums.create_zero_tensor(_nm(), list(arr.shape), dtype=dt)
    if arr.size:
        np.asarray(t)[...] = arr
    return t


def _mk_maybe_view(arr, use_view, dt, rng):
    """Return a tensor whose logical data is ``arr``; optionally a permuted
    (non-contiguous) view so the einsum sees inflated/out-of-order strides."""
    if not use_view or arr.ndim < 2:
        return _mk(arr, dt)
    perm = list(rng.permutation(arr.ndim))
    if perm == list(range(arr.ndim)):
        perm = perm[::-1]
    t = _mk(np.ascontiguousarray(np.transpose(arr, perm)), dt)
    return t.permute_view(list(np.argsort(perm)))


def _rnd(shape, dt, rng):
    if dt == "complex128":
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


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
    return (a_idx, b_idx, c_idx, extent,
            draw(st.sampled_from(["float64", "complex128"])),
            draw(st.sampled_from([0.0, 1.0])), draw(st.sampled_from([1.0, -2.0])),
            draw(st.booleans()), draw(st.booleans()), draw(st.booleans()))


@given(prob=_einsum_problem())
@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prob=(["j", "l", "i"], ["l", "k", "i"], ["k", "j", "i"],
               {"i": 1, "j": 1, "k": 2, "l": 1}, "float64", 0.0, 1.0, False, False, False))  # batched transposed output
@example(prob=(["b", "i", "k"], ["b", "k", "j"], ["b", "i", "j"],
               {"b": 2, "i": 3, "k": 2, "j": 2}, "complex128", 1.0, 1.0, True, True, True))   # canonical batched matmul
def test_hyp_einsum_diff(prob):
    a_idx, b_idx, c_idx, extent, dt, c_pf, ab_pf, view_a, view_b, passes = prob
    rng = np.random.default_rng(0)
    A0 = _rnd([extent[x] for x in a_idx], dt, rng)
    B0 = _rnd([extent[x] for x in b_idx], dt, rng)
    C0 = _rnd([extent[x] for x in c_idx], dt, rng)
    np_spec = f"{''.join(a_idx)},{''.join(b_idx)}->{''.join(c_idx)}"
    oracle = c_pf * C0 + ab_pf * np.einsum(np_spec, A0, B0)
    es_spec = f"{''.join(c_idx)} <- {''.join(a_idx)} ; {''.join(b_idx)}"
    At = _mk_maybe_view(A0, view_a, dt, rng)
    Bt = _mk_maybe_view(B0, view_b, dt, rng)
    Ct = _mk(C0, dt)
    g = cg.Graph(_nm())
    with cg.capture(g):
        einsums.einsum(es_spec, Ct, At, Bt, c_pf=c_pf, ab_pf=ab_pf)
    if passes:
        g.apply(cg.default_pass_manager())
    g.execute()
    np.testing.assert_allclose(
        np.asarray(Ct), oracle, rtol=1e-9, atol=1e-9,
        err_msg=f"einsums '{es_spec}' dt={dt} c_pf={c_pf} ab_pf={ab_pf} "
                f"view_a={view_a} view_b={view_b} passes={passes} extents={extent}")
