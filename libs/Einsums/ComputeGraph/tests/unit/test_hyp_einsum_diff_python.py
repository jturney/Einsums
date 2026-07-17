# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: einsum (graph-captured and eager) vs numpy.einsum.

Generates VALID two-operand contractions across the whole role model
(batch / M / N / K indices, with K allowed to be 0 so pure outer products are
covered), permutes the index order within each operand AND the output to
exercise transposes and non-contiguous targets, and draws each extent 1..N so
size-1 boundaries are covered. Operands may be passed as non-contiguous
permuted VIEWS, the dtype may be real or complex, and the prefactors exercise
accumulation (c_pf != 0). Each draw runs one of two execution modes:

  * graph: capture into a Graph, optionally apply the default pass manager,
    then execute (the capture/optimize/replay route);
  * eager: call einsums.einsum() with no capture, running immediately through
    the runtime string-einsum dispatch.

numpy.einsum is the oracle. Note both modes go through the same runtime
StringDispatch; the eager mode does not reach the eager *compile-time*
``Indices{}`` template path (that one is C++-only and covered by the
TensorAlgebra OuterProduct unit test).

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
    # extent 0 is deliberately drawable: empty tensors exercise the
    # zero-iteration early-return paths and BLAS calls with m/n/k = 0.
    extent = {ix: draw(st.sampled_from([0, 1, 1, 2, 2, 3])) for ix in letters[:total]}
    a_idx = draw(st.permutations(batch + mids + kids))
    b_idx = draw(st.permutations(batch + kids + nids))
    c_idx = draw(st.permutations(batch + mids + nids))
    # Optionally repeat a letter WITHIN an input operand: 'iik,kj' style
    # diagonal access (bug-1023 was invisible because specs only ever used
    # distinct letters). np.einsum supports repeated input letters, so the
    # oracle stays valid; repeated OUTPUT letters are not expressible in
    # np.einsum and are covered by the C++ EagerParityGaps suite instead.
    if a_idx and draw(st.booleans()):
        dup = draw(st.sampled_from(a_idx))
        ins = draw(st.integers(0, len(a_idx)))
        a_idx = a_idx[:ins] + [dup] + a_idx[ins:]
    if b_idx and draw(st.booleans()):
        dup = draw(st.sampled_from(b_idx))
        ins = draw(st.integers(0, len(b_idx)))
        b_idx = b_idx[:ins] + [dup] + b_idx[ins:]
    # Trace letter: a FRESH letter doubled within one input and absent from
    # the other operand and from C - a self-contraction ("j <- iik ; kj"
    # sums A's diagonal). Distinct from the duplication above, which only
    # repeats letters that also appear elsewhere.
    unused = [x for x in draw(st.permutations(list(_LETTERS))) if x not in letters[:total]]
    if unused and draw(st.booleans()):
        tr = unused[0]
        extent[tr] = draw(st.sampled_from([0, 1, 2, 3]))
        which = draw(st.sampled_from(["a", "b"]))
        if which == "a":
            ins = draw(st.integers(0, len(a_idx)))
            a_idx = a_idx[:ins] + [tr, tr] + a_idx[ins:]
        else:
            ins = draw(st.integers(0, len(b_idx)))
            b_idx = b_idx[:ins] + [tr, tr] + b_idx[ins:]
    # Operand aliasing: pass the SAME tensor object as both A and B. Forces
    # b_idx = a_idx so shapes line up; exercises capture-side handling of one
    # TensorId appearing as both inputs of a single einsum node.
    alias_ab = draw(st.booleans()) and not any(x in c_idx and a_idx.count(x) != b_idx.count(x) for x in set(a_idx + b_idx))
    if alias_ab:
        b_idx = list(a_idx)
    dt = draw(st.sampled_from(["float64", "complex128"]))
    # Complex dtypes may carry COMPLEX prefactors (stored as PrefactorScalar,
    # a variant type - no narrowing through capture/replay).
    if dt == "complex128":
        c_pf = draw(st.sampled_from([0.0, 1.0, 1.0 + 2.0j]))
        ab_pf = draw(st.sampled_from([1.0, -2.0, 0.5 - 1.0j]))
    else:
        c_pf = draw(st.sampled_from([0.0, 1.0]))
        ab_pf = draw(st.sampled_from([1.0, -2.0]))
    return (a_idx, b_idx, c_idx, extent, alias_ab,
            dt, c_pf, ab_pf,
            # view_a, view_b, passes, eager
            draw(st.booleans()), draw(st.booleans()), draw(st.booleans()), draw(st.booleans()))


@given(prob=_einsum_problem())
@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prob=(["j", "l", "i"], ["l", "k", "i"], ["k", "j", "i"],
               {"i": 1, "j": 1, "k": 2, "l": 1}, False, "float64", 0.0, 1.0, False, False, False, False))  # batched transposed output (graph)
@example(prob=(["b", "i", "k"], ["b", "k", "j"], ["b", "i", "j"],
               {"b": 2, "i": 3, "k": 2, "j": 2}, False, "complex128", 1.0, 1.0, True, True, True, False))   # canonical batched matmul (graph)
@example(prob=(["a", "c"], ["b"], ["a", "b", "c"],
               {"a": 3, "b": 3, "c": 3}, False, "float64", 0.0, 1.0, False, False, False, True))            # eager outer product, non-contiguous output
@example(prob=(["a", "c"], ["b"], ["a", "b", "c"],
               {"a": 3, "b": 3, "c": 3}, False, "float64", 0.0, 1.0, False, False, True, False))            # graph+passes outer product, non-contiguous output
@example(prob=(["i", "i"], ["j", "j"], ["i", "j"],
               {"i": 3, "j": 4}, False, "float64", 0.0, 1.0, False, False, False, False))                   # Hadamard diagonals (bug-1023, graph)
@example(prob=(["i", "i", "k"], ["k", "j"], ["i", "j"],
               {"i": 3, "j": 2, "k": 2}, False, "complex128", 1.0, -2.0, False, False, True, False))        # diagonal-contract + passes (bug-1023)
def test_hyp_einsum_diff(prob):
    a_idx, b_idx, c_idx, extent, alias_ab, dt, c_pf, ab_pf, view_a, view_b, passes, eager = prob
    rng = np.random.default_rng(0)
    A0 = _rnd([extent[x] for x in a_idx], dt, rng)
    B0 = A0 if alias_ab else _rnd([extent[x] for x in b_idx], dt, rng)
    C0 = _rnd([extent[x] for x in c_idx], dt, rng)
    np_spec = f"{''.join(a_idx)},{''.join(b_idx)}->{''.join(c_idx)}"
    oracle = c_pf * C0 + ab_pf * np.einsum(np_spec, A0, B0)
    es_spec = f"{''.join(c_idx)} <- {''.join(a_idx)} ; {''.join(b_idx)}"
    At = _mk_maybe_view(A0, view_a, dt, rng)
    # Aliased operands: the SAME tensor object appears as both inputs, so
    # the capture sees one TensorId twice in a single einsum node.
    Bt = At if alias_ab else _mk_maybe_view(B0, view_b, dt, rng)
    Ct = _mk(C0, dt)
    if eager:
        # Eager: no graph capture. Runs immediately through the runtime
        # string-einsum dispatch, exercising that path directly instead of
        # the graph capture/optimize/replay route. (passes is irrelevant here.)
        einsums.einsum(es_spec, Ct, At, Bt, c_pf=c_pf, ab_pf=ab_pf)
    else:
        g = cg.Graph(_nm())
        with cg.capture(g):
            einsums.einsum(es_spec, Ct, At, Bt, c_pf=c_pf, ab_pf=ab_pf)
        if passes:
            g.apply(cg.default_pass_manager())
        g.execute()
    np.testing.assert_allclose(
        np.asarray(Ct), oracle, rtol=1e-9, atol=1e-9,
        err_msg=f"einsums '{es_spec}' dt={dt} c_pf={c_pf} ab_pf={ab_pf} "
                f"view_a={view_a} view_b={view_b} passes={passes} eager={eager} extents={extent}")
