# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: einsum (graph-captured and eager) vs numpy.einsum.

Generates VALID two-operand contractions across the whole role model
(batch / M / N / K indices, with K allowed to be 0 so pure outer products are
covered), optionally adds a lone reduction index summed in one operand only (a
"weighted trace": link (A & B) \\ C empty while a summed index survives, which
the K-only role model cannot otherwise emit), permutes the index order within
each operand AND the output to
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
from einsums.testing import assert_exact, integer_data

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
    a = rng.standard_normal(shape)
    if dt.startswith("complex"):
        a = a + 1j * rng.standard_normal(shape)
    # Cast to the target dtype so the oracle and the tensor computation see
    # bit-identical inputs (otherwise 32-bit runs compare a float32 result
    # against a float64-input oracle and eat the input-rounding difference).
    return a.astype(dt)


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
    # Lone reduction index ("weighted trace"): a FRESH letter placed ONCE into
    # exactly one operand, absent from the other operand and from C. It is
    # summed (absent from the output) yet shares no link with the other
    # operand, so link = (A & B) \ C is EMPTY while a summed index still
    # survives - e.g. "ij <- ijk ; ij" sums S over k against the weight W.
    # The GEMM role model above cannot emit this: every summed index there is a
    # K index, which by construction sits in BOTH operands (a real link). Every
    # standard contraction (Fock J/K, GEMM, GEMV) has a non-empty link, which
    # is exactly why this class went untested. np.einsum sums it as the oracle.
    # Distinct from the doubled trace letter below, which repeats one letter
    # within an operand (a diagonal self-trace); here the letter occurs once.
    # Draw 0..2 lone indices: two exercise the generic loop's multi-summed-axis
    # decoding (a lone index in A AND one in B, or two on the same operand,
    # alongside any real link), which a single lone letter never reaches.
    used      = set(letters[:total])
    lone_pool = [x for x in draw(st.permutations(list(_LETTERS))) if x not in used]
    n_lone    = draw(st.integers(0, min(2, len(lone_pool))))
    for li in range(n_lone):
        ln = lone_pool[li]
        used.add(ln)
        extent[ln] = draw(st.sampled_from([0, 1, 2, 3]))
        if draw(st.booleans()):
            ins   = draw(st.integers(0, len(a_idx)))
            a_idx = a_idx[:ins] + [ln] + a_idx[ins:]
        else:
            ins   = draw(st.integers(0, len(b_idx)))
            b_idx = b_idx[:ins] + [ln] + b_idx[ins:]
    # Optionally repeat a letter WITHIN an input operand: 'iik,kj' style
    # diagonal access (a silent miscompilation was invisible while specs only used
    # distinct letters). np.einsum supports repeated input letters, so the
    # oracle stays valid; repeated OUTPUT letters are not expressible in
    # np.einsum and are covered by the C++ EagerParityGaps suite instead.
    # Insert 1..2 extra copies so a letter can reach multiplicity 3 (a triple
    # diagonal 'iii', not just the doubled 'ii'); numpy sums a higher-order
    # diagonal the same way, so the oracle stays valid.
    if a_idx and draw(st.booleans()):
        dup = draw(st.sampled_from(a_idx))
        for _ in range(draw(st.integers(1, 2))):
            ins   = draw(st.integers(0, len(a_idx)))
            a_idx = a_idx[:ins] + [dup] + a_idx[ins:]
    if b_idx and draw(st.booleans()):
        dup = draw(st.sampled_from(b_idx))
        for _ in range(draw(st.integers(1, 2))):
            ins   = draw(st.integers(0, len(b_idx)))
            b_idx = b_idx[:ins] + [dup] + b_idx[ins:]
    # Trace letter: a FRESH letter doubled within one input and absent from
    # the other operand and from C - a self-contraction ("j <- iik ; kj"
    # sums A's diagonal). Distinct from the duplication above, which only
    # repeats letters that also appear elsewhere.
    unused = [x for x in draw(st.permutations(list(_LETTERS))) if x not in used]
    if unused and draw(st.booleans()):
        tr = unused[0]
        extent[tr] = draw(st.sampled_from([0, 1, 2, 3]))
        which = draw(st.sampled_from(["a", "b"]))
        reps  = draw(st.integers(2, 3))  # doubled self-trace, or a tripled diagonal trace
        if which == "a":
            ins   = draw(st.integers(0, len(a_idx)))
            a_idx = a_idx[:ins] + [tr] * reps + a_idx[ins:]
        else:
            ins   = draw(st.integers(0, len(b_idx)))
            b_idx = b_idx[:ins] + [tr] * reps + b_idx[ins:]
    # Operand aliasing: pass the SAME tensor object as both A and B. Forces
    # b_idx = a_idx so shapes line up; exercises capture-side handling of one
    # TensorId appearing as both inputs of a single einsum node.
    alias_ab = draw(st.booleans()) and not any(x in c_idx and a_idx.count(x) != b_idx.count(x) for x in set(a_idx + b_idx))
    if alias_ab:
        b_idx = list(a_idx)
    dt = draw(st.sampled_from(["float64", "complex128", "float32", "complex64"]))
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


def _run_einsum_diff(prob, exact):
    """Execute one drawn contraction and compare to numpy.

    ``exact=False`` uses random float data and a dtype-scaled tolerance.
    ``exact=True`` snaps to the integer regime (float64/complex128, integer
    prefactors, integer_data) so the result is bit-exact and compared with
    ``assert_exact`` - which turns a dropped axis or a transposed tail block
    into a guaranteed mismatch instead of a small error a tolerance can hide.
    The drawn dtype's complex-ness and the prefactors' accumulate/sign
    structure are preserved through the snap.
    """
    a_idx, b_idx, c_idx, extent, alias_ab, dt, c_pf, ab_pf, view_a, view_b, passes, eager = prob
    rng = np.random.default_rng(0)
    if exact:
        dt    = "complex128" if dt.startswith("complex") else "float64"
        c_pf  = 0.0 if c_pf == 0 else 1.0
        ab_pf = -2.0 if np.real(ab_pf) < 0 else 1.0
        gen   = lambda shape: integer_data(shape, dt, rng)
    else:
        gen = lambda shape: _rnd(shape, dt, rng)
    A0 = gen([extent[x] for x in a_idx])
    B0 = A0 if alias_ab else gen([extent[x] for x in b_idx])
    C0 = gen([extent[x] for x in c_idx])
    np_spec = f"{''.join(a_idx)},{''.join(b_idx)}->{''.join(c_idx)}"
    oracle  = c_pf * C0 + ab_pf * np.einsum(np_spec, A0, B0)
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
    if exact:
        assert_exact(np.asarray(Ct), oracle)
    else:
        rt, at = (2e-3, 1e-4) if dt in ("float32", "complex64") else (1e-9, 1e-9)
        np.testing.assert_allclose(
            np.asarray(Ct), oracle, rtol=rt, atol=at,
            err_msg=f"einsums '{es_spec}' dt={dt} c_pf={c_pf} ab_pf={ab_pf} "
                    f"view_a={view_a} view_b={view_b} passes={passes} eager={eager} extents={extent}")


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
               {"i": 3, "j": 4}, False, "float64", 0.0, 1.0, False, False, False, False))                   # Hadamard diagonals (regression, graph)
@example(prob=(["i", "i", "k"], ["k", "j"], ["i", "j"],
               {"i": 3, "j": 2, "k": 2}, False, "complex128", 1.0, -2.0, False, False, True, False))        # diagonal-contract + passes (regression)
@example(prob=(["i", "j", "k"], ["i", "j"], ["i", "j"],
               {"i": 2, "j": 3, "k": 4}, False, "float64", 0.0, 1.0, False, False, True, False))            # P1 weighted trace: C_ij = (sum_k S_ijk) W_ij, empty link (graph+passes)
@example(prob=(["i", "j", "k"], ["j", "k"], ["j", "k"],
               {"i": 2, "j": 3, "k": 4}, False, "float64", 0.0, 1.0, False, False, False, False))           # P2 weighted trace: i summed, only in S (graph)
@example(prob=(["i", "j"], ["i"], ["i"],
               {"i": 3, "j": 4}, False, "float64", 0.0, 1.0, False, False, False, True))                    # P3 weighted trace: rank-2 x rank-1, j summed only in A (eager)
@example(prob=(["j", "l"], ["m", "l", "k"], ["j", "k"],
               {"j": 2, "l": 3, "k": 2, "m": 4}, False, "float64", 0.0, 1.0, False, False, True, False))     # link (l) + lone summed (m in B only): fast path dropped m before the guard (graph+passes)
@example(prob=(["i", "p", "l"], ["q", "l", "j"], ["i", "j"],
               {"i": 2, "p": 3, "l": 4, "q": 2, "j": 3}, False, "float64", 0.0, 1.0, False, False, True, False))   # two lone (p in A, q in B) + link (l): multi-summed-axis decoding (graph+passes)
@example(prob=(["i", "p", "q", "l"], ["l", "j"], ["i", "j"],
               {"i": 2, "p": 2, "q": 3, "l": 2, "j": 3}, False, "float64", 0.0, 1.0, False, False, False, False))  # two lone same operand (p, q in A) + link (l) (graph)
@example(prob=(["i", "p"], ["q", "j"], ["i", "j"],
               {"i": 2, "p": 3, "q": 2, "j": 3}, False, "float64", 0.0, 1.0, False, False, False, True))          # lone in A (p) + lone in B (q), empty link (eager)
@example(prob=(["i", "j"], ["i"], [],
               {"i": 3, "j": 4}, False, "float64", 0.0, 1.0, False, False, False, True))                          # scalar output with a lone summed index (j in A only) (eager)
def test_hyp_einsum_diff(prob):
    _run_einsum_diff(prob, exact=False)


@given(prob=_einsum_problem())
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prob=(["j", "l"], ["m", "l", "k"], ["j", "k"],
               {"j": 2, "l": 3, "k": 2, "m": 4}, False, "float64", 0.0, 1.0, False, False, True, False))     # link + lone summed, exact
@example(prob=(["i", "p", "l"], ["q", "l", "j"], ["i", "j"],
               {"i": 2, "p": 3, "l": 4, "q": 2, "j": 3}, False, "float64", 0.0, 1.0, False, False, True, False))   # two lone + link, exact
@example(prob=(["i", "j", "k"], ["i", "j"], ["i", "j"],
               {"i": 2, "j": 3, "k": 4}, False, "float64", 0.0, 1.0, False, False, False, False))            # P1 weighted trace, exact
def test_hyp_einsum_diff_exact(prob):
    # Exact-integer differential: integer data + integer prefactors make the
    # contraction bit-exact, so assert_exact catches indexing/tail bugs that a
    # relative tolerance can mask. The @example dtypes/prefactors are snapped to
    # the exact regime inside _run_einsum_diff, so pins run in exact mode too.
    _run_einsum_diff(prob, exact=True)
