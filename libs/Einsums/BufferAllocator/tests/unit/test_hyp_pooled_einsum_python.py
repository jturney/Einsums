# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Hypothesis differential: pooled operands vs owned operands vs numpy.

A pooled tensor is supposed to be indistinguishable from an owned one to every
consumer - the storage mode varies at runtime, the type does not. This draws
two-operand contractions and runs each one TWICE over identical data, once with
every operand carved from a :class:`einsums.MemoryPool` and once with ordinary
owned tensors, then checks both against ``numpy.einsum``.

Three things this reaches that the unit tests do not:

  * the whole dispatch cascade (BLAS specializations, packed GEMM, the generic
    loop) sees pooled storage, chosen by the shapes hypothesis draws rather
    than by shapes someone picked;
  * capture, the pass manager and replay run on pooled operands, so a pass that
    reads a tensor's storage mode - MemoryPlanning's arena, InplaceOptimization -
    is exercised against them;
  * carve reuse is under test at the same time, because every draw frees its
    tensors into a pool the next draw carves from. A contraction that read a
    neighbour's bytes would show up as a numeric mismatch, not as a leak.

Aliased operands (A and B the same tensor) and in-place accumulation
(``c_pf != 0``) are drawn too, since both change which alias roots the graph
sees and pooled carves are meant to stay independent roots.
"""

from __future__ import annotations

import gc
import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, tolerance_for

_ctr = itertools.count()

#: One pool for the module, so every draw carves from an arena the previous
#: draws have already used and released. A fresh pool per draw would only ever
#: test cold, zero-filled OS pages, which is the case least likely to fail.
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = einsums.MemoryPool(64 * 1024 * 1024, "hyp-einsum")
    return _POOL


def _nm() -> str:
    return f"hpe{next(_ctr)}"


def _owned(arr, dtype):
    t = einsums.create_zero_tensor(_nm(), list(arr.shape), dtype=dtype)
    if arr.size:
        np.asarray(t)[...] = arr
    return t


def _pooled(arr, dtype):
    t = _pool().empty(list(arr.shape), dtype=dtype, name=_nm())
    if arr.size:
        np.asarray(t)[...] = arr
    return t


@st.composite
def _problem(draw):
    # Roles, in the two-operand model the dispatcher keys on: batch indices in
    # all three operands, M only in A and C, N only in B and C, K contracted.
    # Each may be absent, so pure outer products (no K) and pure GEMMs (no
    # batch) are both drawn.
    have = {r: draw(st.booleans()) for r in ("b", "m", "n", "k")}
    # Every operand needs at least one index: einsums rejects an empty operand
    # spec outright, and a contracted index satisfies both A and B at once.
    if not (have["b"] or have["m"] or have["k"]) or not (have["b"] or have["k"] or have["n"]):
        have["k"] = True
    extent = {r: draw(st.integers(min_value=0, max_value=4)) for r in have}

    a_idx = [r for r in ("b", "m", "k") if have[r]]
    b_idx = [r for r in ("b", "k", "n") if have[r]]
    c_idx = [r for r in ("b", "m", "n") if have[r]]

    a_idx = draw(st.permutations(a_idx))
    b_idx = draw(st.permutations(b_idx))
    c_idx = draw(st.permutations(c_idx))

    return (a_idx, b_idx, c_idx, extent,
            draw(st.sampled_from(ALL_DTYPES)),
            draw(st.sampled_from([0.0, 1.0])),      # c_pf: overwrite or accumulate
            draw(st.sampled_from([1.0, -2.0])),     # ab_pf
            draw(st.booleans()),                    # alias A and B
            draw(st.booleans()),                    # apply the pass manager
            draw(st.booleans()))                    # eager instead of captured


def _run(spec, mk, dtype, c_pf, ab_pf, A0, B0, C0, alias, passes, eager):
    At = mk(A0, dtype)
    Bt = At if alias else mk(B0, dtype)
    Ct = mk(C0, dtype)
    if eager:
        einsums.einsum(spec, Ct, At, Bt, c_pf=c_pf, ab_pf=ab_pf)
    else:
        g = cg.Graph(_nm())
        with cg.capture(g):
            einsums.einsum(spec, Ct, At, Bt, c_pf=c_pf, ab_pf=ab_pf)
        if passes:
            g.apply(cg.default_pass_manager())
        g.execute()
    return np.array(np.asarray(Ct), copy=True)


@given(prob=_problem())
@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large,
                                 HealthCheck.filter_too_much])
def test_hyp_pooled_einsum_matches_owned(prob):
    a_idx, b_idx, c_idx, extent, dtype, c_pf, ab_pf, alias, passes, eager = prob

    rng = np.random.default_rng(abs(hash((tuple(a_idx), tuple(b_idx), tuple(c_idx),
                                          tuple(sorted(extent.items())), dtype))) % (2**32))

    def data(idx):
        shape = [extent[x] for x in idx]
        out = rng.integers(-4, 5, size=shape).astype(dtype)
        if np.iscomplexobj(out):
            out = out + 1j * rng.integers(-4, 5, size=shape)
        return out.astype(dtype)

    # Aliasing means ONE tensor stands in for both operands, so the two index
    # lists must match in order as well as content: a permuted spec over the
    # same tensor would claim extents it does not have.
    alias = alias and a_idx == b_idx

    A0 = data(a_idx)
    B0 = A0 if alias else data(b_idx)
    C0 = data(c_idx) if c_idx else data([])

    np_spec = f"{''.join(a_idx)},{''.join(b_idx)}->{''.join(c_idx)}"
    oracle = c_pf * C0 + ab_pf * np.einsum(np_spec, A0, B0)
    es_spec = f"{''.join(c_idx)} <- {''.join(a_idx)} ; {''.join(b_idx)}"

    args = (es_spec, dtype, c_pf, ab_pf, A0, B0, C0, alias, passes, eager)
    from_pool = _run(args[0], _pooled, *args[1:])
    from_heap = _run(args[0], _owned, *args[1:])

    rtol, atol = tolerance_for(dtype)
    msg = (f"spec={es_spec} dtype={dtype} c_pf={c_pf} ab_pf={ab_pf} alias={alias} "
           f"passes={passes} eager={eager} extents={extent}")
    # Against the owned run first: a difference there is the pooling itself,
    # with the oracle held constant.
    np.testing.assert_allclose(from_pool, from_heap, rtol=rtol, atol=atol,
                               err_msg=f"pooled != owned: {msg}")
    np.testing.assert_allclose(from_pool, oracle, rtol=rtol, atol=atol,
                               err_msg=f"pooled != numpy: {msg}")


def test_pool_is_empty_after_the_sweep():
    """Every draw's carves came back.

    Runs after the fuzzer in file order, and is the leak check the differential
    cannot be: a pooled operand that outlived its draw would still compare
    equal, but would leave the pool holding bytes.
    """
    # Hypothesis holds a falsifying example's frames, and a traceback keeps the
    # tensors in them alive; collect first so the check reads real leaks only.
    gc.collect()
    pool = _pool()
    assert pool.live_borrows == 0
    assert pool.bytes_used == 0
