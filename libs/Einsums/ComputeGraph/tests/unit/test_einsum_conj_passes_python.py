# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Conjugated einsums through the optimization passes (Tier 1).

The single-einsum conj tests never exercise the multi-einsum rewriting passes
(LinearCombinationContractionFolding, GEMMBatching, ChainParenthesization,
DistributiveFactoring) because those only fire on graphs with several related
contractions. Those passes don't thread conj_a/conj_b through their rewrites, so
they conservatively skip conjugated einsums — this test builds graphs shaped to
trigger them (the 2J-K linear-combination idiom, a GEMM chain, a batchable set),
runs the default pass manager, and checks the result still matches numpy. A pass
that silently folded/batched a conj einsum and dropped the flag would diverge.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"ecp{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


@given(pattern=st.sampled_from(["linear_comb", "chain", "batch"]),
       conj_flags=st.tuples(st.booleans(), st.booleans(), st.booleans()),
       dt=st.sampled_from(["float64", "complex128"]), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=400, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_einsum_conj_through_passes(pattern, conj_flags, dt, seed):
    rng = np.random.default_rng(seed)
    cplx = dt == "complex128"
    f0, f1, f2 = conj_flags

    def cj(x, f):
        return np.conj(x) if (f and cplx) else x

    g = cg.Graph(f"ecp{next(_ctr)}")

    if pattern == "linear_comb":
        # 2J-K idiom: O = 2*(opA . opB) - (opA . opC), two accumulating contractions
        # sharing operand A — the shape LinearCombinationContractionFolding targets.
        i, k, j = (int(rng.integers(2, 5)) for _ in range(3))
        A, B, C = _rnd((i, k), cplx, rng), _rnd((k, j), cplx, rng), _rnd((k, j), cplx, rng)
        oracle = 2.0 * (cj(A, f0) @ cj(B, f1)) - (cj(A, f0) @ cj(C, f2))
        At, Bt, Ct, O = _mk(A, dt), _mk(B, dt), _mk(C, dt), _mk(np.zeros((i, j)), dt)
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", O, At, Bt, c_pf=0.0, ab_pf=2.0, conj_a=f0, conj_b=f1)
            einsums.einsum("ij <- ik ; kj", O, At, Ct, c_pf=1.0, ab_pf=-1.0, conj_a=f0, conj_b=f2)
        out = O
    elif pattern == "chain":
        # GEMM chain: T = opA . opB ; O = op(T) . D  (ChainParenthesization target).
        i, k, j, m = (int(rng.integers(2, 5)) for _ in range(4))
        A, B, D = _rnd((i, k), cplx, rng), _rnd((k, j), cplx, rng), _rnd((j, m), cplx, rng)
        tmp = cj(A, f0) @ cj(B, f1)
        oracle = cj(tmp, f2) @ D
        At, Bt, Dt = _mk(A, dt), _mk(B, dt), _mk(D, dt)
        T, O = _mk(np.zeros((i, j)), dt), _mk(np.zeros((i, m)), dt)
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", T, At, Bt, conj_a=f0, conj_b=f1)
            einsums.einsum("im <- ij ; jm", O, T, Dt, conj_a=f2)
        out = O
    else:  # batch: O1 = opA1 . opB, O2 = opA2 . opB  (same B; GEMMBatching target)
        i, k, j = (int(rng.integers(2, 5)) for _ in range(3))
        A1, A2, B = _rnd((i, k), cplx, rng), _rnd((i, k), cplx, rng), _rnd((k, j), cplx, rng)
        o1 = cj(A1, f0) @ cj(B, f2)
        o2 = cj(A2, f1) @ cj(B, f2)
        oracle = o1 + o2
        A1t, A2t, Bt = _mk(A1, dt), _mk(A2, dt), _mk(B, dt)
        O1, O2 = _mk(np.zeros((i, j)), dt), _mk(np.zeros((i, j)), dt)
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", O1, A1t, Bt, conj_a=f0, conj_b=f2)
            einsums.einsum("ij <- ik ; kj", O2, A2t, Bt, conj_a=f1, conj_b=f2)
        cg.default_pass_manager().run(g)
        g.execute()
        np.testing.assert_allclose(np.asarray(O1) + np.asarray(O2), oracle, rtol=1e-6, atol=1e-8,
            err_msg=f"batch conj={conj_flags} dt={dt} seed={seed}")
        return

    cg.default_pass_manager().run(g)
    g.execute()
    np.testing.assert_allclose(np.asarray(out), oracle, rtol=1e-6, atol=1e-8,
        err_msg=f"pattern={pattern} conj={conj_flags} dt={dt} seed={seed}")
