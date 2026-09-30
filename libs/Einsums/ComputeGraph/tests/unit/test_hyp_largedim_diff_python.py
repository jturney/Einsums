# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: larger dimension SIZES through packed-gemm blocking.

The other einsum/gemm harnesses use tiny extents that stay in the small/direct
BLAS path. This one uses sizes around the BLIS micro-kernel tiles (MR=4/8, NR=6)
and across the KC>=64 cache block -- with ragged remainders (MR/NR +/- 1) -- via
gemm-shaped, multi-N and multi-K einsum (which route through pack_A/pack_B + the
MRxNR micro-kernel), plus plain gemm (vendor BLAS). numpy is the oracle.

Clean when mined (700 examples 0 failures); guards the blocking/remainder paths.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from einsums.testing import assert_exact, integer_data
from _dtype_draws import DTYPES, assert_rounding_close, random_array, wide

_ctr = itertools.count()
_CAP = 50000


# Sizes around MR(4)/NR(6) multiples +/- 1 and across KC=64 (capped for CI speed).
_SZ = st.sampled_from([1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 23, 24, 31, 32, 33, 48, 64])
_SZK = st.sampled_from([1, 4, 6, 7, 8, 16, 17, 32, 63, 64, 65, 96])
_DT = DTYPES


# gemm/mm/multiN/multiK route through pack_A/pack_B + micro-kernel; einsum_lone
# (a lone summed index alongside a real link) and einsum_diag (a diagonal input)
# route through the generic loop instead - crossing the same boundary SIZES with
# the non-BLAS patterns, which the GEMM-only ops never reached.
_OPS = ["gemm", "einsum_mm", "einsum_multiN", "einsum_multiK", "einsum_lone", "einsum_diag"]


def _run_largedim(op, m, n, k, p, dt, seed, exact):
    rng = np.random.default_rng(seed)
    # Operands in dt, held in double precision so numpy's oracle does not round
    # at dt as well.
    gen = (lambda shape: integer_data(shape, dt, rng).astype(wide(dt))) if exact else (lambda shape: random_array(shape, dt, rng))

    def check(Ct, np_spec, A0, B0):
        oracle = np.einsum(np_spec, A0, B0)
        scale = np.einsum(np_spec, np.abs(A0), np.abs(B0))
        if exact:
            # Integer data stays exact while every partial sum does, which the
            # sum of absolute terms bounds: 2**24 for single precision. These
            # extents keep it near 2**15, so every dtype qualifies.
            assert scale.max(initial=0.0) < 2.0**24
            assert_exact(np.asarray(Ct), oracle)
        else:
            assert_rounding_close(Ct, oracle, dt, scale, err_msg=f"{op} m={m} n={n} k={k} p={p} dt={dt} seed={seed}")

    def run(spec, Ct, At, Bt):
        g = cg.Graph(f"ld{next(_ctr)}")
        with cg.capture(g):
            einsums.einsum(spec, Ct, At, Bt)
        g.execute()

    if op == "gemm":
        assume(m * k + k * n + m * n <= _CAP)
        A0, B0 = gen((m, k)), gen((k, n))
        At, Bt, Ct = einsums.asarray(A0, dtype=dt), einsums.asarray(B0, dtype=dt), einsums.asarray(np.zeros((m, n)), dtype=dt)
        g = cg.Graph(f"ld{next(_ctr)}")
        with cg.capture(g):
            einsums.linalg.gemm(1.0, At, Bt, 0.0, Ct)
        g.execute()
        check(Ct, "ik,kj->ij", A0, B0)
    elif op == "einsum_mm":
        assume(m * k + k * n + m * n <= _CAP)
        A0, B0 = gen((m, k)), gen((k, n))
        At, Bt, Ct = einsums.asarray(A0, dtype=dt), einsums.asarray(B0, dtype=dt), einsums.asarray(np.zeros((m, n)), dtype=dt)
        run("ij <- ik ; kj", Ct, At, Bt)
        check(Ct, "ik,kj->ij", A0, B0)
    elif op == "einsum_multiN":  # N = {j, p}
        assume(m * k + k * n * p + m * n * p <= _CAP)
        A0, B0 = gen((m, k)), gen((k, n, p))
        At, Bt, Ct = einsums.asarray(A0, dtype=dt), einsums.asarray(B0, dtype=dt), einsums.asarray(np.zeros((m, n, p)), dtype=dt)
        run("ijp <- ik ; kjp", Ct, At, Bt)
        check(Ct, "ik,kjp->ijp", A0, B0)
    elif op == "einsum_multiK":  # K = {k, p}
        assume(m * k * p + k * p * n + m * n <= _CAP)
        A0, B0 = gen((m, k, p)), gen((k, p, n))
        At, Bt, Ct = einsums.asarray(A0, dtype=dt), einsums.asarray(B0, dtype=dt), einsums.asarray(np.zeros((m, n)), dtype=dt)
        run("ij <- ikp ; kpj", Ct, At, Bt)
        check(Ct, "ikp,kpj->ij", A0, B0)
    elif op == "einsum_lone":  # link k + lone p (in B only): C_ij = sum_k sum_p A_ik B_pkj
        assume(m * k + p * k * n + m * n <= _CAP)
        A0, B0 = gen((m, k)), gen((p, k, n))
        At, Bt, Ct = einsums.asarray(A0, dtype=dt), einsums.asarray(B0, dtype=dt), einsums.asarray(np.zeros((m, n)), dtype=dt)
        run("ij <- ik ; pkj", Ct, At, Bt)
        check(Ct, "ik,pkj->ij", A0, B0)
    else:  # einsum_diag: diagonal over p in A, then contract p with B
        assume(m * k * k + k * n + m * n <= _CAP)
        A0, B0 = gen((m, k, k)), gen((k, n))
        At, Bt, Ct = einsums.asarray(A0, dtype=dt), einsums.asarray(B0, dtype=dt), einsums.asarray(np.zeros((m, n)), dtype=dt)
        run("ij <- ipp ; pj", Ct, At, Bt)
        check(Ct, "ipp,pj->ij", A0, B0)


@given(op=st.sampled_from(_OPS),
       m=_SZ, n=_SZ, k=_SZK, p=st.sampled_from([1, 2, 3, 5, 7]), dt=_DT, seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(180), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_largedim_diff(op, m, n, k, p, dt, seed):
    _run_largedim(op, m, n, k, p, dt, seed, exact=False)


@given(op=st.sampled_from(_OPS),
       m=_SZ, n=_SZ, k=_SZK, p=st.sampled_from([1, 2, 3, 5, 7]), dt=_DT, seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(150), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_largedim_diff_exact(op, m, n, k, p, dt, seed):
    _run_largedim(op, m, n, k, p, dt, seed, exact=True)
