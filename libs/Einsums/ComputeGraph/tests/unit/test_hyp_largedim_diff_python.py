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
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()
_CAP = 50000


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"ld{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


# Sizes around MR(4)/NR(6) multiples +/- 1 and across KC=64 (capped for CI speed).
_SZ = st.sampled_from([1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 23, 24, 31, 32, 33, 48, 64])
_SZK = st.sampled_from([1, 4, 6, 7, 8, 16, 17, 32, 63, 64, 65, 96])
_DT = st.sampled_from(["float64", "complex128"])


@given(op=st.sampled_from(["gemm", "einsum_mm", "einsum_multiN", "einsum_multiK"]),
       m=_SZ, n=_SZ, k=_SZK, p=st.sampled_from([1, 2, 3, 5, 7]), dt=_DT, seed=st.integers(0, 2**31 - 1))
@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_largedim_diff(op, m, n, k, p, dt, seed):
    rng = np.random.default_rng(seed)
    isc = (dt == "complex128")
    if op == "gemm":
        assume(m * k + k * n + m * n <= _CAP)
        A0, B0 = _rnd((m, k), isc, rng), _rnd((k, n), isc, rng)
        At, Bt, Ct = _mk(A0, dt), _mk(B0, dt), _mk(np.zeros((m, n)), dt)
        g = cg.Graph(f"ld{next(_ctr)}")
        with cg.capture(g):
            einsums.linalg.gemm(1.0, At, Bt, 0.0, Ct)
        g.execute()
        np.testing.assert_allclose(np.asarray(Ct), A0 @ B0, rtol=1e-6, atol=1e-7)
    elif op == "einsum_mm":
        assume(m * k + k * n + m * n <= _CAP)
        A0, B0 = _rnd((m, k), isc, rng), _rnd((k, n), isc, rng)
        At, Bt, Ct = _mk(A0, dt), _mk(B0, dt), _mk(np.zeros((m, n)), dt)
        g = cg.Graph(f"ld{next(_ctr)}")
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", Ct, At, Bt)
        g.execute()
        np.testing.assert_allclose(np.asarray(Ct), A0 @ B0, rtol=1e-6, atol=1e-7)
    elif op == "einsum_multiN":  # N = {j, p}
        assume(m * k + k * n * p + m * n * p <= _CAP)
        A0, B0 = _rnd((m, k), isc, rng), _rnd((k, n, p), isc, rng)
        At, Bt, Ct = _mk(A0, dt), _mk(B0, dt), _mk(np.zeros((m, n, p)), dt)
        g = cg.Graph(f"ld{next(_ctr)}")
        with cg.capture(g):
            einsums.einsum("ijp <- ik ; kjp", Ct, At, Bt)
        g.execute()
        np.testing.assert_allclose(np.asarray(Ct), np.einsum("ik,kjp->ijp", A0, B0), rtol=1e-6, atol=1e-7)
    else:  # einsum_multiK: K = {k, p}
        assume(m * k * p + k * p * n + m * n <= _CAP)
        A0, B0 = _rnd((m, k, p), isc, rng), _rnd((k, p, n), isc, rng)
        At, Bt, Ct = _mk(A0, dt), _mk(B0, dt), _mk(np.zeros((m, n)), dt)
        g = cg.Graph(f"ld{next(_ctr)}")
        with cg.capture(g):
            einsums.einsum("ij <- ikp ; kpj", Ct, At, Bt)
        g.execute()
        np.testing.assert_allclose(np.asarray(Ct), np.einsum("ikp,kpj->ij", A0, B0), rtol=1e-6, atol=1e-7)
