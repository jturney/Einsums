# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: native einsum conjugation (conj_a / conj_b) vs numpy.

einsum(..., conj_a=True) conjugates A's elements in the contraction (likewise
conj_b for B); for real dtypes it's a no-op. The oracle is numpy.einsum with the
operands conjugated. Exercises eager, graph-captured, and graph-with-optimization-
passes execution (the last guards that the rewriting passes -- folding, batching,
chain reorder, etc. -- don't drop or mis-handle the conjugation flags).
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()

# (einsum spec, numpy spec) pairs spanning the dispatch backends.
_SPECS = [
    ("ij <- ik ; kj", "ik,kj->ij"),      # rank-2 gemm (generic conj path)
    ("ji <- ik ; kj", "ik,kj->ji"),      # transposed output
    ("ijp <- ik ; kjp", "ik,kjp->ijp"),  # rank-3 multi-N (packed native conj)
    ("ij <- ikp ; kpj", "ikp,kpj->ij"),  # multi-K (packed)
    (" <- i ; i", "i,i->"),              # dot
    ("ij <- i ; j", "i,j->ij"),          # outer / ger
]


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"ec{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


def _shape_for(idx, ext):
    return tuple(ext[c] for c in idx)


@given(spec_pair=st.sampled_from(_SPECS), conj_a=st.booleans(), conj_b=st.booleans(),
       cplx=st.booleans(), mode=st.sampled_from(["eager", "graph", "graph_passes"]),
       seed=st.integers(0, 2**31 - 1))
@settings(max_examples=500, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_einsum_conj(spec_pair, conj_a, conj_b, cplx, mode, seed):
    spec, np_spec = spec_pair
    a_idx, rest = spec.split(" <- ")[1].split(" ; ")[0], spec.split(" ; ")[1]
    c_idx = spec.split(" <- ")[0]
    b_idx = rest
    # Scalar-output einsum (empty C indices, e.g. dot) isn't supported by the
    # graph-capture shape validator yet — a pre-existing limitation unrelated to
    # conjugation. Exercise those specs eagerly only.
    if not c_idx and mode != "eager":
        return
    rng = np.random.default_rng(seed)
    dt = "complex128" if cplx else "float64"
    ext = {ch: rng.integers(1, 4) for ch in set(a_idx + b_idx + c_idx)}
    A0 = _rnd(_shape_for(a_idx, ext), cplx, rng)
    B0 = _rnd(_shape_for(b_idx, ext), cplx, rng)
    An = np.conj(A0) if conj_a else A0
    Bn = np.conj(B0) if conj_b else B0
    oracle = np.einsum(np_spec, An, Bn)
    c_shape = _shape_for(c_idx, ext) if c_idx else (1,)

    At, Bt = _mk(A0, dt), _mk(B0, dt)
    C = _mk(np.zeros(c_shape), dt)
    if mode == "eager":
        einsums.einsum(spec, C, At, Bt, conj_a=conj_a, conj_b=conj_b)
    else:
        g = cg.Graph(f"ec{next(_ctr)}")
        with cg.capture(g):
            einsums.einsum(spec, C, At, Bt, conj_a=conj_a, conj_b=conj_b)
        if mode == "graph_passes":
            cg.default_pass_manager().run(g)
        g.execute()
    got = np.asarray(C)
    if not c_idx:
        got = got.ravel()[0]
    np.testing.assert_allclose(got, oracle, rtol=1e-6, atol=1e-8,
        err_msg=f"spec={spec} conj_a={conj_a} conj_b={conj_b} cplx={cplx} mode={mode} seed={seed}")
