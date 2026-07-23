# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: native einsum conjugation (conj_a / conj_b) vs numpy.

einsum(..., conj_a=True) conjugates A's elements in the contraction (likewise
conj_b for B); for real dtypes it's a no-op. The oracle is numpy.einsum with the
operands conjugated. Sweeps all four dtypes (float32/64, complex64/128), dense and
strided-view operands, the kwarg vs conj(...)-spec surfaces, and eager / graph /
graph-with-optimization-passes execution (the last guards that the rewriting passes
-- folding, batching, chain reorder, etc. -- don't drop or mis-handle the flags).
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from _sanitizer_scaling import sanitizer_examples
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
    ("ij <- ii ; jj", "ii,jj->ij"),      # repeated letters: diagonals x conj
    ("ij <- iij ; jji", "iij,jji->ij"),  # repeated letters in rank-3 operands x conj
    ("j <- iik ; kj", "iik,kj->j"),      # trace letter (self-contraction) x conj
    ("ij <- ijk ; ij", "ijk,ij->ij"),    # lone summed index (k in A only, empty link) x conj
    ("jk <- jl ; mlk", "jl,mlk->jk"),    # shared link (l) + lone summed (m in B only) x conj
    ("ij <- ij ; ijm", "ij,ijm->ij"),    # lone summed index in B (m) x conj
]

_DTYPES = ["float32", "float64", "complex64", "complex128"]


def _mk(a, dt):
    a = np.asarray(a)
    t = einsums.create_zero_tensor(f"ec{next(_ctr)}", list(a.shape), dtype=dt)
    if a.size:
        np.asarray(t)[...] = a
    return t


def _mkv(arr, use_view, dt, rng):
    """A tensor holding ``arr``, optionally as a strided (permuted) view."""
    if not use_view or arr.ndim < 2:
        return _mk(arr, dt)
    perm = list(rng.permutation(arr.ndim))
    if perm == list(range(arr.ndim)):
        perm = perm[::-1]
    return _mk(np.ascontiguousarray(np.transpose(arr, perm)), dt).permute_view(list(np.argsort(perm)))


def _rnd(shape, cplx, rng):
    if cplx:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


def _shape_for(idx, ext):
    return tuple(ext[c] for c in idx)


def _tol(dt):
    return (2e-3, 2e-4) if dt in ("float32", "complex64") else (1e-6, 1e-8)


@given(spec_pair=st.sampled_from(_SPECS), conj_a=st.booleans(), conj_b=st.booleans(),
       dt=st.sampled_from(_DTYPES), mode=st.sampled_from(["eager", "graph", "graph_passes"]),
       via=st.sampled_from(["kwarg", "spec"]), va=st.booleans(), vb=st.booleans(),
       seed=st.integers(0, 2**31 - 1))
@settings(max_examples=sanitizer_examples(800), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_hyp_einsum_conj(spec_pair, conj_a, conj_b, dt, mode, via, va, vb, seed):
    spec, np_spec = spec_pair
    a_idx, rest = spec.split(" <- ")[1].split(" ; ")[0], spec.split(" ; ")[1]
    c_idx = spec.split(" <- ")[0]
    b_idx = rest
    # Scalar-output einsum (empty C indices, e.g. dot) isn't supported by the
    # graph-capture shape validator yet; a pre-existing limitation unrelated to
    # conjugation. Exercise those specs eagerly only.
    if not c_idx and mode != "eager":
        return
    rng = np.random.default_rng(seed)
    cplx = dt in ("complex64", "complex128")
    ext = {ch: rng.integers(1, 4) for ch in set(a_idx + b_idx + c_idx)}
    A0 = _rnd(_shape_for(a_idx, ext), cplx, rng)
    B0 = _rnd(_shape_for(b_idx, ext), cplx, rng)
    An = np.conj(A0) if conj_a else A0
    Bn = np.conj(B0) if conj_b else B0
    oracle = np.einsum(np_spec, An, Bn)
    c_shape = _shape_for(c_idx, ext) if c_idx else (1,)

    At, Bt = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng)
    C = _mk(np.zeros(c_shape), dt)
    # Express conjugation either as conj_a/conj_b kwargs or as conj(...) wrappers
    # in the spec string; both must agree with the oracle.
    if via == "spec":
        aw = f"conj({a_idx})" if conj_a else a_idx
        bw = f"conj({b_idx})" if conj_b else b_idx
        spec_used, kw = f"{c_idx} <- {aw} ; {bw}", {}
    else:
        spec_used, kw = spec, dict(conj_a=conj_a, conj_b=conj_b)
    if mode == "eager":
        einsums.einsum(spec_used, C, At, Bt, **kw)
    else:
        g = cg.Graph(f"ec{next(_ctr)}")
        with cg.capture(g):
            einsums.einsum(spec_used, C, At, Bt, **kw)
        if mode == "graph_passes":
            cg.default_pass_manager().run(g)
        g.execute()
    got = np.asarray(C)
    if not c_idx:
        got = got.ravel()[0]
    rtol, atol = _tol(dt)
    np.testing.assert_allclose(got, oracle, rtol=rtol, atol=atol,
        err_msg=f"spec={spec_used} conj_a={conj_a} conj_b={conj_b} dt={dt} mode={mode} via={via} va={va} vb={vb} seed={seed}")
