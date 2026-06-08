# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: a multi-node graph program THROUGH the pass pipeline.

Generates a random straight-line program of RMW ops (gemm / axpy / axpby /
direct_product / scale) over a small pool of NxN tensors, replays it on numpy
for the oracle, then builds the same program in a graph, applies the FULL default
pass manager (CSE, reorder, fusion, in-place, memory planning, scale absorption,
contraction folding, ...), executes, and compares every tensor.

Unlike the single-op harnesses, this exercises the passes that only fire across
*chains* of ops, and the executor's read-modify-write ordering. N includes the
degenerate 1; gemm keeps its output distinct from its inputs (BLAS requires it).

The ``@example`` entries pin the in-place / self-aliasing reducers that were once
miscomputed (axpby/direct_product scaled the output by beta before reading an
input that aliased it):
  * axpby(2, A, 0, A)            -> (2+0)*A
  * direct_product(1, A, B, 0, A) -> A = A * B  (in-place Hadamard)
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()


def _nm() -> str:
    return f"pd{next(_ctr)}"


def _mk(a):
    t = einsums.create_zero_tensor(_nm(), list(a.shape), dtype="float64")
    if a.size:
        np.asarray(t)[...] = a
    return t


_SC = st.sampled_from([1.0, -2.0, 0.5])
_BETA = st.sampled_from([0.0, 1.0])


@st.composite
def _program(draw):
    n = draw(st.integers(1, 3))
    ntens = draw(st.integers(2, 4))
    nsteps = draw(st.integers(2, 8))
    steps = []
    for _ in range(nsteps):
        kind = draw(st.sampled_from(["gemm", "axpy", "axpby", "dirprod", "scale"]))
        if kind == "gemm":
            c = draw(st.integers(0, ntens - 1))
            a = draw(st.integers(0, ntens - 1))
            b = draw(st.integers(0, ntens - 1))
            if a == c:
                a = (a + 1) % ntens
            if b == c:
                b = (b + 1) % ntens
            steps.append((kind, draw(_SC), a, b, draw(_BETA), c, draw(st.booleans()), draw(st.booleans())))
        elif kind == "scale":
            steps.append((kind, draw(_SC), draw(st.integers(0, ntens - 1))))
        elif kind == "axpy":
            steps.append((kind, draw(_SC), draw(st.integers(0, ntens - 1)), draw(st.integers(0, ntens - 1))))
        elif kind == "axpby":
            steps.append((kind, draw(_SC), draw(st.integers(0, ntens - 1)), draw(_BETA), draw(st.integers(0, ntens - 1))))
        else:  # dirprod
            steps.append((kind, draw(_SC), draw(st.integers(0, ntens - 1)), draw(st.integers(0, ntens - 1)),
                          draw(_BETA), draw(st.integers(0, ntens - 1))))
    return n, ntens, steps


def _replay_numpy(arrs, steps):
    for s in steps:
        k = s[0]
        if k == "gemm":
            _, al, a, b, be, c, ta, tb = s
            opA = arrs[a].T if ta else arrs[a]
            opB = arrs[b].T if tb else arrs[b]
            arrs[c] = al * (opA @ opB) + be * arrs[c]
        elif k == "scale":
            _, al, a = s
            arrs[a] = al * arrs[a]
        elif k == "axpy":
            _, al, x, y = s
            arrs[y] = arrs[y] + al * arrs[x]
        elif k == "axpby":
            _, al, x, be, y = s
            arrs[y] = al * arrs[x] + be * arrs[y]
        else:
            _, al, a, b, be, c = s
            arrs[c] = al * arrs[a] * arrs[b] + be * arrs[c]


def _build_graph(tens, steps, g):
    with cg.capture(g):
        for s in steps:
            k = s[0]
            if k == "gemm":
                _, al, a, b, be, c, ta, tb = s
                einsums.linalg.gemm(al, tens[a], tens[b], be, tens[c], trans_a=ta, trans_b=tb)
            elif k == "scale":
                _, al, a = s
                einsums.linalg.scale(al, tens[a])
            elif k == "axpy":
                _, al, x, y = s
                einsums.linalg.axpy(al, tens[x], tens[y])
            elif k == "axpby":
                _, al, x, be, y = s
                einsums.linalg.axpby(al, tens[x], be, tens[y])
            else:
                _, al, a, b, be, c = s
                einsums.linalg.direct_product(al, tens[a], tens[b], be, tens[c])


@given(prog=_program())
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prog=(1, 2, [("axpby", 2.0, 0, 0.0, 0)]))                      # axpby self-alias -> (2+0)*A
@example(prog=(2, 2, [("dirprod", 1.0, 0, 1, 0.0, 0)]))                 # in-place Hadamard A = A*B
@example(prog=(2, 3, [("gemm", 1.0, 0, 1, 0.0, 2, False, False),                   # duplicate -> CSE
                      ("gemm", 1.0, 0, 1, 0.0, 2, False, False)]))
def test_hyp_program_diff(prog):
    n, ntens, steps = prog
    rng = np.random.default_rng(0)
    init = [rng.standard_normal((n, n)) for _ in range(ntens)]
    arrs = [a.copy() for a in init]
    _replay_numpy(arrs, steps)
    tens = [_mk(a) for a in init]
    g = cg.Graph(_nm())
    _build_graph(tens, steps, g)
    g.apply(cg.default_pass_manager())
    g.execute()
    for i in range(ntens):
        np.testing.assert_allclose(np.asarray(tens[i]), arrs[i], rtol=1e-8, atol=1e-8,
            err_msg=f"tensor {i} n={n} ntens={ntens} steps={steps}")
