# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: a multi-node graph program run through the pass pipeline.

Generates a random straight-line program of RMW ops (gemm / axpy / axpby /
direct_product / scale) over a small pool of NxN tensors, replays it on numpy
for the oracle, then builds the same program in a graph, applies the full default
pass manager (CSE, reorder, fusion, in-place, memory planning, scale absorption,
contraction folding, ...), executes, and compares every tensor.

Unlike the single-op harnesses, this exercises the passes that only fire across
chains of ops, and the executor's read-modify-write ordering. N includes the
degenerate 1; gemm keeps its output distinct from its inputs (BLAS requires it).

Every example also draws one of the four dtypes. The prefactors include a complex
value, which on a real dtype contributes its real part only, and the complex
tensors carry genuinely complex data. The oracle replays the same stored inputs
in double precision, so single precision is compared against the exact result at
a tolerance scaled to the largest value the program produced.

The ``@example`` entries pin the in-place / self-aliasing reducers that were once
miscomputed (axpby/direct_product scaled the output by beta before reading an
input that aliased it):
  * axpby(2, A, 0, A)            -> (2+0)*A
  * direct_product(1, A, B, 0, A) -> A = A * B  (in-place Hadamard)
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, example, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES

_ctr = itertools.count()


def _nm() -> str:
    return f"pd{next(_ctr)}"


_SC = st.sampled_from([1.0, -2.0, 0.5, 0.5 - 0.75j])
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


def _typed_steps(steps, dtype):
    """@p steps with each complex prefactor cut to its real part on a real dtype."""
    if np.dtype(dtype).kind == "c":
        return steps
    return [tuple(f.real if isinstance(f, complex) else f for f in s) for s in steps]


def _tolerance(dtype, peak):
    """(rtol, atol) for the comparison: tight in double precision, and in single
    precision a relative bound whose absolute floor scales with @p peak, the
    largest magnitude any step produced. Rounding in an n <= 3 dot product lands
    at about 1e-7 of its operands, which can be far larger than a result that
    cancelled; a miscompiled step is off by O(1) of them."""
    if dtype in ("float32", "complex64"):
        return 1e-4, 1e-4 * max(1.0, peak)
    return 1e-8, 1e-8


def _replay_numpy(arrs, steps):
    """Replay @p steps on @p arrs; returns the largest magnitude seen."""
    peak = max((float(np.max(np.abs(a))) for a in arrs if a.size), default=0.0)
    for s in steps:
        _replay_step(arrs, s)
        peak = max([peak] + [float(np.max(np.abs(a))) for a in arrs if a.size])
    return peak


def _replay_step(arrs, s):
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


@given(prog=_program(), dtype=st.sampled_from(ALL_DTYPES))
@settings(max_examples=sanitizer_examples(300), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prog=(1, 2, [("axpby", 2.0, 0, 0.0, 0)]), dtype="float64")    # axpby self-alias -> (2+0)*A
@example(prog=(2, 2, [("dirprod", 1.0, 0, 1, 0.0, 0)]), dtype="float64")  # in-place Hadamard A = A*B
@example(prog=(2, 3, [("gemm", 1.0, 0, 1, 0.0, 2, False, False),                   # duplicate -> CSE
                      ("gemm", 1.0, 0, 1, 0.0, 2, False, False)]), dtype="float64")
@example(prog=(2, 3, [("gemm", 0.5 - 0.75j, 0, 1, 0.0, 2, True, False),            # complex duplicate -> CSE
                      ("gemm", 0.5 - 0.75j, 0, 1, 0.0, 2, True, False)]), dtype="complex64")
def test_hyp_program_diff(prog, dtype):
    n, ntens, steps = prog
    steps = _typed_steps(steps, dtype)
    rng = np.random.default_rng(0)
    init = [rng.standard_normal((n, n)) for _ in range(ntens)]
    if np.dtype(dtype).kind == "c":
        init = [a + 1j * rng.standard_normal((n, n)) for a in init]
    init = [a.astype(dtype) for a in init]
    wide = np.complex128 if np.dtype(dtype).kind == "c" else np.float64
    arrs = [a.astype(wide) for a in init]
    peak = _replay_numpy(arrs, steps)
    # Repeated squaring leaves single precision's range (and its resolution
    # relative to the tolerance) long before double's; such a program tests
    # overflow, not the graph, as the differential fuzz shards' cap says.
    assume(dtype not in ("float32", "complex64") or peak <= 1e4)
    rtol, atol = _tolerance(dtype, peak)
    tens = [einsums.asarray(a) for a in init]
    g = cg.Graph(_nm())
    _build_graph(tens, steps, g)
    g.apply(cg.default_pass_manager())
    g.execute()
    for i in range(ntens):
        np.testing.assert_allclose(np.asarray(tens[i]), arrs[i], rtol=rtol, atol=atol,
            err_msg=f"tensor {i} n={n} ntens={ntens} dtype={dtype} steps={steps}")
