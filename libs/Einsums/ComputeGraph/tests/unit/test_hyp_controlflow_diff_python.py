# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: control flow (add_loop / add_conditional) vs numpy.

A loop runs an RMW body subgraph N times (condition = ``it < N-1``, capped by
max_iterations); a conditional runs one of two branch subgraphs chosen by a
predicate; a nested case puts a conditional inside a loop body. The same op
sequence(s) are replayed on numpy the matching number of times / for the chosen
branch and every tensor is compared after ``g.execute()``. Passes may be applied;
N (matrix size) includes the degenerate 1.

The ``@example`` anchor pins the LoopInvariantHoisting miscompile: an accumulating
``direct_product(1, B, B, 1, C)`` (C += B**2) was hoisted out of the loop because
its capture omitted C from the node inputs when beta != 0, so the pass didn't see
that it reads its own destination -> the per-iteration accumulation was dropped
(2-iter result collapsed to 1).
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, example, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg

_ctr = itertools.count()


def _nm():
    return f"cf{next(_ctr)}"


def _mk(a):
    t = einsums.create_zero_tensor(_nm(), list(a.shape), dtype="float64")
    if a.size:
        np.asarray(t)[...] = a
    return t


_SC = st.sampled_from([1.0, -0.5, 0.5])
_BETA = st.sampled_from([0.0, 1.0])


@st.composite
def _steps(draw, ntens):
    out = []
    for _ in range(draw(st.integers(1, 3))):
        k = draw(st.sampled_from(["axpy", "axpby", "dirprod", "scale", "gemm"]))
        if k == "gemm":
            c = draw(st.integers(0, ntens - 1))
            a = draw(st.integers(0, ntens - 1))
            b = draw(st.integers(0, ntens - 1))
            if a == c:
                a = (a + 1) % ntens
            if b == c:
                b = (b + 1) % ntens
            out.append((k, draw(_SC), a, b, draw(_BETA), c))
        elif k == "scale":
            out.append((k, draw(_SC), draw(st.integers(0, ntens - 1))))
        elif k == "axpy":
            out.append((k, draw(_SC), draw(st.integers(0, ntens - 1)), draw(st.integers(0, ntens - 1))))
        elif k == "axpby":
            out.append((k, draw(_SC), draw(st.integers(0, ntens - 1)), draw(_BETA), draw(st.integers(0, ntens - 1))))
        else:
            out.append((k, draw(_SC), draw(st.integers(0, ntens - 1)), draw(st.integers(0, ntens - 1)),
                        draw(_BETA), draw(st.integers(0, ntens - 1))))
    return out


@st.composite
def _program(draw):
    mode = draw(st.sampled_from(["loop", "cond", "nested"]))
    n = draw(st.integers(1, 3))
    ntens = draw(st.integers(2, 3))
    return {"mode": mode, "n": n, "ntens": ntens,
            "niters": draw(st.integers(1, 5)), "pred": draw(st.booleans()),
            "passes": draw(st.booleans()), "seed": draw(st.integers(0, 2**31 - 1)),
            "ts": draw(_steps(ntens)), "es": draw(_steps(ntens))}


def _np_step(arrs, s):
    k = s[0]
    if k == "gemm":
        _, al, a, b, be, c = s; arrs[c] = al * (arrs[a] @ arrs[b]) + be * arrs[c]
    elif k == "scale":
        _, al, a = s; arrs[a] = al * arrs[a]
    elif k == "axpy":
        _, al, x, y = s; arrs[y] = arrs[y] + al * arrs[x]
    elif k == "axpby":
        _, al, x, be, y = s; arrs[y] = al * arrs[x] + be * arrs[y]
    else:
        _, al, a, b, be, c = s; arrs[c] = al * arrs[a] * arrs[b] + be * arrs[c]


def _cg_step(tens, s):
    k = s[0]
    if k == "gemm":
        _, al, a, b, be, c = s; einsums.linalg.gemm(al, tens[a], tens[b], be, tens[c])
    elif k == "scale":
        _, al, a = s; einsums.linalg.scale(al, tens[a])
    elif k == "axpy":
        _, al, x, y = s; einsums.linalg.axpy(al, tens[x], tens[y])
    elif k == "axpby":
        _, al, x, be, y = s; einsums.linalg.axpby(al, tens[x], be, tens[y])
    else:
        _, al, a, b, be, c = s; einsums.linalg.direct_product(al, tens[a], tens[b], be, tens[c])


_LIH = {"mode": "loop", "n": 1, "ntens": 2, "niters": 2, "pred": False, "passes": True, "seed": 0,
        "ts": [("dirprod", 1.0, 1, 1, 1.0, 0)], "es": [("scale", 1.0, 0)]}


@given(prog=_program())
@settings(max_examples=sanitizer_examples(300), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(prog=_LIH)  # accumulating direct_product hoisted out of a loop
def test_hyp_controlflow_diff(prog):
    mode, n, ntens, niters, pred, passes, seed = (prog["mode"], prog["n"], prog["ntens"],
                                                  prog["niters"], prog["pred"], prog["passes"], prog["seed"])
    ts, es = prog["ts"], prog["es"]
    rng = np.random.default_rng(seed)
    init = [rng.standard_normal((n, n)) for _ in range(ntens)]
    arrs = [a.copy() for a in init]
    tens = [_mk(a) for a in init]
    g = cg.Graph(_nm())
    if mode == "loop":
        for _ in range(niters):
            for s in ts:
                _np_step(arrs, s)
        loop = g.add_loop("L", niters, lambda it, N=niters: it < N - 1)
        with cg.capture(loop):
            for s in ts:
                _cg_step(tens, s)
    elif mode == "cond":
        for s in (ts if pred else es):
            _np_step(arrs, s)
        then_g, else_g = g.add_conditional("C", lambda P=pred: P)
        with cg.capture(then_g):
            for s in ts:
                _cg_step(tens, s)
        with cg.capture(else_g):
            for s in es:
                _cg_step(tens, s)
    else:  # nested: loop body holds a conditional (fixed predicate)
        for _ in range(niters):
            for s in (ts if pred else es):
                _np_step(arrs, s)
        loop = g.add_loop("L", niters, lambda it, N=niters: it < N - 1)
        with cg.capture(loop):
            then_g, else_g = loop.add_conditional("C", lambda P=pred: P)
        with cg.capture(then_g):
            for s in ts:
                _cg_step(tens, s)
        with cg.capture(else_g):
            for s in es:
                _cg_step(tens, s)
    if passes:
        g.apply(cg.default_pass_manager())
    g.execute()
    for i in range(ntens):
        np.testing.assert_allclose(np.asarray(tens[i]), arrs[i], rtol=1e-7, atol=1e-9,
            err_msg=f"tensor {i} mode={mode} n={n} ntens={ntens} niters={niters} pred={pred} "
                    f"passes={passes} seed={seed} ts={ts} es={es}")
