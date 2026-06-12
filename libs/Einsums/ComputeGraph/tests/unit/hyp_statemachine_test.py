# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Stateful Hypothesis model of einsums tensor algebra.

A RuleBasedStateMachine that keeps a persistent pool of einsums tensors paired
with numpy oracle arrays. Each rule applies one operation (in place) to both and
asserts they still agree. Because the pool persists across rules, this exercises
read-modify-write chains, where a later op reads what an earlier op wrote. That
is the hazard class that broke the optimization passes historically, and on
failure Hypothesis shrinks the operation sequence, dropping ops that don't
contribute to the bug.

Shapes are drawn over dims {1, 2, 3}, so degenerate (size-1) extents are covered
by default, exactly the class that the two bugs fixed this session lived in.

The pool holds COPIES distinct tensors per shape so rules can pick distinct
operands where the operation requires it (BLAS axpy/gemv/gemm don't accept an
output aliasing an input, and gemv needs x != y). This mirrors the hand-rolled
fuzzer's operand-distinctness rules; without them the harness reports false
positives from unsupported aliasing rather than real bugs.

Run:  pytest hyp_statemachine_test.py -q     (under the ASan DYLD setup)
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

import einsums
import einsums.graph as cg

DIMS = (1, 2, 3)
MAT_SHAPES = [(r, c) for r in DIMS for c in DIMS]
VEC_LENS = list(DIMS)
COPIES = 3
SCALAR = st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False).map(lambda x: round(x, 4))
BETA = st.sampled_from([0.0, 0.5, 1.0])
COPY = st.integers(0, COPIES - 1)
CAP = 1e6
_ctr = itertools.count()


def _nm() -> str:
    return f"sm{next(_ctr)}"


def _mk(arr):
    t = einsums.create_zero_tensor(_nm(), list(arr.shape), dtype="float64")
    np.asarray(t)[...] = arr
    return t


def _other(exclude: set[int]) -> int:
    """Lowest copy index not in `exclude` (COPIES=3 covers up to 2 prior operands)."""
    return next(c for c in range(COPIES) if c not in exclude)


class EinsumsMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(1234)
        # Paired pools keyed by (shape, copy): numpy oracle + einsums tensor.
        self.npm = {(sh, c): rng.standard_normal(sh) for sh in MAT_SHAPES for c in range(COPIES)}
        self.esm = {k: _mk(a) for k, a in self.npm.items()}
        self.npv = {(L, c): rng.standard_normal((L,)) for L in VEC_LENS for c in range(COPIES)}
        self.esv = {k: _mk(a) for k, a in self.npv.items()}

    # ── helpers ────────────────────────────────────────────────────────────
    def _check_m(self, key):
        np.testing.assert_allclose(np.asarray(self.esm[key]), self.npm[key], rtol=1e-6, atol=1e-6,
                                   err_msg=f"matrix {key} diverged")
        if np.max(np.abs(self.npm[key])) > CAP:
            self.npm[key] = self.npm[key] / CAP
            np.asarray(self.esm[key])[...] = self.npm[key]

    def _check_v(self, key):
        np.testing.assert_allclose(np.asarray(self.esv[key]), self.npv[key], rtol=1e-6, atol=1e-6,
                                   err_msg=f"vector {key} diverged")
        if np.max(np.abs(self.npv[key])) > CAP:
            self.npv[key] = self.npv[key] / CAP
            np.asarray(self.esv[key])[...] = self.npv[key]

    # ── rules: each mutates the pool in place via a captured graph ──────────
    @rule(sh=st.sampled_from(MAT_SHAPES), c=COPY, a=SCALAR)
    def scale(self, sh, c, a):
        with cg.capture(g := cg.Graph(_nm())):
            einsums.linalg.scale(a, self.esm[(sh, c)])
        g.execute()
        self.npm[(sh, c)] = a * self.npm[(sh, c)]
        self._check_m((sh, c))

    @rule(sh=st.sampled_from(MAT_SHAPES), cx=COPY, a=SCALAR)
    def axpy(self, sh, cx, a):
        cy = _other({cx})  # y += a*x, distinct copies
        with cg.capture(g := cg.Graph(_nm())):
            einsums.linalg.axpy(a, self.esm[(sh, cx)], self.esm[(sh, cy)])
        g.execute()
        self.npm[(sh, cy)] = self.npm[(sh, cy)] + a * self.npm[(sh, cx)]
        self._check_m((sh, cy))

    @rule(a=SCALAR, data=st.data())
    def view_axpy(self, a, data):
        sh = data.draw(st.sampled_from(MAT_SHAPES))
        R, C = sh
        r0 = data.draw(st.integers(0, R - 1)); r1 = data.draw(st.integers(r0 + 1, R))
        c0 = data.draw(st.integers(0, C - 1)); c1 = data.draw(st.integers(c0 + 1, C))
        src_sh = (r1 - r0, c1 - c0)
        ct = data.draw(COPY)
        cs = data.draw(COPY)
        if src_sh == sh and cs == ct:  # keep src distinct from the view's parent
            cs = _other({ct})
        with cg.capture(g := cg.Graph(_nm())):
            einsums.linalg.axpy(a, self.esm[(src_sh, cs)], cg.view(self.esm[(sh, ct)], [(r0, r1), (c0, c1)]))
        g.execute()
        self.npm[(sh, ct)][r0:r1, c0:c1] += a * self.npm[(src_sh, cs)]
        self._check_m((sh, ct))

    @rule(a=SCALAR, b=BETA, data=st.data())
    def gemm(self, a, b, data):
        m = data.draw(st.sampled_from(DIMS)); k = data.draw(st.sampled_from(DIMS)); n = data.draw(st.sampled_from(DIMS))
        ca = data.draw(COPY); cb = data.draw(COPY)
        # C must be a distinct tensor from A and B.
        forbidden = set()
        if (m, n) == (m, k):
            forbidden.add(ca)
        if (m, n) == (k, n):
            forbidden.add(cb)
        cc = _other(forbidden)
        with cg.capture(g := cg.Graph(_nm())):
            einsums.linalg.gemm(a, self.esm[(m, k), ca], self.esm[(k, n), cb], b, self.esm[(m, n), cc])
        g.execute()
        self.npm[(m, n), cc] = a * (self.npm[(m, k), ca] @ self.npm[(k, n), cb]) + b * self.npm[(m, n), cc]
        self._check_m(((m, n), cc))

    @rule(a=SCALAR, b=BETA, data=st.data())
    def gemv(self, a, b, data):
        m = data.draw(st.sampled_from(DIMS)); n = data.draw(st.sampled_from(DIMS))
        ca = data.draw(COPY); cx = data.draw(COPY)
        cy = cx if m != n else _other({cx})  # x (len n) and y (len m) must differ
        with cg.capture(g := cg.Graph(_nm())):
            einsums.linalg.gemv(a, self.esm[(m, n), ca], self.esv[n, cx], b, self.esv[m, cy])
        g.execute()
        self.npv[m, cy] = a * (self.npm[(m, n), ca] @ self.npv[n, cx]) + b * self.npv[m, cy]
        self._check_v((m, cy))

    @invariant()
    def all_consistent(self):
        for k in self.esm:
            np.testing.assert_allclose(np.asarray(self.esm[k]), self.npm[k], rtol=1e-6, atol=1e-6)


EinsumsMachine.TestCase.settings = settings(
    max_examples=150, stateful_step_count=40, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
TestEinsumsMachine = EinsumsMachine.TestCase
