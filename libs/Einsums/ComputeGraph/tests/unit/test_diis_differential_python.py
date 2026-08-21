#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Differential tests for ``einsums.graph.diis`` against a numpy oracle.

The accelerator's engine is einsums operations (axpby snapshots, dotc inner
products, gesv), and this file checks that machinery against a plain-numpy
implementation of the same algorithm: same error vector (the update step),
same drop-oldest history policy, same B normalization, same bordered solve.
Written before DLPNO's private ``_DIIS`` was deleted, per the M6 definition
of done in DESIGN-hybrid-framework.md. Complements ``test_diis_python.py``,
which pins acceleration against a direct solve, in-place writeback, argument
validation, and tiled amplitudes; this file pins the per-iteration
TRAJECTORY, so an engine change that still converges cannot silently change
what the accelerator computes.

Note what a passing differential does NOT prove: that either side ran (see the
M3 finding). ``test_diis_moves_the_amplitudes`` is the deliberate-perturbation
guard - it asserts the extrapolation changed the amplitudes away from the plain
fixed-point trajectory, so agreement cannot come from two no-ops.
"""

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
from einsums.testing import ALL_DTYPES, assert_close, tolerance_for


class NumpyDIIS:
    """The same algorithm as :class:`einsums.graph.DIISAccelerator`, in numpy.

    Components are kept as a list of flat arrays, mirroring the accelerator's
    per-pair components; B sums the componentwise inner products, which is the
    concatenated-vector inner product.
    """

    def __init__(self, k):
        self.k = k
        self.T = []
        self.E = []

    def step(self, ts, es):
        """Push copies of the current (amplitudes, steps), extrapolate in place."""
        self.T.append([t.copy() for t in ts])
        self.E.append([e.copy() for e in es])
        if len(self.T) > self.k:
            self.T.pop(0)
            self.E.pop(0)

        while len(self.T) >= 2:
            m = len(self.T)
            B = np.zeros((m + 1, m + 1))
            scale = 0.0
            for p in range(m):
                for q in range(p, m):
                    v = sum(np.vdot(ep, eq).real for ep, eq in zip(self.E[p], self.E[q]))
                    B[p, q] = B[q, p] = v
                    scale = max(scale, abs(v))
            if scale > 0.0:
                B[:m, :m] /= scale
            B[:m, m] = B[m, :m] = -1.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(B, rhs)
            except np.linalg.LinAlgError:
                self.T.pop(0)
                self.E.pop(0)
                continue
            for comp, t in enumerate(ts):
                t[...] = sum(c[s] * self.T[s][comp] for s in range(m))
            return


def _problem(rng, n, dtype):
    """A diagonally dominant linear system, so plain Jacobi converges slowly."""
    complex_ = dtype.startswith("complex")
    A = rng.standard_normal((n, n))
    if complex_:
        A = A + 1j * rng.standard_normal((n, n))
    A = A + A.conj().T + 2.0 * n * np.eye(n)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0.0)
    return A.astype(dtype), b.astype(dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_diis_matches_numpy_oracle(dtype):
    """Every iteration of the accelerator equals the numpy oracle's."""
    rng = np.random.default_rng(1234)
    n = 12
    A, b = _problem(rng, n, dtype)
    d = np.diag(A).copy()

    t = einsums.zeros((n,), dtype=dtype)
    s = einsums.zeros((n,), dtype=dtype)
    acc = cg.diis([(t, s)], k=4)

    t_np = [np.zeros(n, dtype=dtype)]
    oracle = NumpyDIIS(k=4)

    rtol, atol = tolerance_for(dtype)
    for _ in range(10):
        # The body's work, done identically on both sides: step = -(A t - b)/d.
        step = (-(A @ np.asarray(t).copy() - b) / d).astype(dtype)
        np.asarray(s)[...] = step
        la.axpby(1.0, s, 1.0, t)
        acc.step()

        step_np = (-(A @ t_np[0] - b) / d).astype(dtype)
        t_np[0] = (t_np[0] + step_np).astype(dtype)
        oracle.step(t_np, [step_np])

        assert_close(t, t_np[0], rtol=10 * rtol, atol=10 * atol)


def test_diis_multiple_pairs_match_concatenated_oracle():
    """Several (amplitude, step) pairs behave as one concatenated vector.

    The pair components mirror DLPNO's bucket stores: different shapes and
    ranks, extrapolated together because B sums the componentwise dots.
    """
    rng = np.random.default_rng(77)
    shapes = [(3, 3, 2), (5,), (2, 4)]
    sizes = [int(np.prod(s)) for s in shapes]
    n = sum(sizes)
    A, b = _problem(rng, n, "float64")
    d = np.diag(A).copy()

    ts = [einsums.zeros(s, dtype="float64") for s in shapes]
    ss = [einsums.zeros(s, dtype="float64") for s in shapes]
    acc = cg.diis(list(zip(ts, ss)), k=3)

    t_flat = np.zeros(n)
    oracle = NumpyDIIS(k=3)
    offsets = np.cumsum([0] + sizes)

    for _ in range(8):
        flat = np.concatenate([np.asarray(t).ravel(order="F") for t in ts])
        step = -(A @ flat - b) / d
        for s_t, lo, hi, shape in zip(ss, offsets[:-1], offsets[1:], shapes):
            np.asarray(s_t)[...] = step[lo:hi].reshape(shape, order="F")
        for t, s_t in zip(ts, ss):
            la.axpby(1.0, s_t, 1.0, t)
        acc.step()

        step_np = -(A @ t_flat - b) / d
        t_flat = t_flat + step_np
        wrapped_t = [t_flat]
        oracle.step(wrapped_t, [step_np])
        t_flat = wrapped_t[0]

        got = np.concatenate([np.asarray(t).ravel(order="F") for t in ts])
        assert_close(got, t_flat, dtype="float64")


def test_diis_moves_the_amplitudes():
    """The deliberate-perturbation guard: DIIS must CHANGE the trajectory.

    A differential that agrees because neither side extrapolated is worthless;
    this pins that the extrapolation engaged (history >= 2 changes t away from
    the plain fixed-point iterate) and that it helps (fewer iterations to
    tolerance than unaccelerated Jacobi).
    """
    rng = np.random.default_rng(5)
    n = 20
    A, b = _problem(rng, n, "float64")
    d = np.diag(A).copy()
    exact = np.linalg.solve(A, b)

    def iterate(use_diis, max_it=200, tol=1e-12):
        t = einsums.zeros((n,), dtype="float64")
        s = einsums.zeros((n,), dtype="float64")
        acc = cg.diis([(t, s)], k=6) if use_diis else None
        plain_second_iterate = None
        for it in range(max_it):
            np.asarray(s)[...] = -(A @ np.asarray(t).copy() - b) / d
            la.axpby(1.0, s, 1.0, t)
            if it == 1:
                plain_second_iterate = np.asarray(t).copy()
            if acc is not None:
                acc.step()
            if np.linalg.norm(A @ np.asarray(t) - b) < tol * np.linalg.norm(b):
                return it + 1, plain_second_iterate, np.asarray(t).copy()
        return max_it, plain_second_iterate, np.asarray(t).copy()

    iters_plain, _, t_plain = iterate(False)
    iters_diis, second_before, t_diis = iterate(True)

    # It engaged: with two history vectors the extrapolant differs from the
    # plain iterate (second_before was captured before acc.step ran on it).
    assert not np.allclose(second_before, t_diis) or iters_diis <= 2
    # It converged to the right answer, faster than unaccelerated.
    assert_close(t_diis, exact, rtol=1e-9, atol=1e-11)
    assert iters_diis < iters_plain


def test_wrap_runs_condition_first_and_skips_step_on_convergence():
    t = einsums.zeros((4,), dtype="float64")
    s = einsums.zeros((4,), dtype="float64")
    np.asarray(t)[...] = 1.0
    np.asarray(s)[...] = 0.5
    acc = cg.diis([(t, s)], k=3)

    calls = []
    predicate = acc.wrap(lambda it: calls.append(it) is None and it < 1)

    assert predicate(0) is True
    assert acc.history_size == 1  # continued -> a history push happened
    before = np.asarray(t).copy()
    assert predicate(1) is False
    assert acc.history_size == 1  # converged -> no push, no extrapolation
    assert np.array_equal(np.asarray(t), before)
    assert calls == [0, 1]


def test_singular_history_drops_oldest_and_recovers():
    """Identical consecutive states make B singular; the oldest pair goes."""
    t = einsums.zeros((6,), dtype="float64")
    s = einsums.zeros((6,), dtype="float64")
    np.asarray(t)[...] = 1.0
    np.asarray(s)[...] = 1e-3  # identical steps -> rank-1 B, singular for m >= 2
    acc = cg.diis([(t, s)], k=4)

    for _ in range(4):
        acc.step()
        assert np.all(np.isfinite(np.asarray(t)))
    # The retry loop must not let the history exceed its cap or go negative.
    assert 1 <= acc.history_size <= 4


def test_diis_inside_a_captured_loop_solves_the_system():
    """The DLPNO shape end to end: body in a graph loop, DIIS in the predicate.

    The body computes the residual, the preconditioned step, and the update
    with captured einsums ops; ``acc.wrap`` supplies the predicate. This is
    the integration the docstring example promises.
    """
    rng = np.random.default_rng(9)
    n = 8
    A_np, b_np = _problem(rng, n, "float64")
    exact = np.linalg.solve(A_np, b_np)

    A = einsums.asarray(A_np.copy(order="F"))
    b = einsums.asarray(b_np)
    d = einsums.asarray(np.diag(A_np).copy())
    t = einsums.zeros((n,), dtype="float64")
    s = einsums.zeros((n,), dtype="float64")
    r = einsums.zeros((n,), dtype="float64")

    state = {"iters": 0}

    def converged(it):
        state["iters"] = it + 1
        return np.linalg.norm(np.asarray(r)) > 1e-12 * np.linalg.norm(b_np)

    acc = cg.diis([(t, s)], k=6)
    g = cg.Graph("diis loop")
    body = g.add_loop("jacobi", 100, acc.wrap(converged))
    with cg.capture(body):
        la.gemv(1.0, A, t, 0.0, r)          # r = A t
        la.axpby(-1.0, b, 1.0, r)           # r -= b
        la.direct_division(-1.0, r, d, 0.0, s)  # s = -r/d
        la.axpby(1.0, s, 1.0, t)            # t += s
    g.execute()

    assert_close(t, exact, rtol=1e-9, atol=1e-11)
    assert state["iters"] < 100
