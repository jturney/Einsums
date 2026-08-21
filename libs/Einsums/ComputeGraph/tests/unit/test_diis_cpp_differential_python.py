#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The C++ DIIS accelerator against the Python one, step for step.

``cg.diis`` now returns :class:`einsums.graph.DIISAccelerator`, a thin shell
over ``DiisAccelerator<T>``; :class:`einsums.graph._PyDiis` is the Python
implementation it replaced, kept as the oracle. The two run the same operations
in the same order on the same eager kernels - axpby snapshots, conjugated dots,
a normalized bordered ``gesv``, axpby extrapolation - so agreement here is
BITWISE, not to a tolerance. A tolerance would hide exactly the kind of drift
this file exists to catch: a reordered accumulation, a different normalization,
a coefficient that took an extra rounding on its way through.

``test_diis_differential_python.py`` pins the algorithm against a numpy oracle
and is what says the trajectory is right; this file says the two
implementations of it are the same one.
"""

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
from einsums.testing import ALL_DTYPES


def _contraction(rng, n, dtype):
    """A linear map with spectral radius ~0.9 and its fixed-point offset."""
    complex_ = dtype.startswith("complex")
    M = rng.standard_normal((n, n))
    if complex_:
        M = M + 1j * rng.standard_normal((n, n))
    M *= 0.9 / np.abs(np.linalg.eigvals(M)).max()
    b = rng.standard_normal(n)
    if complex_:
        b = b + 1j * rng.standard_normal(n)
    return M.astype(dtype), b.astype(dtype)


def _drive(accelerator, shapes, dtype, steps, body):
    """Run ``steps`` passes of ``body`` under ``accelerator``, tracing the amplitudes.

    Returns the per-step amplitude snapshots and the per-step history sizes, both
    read after the accelerator's step, which is where the two implementations
    have to agree.
    """
    ts = [einsums.zeros(s, dtype=dtype) for s in shapes]
    ss = [einsums.zeros(s, dtype=dtype) for s in shapes]
    acc = accelerator(list(zip(ts, ss)))
    trace, depths = [], []
    for it in range(steps):
        body(it, ts, ss)
        for t, s in zip(ts, ss):
            la.axpby(1.0, s, 1.0, t)
        acc.step()
        trace.append([np.asarray(t).copy() for t in ts])
        depths.append(acc.history_size)
    return trace, depths


def _assert_bitwise(got, want, label):
    """Every traced amplitude equal bit for bit, with the deltas if not."""
    for it, (g_step, w_step) in enumerate(zip(got, want)):
        for c, (g, w) in enumerate(zip(g_step, w_step)):
            if not np.array_equal(g, w):
                delta = np.max(np.abs(g.astype("complex128") - w.astype("complex128")))
                scale = max(np.max(np.abs(w)), 1e-300)
                pytest.fail(f"{label}: step {it} component {c} differs; "
                            f"max |delta| {delta:.6e}, relative {delta / scale:.6e}")


def _flat_linear_body(M, b, shapes, dtype):
    """``step = M x + b - x`` over the concatenated components, in numpy."""
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([0] + sizes)

    def body(_it, ts, ss):
        x = np.concatenate([np.asarray(t).ravel(order="F") for t in ts])
        step = (M @ x + b - x).astype(dtype)
        for s_t, lo, hi, shape in zip(ss, offsets[:-1], offsets[1:], shapes):
            np.asarray(s_t)[...] = step[lo:hi].reshape(shape, order="F")

    return body


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_cpp_matches_python_oracle_bitwise(dtype):
    """Ten steps at k=4: history ramp-up, eviction, and snapshot recycling."""
    rng = np.random.default_rng(4242)
    shapes = [(12,)]
    M, b = _contraction(rng, 12, dtype)
    body = _flat_linear_body(M, b, shapes, dtype)

    cpp, cpp_depths = _drive(lambda p: cg.DIISAccelerator(p, k=4), shapes, dtype, 10, body)
    py, py_depths = _drive(lambda p: cg._PyDiis(p, k=4), shapes, dtype, 10, body)

    assert cpp_depths == py_depths
    # k=4 over 10 steps: the history caps at 4, so six of the ten steps evicted
    # and recycled a snapshot.
    assert cpp_depths == [1, 2, 3, 4, 4, 4, 4, 4, 4, 4]
    _assert_bitwise(cpp, py, f"single pair, {dtype}")


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_multiple_pairs_of_unequal_sizes_match(dtype):
    """Several pairs of different shapes and ranks, extrapolated together.

    The shapes mirror DLPNO's bucket stores: B sums the componentwise dots, so
    the pair list behaves as one concatenated vector and the summation order
    over components is part of what has to agree.
    """
    shapes = [(3, 3, 2), (5,), (2, 4), (1,)]
    n = sum(int(np.prod(s)) for s in shapes)
    rng = np.random.default_rng(77)
    M, b = _contraction(rng, n, dtype)
    body = _flat_linear_body(M, b, shapes, dtype)

    cpp, cpp_depths = _drive(lambda p: cg.DIISAccelerator(p, k=3), shapes, dtype, 9, body)
    py, py_depths = _drive(lambda p: cg._PyDiis(p, k=3), shapes, dtype, 9, body)

    assert cpp_depths == py_depths == [1, 2, 3, 3, 3, 3, 3, 3, 3]
    _assert_bitwise(cpp, py, f"four pairs, {dtype}")


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_singular_subspace_retry_matches(dtype):
    """Identical steps every pass make B rank-1; both sides must drop the same pair."""
    shapes = [(6,)]

    def body(_it, _ts, ss):
        np.asarray(ss[0])[...] = 1e-3

    cpp, cpp_depths = _drive(lambda p: cg.DIISAccelerator(p, k=4), shapes, dtype, 6, body)
    py, py_depths = _drive(lambda p: cg._PyDiis(p, k=4), shapes, dtype, 6, body)

    assert cpp_depths == py_depths
    _assert_bitwise(cpp, py, f"singular B, {dtype}")
    assert all(np.all(np.isfinite(np.asarray(c[0]).view("float64"))) for c in cpp)


def test_long_run_through_repeated_eviction_matches():
    """Forty steps at k=2, so nearly every step evicts and recycles."""
    dtype = "float64"
    shapes = [(9,), (4, 2)]
    n = 9 + 8
    rng = np.random.default_rng(31337)
    M, b = _contraction(rng, n, dtype)
    body = _flat_linear_body(M, b, shapes, dtype)

    cpp, cpp_depths = _drive(lambda p: cg.DIISAccelerator(p, k=2), shapes, dtype, 40, body)
    py, py_depths = _drive(lambda p: cg._PyDiis(p, k=2), shapes, dtype, 40, body)

    assert cpp_depths == py_depths
    _assert_bitwise(cpp, py, "k=2 long run")


def test_wrap_matches_the_oracle_including_the_skip_on_convergence():
    """``wrap`` takes the step only when the condition says to keep going."""
    def run(factory):
        t = einsums.zeros((4,), dtype="float64")
        s = einsums.zeros((4,), dtype="float64")
        np.asarray(t)[...] = 1.0
        np.asarray(s)[...] = 0.5
        acc = factory([(t, s)])
        calls = []
        predicate = acc.wrap(lambda it: calls.append(it) is None and it < 1)
        assert predicate(0) is True
        first = (acc.history_size, np.asarray(t).copy())
        before = np.asarray(t).copy()
        assert predicate(1) is False
        return calls, first, acc.history_size, np.asarray(t).copy(), before

    cpp = run(lambda p: cg.DIISAccelerator(p, k=3))
    py = run(lambda p: cg._PyDiis(p, k=3))

    assert cpp[0] == py[0] == [0, 1]
    assert cpp[1][0] == py[1][0] == 1
    assert np.array_equal(cpp[1][1], py[1][1])
    # Converged: no push, no extrapolation, amplitudes untouched.
    assert cpp[2] == py[2] == 1
    assert np.array_equal(cpp[3], cpp[4])
    assert np.array_equal(cpp[3], py[3])


def test_factory_picks_cpp_for_dense_pairs_and_python_otherwise():
    t = einsums.zeros((4,), dtype="float64")
    s = einsums.zeros((4,), dtype="float64")
    assert isinstance(cg.diis([(t, s)]), cg.DIISAccelerator)

    # A view is a first-class operand on the C++ path.
    parent = einsums.zeros((4, 3), dtype="float64")
    assert isinstance(cg.diis([(parent[:, 0], s)]), cg.DIISAccelerator)

    # Mixed dtypes have no single C++ instantiation, so the Python one runs.
    s32 = einsums.zeros((4,), dtype="float32")
    assert isinstance(cg.diis([(t, s32)]), cg._PyDiis)

    # A tiled amplitude is not a dense runtime tensor either.
    tiled = einsums.TiledRuntimeTensorD("X", [[2, 2], [2, 2]])
    tiled.add_tile([0, 0])
    tiled.materialize()
    tiled_step = einsums.TiledRuntimeTensorD("R", [[2, 2], [2, 2]])
    tiled_step.add_tile([0, 0])
    tiled_step.materialize()
    assert isinstance(cg.diis([(tiled, tiled_step)]), cg._PyDiis)


def test_validation_matches_between_the_two():
    t = einsums.zeros((4,), dtype="float64")
    s = einsums.zeros((4,), dtype="float64")
    for factory in (cg.diis, cg.DIISAccelerator, cg._PyDiis):
        with pytest.raises(ValueError):
            factory([], k=8)
        with pytest.raises(ValueError):
            factory([(t, s)], k=1)


def test_captured_diis_step_is_one_node_and_replays():
    """The captured form: ``diis_step`` records a node the hazard scan orders.

    Nothing in the solvers uses this yet - they call ``step()`` from the loop
    predicate - but the node is what will let a whole iteration replay with no
    Python between passes, so it has to work.
    """
    n = 10
    rng = np.random.default_rng(5)
    M_np, b_np = _contraction(rng, n, "float64")
    exact = np.linalg.solve(np.eye(n) - M_np, b_np)

    def run(captured):
        t = einsums.zeros((n,), dtype="float64")
        s = einsums.zeros((n,), dtype="float64")
        M = einsums.asarray(np.ascontiguousarray(M_np))
        b = einsums.asarray(np.ascontiguousarray(b_np))
        acc = cg.DIISAccelerator([(t, s)], k=6)

        g = cg.Graph("diis node")
        with cg.capture(g):
            einsums.einsum("i <- i,j ; j", s, M, t, c_pf=0.0, ab_pf=1.0)  # s = M t
            la.axpby(1.0, b, 1.0, s)                                      # s += b
            la.axpby(-1.0, t, 1.0, s)                                     # s -= t
            la.axpby(1.0, s, 1.0, t)                                      # t += s
            if captured:
                cg.diis_step(acc._acc)
        expected = 5 if captured else 4
        assert g.num_nodes() == expected

        for _ in range(40):
            g.execute()
            if not captured:
                acc.step()
        return np.asarray(t).copy()

    in_graph = run(True)
    from_host = run(False)

    # The node does what the host-side call does, on the same schedule.
    assert np.array_equal(in_graph, from_host)
    np.testing.assert_allclose(in_graph, exact, rtol=1e-7, atol=1e-9)
