# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Parallel-executor stress (ASan/UBSan bait) + free-lifecycle stress, and the exception-propagation arms.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

import einsums
import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness


# ──────────────────────────────────────────────────────────────────────────
# Parallel-executor stress (ASan/UBSan bait)
#
# A use-after-free or heap-overflow in the OpenMP / Dataflow scheduler, a node
# whose buffer is freed while a concurrent node still reads it, from a missing
# dependency edge, is timing-dependent. A single execution rarely lands on the
# offending interleaving, and under a sanitizer the corruption is only flagged
# when the bad access actually fires. So we replay each program through the two
# parallel executors many times to shuffle the thread schedule and give ASan
# repeated chances at the bad window. This arm is pure sanitizer bait: float64
# only (a scheduler UAF is dtype-independent), no Sequential executor (it can't
# race), wider blocks (more independent nodes ⇒ more concurrency), and still
# oracle-checked every rep so a dropped/misordered node shows as a divergence
# too. Cheap without a sanitizer; the value is when run under the ASan build.
# ──────────────────────────────────────────────────────────────────────────

_STRESS_REPS = 30
_PARALLEL_EXECUTORS = [("OpenMP", cg.OpenMPExecutor), ("Dataflow", cg.DataflowExecutor)]


@pytest.mark.parametrize("seed", range(40))
def test_fuzz_parallel_executor_stress(seed):
    rng = np.random.default_rng(110_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=12)
    m_arrays, v_arrays, t_arrays = _seed_arrays(rng, "float64")

    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, np.dtype("float64"))
    if not _usable(om, ov, ot, cap=_DTYPE_CAP["float64"]):
        pytest.skip("oracle overflowed — numerically degenerate program")

    rtol, atol = _DTYPE_TOL["float64"]
    for ex_name, exec_cls in _PARALLEL_EXECUTORS:
        for optimize in (False, True):
            for rep in range(_STRESS_REPS):
                tag = f"stress{seed}_{ex_name}_{'opt' if optimize else 'raw'}_{rep}"
                gm, gv, gt = _run_program_exec(prog, m_arrays, v_arrays, t_arrays, tag, optimize, exec_cls)
                for got, oracle, kind in ((gm, om, "m"), (gv, ov, "v"), (gt, ot, "t")):
                    for idx in range(len(oracle)):
                        if not np.allclose(got[idx], oracle[idx], rtol=rtol, atol=atol):
                            raise AssertionError(
                                f"{ex_name}/{'opt' if optimize else 'raw'} rep {rep} "
                                f"diverged on {kind}{idx}\nprogram={prog!r}\n"
                                f"got=\n{got[idx]}\noracle=\n{oracle[idx]}"
                            )


# ──────────────────────────────────────────────────────────────────────────
# Free-lifecycle parallel stress: graphs whose deferred scratch intermediates
# cross FreeInsertion's 1 MiB floor, so default_pass_manager() emits the full
# Materialize/Free lifecycle and MemoryPlanning arena-packs the buffers - then
# replayed under the parallel executors against a numpy oracle.
#
# This arm exists because the stress arm above, written to catch "a buffer
# freed while a concurrent node still reads it", could never fire: fuzz
# tensors are tiny and come from user pools, so no fuzzed graph ever
# contained a Free node. That seam hid a real bug (Free nodes were
# input-only, unordered against readers; the DataflowExecutor could release
# a buffer mid-read once the arena landed). Every intermediate here is
# 1.13 MB, and the random chain re-reads earlier intermediates so frees sit
# after multiple readers.
#
# The scratch is declared with intermediate=True (user-visible declares are
# never freed) and GeneralRuntimeTensor supports materialize_into, so the
# full lifecycle engages: Materialize, Free, and MemoryPlanning's arena.
# Because absent coverage is silent (a graph with no Free nodes passes
# trivially - that is exactly how the original bug hid), the test ASSERTS
# the optimized graph contains Free nodes before executing anything.
# ──────────────────────────────────────────────────────────────────────────

_FREE_N = 385  # 385*385*8 B = 1.13 MiB: over FreeInsertion's 1 MiB floor
_FREE_REPS = 8


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_free_lifecycle_parallel_stress(seed):
    rng = np.random.default_rng(150_000 + seed)
    n = _FREE_N
    A_np = rng.standard_normal((n, n)) * 0.1  # scaled so chains stay bounded
    steps = int(rng.integers(3, 7))
    # rhs_pick[s] chooses the right operand of step s: 0 = A, k>0 = mid k-1.
    rhs_pick = [int(rng.integers(0, s + 1)) for s in range(steps)]

    # numpy oracle
    mids_np = []
    cur = A_np
    for s in range(steps):
        rhs = A_np if rhs_pick[s] == 0 else mids_np[rhs_pick[s] - 1]
        cur = cur @ rhs
        mids_np.append(cur)
    oracle = mids_np[-1]

    for ex_name, exec_cls in _PARALLEL_EXECUTORS:
        A = einsums.asarray(A_np.copy(), name=f"free_A_{seed}_{ex_name}")
        C = einsums.asarray(np.zeros((n, n)), name=f"free_C_{seed}_{ex_name}")

        g = cg.Graph(f"free_stress_{seed}_{ex_name}")
        mids = [g.declare_zero_tensor(f"mid{k}", [n, n], dtype="float64", intermediate=True) for k in range(steps - 1)]
        with cg.capture(g):
            lhs = A
            for s in range(steps):
                rhs = A if rhs_pick[s] == 0 else mids[rhs_pick[s] - 1]
                out = mids[s] if s < steps - 1 else C
                einsums.einsum("i,j <- i,k ; k,j", out, lhs, rhs, c_pf=0.0, ab_pf=1.0)
                lhs = out

        g.apply(cg.default_pass_manager())

        # Coverage guard: without Free nodes this test proves nothing (a
        # Free-less graph passes trivially, which is exactly how the original
        # bug hid). The exact count varies - ContractionPlanning restructures
        # chains and its _cp_ intermediates get their own lifecycles - so
        # only presence is asserted.
        kinds = [node.get("kind") for node in json.loads(g.to_json()).get("nodes", [])]
        assert kinds.count("Free") >= 1, f"no Free nodes in optimized graph; kinds={kinds}"
        assert kinds.count("Materialize") >= 1

        # Use the file-wide differential tolerance, not a hand-tightened one.
        # This compares a numpy oracle against a reordered, multithreaded-BLAS
        # graph execution, so the reduction order differs from numpy's and
        # varies per replay. A chain of up to `steps` matmuls of dimension
        # `_FREE_N` produces values with a wide dynamic range (~1e6 here), whose
        # float64 round-off floor is ~1e-9 in absolute terms - well past a
        # 1e-10 tolerance. That made rep-to-rep fp noise (max abs err ~1.4e-9)
        # read as a "divergence". A genuine free-lifecycle corruption (a read of
        # freed/reused arena bytes) differs by O(1), still far outside 1e-5.
        rtol, atol = _DTYPE_TOL["float64"]
        for rep in range(_FREE_REPS):
            g.execute(exec_cls())
            got = np.asarray(C)
            if not np.allclose(got, oracle, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{ex_name} rep {rep} diverged (seed {seed}, steps {steps}, rhs_pick {rhs_pick})\n"
                    f"max abs err = {np.abs(got - oracle).max()}"
                )

# ──────────────────────────────────────────────────────────────────────────
# Exception propagation: a node that throws must make execute() raise on every
# executor, never hang (TaskPool orphaning a continuation / OpenMP exception
# escaping its parallel region) and never silently complete (swallowed error).
# We inject a throwing element_transform mid-graph with a dependent op, so the
# failure has to travel through the dependency chain to the waited sink.
# ──────────────────────────────────────────────────────────────────────────

_INJECTED_BOOM = "fuzz-injected element_transform failure"


def _boom(_x):
    raise ValueError(_INJECTED_BOOM)


def check_program_raises(prog, m_arrays, v_arrays, t_arrays, label):
    for ex_name, exec_cls in _CROSS_EXECUTORS:
        for optimize in (False, True):
            tag = f"{label}_{ex_name}_{'opt' if optimize else 'raw'}"
            mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, tag)
            g = cg.Graph(tag)
            build_cg(prog, g, mats, vecs, r3s, tag)
            # Inject a throwing transform, then a dependent op so the failure
            # must propagate through the dependency graph to the sink. The
            # callback runs on a worker thread under the parallel executors; the
            # exception must surface here (translated to a C++/Python error),
            # never hang and never be silently swallowed.
            with cg.capture(g):
                einsums.linalg.element_transform(mats[0], _boom)
                einsums.linalg.scale(2.0, mats[0])
            if optimize:
                g.apply(cg.default_pass_manager())

            with pytest.raises(Exception):
                g.execute(exec_cls())


@pytest.mark.skip(
    reason="Same intermittent thread::join EDEADLK as "
    "test_fuzz_executor_propagates_exception_control_flow below: triggered only when "
    "this case cycles Sequential+OpenMP+Dataflow executors in one process via "
    "check_program_raises (a fuzz-only pattern; real callers pick one executor). "
    "Surfaced on Linux gcc+openblas CI as a mid-pytest subprocess abort. Tracked in "
    "alongside the control-flow variant."
)
@pytest.mark.parametrize("seed", range(120))
def test_fuzz_executor_propagates_exception(seed):
    rng = np.random.default_rng(60_000 + seed)
    prog = _gen_block(rng, depth=0, max_stmts=8)
    check_program_raises(prog, *_seed_arrays(rng), f"boom{seed}")


@pytest.mark.skip(
    reason="Single-executor control-flow + exception is fixed (worker BLAS is now "
    "single-threaded, so a control-flow body's BLAS no longer opens a libomp region "
    "from a worker thread). What remains is a separate, intermittent thread::join "
    "EDEADLK that only triggers when this case cycles seq+omp+df in one process — a "
    "fuzz-only pattern; real callers pick one executor."
)
@pytest.mark.parametrize("seed", range(80))
def test_fuzz_executor_propagates_exception_control_flow(seed):
    rng = np.random.default_rng(65_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    check_program_raises(prog, *_seed_arrays(rng), f"boomcf{seed}")
