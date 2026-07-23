# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Opt-in passes: LinearCombinationContractionFolding + DistributiveFactoring.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness


# ══════════════════════════════════════════════════════════════════════════
# Opt-in passes: LinearCombinationContractionFolding + DistributiveFactoring.
#
# Neither pass is registered in PassManager::create_default() nor listed in
# _SAFE_PASSES, so no other fuzz mode ever reaches them - they were exercised
# only by their own hand-written C++ unit tests. Both are now exposed to Python
# (via the APIARY annotations on their headers) so the fuzzer can drive them.
#
#   LCCF group: C = 2*einsum(k,kij->ij) ; C -= einsum(k,kji->ij)  (same vector A,
#               same rank-3 B read with transposed axes -> the CCSD 2J-K fold)
#   DF group:   R += a1*(A@B1) ; R += a2*(A@B2)      (same A, different B,
#               same index pattern, both accumulate -> DistributiveFactoring)
#
# Both passes fold and EXECUTE correctly on the Python (runtime-tensor) path, so
# both are fuzzed end-to-end below against the numpy oracle.
#
# --- Fixed: DistributiveFactoring runtime execution ---------------------------
# DF's rewrite used to build the "sum of non-shared operands" accumulator with
# Graph::create_tensor_dynamic(), a COMPILE-TIME Tensor<T,Rank>; on a Python graph
# whose tensors are GeneralRuntimeTensor<T> the axpy chain then added a runtime
# source into a fixed-rank destination and threw rank_error. DF now builds a
# RUNTIME accumulator (create_zero_runtime_tensor_dynamic) when its operands are
# runtime, and dispatch_binary/dispatch_unary cast runtime handles correctly, so
# the rewrite executes on runtime tensors and is fuzzed with full execution below.
#
# --- Fixed: DistributiveFactoring program order -------------------------------
# DF used to APPEND the combined node; when a later op read a factored tensor the
# replacement writer landed after the reader and the pass verifier rejected it. DF
# now places the combined node at the first replaced node's slot (like GEMMBatching
# / LCCF), so downstream readers observe the written result.
#
# --- Fixed: counter reset under recursion -------------------------------------
# DF used to expose num_groups / num_eliminated as zero on control-flow graphs:
# recurse_into_subgraphs() was true and run() reset the counters, so PassManager
# recursion into subgraphs clobbered the top-level tally with the LAST subgraph's.
# DF now manages its own descent (recurse_into_subgraphs() false, like
# LoopInvariantHoisting) and resets once at the root, so the counters accumulate
# across the whole tree. Coverage below still probes on a control-flow-FREE group
# for a tight, unambiguous num_groups == 1 signal.
# ------------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════


def _gen_lccf_group(rng):
    """Two vector*rank-3 contractions into the same matrix output, sharing the
    vector and reading the same rank-3 with transposed trailing axes:
    LinearCombinationContractionFolding's canonical shape. Returns (stmts, C)."""
    n = _d(rng, R3_DIMS)  # trailing square dim doubles as the output extent
    k = _d(rng, R3_DIMS)
    Bt = _pick_r3(rng, (k, n, n))
    Av = _pick_vec(rng, k)
    Cm = _pick_mat(rng, (n, n))
    if None in (Bt, Av, Cm):
        return None, None
    grp = [
        ("leinsum", "kij", _scalar(rng), Av, Bt, 0.0, Cm),
        ("leinsum", "kji", _scalar(rng), Av, Bt, 1.0, Cm),
    ]
    return grp, Cm


def _gen_df_group(rng):
    """Two accumulating einsums sharing operand A with different B operands:
    DistributiveFactoring's canonical shape. Returns (stmts, R) or (None, None)."""
    m, k, n = _d(rng), _d(rng), _d(rng)
    B1 = _pick_mat(rng, (k, n))
    B2 = _pick_mat(rng, (k, n), (B1,))
    # The shared operand must differ from the summed operands: DF redirects the
    # non-shared operand's slot (keyed by tensor id), so a shared operand aliasing
    # a summed one (e.g. when m==k==n picks A==B1) is rejected by the pass and
    # would not factor. Exclude B1/B2 so the injected group is always factorable.
    A = _pick_mat(rng, (m, k), (B1, B2))
    R = _pick_mat(rng, (m, n), (A, B1, B2))
    if None in (A, B1, B2, R):
        return None, None
    grp = [
        ("einsum", "ij <- ik ; kj", _scalar(rng), A, B1, 1.0, R, False, False),
        ("einsum", "ij <- ik ; kj", _scalar(rng), A, B2, 1.0, R, False, False),
    ]
    return grp, R


def _optin_group_fires(pass_obj, group, m, v, t, name):
    """Run an opt-in pass over a control-flow-FREE graph holding only the injected
    group, so the pass's num_groups counter is trustworthy (see the secondary
    finding on counter reset under subgraph recursion). Returns num_groups."""
    mats, vecs, r3 = _make_pool(m, v, t, name)
    g = cg.Graph(name)
    build_cg(group, g, mats, vecs, r3, name)
    pm = cg.PassManager()
    pm.add(pass_obj)
    pm.run(g)
    return pass_obj.num_groups


def _run_lccf(prog, m, v, t, name):
    """Apply LinearCombinationContractionFolding EARLY (on the raw einsum group,
    before the default manager collapses it), then the default manager, then
    execute. Returns the resulting pools."""
    mats, vecs, r3 = _make_pool(m, v, t, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3, name)

    lccf = _G.LinearCombinationContractionFolding()
    pm = cg.PassManager()
    pm.add(lccf)
    pm.run(g)

    g.apply(cg.default_pass_manager())
    g.execute()
    return (
        [np.asarray(x).copy() for x in mats],
        [np.asarray(x).copy() for x in vecs],
        [np.asarray(x).copy() for x in r3],
    )


def _run_df(prog, m, v, t, name):
    """Apply DistributiveFactoring EARLY (on the raw einsum group, before the
    default manager restructures it), then the default manager, then execute.
    Returns the resulting pools."""
    mats, vecs, r3 = _make_pool(m, v, t, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3, name)

    df = _G.DistributiveFactoring()
    pm = cg.PassManager()
    pm.add(df)
    pm.run(g)

    g.apply(cg.default_pass_manager())
    g.execute()
    return (
        [np.asarray(x).copy() for x in mats],
        [np.asarray(x).copy() for x in vecs],
        [np.asarray(x).copy() for x in r3],
    )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(100))
def test_fuzz_optin_lccf(seed, dtype):
    """Random program embedding an LCCF-foldable group plus random control-flow
    filler, run through LinearCombinationContractionFolding + the default
    manager, executed and compared to the numpy oracle over every dtype. A
    mismatch is a REAL FINDING in the fold (or its interaction with the default
    pipeline). Coverage: the injected group is built to fold, so the pass MUST
    fire; the arm asserts num_groups >= 1 whenever the group was generated."""
    rng = np.random.default_rng(190_000 + seed)
    m, v, t = _seed_arrays(rng, dtype)
    lccf_grp, _ = _gen_lccf_group(rng)
    filler = _gen_block(rng, depth=2, max_stmts=5)
    prog = (lccf_grp or []) + filler

    rtol, atol = _DTYPE_TOL[dtype]
    cap = _DTYPE_CAP[dtype]
    dt = np.dtype(dtype)
    om = [a.copy() for a in m]
    ov = [a.copy() for a in v]
    ot = [a.copy() for a in t]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, dt)
    if not _usable(om, ov, ot, cap=cap):
        pytest.skip("oracle overflowed — numerically degenerate program")

    # Coverage: prove the fold fired on the (control-flow-free) injected group.
    if lccf_grp is not None:
        probe = _G.LinearCombinationContractionFolding()
        fired = _optin_group_fires(probe, lccf_grp, m, v, t, f"optin_lccf_cov{seed}_{dtype}")
        assert fired >= 1, f"LinearCombinationContractionFolding did not fire on its injected group (seed {seed})"

    (gm, gv, gt) = _run_lccf(prog, m, v, t, f"optin_lccf{seed}_{dtype}")

    for got, oracle, kind in ((gm, om, "m"), (gv, ov, "v"), (gt, ot, "t")):
        for idx in range(len(oracle)):
            if not np.allclose(got[idx], oracle[idx], rtol=rtol, atol=atol):
                raise AssertionError(
                    f"OPT-IN LCCF disagrees with oracle on {kind}{idx} (dtype={dtype})\n"
                    f"program={prog!r}\ngot=\n{got[idx]}\noracle=\n{oracle[idx]}"
                )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(100))
def test_fuzz_optin_distributive_factoring(seed, dtype):
    """Random program embedding a DistributiveFactoring-factorable group plus
    random control-flow filler, run through DistributiveFactoring + the default
    manager, executed and compared to the numpy oracle over every dtype. A
    mismatch is a REAL FINDING in the factoring (or its interaction with the
    default pipeline). Coverage: the injected group is built to factor, so the
    pass MUST fire; the arm asserts num_groups >= 1 whenever the group was
    generated (probed on a control-flow-free graph for an unambiguous count)."""
    rng = np.random.default_rng(200_000 + seed)
    m, v, t = _seed_arrays(rng, dtype)
    df_grp, _ = _gen_df_group(rng)
    filler = _gen_block(rng, depth=2, max_stmts=5)
    prog = (df_grp or []) + filler

    rtol, atol = _DTYPE_TOL[dtype]
    cap = _DTYPE_CAP[dtype]
    dt = np.dtype(dtype)
    om = [a.copy() for a in m]
    ov = [a.copy() for a in v]
    ot = [a.copy() for a in t]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, dt)
    if not _usable(om, ov, ot, cap=cap):
        pytest.skip("oracle overflowed — numerically degenerate program")

    # Coverage: prove the factoring fired on the (control-flow-free) injected group.
    if df_grp is not None:
        probe = _G.DistributiveFactoring()
        fired = _optin_group_fires(probe, df_grp, m, v, t, f"optin_df_cov{seed}_{dtype}")
        assert fired >= 1, f"DistributiveFactoring did not fire on its injected group (seed {seed})"

    (gm, gv, gt) = _run_df(prog, m, v, t, f"optin_df{seed}_{dtype}")

    for got, oracle, kind in ((gm, om, "m"), (gv, ov, "v"), (gt, ot, "t")):
        for idx in range(len(oracle)):
            if not np.allclose(got[idx], oracle[idx], rtol=rtol, atol=atol):
                raise AssertionError(
                    f"OPT-IN DF disagrees with oracle on {kind}{idx} (dtype={dtype})\n"
                    f"program={prog!r}\ngot=\n{got[idx]}\noracle=\n{oracle[idx]}"
                )


def test_distributive_factoring_preserves_program_order():
    """Regression: DistributiveFactoring places its combined node at
    the first replaced node's slot (like GEMMBatching / LCCF), so a later op that
    reads a factored tensor still observes the written result. The graph's own pass
    verifier (check_observed_writes) accepts the rewrite, and execution matches the
    numpy oracle. (Previously DF appended the combined node, landing it after the
    reader, and the verifier rejected it.)"""
    rng = np.random.default_rng(1)
    A0 = rng.standard_normal((3, 3))
    B1_0 = rng.standard_normal((3, 3))
    B2_0 = rng.standard_normal((3, 3))
    R0 = rng.standard_normal((3, 3))
    E0 = rng.standard_normal((3, 3))

    A = einsums.asarray(A0.copy(), name="dfo_A")
    B1 = einsums.asarray(B1_0.copy(), name="dfo_B1")
    B2 = einsums.asarray(B2_0.copy(), name="dfo_B2")
    R = einsums.asarray(R0.copy(), name="dfo_R")
    E = einsums.asarray(E0.copy(), name="dfo_E")
    S = einsums.asarray(np.zeros((3, 3)), name="dfo_S")
    pool = [A, B1, B2, R, E, S]

    g = cg.Graph("df_order")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", R, A, B1, c_pf=1.0, ab_pf=0.5)
        einsums.einsum("ij <- ik ; kj", R, A, B2, c_pf=1.0, ab_pf=0.7)
        einsums.linalg.gemm(1.0, R, E, 0.0, S)  # downstream reader of the factored R
    pm = cg.PassManager()
    df = _G.DistributiveFactoring()
    pm.add(df)
    assert pm.run(g) and df.num_groups == 1  # rewrites and passes the verifier
    g.execute()

    R_np = R0 + 0.5 * (A0 @ B1_0) + 0.7 * (A0 @ B2_0)
    S_np = R_np @ E0
    assert np.allclose(np.asarray(R), R_np, rtol=RTOL, atol=ATOL)
    assert np.allclose(np.asarray(S), S_np, rtol=RTOL, atol=ATOL)
    del pool


def test_distributive_factoring_executes_on_runtime_tensor():
    """Regression: DistributiveFactoring builds a RUNTIME accumulator
    when its operands are runtime tensors, so a DF-rewritten Python graph executes
    and matches the numpy oracle. Same (4,3)*(3,5)->(4,5) shape the C++ unit test
    uses. (Previously the axpy chain added a runtime source into a compile-time
    Tensor<T,Rank> accumulator and threw rank_error at execute.)"""
    rng = np.random.default_rng(0)
    A0 = rng.standard_normal((4, 3))
    B1_0 = rng.standard_normal((3, 5))
    B2_0 = rng.standard_normal((3, 5))

    A = einsums.asarray(A0.copy(), name="df_A")
    B1 = einsums.asarray(B1_0.copy(), name="df_B1")
    B2 = einsums.asarray(B2_0.copy(), name="df_B2")
    R = einsums.asarray(np.zeros((4, 5)), name="df_R")

    g = cg.Graph("df_runtime")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", R, A, B1, c_pf=1.0, ab_pf=0.5)
        einsums.einsum("ij <- ik ; kj", R, A, B2, c_pf=1.0, ab_pf=0.7)
    df = _G.DistributiveFactoring()
    pm = cg.PassManager()
    pm.add(df)
    assert pm.run(g) and df.num_groups == 1

    pool = [A, B1, B2, R]  # keep operands alive through execute
    g.execute()
    R_np = 0.5 * (A0 @ B1_0) + 0.7 * (A0 @ B2_0)
    assert np.allclose(np.asarray(R), R_np, rtol=RTOL, atol=ATOL)
    del pool
