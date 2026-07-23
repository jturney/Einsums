# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Deferred (workspace) tensor materialization differential fuzz.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness


# ══════════════════════════════════════════════════════════════════════════
# Deferred (workspace) tensors: exercise the MaterializationPass / deferred
# allocation path differentially, which the all-eager suites above never touch.
#
# Half the matrices are declared deferred (workspace, zero-initialized at
# materialize time); the rest stay eager with random data so nonzero values
# still flow. Each program runs two ways and both must match an oracle in which
# the deferred matrices start at zero:
#   * RAW:       explicit ws.materialize_all() then execute (no passes).
#   * OPTIMIZED: default manager (its MaterializationPass allocates/zeroes the
#                deferred tensors) then execute.
# This validates that MaterializationPass produces the same result as explicit
# materialization, across the full op / control-flow / view / dtype surface.
# ══════════════════════════════════════════════════════════════════════════


def _deferred_mask(n):
    return [i % 2 == 1 for i in range(n)]


def _make_pool_deferred(m_arrays, v_arrays, t_arrays, name):
    """Odd-indexed tensors in *each* pool (matrices, vectors, rank-3) are
    declared deferred (workspace, zero-initialized); even-indexed ones stay
    eager with random data so nonzero values still flow."""
    ws = _G.Workspace(f"{name}_ws")

    def build(prefix, arrays):
        mask = _deferred_mask(len(arrays))
        out = []
        for idx, arr in enumerate(arrays):
            dt = str(arr.dtype)
            if mask[idx]:
                out.append(ws.declare_zero_tensor(f"{name}_d{prefix}{idx}", list(arr.shape), dt))
            else:
                tn = einsums.create_zero_tensor(f"{name}_{prefix}{idx}", list(arr.shape), dtype=dt)
                np.asarray(tn)[...] = arr
                out.append(tn)
        return out

    return build("m", m_arrays), build("v", v_arrays), build("t", t_arrays), ws


def _referenced(stmts, m, v, t):
    """Pool indices the program reads or writes, split by pool (matrix / vector
    / rank-3). A deferred tensor the program never references is *not*
    materialized by the pass, so its post-execute value is undefined and must be
    excluded from the comparison."""
    for s in stmts:
        k = s[0]
        if k in ("scale", "etransform", "vscale"):
            m.add(s[2])
        elif k == "axpy":
            m.update((s[2], s[3]))
        elif k == "axpby":
            m.update((s[2], s[4]))
        elif k == "gemm":
            m.update((s[2], s[3], s[5]))
        elif k == "einsum":
            m.update((s[3], s[4], s[6]))
        elif k == "perm":
            m.update((s[3], s[4]))
        elif k == "symm":
            m.update((s[1], s[2], s[3]))
        elif k == "gemv":
            m.add(s[2])
            v.update((s[3], s[5]))
        elif k == "ger":
            v.update((s[2], s[3]))
            m.add(s[4])
        elif k == "vaxpy":
            m.update((s[2], s[3]))
        elif k == "vgemm":
            m.update((s[2], s[7], s[9]))
        elif k == "beinsum":
            t.update((s[3], s[4], s[6]))
        elif k == "leinsum":
            v.add(s[3])
            t.add(s[4])
            m.add(s[6])
        elif k == "loop":
            _referenced(s[2], m, v, t)
        elif k == "cond":
            _referenced(s[2], m, v, t)
            _referenced(s[3], m, v, t)
    return m, v, t


def _deferred_oracle_init(m_arrays, v_arrays, t_arrays):
    mm, vm, tm = _deferred_mask(len(m_arrays)), _deferred_mask(len(v_arrays)), _deferred_mask(len(t_arrays))
    om = [np.zeros_like(a) if mm[i] else a.copy() for i, a in enumerate(m_arrays)]
    ov = [np.zeros_like(a) if vm[i] else a.copy() for i, a in enumerate(v_arrays)]
    ot = [np.zeros_like(a) if tm[i] else a.copy() for i, a in enumerate(t_arrays)]
    return om, ov, ot, (mm, vm, tm)


def _deferred_skips(prog, masks):
    mm, vm, tm = masks
    rm, rv, rt = _referenced(prog, set(), set(), set())
    return ({i for i in range(len(mm)) if mm[i] and i not in rm},
            {i for i in range(len(vm)) if vm[i] and i not in rv},
            {i for i in range(len(tm)) if tm[i] and i not in rt})


def _check_deferred(stage, prog, pools, oracle, skips):
    for arrs, oarr, skip, kind in zip(pools, oracle, skips, "mvt"):
        for i in range(len(oarr)):
            if i in skip:  # unused deferred tensor, never materialized
                continue
            got = np.asarray(arrs[i])
            if not np.allclose(got, oarr[i], rtol=RTOL, atol=ATOL):
                raise AssertionError(f"{stage} disagrees on {kind}{i}\nprogram={prog!r}\ngot=\n{got}\noracle=\n{oarr[i]}")


def check_program_deferred(prog, m_arrays, v_arrays, t_arrays, label):
    om, ov, ot, masks = _deferred_oracle_init(m_arrays, v_arrays, t_arrays)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot)
    if not _usable(om, ov, ot):
        pytest.skip("oracle overflowed — numerically degenerate program")
    oracle = (om, ov, ot)
    skips = _deferred_skips(prog, masks)

    # RAW: materialize the workspace explicitly, no optimization passes.
    mats, vecs, r3, ws = _make_pool_deferred(m_arrays, v_arrays, t_arrays, f"{label}_raw")
    g = cg.Graph(f"{label}_raw")
    build_cg(prog, g, mats, vecs, r3, f"{label}_raw")
    ws.materialize_all()
    g.execute()
    _check_deferred("DEFERRED-RAW", prog, (mats, vecs, r3), oracle, skips)

    # OPTIMIZED: the default manager's MaterializationPass allocates/zeroes.
    mats2, vecs2, r32, _ = _make_pool_deferred(m_arrays, v_arrays, t_arrays, f"{label}_opt")
    g2 = cg.Graph(f"{label}_opt")
    build_cg(prog, g2, mats2, vecs2, r32, f"{label}_opt")
    g2.apply(cg.default_pass_manager())
    g2.execute()
    _check_deferred("DEFERRED-OPTIMIZED", prog, (mats2, vecs2, r32), oracle, skips)


def check_program_deferred_replay(prog, m_arrays, v_arrays, t_arrays, label):
    """Deferred tensors + re-execution. The optimized graph carries Initialize
    nodes that re-zero each deferred tensor at the start of every execute, so on
    replay the deferred (scratch) tensors reset to zero while eager tensors carry
    over, verified empirically. Execute twice and compare to an oracle applied
    twice with the deferred tensors reset to zero before each application.

    Only the optimized path is meaningful: the explicit-materialize_all RAW path
    zeroes once and would NOT reset deferred tensors on replay (a different, and
    not the intended, re-execution semantics)."""
    om, ov, ot, masks = _deferred_oracle_init(m_arrays, v_arrays, t_arrays)
    mm, vm, tm = masks
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot)  # application 1
        for arrs, mask in ((om, mm), (ov, vm), (ot, tm)):  # Initialize re-zeroes deferred
            for i in range(len(arrs)):
                if mask[i]:
                    arrs[i] = np.zeros_like(arrs[i])
        interp_np(prog, om, ov, ot)  # application 2
    if not _usable(om, ov, ot):
        pytest.skip("oracle overflowed — numerically degenerate program")
    skips = _deferred_skips(prog, masks)

    mats, vecs, r3, _ = _make_pool_deferred(m_arrays, v_arrays, t_arrays, f"{label}_rep")
    g = cg.Graph(f"{label}_rep")
    build_cg(prog, g, mats, vecs, r3, f"{label}_rep")
    g.apply(cg.default_pass_manager())
    g.execute()
    g.execute()
    _check_deferred("DEFERRED-REPLAY", prog, (mats, vecs, r3), (om, ov, ot), skips)


def test_regression_deferred_loop_accumulate_then_read():
    """A deferred tensor (m1) accumulated inside a loop and then read by a
    parent op. MaterializationPass must materialize+zero it once before the
    loop; emitting a second Initialize before the later read re-zeroes it and
    wipes the loop's accumulation. (m1 is deferred under _deferred_mask.)"""
    prog = [
        ("loop", 3, [("axpby", 0.5, 0, 0.5, 1)]),  # m1 += accumulate from m0
        ("axpy", 1.0, 1, 2),                        # m2 += m1 (reads m1 outside the loop)
    ]
    check_program_deferred(prog, *_square_seed_arrays(np.random.default_rng(31337)), "def_accum")


@pytest.mark.parametrize("seed", fuzz_seeds(250))
def test_fuzz_deferred(seed):
    rng = np.random.default_rng(120_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=6)
    check_program_deferred(prog, *_seed_arrays(rng), f"def{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_deferred_complex(seed):
    rng = np.random.default_rng(130_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=6)
    check_program_deferred(prog, *_seed_arrays(rng, "complex128"), f"cdef{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(200))
def test_fuzz_deferred_replay(seed):
    rng = np.random.default_rng(140_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=6)
    check_program_deferred_replay(prog, *_seed_arrays(rng), f"defr{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_deferred_replay_complex(seed):
    rng = np.random.default_rng(150_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=6)
    check_program_deferred_replay(prog, *_seed_arrays(rng, "complex128"), f"cdefr{seed}")
