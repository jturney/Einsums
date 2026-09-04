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
from _region_invariants import assert_materialization_invariants


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
    """Two of every three pool slots are deferred, and the third is not.

    The pool lays ``COPIES`` tensors of each shape at consecutive indices, so
    deferring all but the first of each group is what puts TWO same-shaped
    deferred tensors in the corpus. The old rule deferred the odd indices,
    which for half the shapes left exactly one, and a shape with one deferred
    tensor cannot express the bug this shard now covers: two deferred parents
    sliced at the same offset presenting views on identical byte spans. One
    eager copy per shape stays, so nonzero values still flow into the deferred
    ones.
    """
    return [i % COPIES != 0 for i in range(n)]


def _make_pool_deferred(m_arrays, v_arrays, t_arrays, name, graph=None):
    """The deferred half of each pool, declared on a Workspace or on a Graph.

    ``graph=None`` declares them on a Workspace, which the caller materializes
    itself. Passing a graph declares them as graph-owned intermediates instead,
    which is the path ``Materialization`` allocates and the one whose usage
    analysis has to resolve a view back to its parent; the returned workspace is
    then a placeholder with nothing in it.
    """
    ws = _G.Workspace(f"{name}_ws")

    def build(prefix, arrays):
        mask = _deferred_mask(len(arrays))
        out = []
        for idx, arr in enumerate(arrays):
            dt = str(arr.dtype)
            if mask[idx]:
                dname = f"{name}_d{prefix}{idx}"
                if graph is None:
                    out.append(ws.declare_zero_tensor(dname, list(arr.shape), dt))
                else:
                    out.append(graph.declare_zero_tensor(dname, list(arr.shape), intermediate=True, dtype=dt))
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
        if k in ("scale", "etransform", "vscale", "ivscale"):
            m.add(s[2])
        elif k == "ivaxpy":
            v.add(s[2])
            m.add(s[3])
        elif k == "i3scale":
            t.add(s[2])
        elif k == "i3axpy":
            m.add(s[2])
            t.add(s[3])
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
        elif k in ("vvscale", "tvscale"):
            m.add(s[2])
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


def _all_deferred_skips(masks):
    """Every deferred slot, whatever the program does with it."""
    return tuple({i for i in range(len(mask)) if mask[i]} for mask in masks)


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
    assert_materialization_invariants(g2, f"{label} workspace-declared")

    # GRAPH-DECLARED: the same deferred half declared on the graph as OWNED
    # INTERMEDIATES rather than on a workspace. Materialization then owns the
    # whole lifecycle, and its "is this tensor used" question has to resolve a
    # view back to the parent it slices, which is where the alias merge that
    # motivated this arm lived.
    #
    # Only the ordinary tensors are compared here, for the reason the graph
    # scratch shard gives: a graph-owned intermediate is not observable, so
    # DeadNodeElimination may legitimately drop a write to one nothing reads and
    # FreeInsertion may release it. What flowed THROUGH a scratch still reaches
    # an ordinary tensor, so a wrong value in one is not hidden by this; a
    # scratch nothing reads contributes nothing to the answer by definition.
    g3 = cg.Graph(f"{label}_gd")
    mats3, vecs3, r33, _ = _make_pool_deferred(m_arrays, v_arrays, t_arrays, f"{label}_gd", graph=g3)
    build_cg(prog, g3, mats3, vecs3, r33, f"{label}_gd")
    g3.apply(cg.default_pass_manager())
    g3.execute()
    _check_deferred("DEFERRED-GRAPH-DECLARED", prog, (mats3, vecs3, r33), oracle, _all_deferred_skips(masks))
    assert_materialization_invariants(g3, f"{label} graph-declared")


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
    prog = _gen_block(rng, depth=2, max_stmts=6, rank_views=True)
    check_program_deferred(prog, *_seed_arrays(rng), f"def{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_deferred_complex(seed):
    rng = np.random.default_rng(130_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=6, rank_views=True)
    check_program_deferred(prog, *_seed_arrays(rng, "complex128"), f"cdef{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(200))
def test_fuzz_deferred_replay(seed):
    rng = np.random.default_rng(140_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=6, rank_views=True)
    check_program_deferred_replay(prog, *_seed_arrays(rng), f"defr{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_deferred_replay_complex(seed):
    rng = np.random.default_rng(150_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=6, rank_views=True)
    check_program_deferred_replay(prog, *_seed_arrays(rng, "complex128"), f"cdefr{seed}")


def test_regression_two_deferred_parents_seen_through_dropped_views():
    """Two same-shaped deferred tensors, sliced at the same non-zero offset.

    The rank-reducing views present identical extents, and while a deferred
    parent's view carried the shell's sentinel address plus its slice offset
    they presented identical byte SPANS as well, so the pointer-derived alias
    linking merged the two parents into one root. Every use of the loser was
    credited to the winner, and Materialization, asked whether anything used it,
    was told no and left it unallocated.

    Pinned from Python as well as from C++ because this is the shape the
    generator now draws, and a fuzzer that cannot reach its own regression is
    a fuzzer nobody can debug.
    """
    r3_mask = _deferred_mask(len(R3_SHAPES))
    first, second = [i for i in R3_BY_SHAPE[(2, 2, 2)] if r3_mask[i]][:2]
    mat_mask = _deferred_mask(len(MAT_SHAPES))
    mat = next(i for i in MAT_BY_SHAPE[(2, 2)] if not mat_mask[i])
    prog = [
        ("i3axpy", 1.0, mat, first, 1, 0, 2, 0, 2),    # t[first][0:2, 0:2, 1] += m
        ("i3axpy", 2.0, mat, second, 1, 0, 2, 0, 2),   # t[second][0:2, 0:2, 1] += 2*m
        ("i3scale", 0.5, first, 1, 0, 2, 0, 2),
    ]
    rng = np.random.default_rng(4242)
    m = [rng.standard_normal(sh) for sh in MAT_SHAPES]
    v = [rng.standard_normal((L,)) for L in VEC_LENS]
    t = [rng.standard_normal(sh) for sh in R3_SHAPES]
    check_program_deferred(prog, m, v, t, "def_two_parents")


def test_the_deferred_corpus_draws_dropped_views_of_two_parents_per_shape():
    """The corpus guard, because both halves of the shape are drawn rather than
    written down: an axis dropped at a non-zero offset, and two deferred tensors
    of one shape for the views to be confused between."""
    kinds = set()
    for seed in range(60):
        rng = np.random.default_rng(120_000 + seed)
        for stmt in _gen_block(rng, depth=2, max_stmts=6, rank_views=True):
            kinds.add(stmt[0])
            if stmt[0] in ("loop", "cond"):
                for sub in stmt[2:]:
                    if isinstance(sub, list):
                        kinds.update(x[0] for x in sub)
    assert {"ivscale", "ivaxpy", "i3scale", "i3axpy"} & kinds, kinds

    for sizes in (len(MAT_SHAPES), len(VEC_LENS), len(R3_SHAPES)):
        mask = _deferred_mask(sizes)
        # The pool lays COPIES tensors of each shape consecutively.
        assert sum(mask[:COPIES]) >= 2, "a shape with fewer than two deferred copies"

    # And the two halves TOGETHER, which is the shape the regression above pins:
    # two deferred tensors of one shape, each sliced. Neither half alone reaches
    # it, and the corpus drew it in none of its programs before this change.
    mat_mask = _deferred_mask(len(MAT_SHAPES))
    r3_mask = _deferred_mask(len(R3_SHAPES))
    reached = 0
    for seed in range(250):
        rng = np.random.default_rng(120_000 + seed)
        sliced: dict = {}

        def note(pool_shapes, mask, index):
            if mask[index]:
                sliced.setdefault(pool_shapes[index], set()).add(index)

        def walk(stmts):
            for stmt in stmts:
                kind = stmt[0]
                if kind in ("vscale", "vvscale", "tvscale", "ivscale", "vgemm"):
                    note(MAT_SHAPES, mat_mask, stmt[2])
                elif kind in ("vaxpy", "ivaxpy"):
                    note(MAT_SHAPES, mat_mask, stmt[3])
                elif kind == "i3scale":
                    note(R3_SHAPES, r3_mask, stmt[2])
                elif kind == "i3axpy":
                    note(R3_SHAPES, r3_mask, stmt[3])
                elif kind == "loop":
                    walk(stmt[2])
                elif kind == "cond":
                    walk(stmt[2])
                    walk(stmt[3])

        walk(_gen_block(rng, depth=2, max_stmts=6, rank_views=True))
        if any(len(group) > 1 for group in sliced.values()):
            reached += 1
    assert reached > 20, f"only {reached} of 250 programs slice two same-shaped deferred parents"
