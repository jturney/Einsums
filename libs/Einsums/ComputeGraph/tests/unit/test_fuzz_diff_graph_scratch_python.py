# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Parent-declared graph scratch consumed inside / after control flow.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

import einsums
import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness
from _region_invariants import assert_materialization_invariants


# ──────────────────────────────────────────────────────────────────────────
# Parent-declared graph scratch consumed inside / after control flow.
#
# The generator's whole tensor pool is user-visible (created eager, never
# freed), so a random program NEVER declares an intermediate=True graph scratch
# that is (re)computed and consumed only inside a generated loop / conditional
# body, or defined in both branches of a conditional and consumed AFTER it.
# That exact shape is behind three recent bugs (FreeInsertion Part A
# last_use/first_writer subtree fixes, hoisted lifecycle nodes): the default
# pipeline must hoist the scratch's Materialize above the control-flow node,
# zero it once, and free it after the true last read. These two arms declare
# 1-2 square scratch tensors in the PARENT graph, drive them through generated
# control flow, and demand the default-optimized graph (Sequential + both
# parallel executors) matches a numpy oracle that models the scratch as a
# plain zero-initialized array.
#
# Because a program that never materializes its scratch would pass trivially
# (that is how the FreeInsertion bug hid), each build ASSERTS the optimized
# graph carries a Materialize node for every declared scratch before executing.
# Only the ordinary (user-visible) tensors are compared: FreeInsertion may free
# the scratch, leaving its post-execute buffer undefined.
# ──────────────────────────────────────────────────────────────────────────

_SCRATCH_N = 3        # square dimension for the graph-scratch pool
_SCRATCH_ORDS = 4     # number of ordinary (user-visible) square matrices


def _distinct(rng, ords, exclude):
    """A pool index not in ``exclude`` (contraction/transpose outputs may not
    alias an operand). ords is always large enough for a fresh pick here."""
    cands = [o for o in ords if o not in exclude]
    return int(rng.choice(cands)) if cands else None


def _gen_square_primitive(rng, ords):
    """A random primitive over a square n x n pool (ordinary tensor indices only).
    Kept self-contained because _gen_primitive indexes the global MAT_SHAPES pool.
    Contraction/transpose outputs are kept distinct from their inputs (in-place
    einsum/gemm/perm on a contraction is unsupported)."""
    roll = int(rng.integers(0, 6))
    a = _scalar(rng)
    x, y = int(rng.choice(ords)), int(rng.choice(ords))
    if roll == 0:
        return ("scale", a, x)
    if roll == 1:
        return ("axpy", a, x, y)
    if roll == 2:
        return ("axpby", a, x, _scalar(rng), y)
    if roll == 3:
        z = _distinct(rng, ords, {x, y})
        return ("gemm", a, x, y, float(rng.integers(0, 2)), z) if z is not None else ("scale", a, x)
    if roll == 4:
        z = _distinct(rng, ords, {x, y})
        return ("einsum", _SQ, a, x, y, float(rng.integers(0, 2)), z, False, False) if z is not None else ("scale", a, x)
    y = _distinct(rng, ords, {x})
    return ("perm", a, float(rng.integers(0, 2)), x, y) if y is not None else ("scale", a, x)


def _write_scratch(rng, dst, ords):
    """An op that OVERWRITES dst (a scratch index) from ordinary tensors. dst is a
    scratch index, disjoint from every ordinary operand, so no aliasing arises."""
    o1, o2 = int(rng.choice(ords)), int(rng.choice(ords))
    roll = int(rng.integers(0, 4))
    if roll == 0:
        return ("gemm", _scalar(rng), o1, o2, 0.0, dst)
    if roll == 1:
        return ("einsum", _SQ, _scalar(rng), o1, o2, 0.0, dst, False, False)
    if roll == 2:
        return ("perm", _scalar(rng), 0.0, o1, dst)
    return ("axpby", _scalar(rng), o1, 0.0, dst)


def _view_write_scratch(rng, dst, n):
    """Touch part of the scratch THROUGH a view.

    The shard declared same-shaped graph scratch from the start and never took a
    view of one, which is precisely where the alias merge that shipped lived: a
    view of a deferred parent that registered an address, so two same-shaped
    parents sliced alike presented one byte span and the pointer derivation
    merged them. Four spellings, because the derivations differ on them: a plain
    range, a range of a range, a range of a transpose, and an axis DROPPED
    rather than sliced.
    """
    a = _scalar(rng)
    roll = int(rng.integers(0, 4))
    if roll == 0:
        r0 = int(rng.integers(0, n))
        c0 = int(rng.integers(0, n))
        return ("vscale", a, dst, r0, n, c0, n)
    if roll == 1:
        r0 = int(rng.integers(0, n))
        return ("vvscale", a, dst, r0, n, 0, n, 0, n - r0, 0, n)
    if roll == 2:
        c0 = int(rng.integers(0, n))
        return ("tvscale", a, dst, 0, n, c0, n)
    row = int(rng.integers(0, n))
    c0 = int(rng.integers(0, n))
    return ("ivscale", a, dst, row, c0, n)


def _view_consume_scratch(rng, src, ords, n):
    """Read the scratch through a full-cover view.

    A view whose box equals its parent's is the case that once got no alias edge
    at all, so it is worth drawing on purpose rather than hoping a random offset
    lands on it.
    """
    o1 = int(rng.choice(ords))
    o2 = _distinct(rng, ords, {o1})
    if o2 is None:
        return ("axpy", _scalar(rng), src, o1)
    return ("vgemm", _scalar(rng), src, 0, n, 0, n, o1, float(rng.integers(0, 2)), o2)


def _consume_scratch(rng, src, ords):
    """An op that READS src (a scratch index) into an ordinary tensor. The output
    is kept distinct from the ordinary input for the contraction case."""
    roll = int(rng.integers(0, 3))
    if roll == 0:
        return ("axpy", _scalar(rng), src, int(rng.choice(ords)))
    if roll == 1:
        o1 = int(rng.choice(ords))
        o2 = _distinct(rng, ords, {o1})
        if o2 is None:
            return ("axpy", _scalar(rng), src, o1)
        return ("gemm", _scalar(rng), src, o1, float(rng.integers(0, 2)), o2)
    return ("perm", _scalar(rng), float(rng.integers(0, 2)), src, int(rng.choice(ords)))


def _gen_scratch_body_program(rng, ords, scratch):
    """Arm A: each scratch tensor is (re)computed and consumed strictly INSIDE a
    generated loop or conditional body; nothing outside the body references it."""
    stmts = []
    for si in scratch:
        body = [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
        body.append(_write_scratch(rng, si, ords))
        loop = rng.random() < 0.5
        if loop and rng.random() < 0.5:
            # loop-carried accumulation: scratch += ordinary each iteration
            body.append(("axpby", _scalar(rng), int(rng.choice(ords)), 1.0, si))
        if rng.random() < 0.5:
            body.append(_view_write_scratch(rng, si, _SCRATCH_N))
        body.append(_view_consume_scratch(rng, si, ords, _SCRATCH_N)
                    if rng.random() < 0.5 else _consume_scratch(rng, si, ords))
        body += [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
        if loop:
            stmts.append(("loop", int(rng.integers(2, 4)), body))
        else:
            # both branches recompute+consume so the scratch is defined whichever runs
            other = [_write_scratch(rng, si, ords), _consume_scratch(rng, si, ords)]
            stmts.append(("cond", bool(rng.integers(0, 2)), body, other))
    return stmts


def _gen_scratch_after_cond_program(rng, ords, scratch):
    """Arm C: BOTH branches of a conditional define each scratch (overwrite), and
    the scratch is consumed AFTER the conditional, so its value is well-defined
    whichever branch runs. Stresses materialize hoisting above the branch and the
    free of a tensor whose last read follows the conditional (blind spot #3)."""
    then, els = [], []
    for si in scratch:
        then.append(_write_scratch(rng, si, ords))
        els.append(_write_scratch(rng, si, ords))
    then += [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
    els += [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
    stmts = [("cond", bool(rng.integers(0, 2)), then, els)]
    for si in scratch:
        if rng.random() < 0.5:
            stmts.append(_view_write_scratch(rng, si, _SCRATCH_N))
        stmts.append(_view_consume_scratch(rng, si, ords, _SCRATCH_N)
                     if rng.random() < 0.5 else _consume_scratch(rng, si, ords))
    stmts += [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
    return stmts


def _make_pool_graph_scratch(m_arrays, graph, name, n_scratch, n):
    """Ordinary matrices (eager, user-visible) followed by n_scratch graph-declared
    intermediate=True scratch tensors, appended at indices [len(m_arrays), ...)."""
    mats = []
    for idx, arr in enumerate(m_arrays):
        tn = einsums.create_zero_tensor(f"{name}_m{idx}", list(arr.shape), dtype=str(arr.dtype))
        np.asarray(tn)[...] = arr
        mats.append(tn)
    scratch_names = []
    for k in range(n_scratch):
        sn = f"{name}_sc{k}"
        mats.append(graph.declare_zero_tensor(sn, [n, n], intermediate=True, dtype=str(m_arrays[0].dtype)))
        scratch_names.append(sn)
    return mats, scratch_names


def _assert_scratch_materialized(graph, scratch_names, label):
    labels = " ".join(
        nd.get("label", "") for nd in json.loads(graph.to_json()).get("nodes", []) if nd.get("kind") == "Materialize"
    )
    for sn in scratch_names:
        assert sn in labels, f"{label}: scratch {sn} was not materialized (Materialize labels: {labels!r})"


def _check_scratch_program(prog, m_arrays, n_scratch, n, label):
    B = len(m_arrays)
    om = [a.copy() for a in m_arrays] + [np.zeros((n, n)) for _ in range(n_scratch)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, [], [], np.dtype("float64"))
    if not _usable(om[:B], cap=_DTYPE_CAP["float64"]):
        pytest.skip("oracle overflowed — numerically degenerate program")
    oracle_ord = om[:B]

    for ex_name, exec_cls in _CROSS_EXECUTORS:
        g = cg.Graph(f"{label}_{ex_name}")
        mats, scratch_names = _make_pool_graph_scratch(m_arrays, g, f"{label}_{ex_name}", n_scratch, n)
        build_cg(prog, g, mats, [], [], f"{label}_{ex_name}")
        g.apply(cg.default_pass_manager())
        _assert_scratch_materialized(g, scratch_names, f"{label}/{ex_name}")
        assert_materialization_invariants(g, f"{label}/{ex_name}")
        g.execute() if ex_name == "Sequential" else g.execute(exec_cls())
        for idx in range(B):
            got = np.asarray(mats[idx])
            if not np.allclose(got, oracle_ord[idx], rtol=RTOL, atol=ATOL):
                raise AssertionError(
                    f"GRAPH-SCRATCH {ex_name} disagrees on m{idx}\n"
                    f"program={prog!r}\ngot=\n{got}\noracle=\n{oracle_ord[idx]}"
                )


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_graph_scratch_in_control_flow(seed):
    """Arm A: parent-declared scratch produced and consumed inside a loop /
    conditional body (default pipeline, Sequential + parallel executors)."""
    rng = np.random.default_rng(170_000 + seed)
    n = _SCRATCH_N
    m_arrays = [rng.standard_normal((n, n)) * 0.5 for _ in range(_SCRATCH_ORDS)]
    n_scratch = int(rng.integers(1, 3))
    ords = list(range(_SCRATCH_ORDS))
    scratch = list(range(_SCRATCH_ORDS, _SCRATCH_ORDS + n_scratch))
    prog = _gen_scratch_body_program(rng, ords, scratch)
    _check_scratch_program(prog, m_arrays, n_scratch, n, f"scr{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_graph_scratch_after_conditional(seed):
    """Arm C: parent-declared scratch defined in both branches of a conditional
    and consumed after it (default pipeline, Sequential + parallel executors)."""
    rng = np.random.default_rng(180_000 + seed)
    n = _SCRATCH_N
    m_arrays = [rng.standard_normal((n, n)) * 0.5 for _ in range(_SCRATCH_ORDS)]
    n_scratch = int(rng.integers(1, 3))
    ords = list(range(_SCRATCH_ORDS))
    scratch = list(range(_SCRATCH_ORDS, _SCRATCH_ORDS + n_scratch))
    prog = _gen_scratch_after_cond_program(rng, ords, scratch)
    _check_scratch_program(prog, m_arrays, n_scratch, n, f"scrc{seed}")


def test_the_scratch_corpus_takes_views_of_the_scratch():
    """The corpus guard: a shard that declares same-shaped scratch and never
    slices it cannot reach the alias question it is closest to."""
    kinds = set()

    def walk(stmts):
        for stmt in stmts:
            kinds.add(stmt[0])
            if stmt[0] == "loop":
                walk(stmt[2])
            elif stmt[0] == "cond":
                walk(stmt[2])
                walk(stmt[3])

    ords = list(range(_SCRATCH_ORDS))
    scratch = list(range(_SCRATCH_ORDS, _SCRATCH_ORDS + 2))
    for seed in range(40):
        rng = np.random.default_rng(170_000 + seed)
        walk(_gen_scratch_body_program(rng, ords, scratch))
        rng = np.random.default_rng(180_000 + seed)
        walk(_gen_scratch_after_cond_program(rng, ords, scratch))
    for kind in ("vscale", "vvscale", "tvscale", "ivscale", "vgemm"):
        assert kind in kinds, f"{kind} never drawn: {sorted(kinds)}"
