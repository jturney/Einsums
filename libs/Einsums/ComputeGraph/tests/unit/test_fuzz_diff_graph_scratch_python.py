# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Parent-declared graph scratch consumed inside / after control flow, and
eager graph-owned intermediates written only through views.

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


# ──────────────────────────────────────────────────────────────────────────
# Eager graph-owned intermediates written ONLY through views.
#
# ConstantFolding executes a node at pass time when every input is a
# MATERIALIZED graph-owned intermediate that no node writes. The arms above
# never reach that question: their scratch is a deferred ``declare_*`` shell,
# which the pass refuses to touch, and it is always written whole before any
# view of it is. The bug that shipped lived exactly in the gap: an eager
# intermediate T written only by an axpy into a slice of T looked unwritten,
# so a later reader of T was folded with T's pre-write contents.
#
# This arm creates 1-3 intermediates with ``graph.create_zero_tensor`` (eager,
# intermediate=True), optionally seeds them with data so even a scale through
# a view changes the value, and writes most of them only through slices or
# full-cover views with value-carrying ops (axpy, axpby, gemm into the view).
# The rest stay untouched, so the pass has something it SHOULD fold and a
# broken "is it written?" answer cannot hide behind the pass never firing.
# Readers come after the writes and often take every input from the eager
# intermediates, which is the shape that makes the reader itself foldable.
#
# View-write opcodes local to this arm (box = (r0, r1, c0, c1)):
#
#   ("gvaxpy",  a, src, sbox, dst, dbox)     m[dst][dbox] += a*m[src][sbox]
#   ("gvaxpby", a, src, sbox, b, dst, dbox)  m[dst][dbox] = a*m[src][sbox] + b*m[dst][dbox]
#   ("gvgemm",  a, A, ar0, B, bc0, b, dst, dbox)
#         m[dst][dbox] = a*(m[A][ar0:ar0+h, :] @ m[B][:, bc0:bc0+w]) + b*m[dst][dbox]
#
# A full source box is emitted as the plain tensor rather than a view of it,
# so the view-to-view and tensor-to-view overloads are both exercised.
# ──────────────────────────────────────────────────────────────────────────

_EAGER_VIEW_WRITES = ("gvaxpy", "gvaxpby", "gvgemm")


def _draw_box(rng, n):
    """A random sub-block, or the full cover a third of the time: a view whose
    box equals its parent's is its own alias case and a random offset rarely
    lands on it."""
    if rng.random() < 1 / 3:
        return (0, n, 0, n)
    h = int(rng.integers(1, n + 1))
    w = int(rng.integers(1, n + 1))
    r0 = int(rng.integers(0, n - h + 1))
    c0 = int(rng.integers(0, n - w + 1))
    return (r0, r0 + h, c0, c0 + w)


def _same_shaped_box(rng, box, n):
    """A box of the same extent as ``box`` at a random offset, for the source."""
    h, w = box[1] - box[0], box[3] - box[2]
    r0 = int(rng.integers(0, n - h + 1))
    c0 = int(rng.integers(0, n - w + 1))
    return (r0, r0 + h, c0, c0 + w)


def _view_write_eager(rng, dst, srcs, n):
    """One write into ``dst`` that goes only through a view of it. ``srcs``
    excludes ``dst``, so source and destination never share a buffer."""
    a = _scalar(rng)
    dbox = _draw_box(rng, n)
    roll = int(rng.integers(0, 4))
    if roll == 0:
        return ("gvaxpy", a, int(rng.choice(srcs)), _same_shaped_box(rng, dbox, n), dst, dbox)
    if roll == 1:
        return ("gvaxpby", a, int(rng.choice(srcs)), _same_shaped_box(rng, dbox, n), _scalar(rng), dst, dbox)
    if roll == 2:
        h, w = dbox[1] - dbox[0], dbox[3] - dbox[2]
        return ("gvgemm", a, int(rng.choice(srcs)), int(rng.integers(0, n - h + 1)),
                int(rng.choice(srcs)), int(rng.integers(0, n - w + 1)), float(rng.integers(0, 2)), dst, dbox)
    # A scale through a view only changes a seeded intermediate, but it is the
    # write every other arm already draws, so keep it in the mix.
    return ("vscale", a, dst, *dbox)


def _read_eager(rng, eager, ords, all_owned):
    """A whole-tensor read of the eager intermediates into a caller-owned
    result. With ``all_owned`` every input is an eager intermediate and the
    result's prior value is discarded, which leaves ConstantFolding free to
    evaluate the node at pass time if it believes its inputs never change."""
    a = _scalar(rng)
    e1, e2 = int(rng.choice(eager)), int(rng.choice(eager))
    o = int(rng.choice(ords))
    if all_owned:
        roll = int(rng.integers(0, 4))
        if roll == 0:
            return ("gemm", a, e1, e2, 0.0, o)
        if roll == 1:
            return ("einsum", _SQ, a, e1, e2, 0.0, o, False, False)
        if roll == 2:
            return ("perm", a, 0.0, e1, o)
        return ("axpby", a, e1, 0.0, o)
    roll = int(rng.integers(0, 3))
    if roll == 0:
        return ("axpy", a, e1, o)
    o2 = _distinct(rng, ords, {o})
    if roll == 1:
        return ("gemm", a, e1, o, float(rng.integers(0, 2)), o2)
    return ("gemm", a, o, e1, float(rng.integers(0, 2)), o2)


def _gen_eager_view_program(rng, ords, eager, n):
    """Arm E: view-only writes of eager intermediates, then whole reads."""
    stmts = [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
    # At least one intermediate is always view-written; an untouched one is
    # drawn alongside it often enough that the pass also has a real fold.
    written = [e for e in eager if rng.random() < 0.75] or [eager[0]]
    for dst in written:
        srcs = ords + [e for e in eager if e != dst]
        writes = [_view_write_eager(rng, dst, srcs, n) for _ in range(int(rng.integers(1, 4)))]
        if rng.random() < 0.25:
            # A writer inside a loop body must still count against the parent's
            # buffer; a per-graph writer scan would miss it the same way.
            stmts.append(("loop", int(rng.integers(2, 4)), writes))
        else:
            stmts += writes
        stmts += [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
    for _ in range(int(rng.integers(1, 4))):
        stmts.append(_read_eager(rng, eager, ords, all_owned=rng.random() < 0.6))
    stmts += [_gen_square_primitive(rng, ords) for _ in range(int(rng.integers(0, 2)))]
    return stmts


def _box_slice(box):
    return (slice(box[0], box[1]), slice(box[2], box[3]))


def _interp_eager(stmts, m):
    """numpy oracle: the local view-write opcodes here, everything else in
    ``interp_np``. Control flow recurses here so a loop body may hold both."""
    for s in stmts:
        k = s[0]
        if k == "gvaxpy":
            _, a, src, sbox, dst, dbox = s
            m[dst][_box_slice(dbox)] += a * m[src][_box_slice(sbox)]
        elif k == "gvaxpby":
            _, a, src, sbox, b, dst, dbox = s
            m[dst][_box_slice(dbox)] = a * m[src][_box_slice(sbox)] + b * m[dst][_box_slice(dbox)]
        elif k == "gvgemm":
            _, a, A, ar0, B, bc0, b, dst, dbox = s
            h, w = dbox[1] - dbox[0], dbox[3] - dbox[2]
            prod = m[A][ar0:ar0 + h, :] @ m[B][:, bc0:bc0 + w]
            m[dst][_box_slice(dbox)] = a * prod + b * m[dst][_box_slice(dbox)]
        elif k == "loop":
            for _ in range(s[1]):
                _interp_eager(s[2], m)
        else:
            interp_np([s], m, [], [], np.dtype("float64"))


def _src_operand(t, box, n):
    return t if box == (0, n, 0, n) else cg.view(t, [(box[0], box[1]), (box[2], box[3])])


def _emit_eager(s, m, n):
    k = s[0]
    if k == "gvaxpy":
        _, a, src, sbox, dst, dbox = s
        einsums.linalg.axpy(a, _src_operand(m[src], sbox, n), cg.view(m[dst], [dbox[:2], dbox[2:]]))
    elif k == "gvaxpby":
        _, a, src, sbox, b, dst, dbox = s
        einsums.linalg.axpby(a, _src_operand(m[src], sbox, n), b, cg.view(m[dst], [dbox[:2], dbox[2:]]))
    elif k == "gvgemm":
        _, a, A, ar0, B, bc0, b, dst, dbox = s
        h, w = dbox[1] - dbox[0], dbox[3] - dbox[2]
        einsums.linalg.gemm(a, _src_operand(m[A], (ar0, ar0 + h, 0, n), n),
                            _src_operand(m[B], (0, n, bc0, bc0 + w), n), b, cg.view(m[dst], [dbox[:2], dbox[2:]]))
    else:
        _emit_primitive(s, m, [], [])


def _build_eager(stmts, graph, m, n, tag):
    """``build_cg`` for this arm: straight runs are captured into ``graph``,
    loops get their own body graph."""
    run = []

    def flush():
        if run:
            with cg.capture(graph):
                for s in run:
                    _emit_eager(s, m, n)
            run.clear()

    for i, s in enumerate(stmts):
        if s[0] != "loop":
            run.append(s)
            continue
        flush()
        cnt = s[1]
        body = graph.add_loop(f"{tag}_loop{i}", cnt, lambda it, c=cnt: it < c - 1)
        _build_eager(s[2], body, m, n, f"{tag}_l{i}")
    flush()


def _check_eager_view_program(prog, m_arrays, e_arrays, n, label):
    B = len(m_arrays)
    om = [a.copy() for a in m_arrays] + [a.copy() for a in e_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _interp_eager(prog, om)
    if not _usable(om[:B], cap=_DTYPE_CAP["float64"]):
        pytest.skip("oracle overflowed: numerically degenerate program")

    for ex_name, exec_cls in _CROSS_EXECUTORS:
        tag = f"{label}_{ex_name}"
        g = cg.Graph(tag)
        mats = []
        for idx, arr in enumerate(m_arrays):
            tn = einsums.create_zero_tensor(f"{tag}_m{idx}", [n, n], dtype="float64")
            np.asarray(tn)[...] = arr
            mats.append(tn)
        for idx, arr in enumerate(e_arrays):
            # Graph-owned AND materialized: the only kind of tensor
            # ConstantFolding will evaluate a node over.
            tn = g.create_zero_tensor(f"{tag}_e{idx}", [n, n], intermediate=True, dtype="float64")
            np.asarray(tn)[...] = arr
            mats.append(tn)
        _build_eager(prog, g, mats, n, tag)
        g.apply(cg.default_pass_manager())
        assert_materialization_invariants(g, f"{label}/{ex_name}")
        g.execute() if ex_name == "Sequential" else g.execute(exec_cls())
        for idx in range(B):
            got = np.asarray(mats[idx])
            if not np.allclose(got, om[idx], rtol=RTOL, atol=ATOL):
                raise AssertionError(
                    f"EAGER-VIEW {ex_name} disagrees on m{idx}\n"
                    f"program={prog!r}\ngot=\n{got}\noracle=\n{om[idx]}"
                )


def _eager_seed(rng, n):
    m_arrays = [rng.standard_normal((n, n)) * 0.5 for _ in range(_SCRATCH_ORDS)]
    n_eager = int(rng.integers(1, 4))
    # Half the intermediates start at zero, as a fresh create_zero_tensor does;
    # the rest carry data so a scale through a view is a visible write too.
    e_arrays = [rng.standard_normal((n, n)) * 0.5 if rng.random() < 0.5 else np.zeros((n, n))
                for _ in range(n_eager)]
    ords = list(range(_SCRATCH_ORDS))
    eager = list(range(_SCRATCH_ORDS, _SCRATCH_ORDS + n_eager))
    return m_arrays, e_arrays, ords, eager


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_eager_intermediate_written_through_views(seed):
    """Arm E: eager graph-owned intermediates written only through views, then
    read whole (default pipeline, Sequential + parallel executors)."""
    rng = np.random.default_rng(190_000 + seed)
    n = _SCRATCH_N
    m_arrays, e_arrays, ords, eager = _eager_seed(rng, n)
    prog = _gen_eager_view_program(rng, ords, eager, n)
    _check_eager_view_program(prog, m_arrays, e_arrays, n, f"eview{seed}")


def _all_owned_reader_inputs(s, eager):
    """The inputs of ``s`` if it is a reader ConstantFolding could evaluate:
    every operand an eager intermediate and the result's prior value unread.
    Empty otherwise."""
    k = s[0]
    if k == "gemm" and s[4] == 0.0:
        ins = {s[2], s[3]}
    elif k == "einsum" and s[5] == 0.0:
        ins = {s[3], s[4]}
    elif k == "perm" and s[2] == 0.0:
        ins = {s[3]}
    elif k == "axpby" and s[3] == 0.0:
        ins = {s[2]}
    else:
        return set()
    return ins if ins <= set(eager) else set()


def test_the_eager_view_corpus_reaches_the_fold_question():
    """The corpus guard for arm E: the shape behind the view-only-write fold bug
    is an intermediate whose every write goes through a view with a value
    carrying op, followed by a reader whose every input is graph-owned. A
    generator change that stops producing it would leave the arm passing while
    testing nothing."""
    kinds = set()
    full_box = False
    hits = 0
    for seed in range(40):
        rng = np.random.default_rng(190_000 + seed)
        _, _, ords, eager = _eager_seed(rng, _SCRATCH_N)
        prog = _gen_eager_view_program(rng, ords, eager, _SCRATCH_N)
        flat = []
        for s in prog:
            kinds.add(s[0])
            flat += s[2] if s[0] == "loop" else [s]
        for s in flat:
            kinds.add(s[0])
            if s[0] in _EAGER_VIEW_WRITES and s[-1] == (0, _SCRATCH_N, 0, _SCRATCH_N):
                full_box = True
        view_written = {s[-2] for s in flat if s[0] in _EAGER_VIEW_WRITES}
        for s in flat:
            owned = _all_owned_reader_inputs(s, eager)
            if owned and owned & view_written:
                hits += 1
    for kind in _EAGER_VIEW_WRITES + ("loop",):
        assert kind in kinds, f"{kind} never drawn: {sorted(kinds)}"
    assert full_box, "no full-cover view write drawn"
    assert hits >= 10, f"only {hits} all-owned readers of a view-written intermediate drawn"


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
