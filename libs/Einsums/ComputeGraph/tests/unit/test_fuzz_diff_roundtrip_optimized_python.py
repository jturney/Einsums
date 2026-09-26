# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: an OPTIMIZED program survives a save/load round trip.

``test_fuzz_diff_roundtrip_python.py`` saves the graph exactly as captured. That
is the easy file to write: every node is one the user asked for, spelled the way
capture spelled it. The file a real workflow saves is the one after
``default_pass_manager()``, which is a different population of nodes: fused
einsums with rewritten index lists and folded prefactors, permutes a pass
introduced, gemm hints a pass computed, slot redirects, intermediates the
memory planner shares. None of those is reachable from the captured-graph shard,
and each is a field the writer can drop or the reader can rebuild wrongly
without either one raising.

So each trial runs three ways and demands they agree:

  1. the numpy oracle,
  2. the optimized graph executed in memory,
  3. the optimized graph written to a file, read back, bound by manifest name
     to a fresh copy of the operands, and executed.

(3) against (1) catches a round trip that computes the wrong answer. (3) against
(2) catches one that computes a DIFFERENT answer from the graph it was saved
from, which is the narrower claim a file makes and the one a tolerance against
numpy can hide when both are close to the truth.

A refused save is acceptable, because the library refuses exactly the nodes it
cannot rebuild and that list lives in one place. What is not acceptable is a
shard that passes because everything was refused, so the refusals are counted
and ``test_optimized_roundtrip_corpus_actually_round_trips`` asserts a real
fraction of the corpus was written, read and executed.

Views are drawn as well, from the harness's plain, chained and rank-dropping
generators. Today every ``View`` node is refused at save, so those programs
only contribute to the refusal tally; the guard below them asserts the refusal
names the view, so the day views become saveable this shard starts comparing
them with no edit.

Loops and conditionals are not drawn: the Python binding can only build them
with a callable predicate, which a file cannot hold, so every one would be a
refusal and would only dilute the corpus.

The shared harness lives in _fuzz_diff_common.py.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import einsums.graph as cg
from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import (
    _DTYPE_CAP,
    _DTYPE_TOL,
    _gen_block,
    _make_pool,
    _oracle,
    _seed_arrays,
    _square_seed_arrays,
    _usable,
    build_cg,
    fuzz_seeds,
    interp_np,
)

# Statement kinds whose captured nodes are reconstructible today; see the
# captured-graph shard for why the filter is a density choice and not a
# correctness one.
_SAVEABLE_KINDS = frozenset({"scale", "axpy", "axpby", "gemm", "einsum", "beinsum", "perm"})

# Statement kinds that capture a View node. Kept separate so a refusal of a
# program holding one can be told apart from a refusal nothing explains.
_VIEW_KINDS = frozenset(
    {"vgemm", "vscale", "vaxpy", "vvscale", "tvscale", "ivscale", "ivaxpy", "i3scale", "i3axpy"}
)

_STATS = {
    "attempted": 0,
    "round_tripped": 0,
    "refused": 0,
    "with_views_attempted": 0,
    "with_views_round_tripped": 0,
    # A program whose CAPTURED graph saves but whose optimized graph does not:
    # a pass introduced a node the writer cannot carry. Counted, not failed,
    # because it is a coverage gap rather than a wrong answer, but the report
    # names it.
    "refused_only_after_passes": 0,
    # Round-tripped trials whose optimized file differs from the captured one,
    # so the corpus guard can tell a shard that tests optimized files from one
    # whose passes happen to leave every program untouched.
    "rewritten": 0,
}
_REFUSED_AFTER_PASSES = []


def _pools_of(mats, vecs, r3s):
    return ([np.asarray(x).copy() for x in mats],
            [np.asarray(x).copy() for x in vecs],
            [np.asarray(x).copy() for x in r3s])


def _body(path):
    """The part of a file that describes the computation, without provenance.

    The provenance block records which structural passes ran, so it differs
    between a captured and an optimized save of the SAME graph; comparing it
    would count every trial as rewritten.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return {key: doc.get(key) for key in ("manifest", "tensors", "slot_redirects", "nodes")}


def _saves(graph, path):
    try:
        cg.save_graph(graph, path)
    except RuntimeError as exc:
        return str(exc)
    return None


def _run_optimized_roundtrip(prog, m_arrays, v_arrays, t_arrays, name):
    """Run the optimized graph in memory, then save, load, rebind and run it.

    Returns ``(in_memory, loaded, None)``, or ``(in_memory, None, refusal)``
    with the refusal message when the save refuses.
    """
    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3s, name)

    with tempfile.TemporaryDirectory() as scratch:
        captured_path = os.path.join(scratch, f"{name}.captured.eig.json")
        captured_refusal = _saves(g, captured_path)

        g.apply(cg.default_pass_manager())
        path = os.path.join(scratch, f"{name}.eig.json")
        # Save BEFORE executing, so the file holds the graph as the passes left
        # it and not whatever an executor cached on the nodes during a replay.
        refusal = _saves(g, path)
        loaded = None if refusal is not None else cg.load_graph(path)

        if captured_refusal is None and refusal is not None:
            _STATS["refused_only_after_passes"] += 1
            _REFUSED_AFTER_PASSES.append((prog, refusal))
        if captured_refusal is None and refusal is None and _body(captured_path) != _body(path):
            # The passes changed what the file says, so this trial exercised a
            # file capture could never have written.
            _STATS["rewritten"] += 1

    g.execute()
    in_memory = _pools_of(mats, vecs, r3s)
    if loaded is None:
        return in_memory, None, refusal

    # A second pool under the SAME names, seeded identically: the manifest
    # binds by name and `_make_pool` derives a tensor's name from the pool's.
    b_mats, b_vecs, b_r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    by_name = {}
    for prefix, pool in (("m", b_mats), ("v", b_vecs), ("t", b_r3s)):
        for idx, tensor in enumerate(pool):
            by_name[f"{name}_{prefix}{idx}"] = tensor

    for key in loaded.manifest_names():
        # A manifest slot the pool cannot supply is not a skip here: every
        # caller tensor of an optimized graph came from the pool, so a name the
        # pool does not know means the writer invented or renamed a slot, and
        # executing against the loader's placeholder would prove nothing.
        assert key in by_name, (
            f"the optimized graph's file names a manifest slot {key!r} no caller tensor has\n"
            f"program={prog!r}\nmanifest={loaded.manifest_names()!r}"
        )
        loaded.bind(key, by_name[key])

    loaded.execute()
    return in_memory, _pools_of(b_mats, b_vecs, b_r3s), None


def _has_kind(stmts, kinds):
    return any(s[0] in kinds for s in stmts)


def _compare(got, expected, prog, stage, dtype, rtol, atol):
    for kind, gs, es in zip("mvt", got, expected):
        for idx in range(len(es)):
            if not np.allclose(gs[idx], es[idx], rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{stage} on {kind}{idx} (dtype={dtype})\n"
                    f"program={prog!r}\ngot=\n{gs[idx]}\nexpected=\n{es[idx]}"
                )


def check_program_optimized_roundtrip(prog, m_arrays, v_arrays, t_arrays, label, dtype="float64"):
    """Oracle, optimized in memory, and optimized after a round trip all agree."""
    rtol, atol = _DTYPE_TOL[dtype]
    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, np.dtype(dtype))
    if not _usable(om, ov, ot, cap=_DTYPE_CAP[dtype]):
        pytest.skip("oracle overflowed, a numerically degenerate program")

    with_views = _has_kind(prog, _VIEW_KINDS)
    _STATS["attempted"] += 1
    _STATS["with_views_attempted"] += with_views

    in_memory, loaded, refusal = _run_optimized_roundtrip(prog, m_arrays, v_arrays, t_arrays, f"{label}_ort")

    # The in-memory arm stands on its own: the optimized graph must agree with
    # numpy whether or not it can be saved, so a refusal still checks it.
    _compare(in_memory, (om, ov, ot), prog, "OPTIMIZED (in memory) disagrees with oracle", dtype, rtol, atol)

    if loaded is None:
        _STATS["refused"] += 1
        pytest.skip("the optimized graph holds a node the IR cannot carry yet")

    _STATS["round_tripped"] += 1
    _STATS["with_views_round_tripped"] += with_views
    _compare(loaded, (om, ov, ot), prog, "OPTIMIZED + ROUND-TRIPPED disagrees with oracle", dtype, rtol, atol)
    # Against the graph it was saved from. The same tolerance, not bitwise: the
    # loaded graph re-plans its executor, and a different reduction order or
    # thread split is not a round-trip defect.
    _compare(loaded, in_memory, prog, "OPTIMIZED + ROUND-TRIPPED disagrees with the in-memory optimized graph",
             dtype, rtol, atol)


def _all_saveable(stmts):
    return all(s[0] in _SAVEABLE_KINDS for s in stmts)


def _draw_saveable(rng, max_stmts, tries=32):
    """Draw until the flat program holds only saveable kinds, or give up."""
    for _ in range(tries):
        prog = _gen_block(rng, depth=0, max_stmts=max_stmts)
        if prog and _all_saveable(prog):
            return prog
    return None


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(80))
def test_fuzz_optimized_roundtrip_flat(seed, dtype):
    rng = np.random.default_rng(20_000 + seed)
    prog = _draw_saveable(rng, max_stmts=8)
    if prog is None:
        pytest.skip("no saveable program drawn for this seed")
    check_program_optimized_roundtrip(prog, *_seed_arrays(rng, dtype), f"ortflat{seed}", dtype=dtype)


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_optimized_roundtrip_long_chains(seed):
    # Longer programs give the fusion and CSE passes something to fuse, which is
    # where the optimized file differs most from the captured one.
    rng = np.random.default_rng(30_000 + seed)
    prog = _draw_saveable(rng, max_stmts=16)
    if prog is None:
        pytest.skip("no saveable program drawn for this seed")
    check_program_optimized_roundtrip(prog, *_seed_arrays(rng), f"ortlong{seed}")


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_optimized_roundtrip_with_views(seed):
    # Unfiltered draws with every view generator on, so plain sub-blocks,
    # chained views, transposed views and rank-dropping views all appear.
    rng = np.random.default_rng(40_000 + seed)
    prog = _gen_block(rng, depth=0, max_stmts=6, rich_views=True, rank_views=True)
    check_program_optimized_roundtrip(prog, *_seed_arrays(rng), f"ortview{seed}")


def test_optimized_roundtrip_of_a_pinned_fusable_program():
    """One program that is saveable by construction and that the passes rewrite.

    Two einsums into one target and a scale of it give the fusion passes a
    chain to fold; the square pool keeps every statement shape valid. This case
    cannot skip, so a regression that made optimized graphs unsaveable fails
    here rather than only moving the corpus guard.
    """
    rng = np.random.default_rng(5151)
    m, v, t = _square_seed_arrays(rng)
    prog = [
        ("einsum", "ij <- ik ; kj", 1.0, 0, 1, 0.0, 2, False, False),
        ("einsum", "ij <- ik ; kj", 0.5, 1, 0, 1.0, 2, False, False),
        ("scale", 2.0, 2),
        ("axpby", 0.25, 2, 0.75, 3),
        ("perm", 1.5, 0.0, 3, 0),
    ]
    oracle = _oracle(prog, m, v, t)
    in_memory, loaded, refusal = _run_optimized_roundtrip(prog, m, v, t, "ortpinned")
    assert loaded is not None, f"an optimized einsum/scale/axpby/permute program must be saveable: {refusal}"
    _compare(in_memory, oracle, prog, "OPTIMIZED (in memory) disagrees with oracle", "float64", 1e-10, 1e-10)
    _compare(loaded, oracle, prog, "OPTIMIZED + ROUND-TRIPPED disagrees with oracle", "float64", 1e-10, 1e-10)


def test_view_refusal_names_the_view():
    """A view program is refused BECAUSE of its view, not for some other reason.

    The view arm above counts refusals without reading them. This pins the
    reason, so a refusal that starts naming something else (a pass that stopped
    folding a view into its consumer and left an unsaveable node behind, say)
    shows up as a change here rather than hiding in the tally.
    """
    rng = np.random.default_rng(6161)
    m, v, t = _square_seed_arrays(rng)
    mats, vecs, r3s = _make_pool(m, v, t, "ortvref")
    g = cg.Graph("ortvref")
    build_cg([("vscale", 0.5, 0, 0, 2, 0, 2)], g, mats, vecs, r3s, "ortvref")
    g.apply(cg.default_pass_manager())
    report = g.serializability_report()
    if not report:
        pytest.skip("views became saveable; the view arm now compares them")
    assert all(b.kind_name == "View" for b in report), [(b.label, b.kind_name, b.reason) for b in report]


def test_optimized_roundtrip_corpus_actually_round_trips():
    """The refusal rule must not swallow the corpus.

    Ordered last in the file so the parametrized cases above have run. Of the
    trials drawn from saveable kinds, most must survive the passes and the file:
    a pass that started leaving an unsaveable node behind on ordinary programs
    would otherwise turn every trial into a skip and this shard into a no-op.
    """
    attempted = _STATS["attempted"]
    tripped = _STATS["round_tripped"]
    assert attempted > 0, "no optimized round-trip trial ran at all"
    assert tripped > 0, f"every one of {attempted} trials was refused; nothing was round-tripped"
    rewritten = _STATS["rewritten"]
    assert rewritten >= 0.2 * tripped, (
        f"the passes changed only {rewritten} of {tripped} round-tripped files, so the shard is mostly "
        "re-testing captured graphs the unoptimized round-trip shard already covers"
    )
    no_view_attempted = attempted - _STATS["with_views_attempted"]
    no_view_tripped = tripped - _STATS["with_views_round_tripped"]
    assert no_view_tripped >= 0.8 * no_view_attempted, (
        f"only {no_view_tripped} of {no_view_attempted} view-free optimized programs round-tripped; "
        f"{_STATS['refused_only_after_passes']} were saveable as captured and refused only after the passes:\n"
        + "\n".join(f"{p!r}\n  -> {r.splitlines()[-1] if r else r}" for p, r in _REFUSED_AFTER_PASSES[:5])
    )
