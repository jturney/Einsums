# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: an OPTIMIZED program survives a save/load round trip.

``test_fuzz_diff_roundtrip_python.py`` saves the graph exactly as captured. That
is the easy file to write: every node is one the user asked for, spelled the way
capture spelled it. The file a real workflow saves is the one the structural
passes leave, which is the only form a file holds (see GraphIR.hpp): fused
einsums with rewritten index lists and folded prefactors, permutes a pass
introduced, gemm hints a pass derived again, slot redirects. None of those is
reachable from the captured-graph shard, and each is a field the writer can drop
or the reader can rebuild wrongly without either one raising.

So each trial follows the documented workflow and runs three ways that must agree:

  1. the numpy oracle,
  2. the graph after the structural passes, carried on in memory through the
     resource and tuning passes and executed,
  3. the same graph written to a file at that point, read back, bound by
     manifest name to a fresh copy of the operands, put through the same
     resource and tuning passes, and executed.

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

The passes rewrite only tensors the graph owns, because a caller reads whatever
it handed in, so a program over the caller's tensors alone gives them little to
change. The scratch arm draws programs over graph-owned intermediates, and the
corpus guard asserts that most of its files differ from their captured form.

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
from _pytest.outcomes import Skipped

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import (
    _DTYPE_CAP,
    _DTYPE_TOL,
    _build_with_scratch,
    _gen_block,
    _gen_scratch_motif,
    _make_pool,
    _oracle,
    _oracle_typed,
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
    # Round-tripped trials whose saved file differs from the captured one.
    # Counted for every arm but asserted only for the scratch arm: the passes
    # leave a write to a caller's tensor alone, so a program over the caller's
    # tensors alone has little for them to rewrite.
    "rewritten": 0,
    # The scratch arm, kept apart so the guard can ask it alone whether the
    # passes changed the files it saves.
    "scratch_attempted": 0,
    "scratch_round_tripped": 0,
    "scratch_rewritten": 0,
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


def _phase(populate):
    """A pass manager holding one phase of the default pipeline."""
    manager = cg.PassManager()
    getattr(manager, populate)()
    return manager


def _finish(graph):
    """The phases a loaded graph still needs, in the order the default pipeline runs them."""
    graph.apply(_phase("populate_resource"))
    graph.apply(_phase("populate_tuning"))


def _run_optimized_roundtrip(prog, m_arrays, v_arrays, t_arrays, name, with_scratch=False):
    """Optimize, then run the graph in memory and through a file, the documented way.

    The structural passes run first, and the file is written there, because that
    is the only form a file holds. Both the graph in memory and the one read back
    then go through the resource and tuning passes. With @p with_scratch, every
    scratch slot @p prog names is a graph-owned intermediate, and only the
    caller's tensors are returned.

    Returns ``(in_memory, loaded, None)``, or ``(in_memory, None, refusal)``
    with the refusal message when the save refuses.
    """
    if with_scratch:
        g, mats, vecs, r3s = _build_with_scratch(prog, m_arrays, v_arrays, t_arrays, name)
    else:
        mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
        g = cg.Graph(name)
        build_cg(prog, g, mats, vecs, r3s, name)

    with tempfile.TemporaryDirectory() as scratch:
        captured_path = os.path.join(scratch, f"{name}.captured.eig.json")
        captured_refusal = _saves(g, captured_path)

        g.apply(_phase("populate_structural"))
        path = os.path.join(scratch, f"{name}.eig.json")
        refusal = _saves(g, path)
        loaded = None if refusal is not None else cg.load_graph(path)

        if captured_refusal is None and refusal is not None:
            _STATS["refused_only_after_passes"] += 1
            _REFUSED_AFTER_PASSES.append((prog, refusal))
        if captured_refusal is None and refusal is None and _body(captured_path) != _body(path):
            # The passes changed what the file says, so this trial exercised a
            # file capture could never have written.
            _STATS["scratch_rewritten" if with_scratch else "rewritten"] += 1

    _finish(g)
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

    _finish(loaded)
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


def check_program_optimized_roundtrip(prog, m_arrays, v_arrays, t_arrays, label, dtype="float64", scratch=False):
    """Oracle, optimized in memory, and optimized after a round trip all agree.

    With @p scratch the program's scratch slots are graph-owned intermediates,
    and the trial is tallied apart from the caller-tensor arms.
    """
    rtol, atol = _DTYPE_TOL[dtype]
    if scratch:
        om, ov, ot = _oracle_typed(prog, m_arrays, v_arrays, t_arrays, dtype)
    else:
        om = [a.copy() for a in m_arrays]
        ov = [a.copy() for a in v_arrays]
        ot = [a.copy() for a in t_arrays]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            interp_np(prog, om, ov, ot, np.dtype(dtype))
        if not _usable(om, ov, ot, cap=_DTYPE_CAP[dtype]):
            pytest.skip("oracle overflowed, a numerically degenerate program")

    with_views = _has_kind(prog, _VIEW_KINDS)
    if scratch:
        _STATS["scratch_attempted"] += 1
    else:
        _STATS["attempted"] += 1
        _STATS["with_views_attempted"] += with_views

    in_memory, loaded, refusal = _run_optimized_roundtrip(prog, m_arrays, v_arrays, t_arrays, f"{label}_ort", scratch)

    # The in-memory arm stands on its own: the optimized graph must agree with
    # numpy whether or not it can be saved, so a refusal still checks it.
    _compare(in_memory, (om, ov, ot), prog, "OPTIMIZED (in memory) disagrees with oracle", dtype, rtol, atol)

    if loaded is None:
        _STATS["refused"] += 1
        pytest.skip("the optimized graph holds a node the IR cannot carry yet")

    if scratch:
        _STATS["scratch_round_tripped"] += 1
    else:
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


_FLAT_SEEDS = fuzz_seeds(80)
_LONG_SEEDS = fuzz_seeds(60)
_VIEW_SEEDS = fuzz_seeds(60)
_SCRATCH_SEEDS = fuzz_seeds(60)


def _flat_case(seed, dtype):
    rng = np.random.default_rng(20_000 + seed)
    prog = _draw_saveable(rng, max_stmts=8)
    if prog is None:
        pytest.skip("no saveable program drawn for this seed")
    check_program_optimized_roundtrip(prog, *_seed_arrays(rng, dtype), f"ortflat{seed}", dtype=dtype)


def _long_chain_case(seed):
    # Longer programs give the fusion and CSE passes something to fuse, which is
    # where the optimized file differs most from the captured one.
    rng = np.random.default_rng(30_000 + seed)
    prog = _draw_saveable(rng, max_stmts=16)
    if prog is None:
        pytest.skip("no saveable program drawn for this seed")
    check_program_optimized_roundtrip(prog, *_seed_arrays(rng), f"ortlong{seed}")


def _view_case(seed):
    # Unfiltered draws with every view generator on, so plain sub-blocks,
    # chained views, transposed views and rank-dropping views all appear.
    rng = np.random.default_rng(40_000 + seed)
    prog = _gen_block(rng, depth=0, max_stmts=6, rich_views=True, rank_views=True)
    check_program_optimized_roundtrip(prog, *_seed_arrays(rng), f"ortview{seed}")


def _scratch_case(seed, dtype):
    # A motif over graph-owned scratch that a structural pass is built to
    # rewrite (a transpose PermuteFusion folds, a contraction CSE deduplicates,
    # a scale ScaleAbsorption absorbs), after a short caller-tensor program.
    # These are the files the shard exists for.
    rng = np.random.default_rng(50_000 + seed)
    prog = (_draw_saveable(rng, max_stmts=4) or []) + _gen_scratch_motif(rng)
    check_program_optimized_roundtrip(prog, *_seed_arrays(rng, dtype), f"ortscr{seed}", dtype=dtype, scratch=True)


def _corpus():
    """Every case the parametrized tests below run, as thunks, for the guard to run again."""
    for seed in _FLAT_SEEDS:
        for dtype in ALL_DTYPES:
            yield lambda seed=seed, dtype=dtype: _flat_case(seed, dtype)
    for seed in _LONG_SEEDS:
        yield lambda seed=seed: _long_chain_case(seed)
    for seed in _VIEW_SEEDS:
        yield lambda seed=seed: _view_case(seed)
    for seed in _SCRATCH_SEEDS:
        for dtype in ALL_DTYPES:
            yield lambda seed=seed, dtype=dtype: _scratch_case(seed, dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", _FLAT_SEEDS)
def test_fuzz_optimized_roundtrip_flat(seed, dtype):
    _flat_case(seed, dtype)


@pytest.mark.parametrize("seed", _LONG_SEEDS)
def test_fuzz_optimized_roundtrip_long_chains(seed):
    _long_chain_case(seed)


@pytest.mark.parametrize("seed", _VIEW_SEEDS)
def test_fuzz_optimized_roundtrip_with_views(seed):
    _view_case(seed)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", _SCRATCH_SEEDS)
def test_fuzz_optimized_roundtrip_scratch(seed, dtype):
    _scratch_case(seed, dtype)


def test_optimized_roundtrip_of_a_pinned_fusable_program():
    """One program that is saveable by construction.

    Two einsums into one target and a scale of it, the chain the fusion passes
    fold when the target is the graph's own; the square pool keeps every
    statement shape valid. This case
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


def test_a_tuned_graph_saves_only_when_the_save_allows_it(tmp_path):
    """The Python spelling of the override: ``allow_machine_dependent``.

    Mirrors the SaveLoad case in C++. Reorder decides an order for one machine's memory, so a
    save refuses the graph it changed and names the pass, and the keyword writes it anyway; what
    loads back is a plain graph that no pass has changed.
    """
    n = 8
    A = einsums.create_random_tensor("A", [n, n])
    B = einsums.create_random_tensor("B", [n, n])
    R = einsums.create_zero_tensor("R", [n, n])
    S = einsums.create_zero_tensor("S", [n, n])

    g = cg.Graph("orttuned")
    X = g.scratch("X", [n, n])
    Y = g.scratch("Y", [n, n])
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", X, A, B)
        einsums.einsum("ij <- ik ; kj", Y, B, A)
        einsums.einsum("ij <- ik ; kj", R, X, A)
        einsums.einsum("ij <- ik ; kj", S, Y, B)

    pm = cg.PassManager()
    pm.add(cg.Reorder())
    assert g.apply(pm), "Reorder was expected to move the second product ahead"
    assert g.machine_dependent_passes() == ["Reorder"]

    path = str(tmp_path / "orttuned.eig.json")
    with pytest.raises(RuntimeError, match=r"machine-dependent passes \(Reorder\)"):
        cg.save_graph(g, path)
    assert not os.path.exists(path)

    cg.save_graph(g, path, allow_machine_dependent=True)
    assert cg.load_graph(path).machine_dependent_passes() == []


def test_optimized_roundtrip_corpus_actually_round_trips():
    """The refusal rule must not swallow the corpus.

    Of the trials drawn from saveable kinds, most must survive the passes and the
    file: a pass that started leaving an unsaveable node behind on ordinary
    programs would otherwise turn every trial into a skip and this shard into a
    no-op. Runs the corpus itself rather than reading what the cases above
    counted: under pytest-xdist those ran in other processes, and under ``-k``
    they may not have run at all.
    """
    before = dict(_STATS)
    refused_before = len(_REFUSED_AFTER_PASSES)
    for case in _corpus():
        try:
            case()
        except Skipped:
            pass
    stats = {key: _STATS[key] - before[key] for key in _STATS}
    refused_after_passes = _REFUSED_AFTER_PASSES[refused_before:]

    attempted = stats["attempted"]
    tripped = stats["round_tripped"]
    assert attempted > 0, "no optimized round-trip trial ran at all"
    assert tripped > 0, f"every one of {attempted} trials was refused; nothing was round-tripped"
    scratch_tripped = stats["scratch_round_tripped"]
    assert scratch_tripped >= 0.6 * stats["scratch_attempted"], (
        f"only {scratch_tripped} of {stats['scratch_attempted']} scratch programs round-tripped"
    )
    assert stats["scratch_rewritten"] >= 0.6 * scratch_tripped, (
        f"the passes changed only {stats['scratch_rewritten']} of {scratch_tripped} round-tripped scratch files, so "
        "the shard is mostly re-testing captured graphs the unoptimized round-trip shard already covers"
    )
    no_view_attempted = attempted - stats["with_views_attempted"]
    no_view_tripped = tripped - stats["with_views_round_tripped"]
    assert no_view_tripped >= 0.8 * no_view_attempted, (
        f"only {no_view_tripped} of {no_view_attempted} view-free optimized programs round-tripped; "
        f"{stats['refused_only_after_passes']} were saveable as captured and refused only after the passes:\n"
        + "\n".join(f"{p!r}\n  -> {r.splitlines()[-1] if r else r}" for p, r in refused_after_passes[:5])
    )
