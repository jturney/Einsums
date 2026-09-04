# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: the two alias derivations must schedule identically.

``Graph.link_alias_storage`` recovers which tensors share storage from
registration-time data pointers and strides. A graph read from a file has none,
so the same relation has to be derivable from ``View`` nodes and manifest
declarations alone - and that equivalence is to be
*tested over the differential fuzz corpus* rather than hoped for, because both
of the alias bugs this module has shipped (the full-cover one and the 32-hop
``resolve_alias`` cap) were subtly incomplete relations that surfaced as a race
or a wrong number rather than as an error.

What is asserted here is the SCHEDULE - ``schedule_edge_count`` and
``schedule_level_sizes`` - not any value. That is deliberate, and
``AliasOrderSharedScratch.cpp`` gives the argument at length: a missing hazard
edge is a race that reproduces probabilistically, so a value check would be a
flaky test that passes for the wrong reason, while the level structure the edge
belongs to is deterministic.

Three shapes of assertion. The first two are about the two derivations
agreeing; the third is about the edge set they agree on being the RIGHT one.

* **Equality** over programs whose aliases all come from captured ``cg.view``
  chains. Both derivations are defined there, so anything but equality is a
  defect in one of them.
* **Direction** over programs where structural is legitimately coarser (a
  manifest declaration names a buffer and carries no region). There the demand
  is ``structural >= pointer`` on edges, never ``<``. An extra edge costs
  parallelism; a missing edge is the race.
* **Minimality** of the edge set itself, through
  ``Graph.unjustified_hazard_edges``. Both derivations agreeing says nothing
  about whether either is right, and the direction above deliberately tolerates
  an extra edge, so nothing here could see an alias relation that is merely too
  COARSE. That is the direction the deferred-parent merge failed in: it cost
  every hazard between two unrelated tensors, moved no number, and was found by
  a materialization counter rather than by any oracle. The minimum is derived
  from byte overlap and from the graph's own ``View`` nodes, never from
  ``TensorHandle::aliases``, which is the relation under test.

The corpus is drawn with ``rich_views=True``, which adds a sub-block-of-a-
sub-block and a sub-block-of-a-transpose to the generator. Those are exactly the
two compositions the structural derivation had to learn (chaining and
permutation), and the plain corpus never draws either, so without them the
equality would hold vacuously over the shapes that matter most.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness


def _schedule(g):
    """The two deterministic numbers a change to the dependence machinery moves."""
    return g.schedule_edge_count(), tuple(g.schedule_level_sizes())


def _pointer_schedule(g):
    g.clear_alias_links()
    g.link_alias_storage()
    return _schedule(g)


def _structural_schedule(g):
    g.clear_alias_links()
    g.link_alias_structural()
    return _schedule(g)


def _program(seed, depth=0, max_stmts=8):
    rng = np.random.default_rng(seed)
    return rng, _gen_block(rng, depth=depth, max_stmts=max_stmts, rich_views=True)


def _graph_for(prog, rng, name):
    """Build the program into a graph, without executing it.

    Nothing here runs a kernel: the property under test is a property of the
    dependence graph, and building one is far cheaper than replaying it, which
    is what lets this shard cover a wide corpus.
    """
    g, _mats, _vecs, _r3 = _build(prog, *_seed_arrays(rng), name)
    return g


@pytest.mark.parametrize("seed", fuzz_seeds(120))
def test_alias_derivations_schedule_identically(seed):
    """Tier: exact schedule equality over ``cg.view``-only alias relations."""
    rng, prog = _program(seed)
    g = _graph_for(prog, rng, f"alias_flat{seed}")

    pointer = _pointer_schedule(g)
    structural = _structural_schedule(g)
    assert structural == pointer, (
        "the structural alias derivation schedules differently from the "
        f"pointer-derived one\nprogram={prog!r}\npointer={pointer}\nstructural={structural}"
    )


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_alias_derivations_schedule_identically_nested(seed):
    """The same, with loops and conditionals in the corpus.

    A ``Loop`` body and a ``Conditional`` branch are separate graphs with their
    own handles, and both derivations have to descend into them. A body left
    unlinked answers "this view aliases nothing" and its hazard edges vanish
    silently, which is invisible to a straight-line program.
    """
    rng, prog = _program(seed, depth=2, max_stmts=5)
    g = _graph_for(prog, rng, f"alias_nested{seed}")

    pointer = _pointer_schedule(g)
    structural = _structural_schedule(g)
    assert structural == pointer, (
        "the structural alias derivation schedules differently from the "
        f"pointer-derived one\nprogram={prog!r}\npointer={pointer}\nstructural={structural}"
    )


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_pointer_derivation_adds_nothing_to_a_structural_graph(seed):
    """The equivalence stated the other way round.

    ``link_alias_structural`` deliberately does not mark the pointer derivation
    as done, so running it afterwards actually re-runs the containment search
    over the relation structural just installed. If the two disagree anywhere,
    the second pass moves the schedule.
    """
    rng, prog = _program(seed)
    g = _graph_for(prog, rng, f"alias_bothways{seed}")

    structural = _structural_schedule(g)
    g.link_alias_storage()
    assert _schedule(g) == structural, (
        "running the pointer derivation on top of a structurally linked graph "
        f"changed the schedule\nprogram={prog!r}\nstructural={structural}\nafter={_schedule(g)}"
    )


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_structural_never_drops_an_edge_the_pointer_path_found(seed):
    """The directional guarantee, over the corpus the equality tier excludes.

    This runs the SAME programs but compares against the graph's normal state -
    capture's own links plus the containment search - rather than against a
    cleared re-derivation. Where the two differ at all, the demand is one-sided:
    structural may be coarser and cost edges, it may never be finer and drop
    one. That direction is the safety property; equality is the sharpness one.
    """
    rng, prog = _program(seed)
    g = _graph_for(prog, rng, f"alias_direction{seed}")

    baseline_edges, _ = _pointer_schedule(g)
    structural_edges, _ = _structural_schedule(g)
    assert structural_edges >= baseline_edges, (
        "the structural derivation emitted FEWER hazard edges than the pointer "
        "one, which means it proved a disjointness the addresses do not support "
        f"- a dropped edge is a data race\nprogram={prog!r}"
    )


@pytest.mark.parametrize("seed", fuzz_seeds(40))
def test_rich_view_programs_agree_with_the_numpy_oracle(seed):
    """The corpus has to be RIGHT before an equivalence over it means anything.

    The two chained-view primitives are new, so this replays them against the
    numpy oracle the way every other shard does. It is not a duplicate of the
    schedule tiers: it checks that a chained or permuted view writes where its
    box says it does, which is the assumption every disjointness proof in this
    file rests on. A box that is right about the schedule and wrong about the
    memory would pass every other test here.
    """
    rng = np.random.default_rng(seed)
    prog = _gen_block(rng, depth=0, max_stmts=6, rich_views=True)
    check_program(prog, *_seed_arrays(rng), f"alias_values{seed}")


def test_rich_view_corpus_actually_draws_chained_views():
    """The corpus guard.

    An equivalence over a corpus that never exercises the composition is
    vacuously true, and this shard exists precisely because chained and permuted
    views are what the structural derivation had to learn. So the generator is
    asserted to produce both, rather than trusted to.
    """
    seen = set()
    rng = np.random.default_rng(0)
    for _ in range(400):
        stmt = _gen_chained_view(rng)
        seen.add(stmt[0])
    assert seen == {"vvscale", "tvscale"}

    kinds = set()
    for seed in range(40):
        rng = np.random.default_rng(seed)
        for stmt in _gen_block(rng, depth=0, max_stmts=8, rich_views=True):
            kinds.add(stmt[0])
    assert "vvscale" in kinds
    assert "tvscale" in kinds


def test_plain_corpus_is_untouched_by_the_rich_view_option():
    """Existing shards' corpora are defined by their seeds, so the opt-in must
    not perturb the default roll. Same seed, same program."""
    for seed in range(20):
        plain = _gen_block(np.random.default_rng(seed), depth=1, max_stmts=6)
        again = _gen_block(np.random.default_rng(seed), depth=1, max_stmts=6, rich_views=False)
        assert plain == again
        for stmt in plain:
            assert stmt[0] not in ("vvscale", "tvscale")


# ──────────────────────────────────────────────────────────────────────────
# Minimality
#
# The tiers above compare two derivations of one relation. This one compares
# the relation against the memory, which is the only thing that can say the
# agreed answer is too coarse rather than merely consistent.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", fuzz_seeds(120))
def test_no_hazard_edge_orders_two_disjoint_accesses(seed):
    """Every hazard edge is justified by storage, a view chain or a parameter.

    A pair with no byte overlap, no ``View`` relating them and no shared
    parameter has no reason to be ordered, and an edge between them is a level
    scheduler's width spent on nothing. No value moves, so this is the only
    oracle that can see it.
    """
    rng, prog = _program(seed)
    g = _graph_for(prog, rng, f"alias_minimal{seed}")
    g.link_alias_storage()
    unjustified = g.unjustified_hazard_edges()
    assert not unjustified, (
        "hazard edges nothing about the storage justifies, which means the alias relation "
        f"merged tensors that share no memory\nprogram={prog!r}\n" + "\n".join(unjustified[:8])
    )


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_no_hazard_edge_orders_two_disjoint_accesses_nested(seed):
    """The same, with loops and conditionals: a control-flow node's effective
    I/O is the subtree's, so a merged root inside a body reaches the parent."""
    rng, prog = _program(seed, depth=2, max_stmts=5)
    g = _graph_for(prog, rng, f"alias_minimal_nested{seed}")
    g.link_alias_storage()
    unjustified = g.unjustified_hazard_edges()
    assert not unjustified, (
        "hazard edges nothing about the storage justifies\n"
        f"program={prog!r}\n" + "\n".join(unjustified[:8])
    )


def test_two_deferred_parents_are_not_ordered_against_each_other():
    """The shape the oracle exists for, pinned rather than left to the corpus.

    Two same-shaped deferred tensors sliced at the same offset. Their writers
    touch no common memory and no ``View`` relates them, so nothing may order
    them; while a view of a deferred parent registered the shell's sentinel plus
    its offset as an address, the pointer derivation merged the parents and
    every access to one was ordered against every access to the other.
    """
    graph = cg.Graph("two-deferred-parents")
    first = graph.declare_zero_tensor("first", [2, 2, 3, 3], intermediate=True, dtype="float64")
    second = graph.declare_zero_tensor("second", [2, 2, 3, 3], intermediate=True, dtype="float64")
    src = einsums.create_zero_tensor("src", [3, 3], dtype="float64")
    np.asarray(src)[...] = np.arange(9.0).reshape(3, 3)

    drop_at_one = [(2, 1, 0), (2, 1, 0), (0, 0, 0), (0, 0, 0)]
    with cg.capture(graph):
        einsums.linalg.axpby(1.0, src, 0.0, cg.view_indexed(first, drop_at_one))
        einsums.linalg.axpby(2.0, src, 0.0, cg.view_indexed(second, drop_at_one))

    graph.link_alias_storage()
    assert not graph.unjustified_hazard_edges()
    # And the schedule shows it: the two writes are independent, so they share a
    # level rather than chaining. Merging the parents put them on levels of one.
    assert max(graph.schedule_level_sizes()) >= 2
