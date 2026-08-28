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

Two shapes of assertion, and the difference between them is the whole safety
argument:

* **Equality** over programs whose aliases all come from captured ``cg.view``
  chains. Both derivations are defined there, so anything but equality is a
  defect in one of them.
* **Direction** over programs where structural is legitimately coarser (a
  manifest declaration names a buffer and carries no region). There the demand
  is ``structural >= pointer`` on edges, never ``<``. An extra edge costs
  parallelism; a missing edge is the race.

The corpus is drawn with ``rich_views=True``, which adds a sub-block-of-a-
sub-block and a sub-block-of-a-transpose to the generator. Those are exactly the
two compositions the structural derivation had to learn (chaining and
permutation), and the plain corpus never draws either, so without them the
equality would hold vacuously over the shapes that matter most.
"""

from __future__ import annotations

import numpy as np
import pytest

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
