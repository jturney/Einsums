# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Random pass order over programs with permutation operators, views and scratch.

The random-pipeline shards shuffle ``_SAFE_PASSES`` over the base corpus, which
never carries an operator, never writes an einsum through a view, and holds no
graph-owned tensor for PermuteFusion or CSE to act on. This shard shuffles
``_RICH_SAFE_PASSES`` (adding PermuteFusion, AntisymmetrizerExpansion,
GEMMBatching and Materialization) over the ``rich_ops`` corpus, one pass per
manager so each pass's own verdict is counted.

The guards at the bottom draw their own trials, so they hold whichever subset of
this file is selected.

The shared harness lives in _fuzz_diff_common.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from _pytest.outcomes import Skipped

from _fuzz_diff_common import *  # shared fuzz/differential harness


def _rich_program(rng, depth, max_stmts):
    return _gen_block(rng, depth=depth, max_stmts=max_stmts, rich_ops=True)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_rich_random_pipeline(seed, dtype):
    rng = np.random.default_rng(240_000 + seed)
    prog = _rich_program(rng, depth=3, max_stmts=6)
    check_program_rich_pipeline(prog, *_seed_arrays(rng, dtype), f"rrp{seed}", rng, dtype=dtype)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_rich_random_pipeline_flat(seed, dtype):
    """Flat programs are where GEMMBatching can fire: a Loop or Conditional
    between two cluster members disqualifies the group."""
    rng = np.random.default_rng(250_000 + seed)
    prog = _rich_program(rng, depth=0, max_stmts=10)
    check_program_rich_pipeline(prog, *_seed_arrays(rng, dtype), f"rrpflat{seed}", rng, dtype=dtype)


@pytest.mark.parametrize("seed", fuzz_seeds(100))
def test_fuzz_rich_random_pipeline_replay(seed):
    rng = np.random.default_rng(260_000 + seed)
    prog = _rich_program(rng, depth=2, max_stmts=8)
    check_program_rich_pipeline(prog, *_seed_arrays(rng), f"rrpreplay{seed}", rng, runs=2)


@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_rich_random_pipeline_chained_and_rank_views(seed):
    """The rich corpus plus views of views, views of a transpose and rank-reducing views.

    CSE, ContractionPlanning and ElementWiseFusion read through views, so a shuffled order
    meets every alias shape the generator can draw rather than only the blocks ``rich_ops``
    names. A separate corpus, so the seeds above keep drawing the programs they always have.
    """
    rng = np.random.default_rng(270_000 + seed)
    prog = _gen_block(rng, depth=2, max_stmts=8, rich_views=True, rank_views=True, rich_ops=True)
    check_program_rich_pipeline(prog, *_seed_arrays(rng), f"rrpviews{seed}", rng)


# ──────────────────────────────────────────────────────────────────────────
# Guards: a shuffled pass that never fires has been shuffled, not tested, and a
# corpus that stopped drawing operators or view writes would leave the added
# passes nothing to get wrong.
# ──────────────────────────────────────────────────────────────────────────

_GUARD_SEEDS = range(80)


def test_rich_pipeline_corpus_carries_operators_and_view_writes():
    have = {"operator": 0, "view_read": 0, "view_write": 0, "scratch": 0}
    for seed in _GUARD_SEEDS:
        census = rich_op_census(_rich_program(np.random.default_rng(240_000 + seed), depth=3, max_stmts=6))
        for key in have:
            have[key] += census[key] > 0
    n = len(_GUARD_SEEDS)
    for key, floor in (("operator", 0.5), ("view_read", 0.3), ("view_write", 0.4), ("scratch", 0.3)):
        assert have[key] >= floor * n, f"only {have[key]} of {n} rich programs carry a {key}: {have}"


def test_added_passes_fire_under_a_shuffled_order():
    """Every pass this shard added to the shuffle rewrites some programs.

    Floors sit well under what the corpus gives today (roughly 70% for
    AntisymmetrizerExpansion, 10% for PermuteFusion and CSE, 8% for
    GEMMBatching on flat programs) so they catch a generator that stopped
    producing the shape, not ordinary drift.
    """
    before = {name: dict(stat) for name, stat in _RICH_PIPELINE_STATS.items()}
    for seed in _GUARD_SEEDS:
        rng = np.random.default_rng(250_000 + seed)
        prog = _rich_program(rng, depth=0, max_stmts=10)
        try:
            check_program_rich_pipeline(prog, *_seed_arrays(rng), f"rrpguard{seed}", rng)
        except Skipped:
            pass

    def fired(name):
        return _RICH_PIPELINE_STATS[name]["modified"] - before[name]["modified"]

    ran = _RICH_PIPELINE_STATS["CSE"]["ran"] - before["CSE"]["ran"]
    assert ran >= 0.9 * len(_GUARD_SEEDS), f"only {ran} usable trials"
    for name, floor in (("AntisymmetrizerExpansion", 0.4), ("PermuteFusion", 0.04), ("CSE", 0.03),
                        ("GEMMBatching", 0.02), ("Materialization", 0.1)):
        assert fired(name) >= floor * ran, f"{name} fired on only {fired(name)} of {ran} programs"


class _FixedOrder:
    """Stands in for the rng's shuffle so a pinned case runs one exact order."""

    def __init__(self, order):
        self.order = list(order)

    def shuffle(self, x):
        x[:] = self.order


def test_dead_node_elimination_keeps_the_materialize_a_batched_gemm_needs():
    """GEMMBatching, then Materialization, then DeadNodeElimination.

    The two identical contractions become one BatchedGemm writing both the dead
    scratch and the live one. Materialization gives the dead one storage, and
    DeadNodeElimination must keep that Materialize: nothing reads the tensor,
    but the surviving batch still writes it. Removing it, as the pass once did,
    fails only at execute, since the graph verifier does not flag it. The
    default order runs
    DeadNodeElimination first, which removes the dead einsum before batching.
    The leading scale is what makes GEMMBatching group the pair at all.
    """
    A, B = MAT_BY_SHAPE[(2, 3)][0], MAT_BY_SHAPE[(3, 3)][0]
    R = MAT_BY_SHAPE[(2, 3)][1]
    dead, live = SCRATCH_BY_SHAPE[(2, 3)]
    product = ("xeinsum", "ij <- ik ; kj", None, 1.0, ("m", A), ("m", B), 0.0)
    prog = [("scale", 0.5, B),
            product + (("m", dead), False, False),
            product + (("m", live), False, False),
            ("axpy", 1.0, live, R)]
    order = _FixedOrder(["GEMMBatching", "Materialization", "DeadNodeElimination"])
    check_program_rich_pipeline(prog, *_seed_arrays(np.random.default_rng(3)), "dne_batched", order)
