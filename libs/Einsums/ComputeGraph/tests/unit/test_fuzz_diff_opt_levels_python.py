# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: ``graph.optimize(O1)`` and ``graph.optimize(O2)`` against numpy.

Every other shard optimizes through ``default_pass_manager``. The levels are
built by ``PassManager::create_for``, a separately written list, and O1 in
particular is a subset whose passes were only ever checked with the rest of the
default pipeline behind them. A pass that leaves something for a later pass to
tidy is wrong at O1 and correct everywhere else, and only running O1 finds it.

Three corpora: the shared flat and control-flow programs, and the ``rich_ops``
programs, which add permutation operators, views read and written, batchable
clusters and graph-owned scratch. The last is where O1 has something to do: on
a pool of nothing but caller-held tensors its passes almost never fire, so the
first two corpora test that O1 leaves a program alone and the third tests that
what it rewrites stays right.

The guards at the bottom draw their own trials, so they hold whichever subset of
this file is selected.

The shared harness lives in _fuzz_diff_common.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from _pytest.outcomes import Skipped

from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness

LEVELS = ("O1", "O2")


def _rich_program(rng, depth, max_stmts):
    return _gen_block(rng, depth=depth, max_stmts=max_stmts, rich_ops=True)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(80))
def test_fuzz_opt_level_flat(seed, dtype, level):
    rng = np.random.default_rng(200_000 + seed)
    prog = _gen_block(rng, depth=0, max_stmts=10)
    check_program_opt_level(prog, *_seed_arrays(rng, dtype), f"olflat{seed}", level, dtype=dtype)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("seed", fuzz_seeds(80))
def test_fuzz_opt_level_control_flow(seed, dtype, level):
    rng = np.random.default_rng(210_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    check_program_opt_level(prog, *_seed_arrays(rng, dtype), f"olcf{seed}", level, dtype=dtype)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(80))
def test_fuzz_opt_level_rich(seed, dtype, level):
    rng = np.random.default_rng(220_000 + seed)
    prog = _rich_program(rng, depth=2, max_stmts=8)
    check_program_opt_level(prog, *_seed_arrays(rng, dtype), f"olrich{seed}", level, dtype=dtype)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_opt_level_rich_replay(seed, level):
    """Optimize once, execute twice. The deferred scratch is materialized by
    the pipeline, so the second run is the one that finds storage a pass assumed
    would only be needed once."""
    rng = np.random.default_rng(230_000 + seed)
    prog = _rich_program(rng, depth=3, max_stmts=6)
    check_program_opt_level(prog, *_seed_arrays(rng), f"olreplay{seed}", level, runs=2)


# ──────────────────────────────────────────────────────────────────────────
# Guards: the arms above pass vacuously if the rich corpus stops carrying what
# it was built to carry, or if a level stops rewriting anything.
# ──────────────────────────────────────────────────────────────────────────

_GUARD_SEEDS = range(60)


def test_opt_level_rich_corpus_carries_operators_views_and_scratch():
    """Each feature appears in a meaningful share of drawn programs.

    Counted per program rather than per statement, since what matters is how
    many trials put the feature in front of the pipeline.
    """
    have = {"operator": 0, "view_read": 0, "view_write": 0, "scratch": 0}
    for seed in _GUARD_SEEDS:
        census = rich_op_census(_rich_program(np.random.default_rng(220_000 + seed), depth=2, max_stmts=8))
        for key in have:
            have[key] += census[key] > 0
    n = len(_GUARD_SEEDS)
    for key, floor in (("operator", 0.5), ("view_read", 0.3), ("view_write", 0.4), ("scratch", 0.3)):
        assert have[key] >= floor * n, f"only {have[key]} of {n} rich programs carry a {key}: {have}"


def test_each_opt_level_modifies_a_share_of_rich_programs():
    """O1 and O2 each change a meaningful share of the rich corpus.

    A level that silently stopped rewriting anything would pass every oracle
    comparison above, so this counts it directly. O1 is the one at risk: over
    caller-held tensors alone its passes find nothing, and the scratch motifs
    are what give them work.
    """
    before = {lvl: dict(_OPT_LEVEL_STATS[lvl]) for lvl in LEVELS}
    for seed in _GUARD_SEEDS:
        for level in LEVELS:
            rng = np.random.default_rng(220_000 + seed)
            prog = _rich_program(rng, depth=2, max_stmts=8)
            try:
                check_program_opt_level(prog, *_seed_arrays(rng), f"olguard{seed}", level)
            except Skipped:
                pass
    for level, floor in (("O1", 0.15), ("O2", 0.6)):
        ran = _OPT_LEVEL_STATS[level]["attempted"] - before[level]["attempted"]
        modified = _OPT_LEVEL_STATS[level]["modified"] - before[level]["modified"]
        assert ran >= 0.9 * len(_GUARD_SEEDS), f"{level}: only {ran} usable trials"
        assert modified >= floor * ran, f"{level} modified only {modified} of {ran} rich programs"


# ──────────────────────────────────────────────────────────────────────────
# PermuteFusion defects the rich corpus found, pinned at both levels. Built
# from fixed pool slots rather than a seed, so each reproduces every run.
# ──────────────────────────────────────────────────────────────────────────

_M44 = MAT_BY_SHAPE[(4, 4)]
_EAGER44 = SCRATCH_BY_SHAPE[(4, 4)][1]
_DEFERRED44 = SCRATCH_BY_SHAPE[(4, 4)][0]


@pytest.mark.parametrize("level", LEVELS)
def test_permute_fusion_ignores_a_write_to_the_source_before_the_consumer(level):
    """S = A^T; A = X @ Y; C = S @ B.

    A fused einsum would read A directly, after the gemm replaced it, and C
    would come out as (X @ Y)^T @ B, which is what PermuteFusion did before it
    checked the gap between a permute and its consumer. The generator reaches
    the same state through Reorder moving an unrelated writer of A between the
    two, which is how it was found.
    """
    A, S = _M44[0], _EAGER44
    X, Y = MAT_BY_SHAPE[(4, 3)][0], MAT_BY_SHAPE[(3, 4)][0]
    B, C = MAT_BY_SHAPE[(4, 2)][0], MAT_BY_SHAPE[(4, 2)][1]
    prog = [("perm", 1.0, 0.0, A, S),
            ("gemm", 1.0, X, Y, 0.0, A),
            ("xeinsum", "ij <- ik ; kj", None, 1.0, ("m", S), ("m", B), 0.0, ("m", C), False, False)]
    check_program_opt_level(prog, *_seed_arrays(np.random.default_rng(1)), "pf_src_write", level)


@pytest.mark.parametrize("level", LEVELS)
def test_permute_fusion_redirect_clobbers_the_source_through_an_earlier_writer(level):
    """S1 = X @ Y; S2 = X @ Y; R += S2; S1 = A^T; C = S1 @ B.

    Fusing the transpose would redirect S1's slot at A, so the first einsum,
    which still writes S1, would write X @ Y into A, destroying the caller's
    input. CSE hides the earlier writer's reader (it redirects R's read of S2
    to S1 through the slot, not the node's inputs), so PermuteFusion has to
    count readers and writers by buffer rather than by id.
    """
    A, R = _M44[0], _M44[1]
    X, Y = MAT_BY_SHAPE[(4, 3)][0], MAT_BY_SHAPE[(3, 4)][0]
    B, C = MAT_BY_SHAPE[(4, 2)][0], MAT_BY_SHAPE[(4, 2)][1]
    product = ("xeinsum", "ij <- ik ; kj", None, 1.0, ("m", X), ("m", Y), 0.0)
    prog = [product + (("m", _EAGER44), False, False),
            product + (("m", _DEFERRED44), False, False),
            ("axpy", 1.0, _DEFERRED44, R),
            ("perm", 1.0, 0.0, A, _EAGER44),
            ("xeinsum", "ij <- ik ; kj", None, 1.0, ("m", _EAGER44), ("m", B), 0.0, ("m", C), False, False)]
    check_program_opt_level(prog, *_seed_arrays(np.random.default_rng(2)), "pf_redirect", level)
