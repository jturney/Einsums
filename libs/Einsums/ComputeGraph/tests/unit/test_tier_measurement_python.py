# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Coverage for the single-pass tier measurement in ``_fuzz_diff_common``.

Sorting the structural-algebraic passes into their Part 5.1 tiers is a
measurement rather than a declaration, and this pins the two properties that
measurement depends on. Neither is about any particular pass being correct;
both are about the instrument reporting something a classification can be
built on.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness
from _fuzz_diff_common import (
    _SQ,
    _sq_pool,
    TIER_CANDIDATES,
    measure_program_single_pass,
)


def _perm_program(rng, shared_temporary):
    """A transpose feeding one einsum, or the same transpose feeding two.

    PermuteFusion folds the first and documents that it declines the second,
    because a shared transposed temporary has more than one reader. The pair is
    numerically identical work either way, which is what makes it a test of the
    instrument rather than of the arithmetic.
    """
    pool = _sq_pool(rng, 8)
    idx = list(range(8))
    rng.shuffle(idx)
    a, b, t, d, e = idx[:5]
    prog = [("perm", 1.0, 0.0, a, t), ("einsum", _SQ, 1.0, t, b, 0.0, d)]
    if shared_temporary:
        prog.append(("einsum", _SQ, 1.0, t, b, 0.0, e))
    return prog, pool


@pytest.mark.parametrize("pass_name", sorted(TIER_CANDIDATES))
def test_every_candidate_constructs_and_reports_a_firing_count(pass_name):
    """Each pass under classification is reachable and counts its own rewrites.

    A pass missing from Python, or present without a counter, cannot be
    measured at all: a differential that finds no difference would be unable to
    say whether the pass was faithful or simply never ran.
    """
    obj = getattr(cg, pass_name)()
    assert isinstance(getattr(obj, TIER_CANDIDATES[pass_name]), int)


def test_fires_on_a_lone_consumer_and_declines_on_a_shared_temporary():
    """The discriminating pair, run over several seeds."""
    fired_any = declined_any = 0
    for seed in range(8):
        prog, pool = _perm_program(np.random.default_rng(4300 + seed), False)
        rec = measure_program_single_pass(prog, pool, [], [], f"lone{seed}", "PermuteFusion")
        if rec is not None and rec["fired"]:
            fired_any += 1

        prog, pool = _perm_program(np.random.default_rng(4300 + seed), True)
        rec = measure_program_single_pass(prog, pool, [], [], f"shared{seed}", "PermuteFusion")
        if rec is not None and rec["fired"] == 0:
            declined_any += 1

    assert fired_any, "the lone-consumer shape should fuse; the corpus proves nothing if not"
    assert declined_any, "the shared-temporary shape should be declined"


def test_an_orphaned_buffer_is_not_counted_as_a_deviation():
    """A buffer whose producer was folded away is excluded, not compared.

    PermuteFusion removes the transpose, so the transposed temporary keeps its
    seed value. Comparing it would report the seed against the computed answer
    and call a faithful pass a 60%-error one, which is exactly what the first
    version of this harness did.
    """
    for seed in range(8):
        prog, pool = _perm_program(np.random.default_rng(4400 + seed), False)
        rec = measure_program_single_pass(prog, pool, [], [], f"orph{seed}", "PermuteFusion")
        if rec is None or not rec["fired"]:
            continue
        assert rec["orphaned"] >= 1, "the folded transpose's output should be excluded"
        assert rec["bitwise"], (
            "folding a transpose into its consumer changed no value the graph still "
            f"produces, so the measurement should see none: {rec}")
        assert rec["max_ulp"] == 0.0 and rec["norm_rel"] == 0.0
