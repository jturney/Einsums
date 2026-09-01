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


#: What "the same answer" is allowed to mean across a kernel change, measured
#: NORM-RELATIVE rather than in ULP.
#:
#: PermuteFusion folds a transpose into its consumer's transa flag, so the fused
#: form runs a different vendor kernel than the explicit-transpose-then-GEMM
#: form, and two implementations of one expression may disagree in the last bit.
#:
#: ULP is the wrong instrument for that bound and this file learned it the hard
#: way: it is measured per element against THAT element's spacing, so a result
#: with entries near zero reports tens or thousands of ULP at a few times
#: machine epsilon of real error. An earlier version of this test asserted 4
#: ULP, picked off one machine, and the Windows leg reported 25 ULP at a
#: max-absolute gap of 4e-16 and a norm-relative gap of 5e-17, which is smaller
#: than ordinary rounding. The number was alarming and the answer was fine.
#:
#: So the bound is on the whole-result norm, which does not care where the small
#: entries are, and it is loose enough to hold on any vendor while still being
#: many orders below the order-one gap an orphaned buffer produces.
_KERNEL_CHANGE_NORM_REL = 1e-12


def test_an_orphaned_buffer_is_not_counted_as_a_deviation():
    """A buffer whose producer was folded away is excluded, not compared.

    PermuteFusion removes the transpose, so the transposed temporary keeps its
    seed value. Comparing it would report the seed against the computed answer
    and call a faithful pass a 60%-error one, which is exactly what the first
    version of this harness did.

    The bar is a norm-relative gap rather than bit equality, and the difference
    between those two numbers is the whole point: without the exclusion the gap
    is of order one, with it the gap is of order the last bit. Asserting zero
    here would be asserting that the vendor computes ``A^T B`` the same way
    whether it is handed a transposed copy or a transa flag, which no BLAS
    promises and Accelerate does not do.
    """
    checked = 0
    for seed in range(8):
        prog, pool = _perm_program(np.random.default_rng(4400 + seed), False)
        rec = measure_program_single_pass(prog, pool, [], [], f"orph{seed}", "PermuteFusion")
        if rec is None or not rec["fired"]:
            continue
        checked += 1
        assert rec["orphaned"] >= 1, "the folded transpose's output should be excluded"
        assert rec["norm_rel"] < _KERNEL_CHANGE_NORM_REL, (
            "folding a transpose into its consumer changed a value the graph still "
            f"produces by more than a kernel swap can explain, or an orphaned buffer "
            f"leaked into the measurement: {rec}")
    assert checked, "the fusion never fired; this test then proves nothing"
