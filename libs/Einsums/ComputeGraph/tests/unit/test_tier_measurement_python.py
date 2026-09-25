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

import einsums
import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness
from _fuzz_diff_common import (
    _SQ,
    _sq_pool,
    TIER_CANDIDATES,
    measure_program_single_pass,
)


def _perm_program(rng, shared_temporary):
    """A transpose into graph scratch feeding one einsum, or the same transpose feeding two.

    PermuteFusion folds the first and documents that it declines the second,
    because a shared transposed temporary has more than one reader. The pair is
    numerically identical work either way, which is what makes it a test of the
    instrument rather than of the arithmetic. The transpose writes graph scratch
    because fusing removes the only write to it, which the pass will not do to a
    buffer the caller holds.
    """
    pool = _sq_pool(rng, 8)
    idx = list(range(8))
    rng.shuffle(idx)
    a, b, d, e = idx[:4]

    def build(g, m, v, t, name):
        w = g.create_zero_tensor(f"{name}_w", [3, 3], dtype="float64")
        with cg.capture(g):
            einsums.permute("ji <- ij", w, m[a])
            einsums.einsum(_SQ, m[d], w, m[b])
            if shared_temporary:
                einsums.einsum(_SQ, m[e], w, m[b])

    return build, pool


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


def test_folding_a_transpose_stays_within_a_kernel_swap():
    """The fused form agrees with the explicit transpose to within a vendor's rounding.

    The bar is a norm-relative gap rather than bit equality: asserting zero here
    would be asserting that the vendor computes ``A^T B`` the same way whether it
    is handed a transposed copy or a transa flag, which no BLAS promises and
    Accelerate does not do. The transposed temporary is graph scratch, so no pool
    buffer loses its writer.
    """
    checked = 0
    for seed in range(8):
        prog, pool = _perm_program(np.random.default_rng(4400 + seed), False)
        rec = measure_program_single_pass(prog, pool, [], [], f"fold{seed}", "PermuteFusion")
        if rec is None or not rec["fired"]:
            continue
        checked += 1
        assert rec["orphaned"] == 0, f"a caller-held buffer lost its writer: {rec}"
        assert rec["norm_rel"] < _KERNEL_CHANGE_NORM_REL, (
            "folding a transpose into its consumer changed a value the graph still "
            f"produces by more than a kernel swap can explain: {rec}")
    assert checked, "the fusion never fired; this test then proves nothing"


def test_a_transpose_into_a_caller_buffer_is_not_folded():
    """A transpose whose output the caller holds keeps its permute.

    Fusing removes the only write to the transpose's output. The pass used to do
    that to a pool buffer, and this harness excluded the stale buffer from the
    comparison as "orphaned" rather than reporting it; a caller reading that
    tensor after execute() got its old contents.
    """
    for seed in range(4):
        rng = np.random.default_rng(4500 + seed)
        pool = _sq_pool(rng, 8)
        prog = [("perm", 1.0, 0.0, 0, 1), ("einsum", _SQ, 1.0, 1, 2, 0.0, 3)]
        rec = measure_program_single_pass(prog, pool, [], [], f"caller{seed}", "PermuteFusion")
        if rec is None:
            continue
        assert rec["fired"] == 0, f"the transpose into a caller buffer was folded: {rec}"
        assert rec["orphaned"] == 0
        assert rec["bitwise"]
