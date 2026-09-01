# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Keeps the tier opportunity generators honest.

What this file guards is COVERAGE, not tolerance. Each generator exists so a
pass can be measured at all, and a generator that stops provoking its pass
turns that pass's row in the classification into a statement about nothing
while every assertion here still passes. So the assertion is that the pass
FIRED.

Deliberately absent: any pinned ULP or bit-equality figure. Those belong in the
classification, they differ by vendor and by toolchain, and asserting one here
would build the detector this project has already been bitten by four times.
The only numeric assertion is a loose sanity bound that catches a generator
comparing something it should not, which is a different failure and one that
shows up as an error of order one rather than of order the last bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from _fuzz_diff_common import *  # shared fuzz/differential harness
from _fuzz_diff_common import TIER_CANDIDATES, measure_program_single_pass
from _tier_opportunities import NO_GENERATOR, OPPORTUNITY_GENERATORS

#: Enough seeds to show the shape is reliable, few enough that the two
#: deliberately large generators do not dominate the suite's runtime.
_SEEDS = 3


def test_every_candidate_has_a_generator_or_a_stated_reason():
    """No pass is quietly left out of the classification."""
    covered = set(OPPORTUNITY_GENERATORS) | set(NO_GENERATOR)
    missing = set(TIER_CANDIDATES) - covered
    assert not missing, (
        f"these passes have neither a generator nor a recorded reason: {sorted(missing)}")

    overlap = set(OPPORTUNITY_GENERATORS) & set(NO_GENERATOR)
    assert not overlap, f"listed as both generated and not: {sorted(overlap)}"


def test_no_generator_entries_explain_themselves():
    """A gap is only acceptable while it says why it is there."""
    for name, reason in NO_GENERATOR.items():
        assert name in TIER_CANDIDATES, f"{name} is not a pass under classification"
        assert len(reason) > 40, f"{name}'s reason is too thin to act on: {reason!r}"


@pytest.mark.parametrize("pass_name", sorted(OPPORTUNITY_GENERATORS))
def test_generator_actually_provokes_its_pass(pass_name):
    """The pass rewrites something on the shape built for it.

    This is the whole contract of a generator. A pass that never fires produces
    a differential with no difference, which reads as "faithful" and means
    "untested".
    """
    gen = OPPORTUNITY_GENERATORS[pass_name]
    fired = usable = 0
    worst = None

    for seed in range(_SEEDS):
        rng = np.random.default_rng(8800 + seed)
        prog, m_arrays, v_arrays, t_arrays = gen(rng)
        rec = measure_program_single_pass(
            prog, m_arrays, v_arrays, t_arrays, f"tieropp_{pass_name}{seed}", pass_name)
        if rec is None:
            continue
        usable += 1
        if rec["fired"]:
            fired += 1
        if worst is None or rec["norm_rel"] > worst["norm_rel"]:
            worst = rec

    assert usable, f"{pass_name}: every trial was skipped as numerically unusable"
    assert fired == usable, (
        f"{pass_name}: fired on {fired} of {usable} trials. Its generator no longer builds "
        f"the shape it responds to, so this pass is not being measured.")

    # A rewrite this far off is not re-association, it is the measurement
    # comparing something it should not - an orphaned buffer, or two runs handed
    # different inputs. Both have happened here; neither is subtle when it does.
    assert worst["norm_rel"] < 1e-6, (
        f"{pass_name}: norm-relative gap of {worst['norm_rel']:.3g} is far past anything "
        f"re-association explains, so the trial is comparing the wrong thing: {worst}")
