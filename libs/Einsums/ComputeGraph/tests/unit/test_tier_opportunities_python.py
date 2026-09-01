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

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

import einsums

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


# ──────────────────────────────────────────────────────────────────────────
# The report
#
# Which tier a pass belongs to cannot be decided on one machine. PermuteFusion
# reads bit-identical against OpenBLAS here and one ULP against Accelerate, so
# a single-toolchain answer is a hypothesis wearing a result's clothes.
#
# Rather than assert anything about the numbers, this writes them out and lets
# every CI leg contribute one. The workflow points EINSUMS_TIER_REPORT_DIR at a
# directory it uploads, exactly as it already does for hung-test stack dumps;
# with the variable unset this is a no-op, so a local run costs nothing and
# leaves nothing behind.
# ──────────────────────────────────────────────────────────────────────────


def _environment():
    """Enough about this machine to tell two legs' reports apart."""
    try:
        configuration = einsums.build_info().get("configuration", "")
    except Exception:  # pragma: no cover - a report is never worth failing over
        configuration = ""
    return {
        "label": os.environ.get("EINSUMS_TIER_LABEL", ""),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "einsums_configuration": configuration,
    }


def test_write_tier_report():
    """Measure every generator and write the result, when CI asked for one.

    Asserts only that the measurement ran. The numbers are evidence for the
    classification, and an assertion on them here would be the toolchain
    detector this file exists to avoid.
    """
    out_dir = os.environ.get("EINSUMS_TIER_REPORT_DIR")

    measurements = []
    for pass_name in sorted(OPPORTUNITY_GENERATORS):
        gen = OPPORTUNITY_GENERATORS[pass_name]
        for seed in range(_SEEDS):
            rng = np.random.default_rng(9900 + seed)
            prog, m_arrays, v_arrays, t_arrays = gen(rng)
            rec = measure_program_single_pass(
                prog, m_arrays, v_arrays, t_arrays, f"tierrep_{pass_name}{seed}", pass_name)
            if rec is None:
                continue
            measurements.append({
                "pass": pass_name,
                "seed": seed,
                "fired": rec["fired"],
                "bitwise": rec["bitwise"],
                "max_abs": rec["max_abs"],
                "max_rel": rec["max_rel"],
                "max_ulp": rec["max_ulp"],
                "norm_rel": rec["norm_rel"],
                "nonfinite_diff": rec["nonfinite_diff"],
                "orphaned": rec["orphaned"],
            })

    assert measurements, "no generator produced a usable trial"

    if not out_dir:
        return

    report = {
        "environment": _environment(),
        "no_generator": sorted(NO_GENERATOR),
        "measurements": measurements,
    }
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = report["environment"]["label"] or platform.platform()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    (directory / f"tier-report-{safe}.json").write_text(json.dumps(report, indent=1))
