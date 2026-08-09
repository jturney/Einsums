#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The hardware cost facts, as Python sees them.

These are bindings over measurements, so the assertions are about shape and
internal consistency rather than values: what a cost model reading them is
entitled to assume.
"""

import os
import subprocess
import sys

import pytest

from einsums import hardware as hw


def test_region_cost_is_a_nonnegative_number():
    cost = hw.omp_region_cost_ns()
    assert isinstance(cost, float)
    assert cost >= 0.0
    # An empty parallel region is microseconds, not milliseconds. A value this
    # large means the measurement ran against a loaded machine and every
    # threshold derived from it is nonsense.
    assert cost < 1e7


def test_thresholds_are_nonnegative():
    assert hw.omp_min_parallel_elements() >= 0
    assert hw.omp_min_parallel_flops() >= 0


def test_thresholds_follow_the_region_cost():
    """Both thresholds are derived from the region cost, so they vanish together.

    A serial build or a single-threaded run measures no region cost, and then
    there is no work size below which parallelizing loses - because nothing is
    parallelized either way. Any nonzero threshold with a zero region cost would
    mean a hardcoded constant had crept back in.
    """
    cost = hw.omp_region_cost_ns()
    elements = hw.omp_min_parallel_elements()
    flops = hw.omp_min_parallel_flops()

    if cost == 0.0:
        assert elements == 0
        assert flops == 0
    else:
        assert elements > 0
        assert flops > 0
        # Same derivation, converted at different rates: flops is the more
        # conservative of the two, so it never sits below the element count.
        assert flops >= elements


def test_values_are_stable_within_a_process():
    """Detected once and cached, so a cost model can call them in a loop."""
    assert hw.omp_region_cost_ns() == hw.omp_region_cost_ns()
    assert hw.omp_min_parallel_elements() == hw.omp_min_parallel_elements()
    assert hw.omp_min_parallel_flops() == hw.omp_min_parallel_flops()


def _region_cost_in_subprocess(env):
    """The region cost as a fresh process sees it, under the given environment."""
    out = subprocess.run(
        [sys.executable, "-c",
         "from einsums import hardware as hw; print(repr(hw.omp_region_cost_ns()))"],
        capture_output=True, text=True, check=True,
        env={**os.environ, **env},
    )
    return float(out.stdout.strip())


CALIBRATION = """# einsums hardware calibration, format 1
omp_region_cost_ns 1 0.000000
omp_region_cost_ns 2 4242.000000
omp_region_cost_ns 4 8484.000000
"""


def test_a_calibrated_value_is_used_and_is_identical_across_processes(tmp_path):
    """The point of calibrating: same machine, same team, same number.

    Measured per process the region cost drifts by tens of percent with whatever
    else the machine is doing, which is harmless for a threshold and not harmless
    for a chooser ranking discrete options against the rate.
    """
    path = tmp_path / "calibration.txt"
    path.write_text(CALIBRATION)
    env = {"EINSUMS_HARDWARE_CALIBRATION": str(path), "OMP_NUM_THREADS": "2"}
    assert [_region_cost_in_subprocess(env) for _ in range(3)] == [4242.0] * 3


def test_the_entry_matching_this_team_size_is_the_one_used(tmp_path):
    """A run at four threads must not be handed the two-thread number."""
    path = tmp_path / "calibration.txt"
    path.write_text(CALIBRATION)
    key = "EINSUMS_HARDWARE_CALIBRATION"
    assert _region_cost_in_subprocess({key: str(path), "OMP_NUM_THREADS": "4"}) == 8484.0


def test_a_team_size_with_no_entry_falls_back_to_measuring(tmp_path):
    """A partial calibration is still useful for the sizes it does cover."""
    path = tmp_path / "calibration.txt"
    path.write_text(CALIBRATION)
    value = _region_cost_in_subprocess(
        {"EINSUMS_HARDWARE_CALIBRATION": str(path), "OMP_NUM_THREADS": "3"})
    assert value not in (4242.0, 8484.0)
    assert value >= 0.0


def test_an_explicit_pin_wins(tmp_path):
    """So a benchmark can hold the rate fixed without touching any file."""
    path = tmp_path / "calibration.txt"
    path.write_text(CALIBRATION)
    assert _region_cost_in_subprocess(
        {"EINSUMS_HARDWARE_CALIBRATION": str(path), "OMP_NUM_THREADS": "2",
         "EINSUMS_OMP_REGION_COST_NS": "12345"}) == 12345.0


def test_a_corrupt_calibration_is_survived(tmp_path):
    """Anything unreadable means measure: the file is never load-bearing."""
    path = tmp_path / "calibration.txt"
    path.write_text("omp_region_cost_ns 2 not-a-number\ngarbage\n")
    assert _region_cost_in_subprocess(
        {"EINSUMS_HARDWARE_CALIBRATION": str(path), "OMP_NUM_THREADS": "2"}) >= 0.0


def test_a_missing_calibration_is_survived_and_none_is_written(tmp_path):
    """The library reads this file. Only calibrate_hardware writes it."""
    path = tmp_path / "absent" / "calibration.txt"
    env = {"EINSUMS_HARDWARE_CALIBRATION": str(path), "OMP_NUM_THREADS": "2"}
    assert _region_cost_in_subprocess(env) >= 0.0
    assert not path.exists(), "the library must never create a calibration file"


def test_the_default_calibration_path_is_reported(tmp_path):
    """So a user can find, inspect or delete it without guessing."""
    assert isinstance(hw.default_calibration_path(), str)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
