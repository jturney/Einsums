#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The hardware cost facts, as Python sees them.

These are bindings over measurements, so the assertions are about shape and
internal consistency rather than values: what a cost model reading them is
entitled to assume.
"""

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
