#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The OpenMP thread ceiling, as Python sees it.

These wrappers are only useful if they read and write the same process-wide
OpenMP runtime that every threaded region in the process consults, so the
cases below check the getter against a direct ``ctypes`` call into that
runtime rather than merely asserting it returns something plausible.
"""

import ctypes

import pytest

from einsums import hardware as hw


def test_get_max_threads_is_a_positive_int():
    threads = hw.get_max_threads()
    assert isinstance(threads, int)
    assert threads >= 1


def test_set_num_threads_round_trips_through_the_getter():
    original = hw.get_max_threads()
    try:
        hw.set_num_threads(1)
        assert hw.get_max_threads() == 1

        hw.set_num_threads(3)
        assert hw.get_max_threads() == 3
    finally:
        hw.set_num_threads(original)
    assert hw.get_max_threads() == original


def test_agrees_with_a_direct_omp_get_max_threads_call():
    """Same process-wide runtime, read two different ways.

    ``ctypes.CDLL(None)`` finds no OpenMP symbol at all in a build without
    OpenMP, or on a platform where the runtime is not already loaded into
    the process; skip gracefully rather than asserting anything about that
    case.
    """
    try:
        fn = ctypes.CDLL(None).omp_get_max_threads
        fn.restype = ctypes.c_int
    except (AttributeError, OSError):
        pytest.skip("no omp_get_max_threads symbol loaded in this process")

    assert hw.get_max_threads() == int(fn())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
