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
import sys

import pytest

from einsums import hardware as hw

# The OpenMP runtimes einsums itself can be linked against: libomp for the LLVM
# toolchains, vcomp for MSVC. Deliberately not libiomp5md - MKL pulls Intel's
# runtime into the same process, and on the Windows CI leg both are loaded at
# once, so asking it for a thread count would answer for a runtime einsums does
# not drive. Order matters for the same reason; the first hit wins.
_WINDOWS_OPENMP_DLLS = ("libomp.dll", "vcomp140.dll", "vcomp.dll")


def _loaded_openmp_runtime():
    """The OpenMP runtime already in this process, or ``None``.

    POSIX spells "search everything already loaded" as ``dlopen(NULL)``, which
    ctypes writes ``CDLL(None)``. Windows has no equivalent and ctypes rejects
    ``None`` outright, so the runtimes have to be named - but only one that
    ``GetModuleHandleW`` already knows about is accepted. Letting ``CDLL`` fall
    back to the search path could load a *second* copy of a runtime and report
    its thread count instead of the running one's, which is the opposite of
    what this file sets out to check.
    """
    try:
        if sys.platform != "win32":
            return ctypes.CDLL(None)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        for name in _WINDOWS_OPENMP_DLLS:
            if kernel32.GetModuleHandleW(name):
                return ctypes.CDLL(name)
    except (AttributeError, OSError, TypeError):
        return None
    return None


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

    A build without OpenMP has no such runtime loaded at all; skip gracefully
    rather than asserting anything about that case.
    """
    lib = _loaded_openmp_runtime()
    if lib is None:
        pytest.skip("no OpenMP runtime loaded in this process")

    try:
        fn = lib.omp_get_max_threads
    except AttributeError:
        pytest.skip("no omp_get_max_threads symbol loaded in this process")
    fn.restype = ctypes.c_int

    assert hw.get_max_threads() == int(fn())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
