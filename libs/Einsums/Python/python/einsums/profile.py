#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Profile Python code alongside Einsums.

The profiler is Waggle's, shared with every other library in the process, and so is this API:
``einsums.profile`` is :mod:`waggle` under the names Einsums has always used, so zones opened here
and the zones Einsums opens inside its library land in one tree. A build with
``EINSUMS_WITH_PROFILER=OFF`` keeps every name callable and records nothing.

Typical usage::

    import einsums.profile as prof

    with prof.section("contract_dgemm"):
        prof.annotate("M", 512)
        prof.annotate("N", 512)
        ...

    @prof.profile
    def build_fock(density):
        ...

    prof.flush()
    prof.print_report(detailed=True)

Use :class:`waggle.Zone` directly for a zone entered often: it registers its site once.
"""

import contextlib as _contextlib
import functools as _functools
import importlib as _importlib

_recording = None


def available():
    """Whether this build records anything: ``False`` for ``EINSUMS_WITH_PROFILER=OFF``."""
    global _recording
    if _recording is None:
        _recording = bool(_importlib.import_module("._core.profile", "einsums").available())
    return _recording


def _waggle():
    import waggle

    return waggle


def section(name, *, file="", line=0, func=""):
    """Scoped profile region, the Python equivalent of ``WAGGLE_ZONE``::

        with section("contract_dgemm"):
            ...

    The optional ``file``, ``line`` and ``func`` parallel the C++ macro's captures, so reports can
    point a Python region at its source. The site is registered once and reused.
    """
    if not available():
        return _contextlib.nullcontext()
    return _waggle().zone(name, file=file, line=line, func=func)


def profile(func=None, /, *, name=None):
    """Record every call of a function as a profile zone; see :func:`waggle.profile`."""
    if func is None:
        return lambda f: profile(f, name=name)
    waggled = _waggle().profile(func, name=name)  # refuses what cannot be timed, either way
    if available():
        return waggled

    @_functools.wraps(func)
    def wrapper(*args, **kwargs):  # nothing records here: only the call
        return func(*args, **kwargs)

    return wrapper


def annotate(key, value):
    """Attach a string, integer or floating-point annotation to the current zone."""
    if available():
        _waggle().annotate(key, value)


def annotate_dims(key, dims):
    """Attach a sequence of dimension sizes as ``<key>.<i>`` annotations."""
    if available():
        _waggle().annotate_dims(key, dims)


def mem_alloc(bytes):
    """Record an allocation of ``bytes`` in the current zone."""
    if available():
        _waggle().mem_alloc(int(bytes))


def mem_free(bytes):
    """Record a free of ``bytes`` in the current zone."""
    if available():
        _waggle().mem_free(int(bytes))


def flush():
    """Drain every thread's recorded events, so ``print_report`` and ``export_json`` see them."""
    if available():
        _waggle().flush()


def print_report(detailed=False):
    """Print the compact (or detailed) report to standard output."""
    if available():
        _waggle().print_report(detailed)


def export_json(path="einsums_profile.json"):
    """Write the aggregated profile to JSON; the path on success, ``None`` otherwise."""
    if available() and _waggle().export_json(path):
        return path
    return None


def set_thread_name(name):
    """Name the calling thread in reports and viewers."""
    if available():
        _waggle().set_thread_name(name)


def current_thread_id():
    """The profiler's id for the calling thread; 0 where nothing records."""
    return _waggle().current_thread_id() if available() else 0


def total_push_count():
    """Zones opened so far, on every thread."""
    return _waggle().total_push_count() if available() else 0


def total_pop_count():
    """Zones closed so far, on every thread."""
    return _waggle().total_pop_count() if available() else 0


def avg_push_overhead_ns():
    """What opening a recorded zone costs, in nanoseconds."""
    return _waggle().push_overhead_ns() if available() else 0.0


def avg_pop_overhead_ns():
    """What closing a recorded zone costs, in nanoseconds."""
    return _waggle().pop_overhead_ns() if available() else 0.0
