#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Profile Python interface.

Surface for ``einsums._core.profile`` plus the ``section`` context manager
that mirrors the C++ ``LabeledSection`` macro, and the ``profile`` decorator
that records every call of a function as a zone. The C-extension submodule
is loaded lazily on first attribute access, so importing this module does
not by itself fire ``einsums::initialize()``.

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
"""

import contextlib as _contextlib
import functools as _functools
import inspect as _inspect
import importlib as _importlib


def _core():
    """Resolve and cache the compiled ``einsums._core.profile`` submodule."""
    return _importlib.import_module("._core.profile", "einsums")


def __getattr__(name):
    """PEP 562 lazy attribute access, mirroring ``einsums.graph``."""
    if name.startswith("_"):
        raise AttributeError(name)
    attr = getattr(_core(), name)
    globals()[name] = attr
    return attr


@_contextlib.contextmanager
def section(name, *, file="", line=0, func=""):
    """Scoped profile region, the Python equivalent of ``LabeledSection``.

    The optional ``file``, ``line``, and ``func`` arguments parallel the
    C++ macro's compile-time captures so reports can link a Python-side
    region to a specific source location. Defaults leave them empty, in
    which case the report falls back to the region name alone.

    Usage::

        with section("contract_dgemm"):
            ...
    """
    push = _core().push
    pop = _core().pop
    push(name, file, line, func)
    try:
        yield
    finally:
        pop()


def profile(func=None, /, *, name=None):
    """Record every call of a function as a profile zone.

    Usable bare, or with a zone name of your own::

        @profile
        def build_fock(density):
            ...

        @profile(name="SCF iteration")
        def iterate(state):
            ...

    The zone is named after the function's qualified name unless ``name`` is
    given, and carries the function's source file, first line and name, so the
    report points at where it is defined. Stack it beneath ``@staticmethod`` or
    ``@classmethod``, so that it wraps the function itself.

    Coroutine and generator functions are refused. A zone must close before any
    zone opened after it on the same thread, and one held open across an
    ``await`` or a ``yield`` would stay open while other code runs and opens and
    closes zones of its own. Time the synchronous work inside them with
    :func:`section` instead.

    The compiled module is looked up on the first call, not here, so that
    decorating a function does not start the runtime when its module is
    imported. In a build without the profiler a call then costs one check.
    """
    if func is None:
        return lambda f: profile(f, name=name)

    if _inspect.iscoroutinefunction(func) or _inspect.isasyncgenfunction(func) or _inspect.isgeneratorfunction(func):
        raise TypeError(
            f"profile cannot time {getattr(func, '__qualname__', func)!r}: it is a coroutine or generator "
            "function, and a zone held open across an await or a yield would not close before zones "
            "opened after it on the same thread. Use profile.section() around its synchronous parts."
        )
    code = getattr(func, "__code__", None)
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    file = code.co_filename if code is not None else ""
    line = code.co_firstlineno if code is not None else 0
    zone = name if name is not None else qualname
    hooks = []  # filled on the first call: (push, pop), or None where nothing records

    @_functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not hooks:
            core = _core()
            hooks.append((core.push, core.pop) if core.available() else None)
        if hooks[0] is None:
            return func(*args, **kwargs)
        push, pop = hooks[0]
        push(zone, file, line, qualname)
        try:
            return func(*args, **kwargs)
        finally:
            pop()

    return wrapper


def annotate_dims(key, dims):
    """Attach a sequence of dimension sizes as ``<key>.<i>`` annotations.

    Mirrors the C++ ``annotate_dims`` helper; emitted entries are
    individual scalar annotations so they show up next to the parent
    region in the report.
    """
    a = _core().annotate
    for i, d in enumerate(dims):
        a(f"{key}.{i}", int(d))
