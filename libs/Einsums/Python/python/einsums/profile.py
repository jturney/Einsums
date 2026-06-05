#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Profile Python interface.

Surface for ``einsums._core.profile`` plus the ``section`` context manager
that mirrors the C++ ``LabeledSection`` macro. The C-extension submodule
is loaded lazily on first attribute access — importing this module does
not by itself fire ``einsums::initialize()``.

Typical usage::

    import einsums.profile as prof

    with prof.section("contract_dgemm"):
        prof.annotate("M", 512)
        prof.annotate("N", 512)
        ...

    prof.flush()
    prof.print_report(detailed=True)
"""

import contextlib as _contextlib
import importlib as _importlib


def _core():
    """Resolve and cache the compiled ``einsums._core.profile`` submodule."""
    return _importlib.import_module("._core.profile", "einsums")


def __getattr__(name):
    """PEP 562 lazy attribute access — mirrors ``einsums.graph``."""
    if name.startswith("_"):
        raise AttributeError(name)
    attr = getattr(_core(), name)
    globals()[name] = attr
    return attr


@_contextlib.contextmanager
def section(name, *, file="", line=0, func=""):
    """Scoped profile region — Python equivalent of ``LabeledSection``.

    The optional ``file``/``line``/``func`` arguments parallel the C++
    macro's compile-time captures so reports can link a Python-side region
    to a specific source location. Defaults leave them empty (the report
    falls back to the region name alone).

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


def annotate_dims(key, dims):
    """Attach a sequence of dimension sizes as ``<key>.<i>`` annotations.

    Mirrors the C++ ``annotate_dims`` helper; emitted entries are
    individual scalar annotations so they show up next to the parent
    region in the report.
    """
    a = _core().annotate
    for i, d in enumerate(dims):
        a(f"{key}.{i}", int(d))
