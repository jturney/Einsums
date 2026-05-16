#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Python package shell for einsums.

Keeping this file dead-simple is load-bearing: importing ``einsums`` (or any
subpackage like ``einsums.rc``) must NOT trigger the C++ runtime to start
up. The compiled extension lives at ``einsums._core`` and is loaded lazily
on first attribute access.

Typical usage::

    import einsums.rc
    einsums.rc.threads = 8         # optional pre-init config

    import einsums
    einsums.gemm(...)              # _core loads here, runtime initializes
"""

import importlib as _importlib

from . import rc  # noqa: F401  (exposed as einsums.rc)


def __getattr__(name):
    """PEP 562 lazy attribute lookup.

    The first time the user touches any compiled symbol (``einsums.gemm``,
    ``einsums.RuntimeTensorD``, ...) we load ``_core``, which fires
    ``einsums::initialize()`` inside its ``PYBIND11_MODULE`` body using the
    current ``einsums.rc`` settings.

    Short-circuits on dunder / private names: Python's import machinery
    itself probes them during ``from . import _core``, and we'd loop.
    """
    if name.startswith("_"):
        raise AttributeError(f"module 'einsums' has no attribute {name!r}")

    # ``importlib.import_module`` returns the loaded module object directly,
    # avoiding the package-attribute lookup that ``from . import _core``
    # would do (and which would re-enter ``__getattr__``).
    core = _importlib.import_module("._core", __name__)
    try:
        attr = getattr(core, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'einsums' has no attribute {name!r}") from exc

    globals()[name] = attr  # cache; future lookups skip __getattr__
    return attr


def __dir__():
    core = _importlib.import_module("._core", __name__)
    return sorted(set(globals()) | set(dir(core)))
