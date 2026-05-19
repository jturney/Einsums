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
from .rc import LogLevel  # noqa: F401  (exposed as einsums.LogLevel)


# ----------------------------------------------------------------------
# Version / build metadata
# ----------------------------------------------------------------------
# ``__version__`` and ``version_info`` follow Python convention (PEP 396,
# sys.version_info-style tuple). ``build_info()`` returns the heavier
# build-time metadata as a dict — kept callable to defer the C++ call to
# ``_core._version.*`` until someone actually asks for it.
#
# Importing this module does NOT load ``_core``; the compiled extension
# is touched lazily on first access via PEP 562.


def _load_version() -> tuple[str, tuple[int, int, int, str]]:
    """Pull version string + info tuple from the compiled ``_core._version``.

    Returns ``(__version__, version_info)``. Cached at module import for
    cheap repeat lookups — but the call itself fires ``_core`` (and thus
    ``einsums::initialize``) the first time it's invoked.
    """
    v = _importlib.import_module("._core._version", __name__)
    return v.full_version_as_string(), (
        v.major_version(),
        v.minor_version(),
        v.patch_version(),
        v.tag(),
    )


def set_log_level(level) -> None:
    """Set the einsums runtime log level.

    Accepts either an :class:`einsums.LogLevel` enum member or an integer
    in the spdlog range ``0..6`` (TRACE..OFF). Out-of-range integers are
    clamped by spdlog. Compile-time disablement takes precedence:
    ``set_log_level(LogLevel.TRACE)`` cannot re-enable TRACE messages in
    a Release build.
    """
    core = _importlib.import_module("._core", __name__)
    if isinstance(level, LogLevel):
        level = level.value
    core.set_log_level(int(level))


def get_log_level() -> "LogLevel":
    """Return the current einsums runtime log level as a :class:`LogLevel`.

    Falls back to the raw integer if the current level doesn't map to a
    :class:`LogLevel` member (e.g. spdlog's CRITICAL / OFF which we don't
    expose in the Python enum yet).
    """
    core = _importlib.import_module("._core", __name__)
    raw = core.get_log_level()
    try:
        return LogLevel(raw)
    except ValueError:
        return raw


def build_info() -> dict[str, str]:
    """Return a dict of build-time metadata.

    Keys: ``version``, ``type``, ``date_time``, ``configuration``,
    ``full``. Values mirror the corresponding ``einsums::*`` C++ helpers
    in ``Einsums/Version.hpp`` — the build type and timestamp are
    captured at compile time, so the dict is constant across the
    lifetime of a process.
    """
    v = _importlib.import_module("._core._version", __name__)
    return {
        "version":       v.full_version_as_string(),
        "type":          v.build_type(),
        "date_time":     v.build_date_time(),
        "configuration": v.configuration_string(),
        "full":          v.full_build_string(),
    }


def __getattr__(name):
    """PEP 562 lazy attribute lookup.

    The first time the user touches any compiled symbol (``einsums.gemm``,
    ``einsums.RuntimeTensorD``, ``einsums.__version__``, ...) we load
    ``_core``, which fires ``einsums::initialize()`` inside its
    ``PYBIND11_MODULE`` body using the current ``einsums.rc`` settings.

    Short-circuits on dunder / private names: Python's import machinery
    itself probes them during ``from . import _core``, and we'd loop.
    The two exceptions are ``__version__`` and ``version_info`` — both
    are part of the public Python surface and are populated lazily from
    the ``_core._version`` submodule on first access.
    """
    if name in ("__version__", "version_info"):
        version_str, info_tuple = _load_version()
        globals()["__version__"] = version_str
        globals()["version_info"] = info_tuple
        return globals()[name]

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
    return sorted(
        set(globals()) | set(dir(core)) |
        {"__version__", "version_info", "build_info", "set_log_level", "get_log_level", "LogLevel"}
    )
