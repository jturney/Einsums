#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Python package shell for einsums.

Importing ``einsums`` (or any subpackage like ``einsums.rc``) loads the
compiled extension ``einsums._core`` so its bindings — types *and* functions
— are registered immediately. This is what lets an external module that
merely *receives* an einsums object (e.g. psi4 returning a
``TiledRuntimeTensor``) work after a plain ``import einsums``, with no need
to ``import einsums._core`` explicitly.

Loading ``_core`` only **registers** the bindings; it does NOT start the C++
runtime. Runtime startup (``einsums::initialize``) is deferred to the first
real use via :func:`_ensure_initialized`, so ``einsums.rc`` can still be
configured first and is read at init time.

Typical usage::

    import einsums.rc
    einsums.rc.threads = 8         # optional pre-init config

    import einsums
    einsums.gemm(...)              # runtime initializes here, using rc

``import einsums.rc`` is optional — ``einsums.rc`` is always present with
default (``None``) values, and the runtime initializes from those defaults
on first use if nothing was configured.
"""

import importlib as _importlib

from . import rc  # noqa: F401  (exposed as einsums.rc)
from .rc import LogLevel  # noqa: F401  (exposed as einsums.LogLevel)

# Eager-load the compiled extension so its bindings are registered as soon as
# the package is imported. This only registers types/functions — _core's
# module init does not start the runtime (see PyEinsumsMain.cpp). Runtime
# startup is deferred to _ensure_initialized() on first real use.
from . import _core as _core  # noqa: F401


def _ensure_initialized() -> None:
    """Start the einsums runtime from ``einsums.rc`` on first real use.

    Idempotent — ``_core._initialize_from_rc()`` no-ops once the runtime is
    running. Called lazily (not at import) so ``einsums.rc`` settings made
    after ``import einsums`` but before the first compute still take effect.
    """
    _core._initialize_from_rc()


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
    _ensure_initialized()
    if isinstance(level, LogLevel):
        level = level.value
    _core.set_log_level(int(level))


def get_log_level() -> "LogLevel":
    """Return the current einsums runtime log level as a :class:`LogLevel`.

    Falls back to the raw integer if the current level doesn't map to a
    :class:`LogLevel` member (e.g. spdlog's CRITICAL / OFF which we don't
    expose in the Python enum yet).
    """
    _ensure_initialized()
    raw = _core.get_log_level()
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


# ----------------------------------------------------------------------
# Capture-aware __getitem__ on RuntimeTensor classes
# ----------------------------------------------------------------------
# Outside of a graph capture, ``t[1:3, :]`` returns a non-graph
# RuntimeTensorView via the codegen's index protocol — the same behavior
# numpy users expect. Inside ``with cg.capture(g):`` though, that view
# isn't registered with the graph, so subsequent ops on it would see an
# off-graph tensor (or fail to find it as a slot).
#
# We monkey-patch __getitem__ on each RuntimeTensor class once at _core
# load time so slice-only access inside capture routes through
# ``cg.view`` automatically. The view it returns aliases the parent
# (TensorHandle::aliases is set), so reads/writes through it propagate
# correctly. Anything we can't translate (integer indices, strided
# slices, mismatched arity) raises a clear error so the user knows to
# either drop out of capture or use ``cg.view`` directly with a custom
# axis spec.

_RUNTIME_TENSOR_CLASS_NAMES = ("RuntimeTensorF", "RuntimeTensorD", "RuntimeTensorC", "RuntimeTensorZ")
_runtime_tensor_getitem_patched = False


def _make_capture_aware_getitem(orig_getitem):
    """Wrap a RuntimeTensor's __getitem__ to dispatch to ``cg.view`` when
    we're inside a graph capture and the key is a pure slice expression."""

    def wrapper(self, key):
        # Defer the import: graph.py imports _core, which imports us, so
        # we can't import at module-load time without a cycle. By the
        # time this wrapper is hit, _core has loaded and graph.py is
        # safely importable.
        from . import graph as _g
        ctx = _g.CaptureContext.current()
        if not ctx.is_capturing():
            return orig_getitem(self, key)

        # Normalize key to a tuple.
        key_tuple = key if isinstance(key, tuple) else (key,)
        rank = self.rank()

        # Expand a single Ellipsis to enough full slices to fill the rank.
        if any(k is Ellipsis for k in key_tuple):
            n_ellipsis = sum(1 for k in key_tuple if k is Ellipsis)
            if n_ellipsis > 1:
                raise IndexError("only one Ellipsis allowed in index")
            ellipsis_pos = next(i for i, k in enumerate(key_tuple) if k is Ellipsis)
            n_explicit = len(key_tuple) - 1
            n_fill = rank - n_explicit
            if n_fill < 0:
                raise IndexError(
                    f"too many indices for tensor of rank {rank}: got {n_explicit} explicit + Ellipsis"
                )
            key_tuple = (
                key_tuple[:ellipsis_pos]
                + (slice(None),) * n_fill
                + key_tuple[ellipsis_pos + 1 :]
            )

        if len(key_tuple) != rank:
            raise IndexError(
                f"cg.view auto-dispatch: index has {len(key_tuple)} elements but tensor has rank {rank}; "
                f"use cg.view directly for partial indexing inside capture"
            )

        ranges = []
        for axis, k in enumerate(key_tuple):
            if isinstance(k, slice):
                if k.step is not None and k.step != 1:
                    raise IndexError(
                        f"cg.view auto-dispatch: axis {axis} has step={k.step}; "
                        f"only step=1 slices are supported"
                    )
                if k.start is None and k.stop is None:
                    ranges.append((-1, -1))  # full axis
                else:
                    lo = 0 if k.start is None else int(k.start)
                    hi = self.dim(axis) if k.stop is None else int(k.stop)
                    ranges.append((lo, hi))
            elif isinstance(k, int):
                raise IndexError(
                    f"cg.view auto-dispatch: integer index at axis {axis} would reduce rank, "
                    f"which cg.view doesn't yet support (rank-preserving slices only). "
                    f"Use a slice like {k}:{k + 1} or call cg.view directly."
                )
            else:
                raise IndexError(
                    f"cg.view auto-dispatch: unsupported index type {type(k).__name__} at axis {axis}"
                )

        return _g.view(self, ranges)

    return wrapper


def _patch_runtime_tensor_getitem(core):
    """Install the capture-aware ``__getitem__`` on each RuntimeTensor class.

    Idempotent — the global ``_runtime_tensor_getitem_patched`` flag
    prevents double-wrapping (which would deepen the call stack and
    confuse error reporting).
    """
    global _runtime_tensor_getitem_patched
    if _runtime_tensor_getitem_patched:
        return
    for cls_name in _RUNTIME_TENSOR_CLASS_NAMES:
        cls = getattr(core, cls_name, None)
        if cls is None:
            continue
        orig = cls.__getitem__
        cls.__getitem__ = _make_capture_aware_getitem(orig)
    _runtime_tensor_getitem_patched = True


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

    # _core is already loaded (eager import above); touching a compiled symbol
    # is the first "real use", so bring the runtime up now (honoring einsums.rc).
    _ensure_initialized()
    _patch_runtime_tensor_getitem(_core)
    try:
        attr = getattr(_core, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'einsums' has no attribute {name!r}") from exc

    globals()[name] = attr  # cache; future lookups skip __getattr__
    return attr


def __dir__():
    return sorted(
        set(globals()) | set(dir(_core)) |
        {"__version__", "version_info", "build_info", "set_log_level", "get_log_level", "LogLevel"}
    )
