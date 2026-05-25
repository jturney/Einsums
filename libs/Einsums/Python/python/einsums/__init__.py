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
import numbers as _numbers

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


# ----------------------------------------------------------------------
# A[o, v, ...] sugar for tiled tensors
# ----------------------------------------------------------------------
# TiledRuntimeTensor.view([s0, s1, ...]) takes one IndexSpace per axis and
# returns a TiledRuntimeTensorView. We add subscript sugar so the natural
# ``A[o, v, o, v]`` (IndexSpace per axis) maps onto it — matching how users
# think about occupied/virtual orbital blocks per irrep.

_TILED_TENSOR_CLASS_NAMES = ("TiledRuntimeTensorF", "TiledRuntimeTensorD", "TiledRuntimeTensorC", "TiledRuntimeTensorZ")
_tiled_tensor_getitem_patched = False


def _tiled_getitem(self, key):
    spaces = list(key) if isinstance(key, tuple) else [key]
    return self.view(spaces)


def _patch_tiled_tensor_getitem(core):
    """Install ``__getitem__`` -> ``view`` on each TiledRuntimeTensor class."""
    global _tiled_tensor_getitem_patched
    if _tiled_tensor_getitem_patched:
        return
    for cls_name in _TILED_TENSOR_CLASS_NAMES:
        cls = getattr(core, cls_name, None)
        if cls is None:
            continue
        cls.__getitem__ = _tiled_getitem
    _tiled_tensor_getitem_patched = True


# ----------------------------------------------------------------------
# numpy-parity ergonomics on RuntimeTensor / RuntimeTensorView
# ----------------------------------------------------------------------
# The compiled tensor classes already implement the buffer protocol, so
# ``np.asarray(t)`` is zero-copy. What they lack is the thin attribute
# layer numpy users reach for reflexively: ``.shape``, ``.ndim``,
# ``.dtype``, ``.T``, ``len(t)``, a useful ``repr``, an explicit
# ``__array__``, and ``@`` (matmul). We install those here, in Python,
# rather than growing the C++ surface — they're pure conveniences built
# on the handful of methods the bindings already expose (``rank``,
# ``dim``, ``transpose_view``, ``name`` where present).
#
# Crucially, ``@`` is NOT delegated to numpy: it dispatches to
# ``einsums.linalg.gemm`` so the operation stays on Einsums code paths
# (and is recorded into the active graph when inside ``cg.capture``).

# Class-name suffix -> numpy dtype. RuntimeTensor{F,D,C,Z} and the View
# variants share the BLAS-style letter: F=float32, D=float64,
# C=complex64, Z=complex128.
_DTYPE_SUFFIX = {"F": "float32", "D": "float64", "C": "complex64", "Z": "complex128"}

_NUMPY_ERGONOMICS_CLASS_NAMES = _RUNTIME_TENSOR_CLASS_NAMES + (
    "RuntimeTensorViewF", "RuntimeTensorViewD", "RuntimeTensorViewC", "RuntimeTensorViewZ",
)
_numpy_ergonomics_patched = False


def _np():
    """Import numpy lazily.

    The ergonomics layer is numpy-interop sugar, so it's reasonable to
    require numpy only when one of these attributes is actually touched —
    keeping ``import einsums`` free of a hard numpy dependency.
    """
    import numpy as np  # noqa: PLC0415
    return np


def _dtype_for_class(cls):
    """numpy dtype for a RuntimeTensor(View) class, keyed on its trailing letter."""
    suffix = cls.__name__[-1]
    name = _DTYPE_SUFFIX.get(suffix)
    if name is None:
        raise TypeError(f"cannot infer dtype for {cls.__name__!r}")
    return _np().dtype(name)


def _tensor_shape(self):
    """``.shape`` -> tuple of axis sizes, built from rank()/dim() so it
    works identically on tensors and views (views don't expose dims())."""
    return tuple(int(self.dim(i)) for i in range(self.rank()))


def _tensor_ndim(self):
    return int(self.rank())


def _tensor_dtype(self):
    return _dtype_for_class(type(self))


def _tensor_T(self):
    """``.T`` -> zero-copy reversed-axis view.

    Outside capture this is the C++ ``transpose_view`` (KEEP_ALIVE-tied to
    this tensor's storage). Inside ``cg.capture`` that raw transpose view
    is invisible to the graph (it shares the parent's data pointer, so the
    capture machinery resolves it to the parent's slot with non-transposed
    dims — a latent crash). So when capturing we route through
    ``cg.permute_view``, which records a graph-registered, parent-aliasing
    view with the correct reversed dims/strides that captured ops can
    safely consume."""
    _g = _importlib.import_module("einsums.graph")
    if _g.current_graph() is not None:
        rank = self.rank()
        return _g.permute_view(self, list(range(rank - 1, -1, -1)))
    return self.transpose_view()


def _permuted_view(self, perm):
    """Zero-copy axis-permuted view, graph-aware.

    Inside ``cg.capture`` routes through ``cg.permute_view`` (graph-registered,
    parent-aliasing); eagerly uses the C++ ``permute_view`` method. Backs
    ``.transpose`` and ``.swapaxes``."""
    _g = _importlib.import_module("einsums.graph")
    if _g.current_graph() is not None:
        return _g.permute_view(self, perm)
    return self.permute_view(perm)


def _normalize_axes(self, axes):
    """Resolve numpy-style ``transpose`` axis args to a perm list.

    Accepts ``transpose()`` (reverse — numpy default), ``transpose(0, 2, 1)``,
    or ``transpose((0, 2, 1))``; negative axes are allowed."""
    rank = self.rank()
    if len(axes) == 0:
        return list(range(rank - 1, -1, -1))
    if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = tuple(axes[0])
    if len(axes) != rank:
        raise ValueError(f".transpose(): got {len(axes)} axes for a rank-{rank} tensor")
    perm = [int(a) % rank for a in axes]
    if sorted(perm) != list(range(rank)):
        raise ValueError(f".transpose(): {tuple(axes)} is not a permutation of 0..{rank - 1}")
    return perm


def _tensor_transpose(self, *axes):
    """``A.transpose(*axes)`` — zero-copy permuted view (numpy semantics)."""
    return _permuted_view(self, _normalize_axes(self, axes))


def _tensor_swapaxes(self, axis1, axis2):
    """``A.swapaxes(i, j)`` — zero-copy view with two axes exchanged."""
    rank = self.rank()
    perm = list(range(rank))
    perm[axis1 % rank], perm[axis2 % rank] = perm[axis2 % rank], perm[axis1 % rank]
    return _permuted_view(self, perm)


def _tensor_copy(self):
    """``A.copy()`` — a fresh dense tensor holding a copy of A (like numpy)."""
    return _copy_of(self, f"{getattr(self, 'name', 'A')}_copy")


def _tensor_len(self):
    rank = self.rank()
    if rank == 0:
        raise TypeError("len() of unsized (rank-0) tensor")
    return int(self.dim(0))


def _tensor_repr(self):
    cls = type(self).__name__
    shape = _tensor_shape(self)
    dtype = _DTYPE_SUFFIX.get(cls[-1], "?")
    name = getattr(self, "name", None)
    if isinstance(name, str) and name:
        return f"{cls}(name={name!r}, shape={shape}, dtype={dtype})"
    return f"{cls}(shape={shape}, dtype={dtype})"


def _tensor_array(self, dtype=None, copy=None):
    """numpy's ``__array__`` protocol.

    Goes through ``memoryview(self)`` (the buffer protocol) rather than
    ``np.asarray(self)`` — the latter would re-enter this very method and
    recurse. The base result aliases tensor storage; we only copy when
    numpy asks (``copy=True``) or a dtype cast forces it.
    """
    np = _np()
    arr = np.asarray(memoryview(self))
    if dtype is not None:
        arr = arr.astype(dtype, copy=bool(copy))
    elif copy:
        arr = arr.copy()
    return arr


def _is_einsums_tensor(obj):
    return type(obj).__name__ in _NUMPY_ERGONOMICS_CLASS_NAMES


def _alloc_output(name, shape, dtype_name):
    """Allocate a fresh zero output tensor for an operator result.

    Inside ``cg.capture`` the output must be owned by the active graph
    (``graph.create_zero_tensor``) so it outlives ``graph.execute()`` — a
    process-owned tensor can be garbage-collected mid-chain (e.g. the first
    intermediate of ``(A + B) - A``), which the executor reports as
    "tensor ... appears to have been destroyed". Outside capture, eager
    process-owned allocation is correct.
    """
    # NB: ``from . import graph`` would hit this package's __getattr__, which
    # forwards "graph" to the C-extension ``_core.graph`` (it has
    # CaptureContext but NOT current_graph). import_module resolves the real
    # Python shell ``einsums/graph.py``.
    _g = _importlib.import_module("einsums.graph")
    g = _g.current_graph()
    if g is not None:
        return g.create_zero_tensor(name, [int(s) for s in shape], dtype=dtype_name)
    return _core.create_zero_tensor(name, [int(s) for s in shape], dtype=dtype_name)


def _tensor_matmul(self, other):
    """``A @ B`` -> ``einsums.linalg.gemm`` into a fresh output tensor.

    Stays on Einsums code paths instead of falling through to numpy. When
    invoked inside ``with cg.capture(g):`` the gemm call is recorded into
    the active graph (the C++ capture context intercepts it), so ``A @ B``
    composes inside a captured workflow just like a direct gemm call would.
    Only rank-2 @ rank-2 with matching dtype is handled; anything else
    raises so the operation never silently leaves Einsums.
    """
    if not _is_einsums_tensor(other):
        raise TypeError(
            "einsums '@' supports only another einsums tensor/view on the "
            f"right-hand side, got {type(other).__name__!r}. Convert it to an "
            "einsums tensor first to keep the operation on Einsums."
        )
    if self.rank() != 2 or other.rank() != 2:
        raise ValueError(
            f"einsums '@' currently supports rank-2 @ rank-2 (gemm); "
            f"got rank {self.rank()} @ rank {other.rank()}"
        )
    if self.dim(1) != other.dim(0):
        raise ValueError(
            f"matmul shape mismatch: {_tensor_shape(self)} @ {_tensor_shape(other)}"
        )
    if _dtype_for_class(type(self)) != _dtype_for_class(type(other)):
        raise TypeError(
            f"einsums '@' requires matching dtypes; got {_dtype_for_class(type(self))} "
            f"and {_dtype_for_class(type(other))}"
        )

    m, n = self.dim(0), other.dim(1)
    out_name = f"{getattr(self, 'name', 'A')}@{getattr(other, 'name', 'B')}"
    # create_*_tensor takes dtype as a string ("float64"), not an np.dtype.
    dtype_name = _DTYPE_SUFFIX[type(self).__name__[-1]]
    C = _alloc_output(out_name, [m, n], dtype_name)
    # beta=0 -> C's zero init is overwritten; alpha=1 -> plain product.
    # Inside cg.capture this records a gemm node instead of executing.
    _core.linalg.gemm(1.0, self, other, 0.0, C)
    return C


# ----------------------------------------------------------------------
# Tier-2 arithmetic operators -> einsums.linalg (axpy / scale / direct_product)
# ----------------------------------------------------------------------
# numpy-style operators that return a NEW tensor (operands never mutated)
# and dispatch to Einsums BLAS-level ops rather than numpy:
#
#   A + B, A - B   -> two axpy into a fresh zero output
#   c * A, A * c   -> axpy(c, A, zeros)           (scalar scaling)
#   A * B          -> direct_product (element-wise / Hadamard, like numpy)
#   A / c          -> axpy(1/c, A, zeros)
#   -A, +A         -> axpy(±1, A, zeros)
#
# In-place forms map straight onto the in-place BLAS ops:
#   A += B, A -= B -> axpy(±1, B, A)
#   A *= c, A /= c -> scale(c | 1/c, A)
#
# All of these record into the active graph when used inside cg.capture
# (the underlying axpy/scale/direct_product are capture-aware), exactly
# like __matmul__. Non-einsums, non-scalar operands raise TypeError so the
# operation never silently falls through to numpy; use np.asarray(t) to
# opt into numpy math explicitly.


def _same_shape(a, b):
    return a.rank() == b.rank() and all(a.dim(i) == b.dim(i) for i in range(a.rank()))


def _zeros_like(self, name):
    """Fresh, fully-allocated zero tensor matching self's shape and dtype.

    Always a dense RuntimeTensor (never a view), so it's a valid output
    target for axpy/scale/direct_product."""
    dtype_name = _DTYPE_SUFFIX[type(self).__name__[-1]]
    shape = [int(self.dim(i)) for i in range(self.rank())]
    return _alloc_output(name, shape, dtype_name)


def _require_same_shape(op, self, other):
    if not _same_shape(self, other):
        raise ValueError(
            f"shape mismatch for '{op}': {_tensor_shape(self)} vs {_tensor_shape(other)}"
        )


def _reject_operand(op, other):
    """Raise a clear TypeError for an unsupported operand so the op never
    silently leaves Einsums via numpy's buffer protocol. Reached only for
    operands that are neither an einsums tensor nor a scalar."""
    raise TypeError(
        f"einsums '{op}' expects an einsums tensor or a scalar, "
        f"got {type(other).__name__!r}. Use np.asarray(t) for numpy math."
    )


def _copy_of(self, name):
    """Fresh graph-/process-owned dense tensor holding a copy of self."""
    out = _zeros_like(self, name)
    _core.linalg.axpy(1.0, self, out)
    return out


def _tensor_add(self, other):
    # scalar -> in-place shift on a copy; tensor -> elementwise sum.
    if isinstance(other, _numbers.Number):
        out = _copy_of(self, f"{getattr(self, 'name', 'A')}+c")
        _core.linalg.shift(other, out)
        return out
    if not _is_einsums_tensor(other):
        _reject_operand("+", other)
    _require_same_shape("+", self, other)
    out = _zeros_like(self, f"{getattr(self, 'name', 'A')}+{getattr(other, 'name', 'B')}")
    _core.linalg.axpy(1.0, self, out)
    _core.linalg.axpy(1.0, other, out)
    return out


def _tensor_sub(self, other):
    if isinstance(other, _numbers.Number):
        out = _copy_of(self, f"{getattr(self, 'name', 'A')}-c")
        _core.linalg.shift(-other, out)
        return out
    if not _is_einsums_tensor(other):
        _reject_operand("-", other)
    _require_same_shape("-", self, other)
    out = _zeros_like(self, f"{getattr(self, 'name', 'A')}-{getattr(other, 'name', 'B')}")
    _core.linalg.axpy(1.0, self, out)
    _core.linalg.axpy(-1.0, other, out)
    return out


def _tensor_radd(self, other):
    # c + A == A + c (only the scalar case reaches __radd__).
    if isinstance(other, _numbers.Number):
        return _tensor_add(self, other)
    _reject_operand("+", other)


def _tensor_rsub(self, other):
    # c - A == (-A) + c.
    if isinstance(other, _numbers.Number):
        out = _zeros_like(self, f"c-{getattr(self, 'name', 'A')}")
        _core.linalg.axpy(-1.0, self, out)  # out = -A
        _core.linalg.shift(other, out)      # out = c - A
        return out
    _reject_operand("-", other)


def _tensor_mul(self, other):
    # scalar -> scaling; tensor -> element-wise (Hadamard), matching numpy '*'.
    if isinstance(other, _numbers.Number):
        out = _zeros_like(self, f"{getattr(self, 'name', 'A')}*c")
        _core.linalg.axpy(other, self, out)
        return out
    if _is_einsums_tensor(other):
        _require_same_shape("*", self, other)
        out = _zeros_like(self, f"{getattr(self, 'name', 'A')}*{getattr(other, 'name', 'B')}")
        _core.linalg.direct_product(1.0, self, other, 0.0, out)
        return out
    _reject_operand("*", other)


def _tensor_rmul(self, other):
    # c * A — only the scalar case reaches here (tensor*tensor uses __mul__).
    if isinstance(other, _numbers.Number):
        return _tensor_mul(self, other)
    _reject_operand("*", other)


def _tensor_truediv(self, other):
    if isinstance(other, _numbers.Number):
        out = _zeros_like(self, f"{getattr(self, 'name', 'A')}/c")
        _core.linalg.axpy(1.0 / other, self, out)
        return out
    _reject_operand("/", other)


def _tensor_neg(self):
    out = _zeros_like(self, f"-{getattr(self, 'name', 'A')}")
    _core.linalg.axpy(-1.0, self, out)
    return out


def _tensor_pos(self):
    # +A returns a copy (numpy semantics), not self.
    out = _zeros_like(self, f"{getattr(self, 'name', 'A')}")
    _core.linalg.axpy(1.0, self, out)
    return out


def _tensor_iadd(self, other):
    # scalar -> in-place shift (no allocation); tensor -> in-place axpy.
    if isinstance(other, _numbers.Number):
        _core.linalg.shift(other, self)
        return self
    if not _is_einsums_tensor(other):
        _reject_operand("+=", other)
    _require_same_shape("+=", self, other)
    _core.linalg.axpy(1.0, other, self)  # in place: self += other
    return self


def _tensor_isub(self, other):
    if isinstance(other, _numbers.Number):
        _core.linalg.shift(-other, self)
        return self
    if not _is_einsums_tensor(other):
        _reject_operand("-=", other)
    _require_same_shape("-=", self, other)
    _core.linalg.axpy(-1.0, other, self)
    return self


def _tensor_imul(self, other):
    # In-place scalar scaling only; tensor '*=' would need a self-aliased
    # direct_product (skip — use `A = A * B` for element-wise).
    if isinstance(other, _numbers.Number):
        _core.linalg.scale(other, self)
        return self
    if _is_einsums_tensor(other):
        raise TypeError("in-place '*=' supports a scalar only; use `A = A * B` for element-wise")
    _reject_operand("*", other)


def _tensor_itruediv(self, other):
    if isinstance(other, _numbers.Number):
        _core.linalg.scale(1.0 / other, self)
        return self
    raise TypeError("in-place '/=' supports a scalar only")


def _patch_numpy_ergonomics(core):
    """Install the numpy-parity attributes on each RuntimeTensor(View) class.

    Idempotent. ``shape``/``ndim``/``dtype``/``T`` are read-only
    properties; the rest are plain methods (``__len__``, ``__repr__``,
    ``__array__``, ``__matmul__``)."""
    global _numpy_ergonomics_patched
    if _numpy_ergonomics_patched:
        return
    for cls_name in _NUMPY_ERGONOMICS_CLASS_NAMES:
        cls = getattr(core, cls_name, None)
        if cls is None:
            continue
        cls.shape = property(_tensor_shape)
        cls.ndim = property(_tensor_ndim)
        cls.dtype = property(_tensor_dtype)
        cls.T = property(_tensor_T)
        cls.__len__ = _tensor_len
        cls.__repr__ = _tensor_repr
        cls.__array__ = _tensor_array
        cls.transpose = _tensor_transpose
        cls.swapaxes = _tensor_swapaxes
        cls.copy = _tensor_copy
        cls.__matmul__ = _tensor_matmul
        # Tier-2 arithmetic operators (dispatch to einsums.linalg).
        cls.__add__ = _tensor_add
        cls.__radd__ = _tensor_radd
        cls.__sub__ = _tensor_sub
        cls.__rsub__ = _tensor_rsub
        cls.__mul__ = _tensor_mul
        cls.__rmul__ = _tensor_rmul
        cls.__truediv__ = _tensor_truediv
        cls.__neg__ = _tensor_neg
        cls.__pos__ = _tensor_pos
        cls.__iadd__ = _tensor_iadd
        cls.__isub__ = _tensor_isub
        cls.__imul__ = _tensor_imul
        cls.__itruediv__ = _tensor_itruediv
    _numpy_ergonomics_patched = True


def _bootstrap():
    """Start the runtime and install the Python-side patches.

    Idempotent. Called from ``__getattr__`` (first compiled-symbol touch)
    and from the Tier-3 constructor aliases below — those are *real*
    module attributes, so they bypass ``__getattr__`` and must bootstrap
    themselves to guarantee the runtime is up and the numpy-ergonomics
    layer (``.shape``, ``@``, …) is on the returned tensor's class.
    """
    _ensure_initialized()
    _patch_runtime_tensor_getitem(_core)
    _patch_tiled_tensor_getitem(_core)
    _patch_numpy_ergonomics(_core)


# ----------------------------------------------------------------------
# Tier-3 numpy-style constructor aliases
# ----------------------------------------------------------------------
# Familiar spellings (einsums.zeros / ones / empty / full / eye / array /
# asarray and the *_like variants) layered over create_zero_tensor and the
# buffer protocol. numpy conventions: shape first, ``dtype=`` keyword
# (numpy dtype or string), default float64. The einsums-specific ``name=``
# is optional and defaults to the constructor name.
#
# einsums has four element types; ``dtype`` is normalized to one of them
# (float32 stays float32; every other real type -> float64; complex64
# stays, every other complex -> complex128). Allocation goes through
# _alloc_output so a constructor used inside cg.capture yields a
# graph-owned tensor (consistent with the operators).


def _normalize_shape(shape):
    """Accept an int or an iterable of ints (numpy-style) -> list[int]."""
    if isinstance(shape, _numbers.Integral):
        return [int(shape)]
    return [int(s) for s in shape]


def _einsums_dtype_str(dtype):
    """Map a dtype spec (string / np.dtype / Python type) to one of the
    four einsums dtype strings."""
    np = _np()
    dt = np.dtype(dtype)
    if dt == np.dtype("float32"):
        return "float32"
    if dt == np.dtype("complex64"):
        return "complex64"
    if np.issubdtype(dt, np.complexfloating):
        return "complex128"
    return "float64"


def zeros(shape, dtype="float64", name=None):
    """Zero-filled tensor of the given shape — like ``numpy.zeros``."""
    _bootstrap()
    return _alloc_output(name or "zeros", _normalize_shape(shape), _einsums_dtype_str(dtype))


def empty(shape, dtype="float64", name=None):
    """Uninitialized-tensor stand-in — like ``numpy.empty``.

    einsums has no uninitialized-allocation primitive, so this is
    currently zero-filled (safe, just not the fastest). Kept for numpy
    parity and forward compatibility."""
    _bootstrap()
    return _alloc_output(name or "empty", _normalize_shape(shape), _einsums_dtype_str(dtype))


def full(shape, fill_value, dtype="float64", name=None):
    """Tensor filled with ``fill_value`` — like ``numpy.full``."""
    _bootstrap()
    t = _alloc_output(name or "full", _normalize_shape(shape), _einsums_dtype_str(dtype))
    t.set_all(fill_value)
    return t


def ones(shape, dtype="float64", name=None):
    """One-filled tensor — like ``numpy.ones``."""
    return full(shape, 1, dtype=dtype, name=name or "ones")


def eye(n, m=None, dtype="float64", name=None):
    """2-D tensor with ones on the main diagonal — like ``numpy.eye``."""
    _bootstrap()
    np = _np()
    cols = n if m is None else m
    t = _alloc_output(name or "eye", [int(n), int(cols)], _einsums_dtype_str(dtype))
    k = min(int(n), int(cols))
    diag = np.arange(k)
    np.asarray(t)[diag, diag] = 1
    return t


def asarray(obj, dtype=None, name=None):
    """Convert ``obj`` to an einsums tensor — like ``numpy.asarray``.

    An einsums tensor with a compatible dtype is returned unchanged (no
    copy); anything else (numpy array, nested list, scalar) is copied into
    a fresh einsums tensor via the buffer protocol. ``dtype`` is honored
    and normalized to an einsums element type."""
    _bootstrap()
    np = _np()
    if _is_einsums_tensor(obj) and (dtype is None or np.dtype(dtype) == obj.dtype):
        return obj
    arr = np.asarray(obj)
    dtype_str = _einsums_dtype_str(arr.dtype if dtype is None else dtype)
    t = _alloc_output(name or "asarray", list(arr.shape), dtype_str)
    np.asarray(t)[...] = arr  # buffer-protocol copy (handles dtype cast + layout)
    return t


def array(obj, dtype=None, name=None):
    """Like :func:`asarray` but always copies (matches ``numpy.array``)."""
    _bootstrap()
    np = _np()
    src = np.asarray(obj)  # also copies an einsums tensor's contents out
    dtype_str = _einsums_dtype_str(src.dtype if dtype is None else dtype)
    t = _alloc_output(name or "array", list(src.shape), dtype_str)
    np.asarray(t)[...] = src
    return t


def _dtype_str_of(t, dtype):
    return _einsums_dtype_str(dtype) if dtype is not None else _DTYPE_SUFFIX[type(t).__name__[-1]]


def zeros_like(t, dtype=None, name=None):
    """Zero tensor matching ``t``'s shape (and dtype unless overridden)."""
    return zeros(_tensor_shape(t), dtype=_dtype_str_of(t, dtype), name=name or "zeros_like")


def ones_like(t, dtype=None, name=None):
    return full(_tensor_shape(t), 1, dtype=_dtype_str_of(t, dtype), name=name or "ones_like")


def empty_like(t, dtype=None, name=None):
    return empty(_tensor_shape(t), dtype=_dtype_str_of(t, dtype), name=name or "empty_like")


def full_like(t, fill_value, dtype=None, name=None):
    return full(_tensor_shape(t), fill_value, dtype=_dtype_str_of(t, dtype), name=name or "full_like")


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
    _bootstrap()
    try:
        attr = getattr(_core, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'einsums' has no attribute {name!r}") from exc

    globals()[name] = attr  # cache; future lookups skip __getattr__
    return attr


def __dir__():
    return sorted(
        set(globals()) | set(dir(_core)) |
        {"__version__", "version_info", "build_info", "set_log_level", "get_log_level", "LogLevel"} |
        {"zeros", "ones", "empty", "full", "eye", "array", "asarray",
         "zeros_like", "ones_like", "empty_like", "full_like"}
    )
