#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""ComputeGraph Python interface.

Surface for ``einsums._core.graph`` plus a few Pythonic helpers such as
``default_pass_manager`` and ``capture``. The C-extension submodule is
loaded lazily on first attribute access, so importing ``einsums.graph``
does not by itself fire ``einsums::initialize()``.
"""

import contextlib as _contextlib
import importlib as _importlib


def _core():
    """Resolve and cache the compiled ``einsums._core.graph`` submodule.

    Helpers defined in this file reference C-extension classes such as
    ``PassManager`` and ``CaptureContext`` by name, but Python only
    triggers ``__getattr__`` for attribute access on the module from
    outside. Unqualified-name lookups inside helpers go straight to
    module globals, which are empty until the lazy loader has cached
    something. So helpers reach the C extension through this small
    accessor instead.
    """
    return _importlib.import_module("._core.graph", "einsums")


def __getattr__(name):
    """PEP 562 lazy attribute access, mirroring ``einsums/__init__.py``.

    Looking up ``einsums.graph.<Name>`` for the first time imports the
    compiled ``einsums._core.graph`` submodule, fetches the attribute,
    caches it in this module's globals so subsequent lookups skip
    ``__getattr__``, and returns it.

    The dunder/private short-circuit avoids re-entry when Python's
    import machinery probes for things like ``__path__``.

    Copy-paste template for new ``einsums.<sub>`` shells: change
    ``"._core.graph"`` to ``"._core.<sub>"``.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    attr = getattr(_core(), name)
    globals()[name] = attr
    return attr


def default_pass_manager():
    """Return a fresh PassManager pre-loaded with the canonical pass list.

    Mirrors the C++ ``cg::PassManager::create_default()`` helper. The
    binding can't expose the static factory directly because PassManager
    holds a vector of non-copyable unique_ptrs, so we construct an empty
    one and call ``populate_default()`` in-place.
    """
    pm = _core().PassManager()
    pm.populate_default()
    return pm


# Python-side stack of graphs currently being captured. The C++
# CaptureContext owns the authoritative capture state, but its ``graph()``
# accessor isn't bound to Python. Operator helpers in
# ``einsums/__init__.py`` need the active graph so they can allocate
# intermediate outputs for expressions like ``A + B`` and ``A @ B`` via
# ``graph.create_zero_tensor`` instead of the process-owned
# ``_core.create_zero_tensor``. Graph-owned intermediates outlive
# ``graph.execute()``; process-owned ones can be GC'd mid-chain, which
# surfaces as "tensor ... appears to have been destroyed".
_capture_graph_stack = []


def current_graph():
    """Return the innermost graph currently being captured, or ``None``.

    Used by the numpy-ergonomics operators to decide where to allocate
    operator outputs. ``None`` means we're not inside ``cg.capture``, so
    eager process-owned allocation is correct.
    """
    return _capture_graph_stack[-1] if _capture_graph_stack else None


@_contextlib.contextmanager
def capture(graph):
    """Context manager wrapping CaptureContext::begin_capture / end_capture.

    Usage::

        import einsums
        import einsums.graph as cg

        g = cg.Graph("my_workflow")
        with cg.capture(g):
            einsums.linalg.gemm(1.0, A, B, 0.0, C)

        g.execute()
    """
    ctx = _core().CaptureContext.current()
    ctx.begin_capture(graph)
    _capture_graph_stack.append(graph)
    try:
        yield graph
    finally:
        _capture_graph_stack.pop()
        ctx.end_capture()
