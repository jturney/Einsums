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

import numpy as _np


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


class DIISAccelerator:
    """Pulay DIIS extrapolation for fixed-point iterations captured as graph loops.

    Accelerates the ``t <- t + step`` update a loop body performs by keeping a
    short history of (amplitude, step) snapshots and replacing the amplitudes
    with the least-squares extrapolant between replays. The error vector is
    the update step itself, the standard coupled-cluster choice; iterations
    whose update the body computes anyway get DIIS for the cost of a K-sized
    host-side solve.

    Construct via :func:`diis` and install with :meth:`wrap`::

        acc = cg.diis([(t1, rd1), (t2, rd2)], k=8)
        body = g.add_loop("iter", 100, acc.wrap(converged))

    Every ``(amplitude, step)`` tensor must be readable between replays: the
    user-visible amplitudes qualify, and the step tensors must be EAGER
    (process-owned), not graph-owned intermediates, which are invisible
    outside the graph. Extrapolated amplitudes are written back IN PLACE
    through ``numpy.asarray`` views, so the pointers the graph captured stay
    valid. Complex dtypes use the conjugated inner product; coefficients are
    real. An ill-conditioned B matrix drops the oldest history pair and
    retries; B is normalized by its largest element before the solve.
    """

    def __init__(self, pairs, k=8):
        if k < 2:
            raise ValueError(f"DIIS needs a history of at least 2, got k={k}")
        self._pairs = list(pairs)
        if not self._pairs:
            raise ValueError("DIIS needs at least one (amplitude, step) pair")
        self._k = k
        self._T = []
        self._E = []

    def wrap(self, condition=None):
        """Return a loop predicate: evaluate ``condition``, then take a DIIS step.

        The returned callable has the ``add_loop`` condition signature
        (iteration index -> bool). ``condition=None`` always continues, i.e.
        the loop runs to its max_iterations. The DIIS step is skipped once
        the condition reports convergence - the loop is over, extrapolating
        would only perturb the converged amplitudes.
        """

        def predicate(it):
            keep_going = condition(it) if condition is not None else True
            if keep_going:
                self.step()
            return keep_going

        return predicate

    def step(self):
        """Push the current (amplitudes, steps) and extrapolate in place."""
        self._push(
            _np.concatenate([_np.asarray(t).ravel() for t, _ in self._pairs]),
            _np.concatenate([_np.asarray(e).ravel() for _, e in self._pairs]),
        )
        ext = self._extrapolate()
        if ext is None:
            return
        offset = 0
        for t, _ in self._pairs:
            view = _np.asarray(t)
            view[...] = ext[offset : offset + view.size].reshape(view.shape)
            offset += view.size

    def _push(self, t_vec, e_vec):
        self._T.append(t_vec)
        self._E.append(e_vec)
        if len(self._T) > self._k:
            self._T.pop(0)
            self._E.pop(0)

    def _extrapolate(self):
        while len(self._T) >= 2:
            m = len(self._T)
            B = _np.empty((m + 1, m + 1))
            B[m, :] = B[:, m] = -1.0
            B[m, m] = 0.0
            for p in range(m):
                for q in range(p, m):
                    B[p, q] = B[q, p] = _np.vdot(self._E[p], self._E[q]).real
            scale = _np.abs(B[:m, :m]).max()
            if scale > 0:
                B[:m, :m] /= scale
            rhs = _np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = _np.linalg.solve(B, rhs)[:m]
            except _np.linalg.LinAlgError:
                self._T.pop(0)
                self._E.pop(0)
                continue
            return sum(ci * Ti for ci, Ti in zip(c, self._T))
        return None


def diis(pairs, k=8):
    """DIIS-accelerate a graph loop's fixed-point iteration.

    ``pairs`` is a list of ``(amplitude, step)`` tensor pairs the loop body
    updates as ``amplitude += step`` each replay; ``k`` is the history depth.
    Returns a :class:`DIISAccelerator` whose :meth:`~DIISAccelerator.wrap`
    produces the loop predicate::

        acc = cg.diis([(t1, rd1), (t2, rd2)])
        body = g.add_loop("ccsd_iter", 100, acc.wrap(converged))
        with cg.capture(body):
            ...  # residuals; rd = r/D; t += rd
    """
    return DIISAccelerator(pairs, k)


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
