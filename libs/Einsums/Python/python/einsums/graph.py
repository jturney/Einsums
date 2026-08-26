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


def _resolve_space(registry, space):
    """Turn one entry of an ``annotate`` spaces argument into a ``SpaceId``.

    Accepts a name, which is the spelling the design puts in user code, or a
    ``SpaceId`` already in hand, which is what a caller that looked one up
    keeps. An unregistered name is an error rather than an implicit
    registration: a space's scale symbol and growth class are what the cost
    model reads, and inventing them from a name would put numbers in a
    scaling report that nobody declared.
    """
    if isinstance(space, str):
        found = registry.find(space)
        if found is None:
            known = ", ".join(sorted(registry.space(i).name for i in registry.ids))
            raise KeyError(
                f"no index space named '{space}' is registered; "
                f"registered spaces are: {known or '(none)'}"
            )
        return found
    return space


def annotate(tensor, spaces, graph=None):
    """Annotate a tensor's axes with the index spaces they range over.

    ``spaces`` is one entry per axis, in axis order, each a registered space
    NAME or a ``SpaceId``. Names are the user-facing spelling: an id is a
    handle into one registry and means nothing against another, while a name
    is what a person writes and what a saved graph carries.

    ``graph`` defaults to the graph currently being captured. Annotation is a
    declaration about a tensor, but it is stored on the graph's handle for
    that tensor, so there has to be a graph to store it on::

        occ = registry.register_space(cg.index_space("occ", "o"))
        virt = registry.register_space(cg.index_space("virt", "v"))

        g = cg.Graph("ccsd")
        with cg.capture(g):
            cg.annotate(T2, ("occ", "occ", "virt", "virt"))
            einsums.einsum("ijab <- ijcd ; cdab", R2, T2, Vvvvv)

    Annotate BEFORE the operations that use the tensor are captured: capture
    reads the annotation off the handle to bind each contraction's letters,
    and one that arrives afterwards does not reach nodes already recorded.
    ``SpacePropagation`` and the diagnostic passes read the handles as they
    stand when they run, so a late annotation still reaches them.

    Returns ``tensor``, so an annotation can wrap an allocation in one line.
    """
    g = graph if graph is not None else current_graph()
    if g is None:
        raise RuntimeError(
            "cg.annotate needs a graph to store the annotation on: pass graph=..., "
            "or call it inside cg.capture(...)"
        )
    registry = g.space_registry
    g.annotate_spaces(tensor, [_resolve_space(registry, s) for s in spaces])
    return tensor


def bind(graph, mapping):
    """Bind several manifest slots at once, as ONE transaction.

    ``mapping`` is ``{manifest name: tensor}``::

        g = cg.load_graph("ccsd.eig")
        cg.bind(g, {"t1": T1, "t2": T2, "fock": F})
        g.optimize()
        g.execute()

    This is not the same as calling ``graph.bind(name, tensor)`` once per slot,
    and the difference matters exactly when it is most wanted. A dim symbol is a
    constraint ACROSS slots: ``nv`` is one extent wherever it appears, so moving a
    graph to a different-sized problem means every slot has to be read before any
    of them is repointed. Bound one at a time, the second call solves against an
    interface the first has already half-moved, and the graph is refused with
    operands that "describe two different problems".

    The single-slot ``graph.bind`` remains right for what it was built for: a
    same-shape pointer swap, where no symbol moves and order cannot matter.

    Nothing is repointed unless every slot reconciles, so a refusal leaves the
    graph exactly as it was. Every tensor in ``mapping`` must stay alive until
    this returns.
    """
    if not isinstance(mapping, dict):
        raise TypeError(f"cg.bind expects a dict of {{name: tensor}}, got {type(mapping).__name__}")
    graph.bind_begin()
    for name, tensor in mapping.items():
        graph.bind_add(name, tensor)
    graph.bind_commit()
    return graph


class _PyDiis:
    """Pure-Python Pulay DIIS, the reference the C++ accelerator is checked against.

    This was the shipping implementation; :class:`DIISAccelerator` replaced it
    with one C++ entry call per step. It stays because it is the oracle the
    differential tests compare against, and because it is what runs for operand
    types the C++ accelerator does not cover - tiled amplitudes, and pairs whose
    components disagree on dtype.

    Accelerates the ``t <- t + step`` update a loop body performs by keeping a
    short history of (amplitude, step) snapshots and replacing the amplitudes
    with the least-squares extrapolant between replays. The error vector is
    the update step itself, the standard coupled-cluster choice; iterations
    whose update the body computes anyway get DIIS for the cost of a K-sized
    host-side solve.

    Every ``(amplitude, step)`` tensor must be readable between replays: the
    user-visible amplitudes qualify, and the step tensors must be EAGER
    (process-owned), not graph-owned intermediates, which are invisible
    outside the graph. Everything tensor-sized runs on einsums operations -
    history snapshots and the extrapolation are ``axpby`` (the amplitudes
    update IN PLACE, so the pointers the graph captured stay valid), the B
    entries are ``dotc`` inner products (conjugated, so complex dtypes are
    exact; coefficients are real), and the K-sized system solves with
    ``gesv``. Only scalar reads cross to Python, and those scalars are
    cached across replays (a step adds one snapshot, so only one row of B
    is new) and staged in a small numpy scratch before the solve - the
    (K+1)^2 assembly is host-side bookkeeping over already-read Python
    floats, and doing it element by element through pybind indexing cost
    more than the dots it was assembling. A singular B drops the oldest
    history pair and retries; B is normalized by its largest element before
    the solve.
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
        self._dots = None  # lazily-created per-pair scalar workspaces for dotc
        # Inner products of live history pairs, keyed by snapshot id. A step
        # adds ONE new snapshot, so only its row of B is new; the rest is
        # cached raw (normalization happens on the per-build copy). Measured
        # on DLPNO's folded loop, rebuilding B from scratch was most of the
        # accelerator's cost.
        self._ids = []
        self._next_id = 0
        self._dot_cache = {}

    @property
    def history_size(self):
        """Snapshots currently held, which ramps to ``k`` and can drop on a singular B."""
        return len(self._T)

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
        ein = _importlib.import_module("einsums")
        np  = _importlib.import_module("numpy")
        la  = ein.linalg

        if self._dots is None:
            self._dots = [ein.zeros((1,), dtype=str(e.dtype)) for _, e in self._pairs]

        # Snapshot the current state: einsums copies, one per pair component.
        # A full history recycles the evicted snapshot's tensors instead of
        # allocating: zeros_like pays an allocation plus a zero-fill pass that
        # the axpby overwrite makes pure waste, once per component per replay.
        recycled = None
        if len(self._T) >= self._k:
            recycled = (self._T.pop(0), self._E.pop(0))
            self._ids.pop(0)
        snap_t, snap_e = [], []
        for c, (t, e) in enumerate(self._pairs):
            st = recycled[0][c] if recycled is not None else ein.zeros_like(t)
            la.axpby(1.0, t, 0.0, st)
            se = recycled[1][c] if recycled is not None else ein.zeros_like(e)
            la.axpby(1.0, e, 0.0, se)
            snap_t.append(st)
            snap_e.append(se)
        self._T.append(snap_t)
        self._E.append(snap_e)
        self._ids.append(self._next_id)
        self._next_id += 1
        live = set(self._ids)
        for key in [key for key in self._dot_cache if key[0] not in live or key[1] not in live]:
            del self._dot_cache[key]

        while len(self._T) >= 2:
            m = len(self._T)
            # B[p,q] = sum over components of Re<e_p, e_q> (dotc conjugates,
            # so this is exact for complex amplitudes). Coefficients are real,
            # so B itself is a real matrix regardless of amplitude dtype. The
            # dots are einsums ops; only cached scalars land in the numpy B,
            # which exists because assembling a (k+1)^2 host-side matrix
            # element by element through pybind indexing swamped the dots.
            B_np = np.zeros((m + 1, m + 1))
            for p in range(m):
                for q in range(p, m):
                    key = (self._ids[p], self._ids[q])
                    v = self._dot_cache.get(key)
                    if v is None:
                        v = 0.0
                        for c, work in enumerate(self._dots):
                            la.dotc(work, self._E[p][c], self._E[q][c])
                            v += complex(work[0]).real
                        self._dot_cache[key] = v
                    B_np[p, q] = B_np[q, p] = v
            scale = float(np.abs(B_np[:m, :m]).max())
            if scale > 0.0:
                B_np[:m, :m] /= scale
            B_np[:m, m] = B_np[m, :m] = -1.0
            B = ein.zeros((m + 1, m + 1), dtype="float64")
            np.asarray(B)[...] = B_np
            rhs = ein.zeros((m + 1,), dtype="float64")
            rhs[m] = -1.0

            try:
                la.gesv(B, rhs)
            except RuntimeError:
                # Singular subspace: forget the oldest pair and retry.
                self._T.pop(0)
                self._E.pop(0)
                self._ids.pop(0)
                continue

            # t <- sum_s c_s T_s, in place on the live amplitude tensors.
            coeff = np.asarray(rhs).copy()
            for c, (t, _) in enumerate(self._pairs):
                la.axpby(float(coeff[0]), self._T[0][c], 0.0, t)
                for s in range(1, m):
                    la.axpby(float(coeff[s]), self._T[s][c], 1.0, t)
            return


# Which C++ accelerator class backs which element type, and which operand
# classes the C++ path accepts. Anything else - tiled amplitudes, a pair whose
# amplitude and step disagree on dtype - routes to ``_PyDiis``, which is
# type-agnostic because it only ever calls einsums operations on the operands.
_DIIS_CLASS_FOR_DTYPE = {
    "float32": "DiisAcceleratorF",
    "float64": "DiisAcceleratorD",
    "complex64": "DiisAcceleratorC",
    "complex128": "DiisAcceleratorZ",
}

_DIIS_DTYPE_FOR_OPERAND = {
    "RuntimeTensorF": "float32", "RuntimeTensorViewF": "float32",
    "RuntimeTensorD": "float64", "RuntimeTensorViewD": "float64",
    "RuntimeTensorC": "complex64", "RuntimeTensorViewC": "complex64",
    "RuntimeTensorZ": "complex128", "RuntimeTensorViewZ": "complex128",
}


def _diis_cpp_dtype(pairs):
    """The single element type the C++ accelerator would carry, or ``None``.

    ``None`` means the pair list is outside what the C++ instantiations cover
    and the caller should fall back to :class:`_PyDiis`.
    """
    dtypes = set()
    for entry in pairs:
        for operand in entry:
            dtype = _DIIS_DTYPE_FOR_OPERAND.get(type(operand).__name__)
            if dtype is None:
                return None
            dtypes.add(dtype)
    return dtypes.pop() if len(dtypes) == 1 else None


class DIISAccelerator:
    """Pulay DIIS extrapolation for fixed-point iterations captured as graph loops.

    Accelerates the ``t <- t + step`` update a loop body performs by keeping a
    short history of (amplitude, step) snapshots and replacing the amplitudes
    with the least-squares extrapolant between replays. The error vector is the
    update step itself, the standard coupled-cluster choice; iterations whose
    update the body computes anyway get DIIS for the cost of a K-sized host-side
    solve.

    Construct via :func:`diis` and install with :meth:`wrap`::

        acc = cg.diis([(t1, rd1), (t2, rd2)], k=8)
        body = g.add_loop("iter", 100, acc.wrap(converged))

    The step itself is one call into ``DiisAccelerator<T>``: the snapshot
    copies, the new row of B, the normalized bordered solve and the
    extrapolation all happen in C++, where they cost no per-operand pybind
    dispatch. What crosses the boundary per step is this one call, whatever the
    pair count - and a DLPNO-(T) iteration hands the accelerator several hundred
    pairs. The algorithm is unchanged from :class:`_PyDiis` op for op, which is
    what the differential tests pin.

    Every ``(amplitude, step)`` tensor must be readable between replays: the
    user-visible amplitudes qualify, and the step tensors must be EAGER
    (process-owned), not graph-owned intermediates, which are invisible outside
    the graph. The accelerator binds to the storage the operands hold when it is
    constructed, so an operand must not be rebound or resized afterwards; this
    object keeps a reference to every one of them, so they stay alive.
    """

    def __init__(self, pairs, k=8):
        if k < 2:
            raise ValueError(f"DIIS needs a history of at least 2, got k={k}")
        self._pairs = list(pairs)
        if not self._pairs:
            raise ValueError("DIIS needs at least one (amplitude, step) pair")
        dtype = _diis_cpp_dtype(self._pairs)
        if dtype is None:
            raise TypeError("the C++ DIIS accelerator needs dense runtime tensors of one dtype")
        core = _core()
        self._acc = getattr(core, _DIIS_CLASS_FOR_DTYPE[dtype])(k)
        for amplitude, step in self._pairs:
            core.diis_add_pair(self._acc, amplitude, step)

    @property
    def history_size(self):
        """Snapshots currently held, which ramps to ``k`` and can drop on a singular B."""
        return self._acc.history_size

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
        self._acc.step()


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

    Pairs of dense runtime tensors of one dtype get the C++ accelerator.
    Anything else - tiled amplitudes, mixed dtypes - gets :class:`_PyDiis`,
    which computes the same thing through per-operand Python calls.
    """
    pairs = list(pairs)
    if k < 2:
        raise ValueError(f"DIIS needs a history of at least 2, got k={k}")
    if not pairs:
        raise ValueError("DIIS needs at least one (amplitude, step) pair")
    if _diis_cpp_dtype(pairs) is None:
        return _PyDiis(pairs, k)
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
