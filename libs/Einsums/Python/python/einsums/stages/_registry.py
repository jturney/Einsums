#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The stage registry, backend selection, and the dispatching wrapper.

A stage is a free function with one or two backends, ``python`` and ``cpp``.
Selection is always explicit: a stage runs the backend it was told to run, and
a missing one is an error rather than a silent fall back to the other. A build
glitch must never quietly change which code computed a published number.
"""

import functools as _functools
import logging as _logging
import os as _os
import threading as _threading

from ._contract import ContractError, DeferredHints, validate_signature

__all__ = [
    "stage",
    "StageError",
    "select",
    "select_default",
    "selected_backend",
    "registered_stages",
    "get_stage",
    "load_stage_module",
]

_log = _logging.getLogger("einsums.stages")

PYTHON = "python"
CPP = "cpp"
_BACKENDS = (PYTHON, CPP)

_ENV_VAR = "EINSUMS_STAGE_BACKEND"


class StageError(RuntimeError):
    """A stage was selected, declared, or dispatched incorrectly."""


class Stage:
    """One registered stage: its contract, its backends, and its selection."""

    __slots__ = (
        "name",
        "qualname",
        "eager",
        "promotable",
        "backends",
        "_selected",
        "_validated",
        "_logged",
        "_python_fn",
    )

    def __init__(self, fn, *, eager: bool, promotable: bool):
        self.name = fn.__name__
        self.qualname = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"
        self.eager = eager
        self.promotable = promotable
        self.backends = {PYTHON: fn}
        self._python_fn = fn
        self._selected: str | None = None
        self._validated = promotable is False  # nothing to validate when never promoted
        self._logged = False

    # -- contract -------------------------------------------------------
    def validate(self) -> None:
        """Run contract validation, tolerating a not-yet-defined annotation."""
        if self._validated:
            return
        try:
            validate_signature(self._python_fn)
        except DeferredHints:
            # A contract defined below the stage that uses it. Leave the stage
            # unvalidated and retry on first call, by which point the module
            # has finished importing.
            return
        self._validated = True

    # -- backends -------------------------------------------------------
    def add_backend(self, name: str, fn) -> None:
        if name not in _BACKENDS:
            raise StageError(f"unknown backend {name!r}; expected one of {_BACKENDS}")
        if name in self.backends and name != PYTHON:
            raise StageError(f"stage {self.name!r} already has a {name} backend")
        self.backends[name] = fn

    def select(self, backend: str) -> None:
        if backend not in _BACKENDS:
            raise StageError(f"unknown backend {backend!r}; expected one of {_BACKENDS}")
        if backend not in self.backends:
            raise StageError(
                f"stage {self.name!r} has no {backend} backend "
                f"(registered: {', '.join(sorted(self.backends))}). "
                f"Load the stage module before selecting it."
            )
        self._selected = backend
        self._logged = False

    @property
    def selected(self) -> str:
        return self._selected or PYTHON

    # -- dispatch -------------------------------------------------------
    def call(self, *args, **kwargs):
        if not self._validated:
            # Retry deferred validation now that the module is fully imported.
            validate_signature(self._python_fn)
            self._validated = True
        backend = self.selected
        fn = self.backends.get(backend)
        if fn is None:
            raise StageError(
                f"stage {self.name!r} selected backend {backend!r}, which is not registered"
            )
        if not self._logged:
            _log.info("stage %s -> %s backend", self.name, backend)
            self._logged = True

        from ._session import active_session

        s = active_session()
        if s is None:
            return fn(*args, **kwargs)
        return s.run_stage(self, fn, args, kwargs)

    def __repr__(self) -> str:
        return (
            f"<Stage {self.name} backends={sorted(self.backends)} "
            f"selected={self.selected}{' eager' if self.eager else ''}"
            f"{'' if self.promotable else ' python-only'}>"
        )


class _Registry:
    """Process-wide stage table. Guarded because stage modules may load late."""

    def __init__(self):
        self._lock = _threading.RLock()
        self._stages: dict[str, Stage] = {}
        self._env_applied = False

    def add(self, st: Stage) -> Stage:
        with self._lock:
            existing = self._stages.get(st.name)
            if existing is not None and existing.qualname != st.qualname:
                raise StageError(
                    f"stage name {st.name!r} is already registered by {existing.qualname}; "
                    f"stage names are process-wide because selection addresses them by name"
                )
            self._stages[st.name] = st
            return st

    def get(self, name: str) -> Stage:
        with self._lock:
            try:
                return self._stages[name]
            except KeyError:
                known = ", ".join(sorted(self._stages)) or "none"
                raise StageError(f"no stage named {name!r} (registered: {known})") from None

    def all(self) -> dict[str, Stage]:
        with self._lock:
            return dict(self._stages)

    def apply_env_once(self) -> None:
        """Apply ``EINSUMS_STAGE_BACKEND`` the first time anything reads selection."""
        with self._lock:
            if self._env_applied:
                return
            self._env_applied = True
        spec = _os.environ.get(_ENV_VAR)
        if spec:
            try:
                apply_backend_spec(spec, source=_ENV_VAR)
            except StageError as exc:
                # Env selection is applied lazily, at whatever call first reads
                # selection, so without this the error reads as a complaint
                # about that call rather than about the environment.
                raise StageError(f"{_ENV_VAR}={spec!r}: {exc}") from exc


_registry = _Registry()


def registered_stages() -> dict[str, Stage]:
    """Every registered stage, by name."""
    return _registry.all()


def get_stage(name: str) -> Stage:
    """The registered stage called *name*."""
    return _registry.get(name)


def stage(fn=None, *, eager: bool = False, promotable: bool = True, loop=None):
    """Register a free function as a stage.

    Usable bare or with arguments::

        @stage
        def prep_sparsity(ref: Reference) -> Domains: ...

        @stage(eager=True)
        def diagonalize(...) -> ...: ...

        @stage(promotable=False)
        def write_checkpoint(...) -> None: ...

    The signature is validated against the closed cross-boundary type list at
    decoration time unless ``promotable=False``, which declares a stage that is
    never intended to reach C++. That is a statement of intent, not an escape
    hatch: such stages are marked in the timing report so a reader can see
    which ones are off the promotion path on purpose.

    Args:
        eager: Force a graph split. The session executes the pending segment,
            runs this stage eagerly, and opens the next segment.
        promotable: False declares a permanently-Python stage and skips
            contract validation.
        loop: Reserved for iterative stages, which map onto
            ``Pipeline::add_loop``. Not yet implemented.
    """
    if loop is not None:
        raise NotImplementedError("loop stages arrive with the iterative-stage work in M6")

    def decorate(f):
        st = Stage(f, eager=eager, promotable=promotable)
        st.validate()
        _registry.add(st)

        @_functools.wraps(f)
        def wrapper(*args, **kwargs):
            return st.call(*args, **kwargs)

        wrapper.stage = st
        return wrapper

    if fn is None:
        return decorate
    if not callable(fn):
        raise StageError("@stage takes no positional arguments; use @stage(eager=True)")
    return decorate(fn)


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
def select(**kwargs: str) -> None:
    """Select a backend per stage: ``select(pno_transform="cpp")``.

    Raises immediately when the stage or the backend is unknown, rather than at
    first call, so a typo in a driver flag fails before any work is done.
    """
    _registry.apply_env_once()
    for name, backend in kwargs.items():
        _registry.get(name).select(backend)


def select_default(backend: str) -> None:
    """Select *backend* for every stage that has it.

    Errors listing the stages that do not, so "all-cpp" is loud rather than
    partial: a run that silently left half the stages in Python is a run whose
    numbers mean something other than what the flag said.
    """
    _registry.apply_env_once()
    if backend not in _BACKENDS:
        raise StageError(f"unknown backend {backend!r}; expected one of {_BACKENDS}")
    stages = _registry.all()
    missing = sorted(n for n, st in stages.items() if backend not in st.backends)
    if missing:
        raise StageError(
            f"select_default({backend!r}) but these stages have no {backend} backend: "
            f"{', '.join(missing)}. Load the stage module that provides them, or select "
            f"the remaining stages individually."
        )
    for st in stages.values():
        st.select(backend)


def selected_backend(name: str) -> str:
    """The backend stage *name* will dispatch to."""
    _registry.apply_env_once()
    return _registry.get(name).selected


# ----------------------------------------------------------------------
# Compiled stage modules
# ----------------------------------------------------------------------
def load_stage_module(module_name: str, *, prefix: str = "stage_"):
    """Import a compiled stage module and register its exports as backend ``cpp``.

    The module is handed to :func:`einsums.sealed.verify_stage_module` before
    anything is read out of it. A module bound to a different libEinsums, or
    compiled against different headers, is refused rather than loaded: it would
    otherwise hand typed Einsums objects across a world boundary, which is the
    one thing the sealed-world rule forbids.

    A refusal is a hard error, never a quiet fall back to the python backend. A
    build glitch must not silently change which code computed a published
    number, so the failure has to be the loud kind.

    Every exported callable named ``<prefix><stage>`` registers as the ``cpp``
    backend of the stage already declared in Python under that name. Matching
    is by name, and a C++ function with no Python counterpart is an error: the
    contract lives on the Python side, so a stage the Python does not declare
    has no contract to be checked against.

    Args:
        module_name: Importable name of the compiled module.
        prefix: Naming convention for exported stage entry points.

    Returns:
        The imported module.
    """
    import importlib

    from .. import sealed

    module = importlib.import_module(module_name)
    sealed.verify_stage_module(module)

    exported = {
        attr[len(prefix):]: getattr(module, attr)
        for attr in dir(module)
        if attr.startswith(prefix) and callable(getattr(module, attr))
    }
    if not exported:
        raise StageError(
            f"{module_name} exports no {prefix}* entry points, so it registers no stages. "
            f"Check that its bindings.cpp was generated by 'python -m einsums.stages promote'."
        )

    known = _registry.all()
    orphans = sorted(set(exported) - set(known))
    if orphans:
        raise StageError(
            f"{module_name} exports {', '.join(orphans)} with no Python counterpart. "
            f"The contract lives on the Python side, so every cpp stage needs a @stage "
            f"declaring its signature. Import the Python stages module first if that is "
            f"all this is."
        )

    for name, fn in exported.items():
        known[name].add_backend(CPP, fn)
    _log.info("loaded %s: cpp backend for %s", module_name, ", ".join(sorted(exported)))
    return module


def apply_backend_spec(spec: str, *, source: str = "spec") -> None:
    """Apply a backend spec string.

    Two forms, matching what a ``--backend`` driver flag would forward:
    ``cpp`` selects that backend everywhere, and
    ``stage=cpp,other=python`` selects per stage.
    """
    spec = spec.strip()
    if not spec:
        return
    if "=" not in spec:
        select_default(spec)
        return
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, backend = item.partition("=")
        name, backend = name.strip(), backend.strip()
        if not name or not backend:
            raise StageError(f"{source}: malformed entry {item!r}; expected stage=backend")
        _registry.get(name).select(backend)
