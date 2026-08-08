#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Cross-boundary contract types, their comparison rules, and signature validation.

A stage's signature is its contract with the C++ side. Only a closed list of
types can cross that boundary, so ``@stage`` validates the signature when the
function is decorated rather than waiting for ``promote`` to discover the
problem. A stage that cannot state its contract is cut at the wrong boundary,
and that verdict is worth delivering while the cut is still cheap to move.

The closed list:

===============================  =====================================
Python annotation                C++ type
===============================  =====================================
``TensorF/D/C/Z``                ``RuntimeTensor<float/double/...>``
``int``                          ``int64_t``
``Annotated[int, index]``        ``std::size_t``
``float``                        ``double``
``bool``                         ``bool``
``str``                          ``std::string``
``list[T]``                      ``std::vector<T'>``
``tuple[T1, T2]``                ``std::pair<T1', T2'>``
frozen ``@contract`` dataclass   aggregate struct
===============================  =====================================
"""

import dataclasses as _dataclasses
import functools as _functools
import typing as _typing

from einsums import _core as _c

__all__ = [
    "ContractError",
    "TensorF",
    "TensorD",
    "TensorC",
    "TensorZ",
    "index",
    "contract",
    "cmp",
    "is_contract",
    "comparison_rules",
    "parallel_groups",
    "validate_signature",
]


class ContractError(TypeError):
    """A stage signature or contract field is not expressible across the boundary."""


# ----------------------------------------------------------------------
# Cross-boundary types
# ----------------------------------------------------------------------
# Aliases rather than marker types on purpose: annotating with the real
# runtime class means type checkers and IDEs see the same thing the
# validator does, and there is no parallel type system to keep in sync.
TensorF = _c.RuntimeTensorF
TensorD = _c.RuntimeTensorD
TensorC = _c.RuntimeTensorC
TensorZ = _c.RuntimeTensorZ

_TENSOR_TYPES = {
    TensorF: "RuntimeTensor<float>",
    TensorD: "RuntimeTensor<double>",
    TensorC: "RuntimeTensor<std::complex<float>>",
    TensorZ: "RuntimeTensor<std::complex<double>>",
}

_SCALAR_TYPES = {
    int: "int64_t",
    float: "double",
    bool: "bool",
    str: "std::string",
}


class _IndexMarker:
    """Marker for ``Annotated[int, index]``, which maps to ``std::size_t``."""

    def __repr__(self) -> str:
        return "einsums.stages.index"


index = _IndexMarker()


# ----------------------------------------------------------------------
# Comparison rules
# ----------------------------------------------------------------------
_CMP_KEY = "einsums.stages.cmp"


@_dataclasses.dataclass(frozen=True)
class ComparisonRule:
    """How a differential test compares one contract field across backends."""

    kind: str
    rtol: float | None = None
    atol: float | None = None
    fn: _typing.Callable | None = None

    def __repr__(self) -> str:
        if self.kind == "custom":
            return f"cmp.custom({getattr(self.fn, '__name__', self.fn)})"
        tol = ", ".join(
            f"{n}={v}" for n, v in (("rtol", self.rtol), ("atol", self.atol)) if v is not None
        )
        return f"cmp.{self.kind}({tol})"


def _field(rule: ComparisonRule):
    return _dataclasses.field(metadata={_CMP_KEY: rule})


class cmp:  # noqa: N801  (a namespace, spelled as the design doc spells it)
    """Field-comparison rule factories for ``@contract`` dataclasses.

    Each returns a ``dataclasses.field`` carrying the rule in its metadata, so
    a contract declares how it should be compared at the same place it
    declares its shape::

        @contract
        @dataclass(frozen=True)
        class PnoBasis:
            X_pno: list[TensorD] = cmp.up_to_sign()
            n_pno: list[int]     = cmp.exact()
    """

    @staticmethod
    def exact():
        """Bit-for-bit equality. The default for non-tensor fields."""
        return _field(ComparisonRule("exact"))

    @staticmethod
    def close(rtol: float | None = None, atol: float | None = None):
        """Approximate equality; tolerances default to ``einsums.testing.tolerance_for``."""
        return _field(ComparisonRule("close", rtol=rtol, atol=atol))

    @staticmethod
    def up_to_sign(rtol: float | None = None, atol: float | None = None):
        """Equality allowing a per-column sign flip, for eigenvector-valued fields."""
        return _field(ComparisonRule("up_to_sign", rtol=rtol, atol=atol))

    @staticmethod
    def up_to_subspace(rtol: float | None = None, atol: float | None = None):
        """Equality of spanned subspaces via projectors, for degenerate eigenvectors."""
        return _field(ComparisonRule("up_to_subspace", rtol=rtol, atol=atol))

    @staticmethod
    def custom(fn: _typing.Callable):
        """Compare with ``fn(a, b)``, which raises or returns False on mismatch."""
        return _field(ComparisonRule("custom", fn=fn))


_CONTRACT_FLAG = "__einsums_contract__"
_PARALLEL_FLAG = "__einsums_parallel__"


def is_contract(tp) -> bool:
    """Whether *tp* is a type declared with :func:`contract`."""
    return getattr(tp, _CONTRACT_FLAG, False) is True


def parallel_groups(tp) -> tuple[tuple[str, ...], ...]:
    """Groups of list fields *tp* declares to be the same length."""
    if not is_contract(tp):
        raise ContractError(f"{tp!r} is not a @contract type")
    return getattr(tp, _PARALLEL_FLAG, ())


def comparison_rules(tp) -> dict[str, ComparisonRule]:
    """Per-field comparison rules for a contract type, defaults filled in."""
    if not is_contract(tp):
        raise ContractError(f"{tp!r} is not a @contract type")
    rules = {}
    for f in _dataclasses.fields(tp):
        rule = f.metadata.get(_CMP_KEY)
        if rule is None:
            # Default per the design: close() for anything tensor-valued,
            # exact for everything else.
            rule = ComparisonRule("close") if _mentions_tensor(f.type) else ComparisonRule("exact")
        rules[f.name] = rule
    return rules


def _mentions_tensor(tp) -> bool:
    """Whether *tp* is, or contains, one of the tensor types."""
    tp = _strip_annotated(tp)[0]
    if tp in _TENSOR_TYPES:
        return True
    args = _typing.get_args(tp)
    return any(_mentions_tensor(a) for a in args)


def contract(cls=None, *, parallel=None):
    """Declare a frozen dataclass as a cross-boundary contract type.

    Validates every field against the closed type list at class-definition
    time, so a contract that cannot cross the boundary is rejected where it is
    written rather than where it is used.

    ``parallel`` names groups of list fields that are always the same length -
    a structure of arrays, indexed by one ordinal::

        @contract(parallel=[("pair", "partner", "sign")])
        @dataclass(frozen=True)
        class CouplingPlan: ...

    That is an invariant the type cannot express and the C++ side cannot infer,
    and it is the difference between an exception and a silent out-of-bounds
    read: every loop over such a group indexes all of its arrays with one
    ordinal, so a short one is read past its end. Declared here, it is checked
    when the contract is constructed AND emitted into the generated pybind
    caster by ``promote``, so both sides enforce the same statement.

    Args:
        cls: The dataclass, when used bare as ``@contract``.
        parallel: Groups of field names that must be the same length. A single
            group may be given as one sequence of names.
    """

    def decorate(target):
        if not _dataclasses.is_dataclass(target):
            raise ContractError(
                f"@contract requires a dataclass; apply @dataclass(frozen=True) to {target.__name__} first"
            )
        if not target.__dataclass_params__.frozen:
            raise ContractError(f"@contract requires frozen=True on {target.__name__}")

        setattr(target, _CONTRACT_FLAG, True)

        hints = _resolve_hints(target)
        for f in _dataclasses.fields(target):
            check_type(
                hints.get(f.name, f.type),
                where=f"{target.__name__}.{f.name}",
                output=False,
            )
        _install_parallel(target, hints, parallel)
        return target

    if cls is None:
        return decorate
    if parallel is not None:
        raise ContractError("@contract takes no positional arguments; use @contract(parallel=[...])")
    return decorate(cls)


def _install_parallel(cls, hints, parallel) -> None:
    """Record and enforce the declared parallel-array groups."""
    if parallel is None:
        return
    groups = list(parallel)
    if groups and all(isinstance(g, str) for g in groups):
        groups = [groups]  # one group, given as a bare sequence of names

    names = {f.name for f in _dataclasses.fields(cls)}
    normalized = []
    for group in groups:
        group = tuple(group)
        if len(group) < 2:
            raise ContractError(
                f"{cls.__name__}: a parallel group needs at least two fields, got {group}"
            )
        for n in group:
            if n not in names:
                raise ContractError(f"{cls.__name__}: parallel names {n!r}, which is not a field")
            if _typing.get_origin(_strip_annotated(hints.get(n))[0]) not in (list, _typing.List):
                raise ContractError(
                    f"{cls.__name__}.{n}: only list fields can be parallel; "
                    f"the group states they are always the same length"
                )
        normalized.append(group)
    setattr(cls, _PARALLEL_FLAG, tuple(normalized))

    # Wraps __init__, NOT __post_init__. @dataclass decides whether to emit a
    # __post_init__ call when it generates __init__, and by the time @contract
    # runs that decision is already made: a __post_init__ installed afterwards
    # is simply never called, and the check silently does nothing. Wrapping the
    # constructor also covers dataclasses.replace, which goes through it.
    original_init = cls.__init__

    @_functools.wraps(original_init)
    def __init__(self, *args, **kwargs):  # noqa: N807
        original_init(self, *args, **kwargs)
        for group in normalized:
            head = group[0]
            n = len(getattr(self, head))
            for other in group[1:]:
                m = len(getattr(self, other))
                if m != n:
                    raise ContractError(
                        f"{cls.__name__}: fields {', '.join(group)} are declared parallel, but "
                        f"{head!r} has {n} entries and {other!r} has {m}."
                    )

    cls.__init__ = __init__


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def _strip_annotated(tp):
    """Return ``(base, metadata)`` for an ``Annotated[...]``, else ``(tp, ())``."""
    if _typing.get_origin(tp) is _typing.Annotated:
        base, *meta = _typing.get_args(tp)
        return base, tuple(meta)
    return tp, ()


def _name(tp) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def check_type(tp, *, where: str, output: bool) -> None:
    """Raise :class:`ContractError` unless *tp* can cross the boundary.

    *output* selects the stricter rules that apply to anything a stage
    returns. A captured stage returns before its graph executes, so a computed
    floating-point value does not exist yet; those come back as rank-0 tensors
    instead. Integers, bools and strings stay legal on the way out because a
    graph cannot be built without knowing its own shapes and counts, so that
    kind of metadata is fixed at capture time by construction.
    """
    if tp is None or tp is type(None):
        raise ContractError(f"{where}: a stage must state its type; bare None is not a contract")

    base, meta = _strip_annotated(tp)

    if any(isinstance(m, _IndexMarker) for m in meta):
        if base is not int:
            raise ContractError(f"{where}: index applies to int, not {_name(base)}")
        return

    if base in _TENSOR_TYPES:
        return

    if base in _SCALAR_TYPES:
        if output and base is float:
            raise ContractError(
                f"{where}: a captured stage cannot return a computed float, because it returns "
                f"before its graph executes. Use a rank-0 TensorD written by the scalar-writing "
                f"dot/norm/trace forms and read it after run()."
            )
        return

    if base is complex:
        raise ContractError(
            f"{where}: complex scalars cross as rank-0 TensorC/TensorZ, not as Python complex"
        )

    origin = _typing.get_origin(base)
    args = _typing.get_args(base)

    if origin in (list, _typing.List):
        if len(args) != 1:
            raise ContractError(f"{where}: list must state its element type, e.g. list[TensorD]")
        check_type(args[0], where=f"{where}[]", output=output)
        return

    if origin in (tuple, _typing.Tuple):
        if len(args) != 2 or Ellipsis in args:
            raise ContractError(
                f"{where}: only a 2-tuple crosses the boundary (it maps to std::pair); "
                f"use a @contract dataclass for anything wider"
            )
        for i, a in enumerate(args):
            check_type(a, where=f"{where}[{i}]", output=output)
        return

    if is_contract(base):
        # Fields were validated by @contract. Re-check them here only when the
        # stricter output rules apply, since @contract cannot know the
        # direction a given use will need.
        if output:
            hints = _resolve_hints(base)
            for f in _dataclasses.fields(base):
                check_type(
                    hints.get(f.name, f.type),
                    where=f"{where}.{f.name}",
                    output=True,
                )
        return

    if _dataclasses.is_dataclass(base):
        raise ContractError(
            f"{where}: {_name(base)} is a dataclass but not a @contract; "
            f"add @contract above @dataclass(frozen=True)"
        )

    raise ContractError(
        f"{where}: {_name(base)} cannot cross the stage boundary. "
        f"Allowed: TensorF/D/C/Z, int, float, bool, str, list[T], tuple[T1, T2], "
        f"and frozen @contract dataclasses. Mark the stage promotable=False if it is "
        f"never intended to reach C++."
    )


class DeferredHints(Exception):
    """A signature referenced a name that is not defined yet."""


def _resolve_hints(obj) -> dict:
    """``typing.get_type_hints`` with unresolved forward references surfaced."""
    try:
        return _typing.get_type_hints(obj, include_extras=True)
    except NameError as exc:
        raise DeferredHints(str(exc)) from exc


def validate_signature(fn) -> None:
    """Validate every parameter and the return type of a stage function.

    Raises :class:`ContractError` on a type that cannot cross, or
    :class:`DeferredHints` when a forward reference is not resolvable yet, which
    the registry retries at first call.
    """
    import inspect

    hints = _resolve_hints(fn)
    sig = inspect.signature(fn)
    qual = getattr(fn, "__qualname__", fn.__name__)

    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ContractError(
                f"{qual}: *{pname} cannot cross the stage boundary; a contract needs a fixed "
                f"parameter list. Group the variable part into a @contract dataclass."
            )
        if pname not in hints:
            # functools.wraps copies __annotations__ off the wrapped function,
            # so a @stage under a @functools.wraps loses every annotation the
            # author wrote and lands here claiming they were never there. The
            # error is correct and the cause is not remotely obvious, so say it.
            wrapped = "" if getattr(fn, "__wrapped__", None) is None else (
                f" This function is wrapped (functools.wraps sets __wrapped__), and wraps copies "
                f"__annotations__ from the wrapped function: that silently replaces the "
                f"annotations you wrote here. Drop the wraps, or annotate the wrapped function."
            )
            raise ContractError(
                f"{qual}: parameter '{pname}' has no annotation, so the stage cannot state its "
                f"contract. Annotate it, or mark the stage promotable=False.{wrapped}"
            )
        check_type(hints[pname], where=f"{qual}({pname})", output=False)

    if "return" not in hints:
        raise ContractError(
            f"{qual}: no return annotation. Use '-> None' for a stage that writes through its "
            f"arguments, or annotate what it returns."
        )
    ret = hints["return"]
    if ret is not None and ret is not type(None):
        check_type(ret, where=f"{qual}() -> ", output=True)
