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


def is_contract(tp) -> bool:
    """Whether *tp* is a type declared with :func:`contract`."""
    return getattr(tp, _CONTRACT_FLAG, False) is True


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


def contract(cls):
    """Declare a frozen dataclass as a cross-boundary contract type.

    Validates every field against the closed type list at class-definition
    time, so a contract that cannot cross the boundary is rejected where it is
    written rather than where it is used.
    """
    if not _dataclasses.is_dataclass(cls):
        raise ContractError(
            f"@contract requires a dataclass; apply @dataclass(frozen=True) to {cls.__name__} first"
        )
    if not cls.__dataclass_params__.frozen:
        raise ContractError(f"@contract requires frozen=True on {cls.__name__}")

    setattr(cls, _CONTRACT_FLAG, True)

    hints = _resolve_hints(cls)
    for f in _dataclasses.fields(cls):
        check_type(
            hints.get(f.name, f.type),
            where=f"{cls.__name__}.{f.name}",
            output=False,
        )
    return cls


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
            raise ContractError(
                f"{qual}: parameter '{pname}' has no annotation, so the stage cannot state its "
                f"contract. Annotate it, or mark the stage promotable=False."
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
