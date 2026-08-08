#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Read a stages module and describe the C++ side of it.

This is the analysis half of ``promote``; :mod:`einsums.stages._emit` renders
what it produces. Everything here is runtime introspection - no libclang, no
parsing of C++ - because Python owns the contract and the C++ is downstream of
it.

Two things are worth knowing before reading further.

**It operates on the module, never on one stage.** A contract's binding
strategy depends on which direction it crosses: a ``type_caster`` converts a
Python object *into* C++ and cannot hand one back, while a ``py::class_``
exposes a C++ object to Python and cannot accept a Python dataclass. So a
contract used as an argument needs a caster, one that is returned needs a
class, and one used both ways needs both - and which of those is true is only
knowable with every stage in hand. A stage name on the command line filters
which skeletons get written; it is not a unit of generation.

**Python owns the prose too, not only the types.** The generated header carries
the contract docstrings and the ``#:`` field comments from the Python source,
so the rationale lives in one place and a regenerated header cannot silently
lose it. ``#:`` comments survive nowhere in the object model, which is why this
module reaches for :func:`inspect.getsource` and :mod:`ast` rather than
``__doc__``.
"""

import ast as _ast
import dataclasses as _dataclasses
import inspect as _inspect
import re as _re
import textwrap as _textwrap
import typing as _typing

from ._contract import (
    TensorC,
    TensorD,
    TensorF,
    TensorZ,
    _IndexMarker,
    _resolve_hints,
    _strip_annotated,
    is_contract,
    parallel_groups,
)
from ._registry import Stage

__all__ = [
    "PromoteError",
    "ContractModel",
    "FieldModel",
    "ParamModel",
    "StageModel",
    "ModuleModel",
    "build_model",
    "classify_stages",
    "contract_includes",
    "load_stages_module",
    "translate",
]


class PromoteError(RuntimeError):
    """The stages module cannot be promoted as it stands."""


# ----------------------------------------------------------------------
# Type translation
# ----------------------------------------------------------------------
_TENSOR_CPP = {
    TensorF: ("einsums::RuntimeTensor<float>", "Einsums/Tensor/RuntimeTensor.hpp"),
    TensorD: ("einsums::RuntimeTensor<double>", "Einsums/Tensor/RuntimeTensor.hpp"),
    TensorC: ("einsums::RuntimeTensor<std::complex<float>>", "Einsums/Tensor/RuntimeTensor.hpp"),
    TensorZ: ("einsums::RuntimeTensor<std::complex<double>>", "Einsums/Tensor/RuntimeTensor.hpp"),
}

_SCALAR_CPP = {
    int: ("std::int64_t", "cstdint"),
    float: ("double", None),
    bool: ("bool", None),
    str: ("std::string", "string"),
}


@_dataclasses.dataclass(frozen=True)
class Translated:
    """One Python annotation, as C++."""

    cpp: str
    #: Headers the type needs, as include spellings without the angle brackets.
    includes: frozenset
    #: Contract types mentioned anywhere inside, outermost first.
    contracts: tuple
    #: Whether a parameter of this type is passed by value rather than by
    #: ``const &``. True only for the arithmetic scalars, where a reference
    #: would be strictly more work for the callee.
    by_value: bool


def translate(tp, *, where: str) -> Translated:
    """Map a validated cross-boundary annotation onto its C++ type.

    The mapping is the closed list in :mod:`einsums.stages._contract`; anything
    outside it has already been refused by ``@stage`` at decoration time. This
    raises anyway rather than emitting something plausible, because a generator
    that guesses produces C++ that compiles and means the wrong thing.
    """
    base, meta = _strip_annotated(tp)

    if any(isinstance(m, _IndexMarker) for m in meta):
        return Translated("std::size_t", frozenset({"cstddef"}), (), True)

    if base in _TENSOR_CPP:
        cpp, header = _TENSOR_CPP[base]
        includes = {header}
        if base in (TensorC, TensorZ):
            includes.add("complex")
        return Translated(cpp, frozenset(includes), (), False)

    if base in _SCALAR_CPP:
        cpp, header = _SCALAR_CPP[base]
        return Translated(
            cpp,
            frozenset() if header is None else frozenset({header}),
            (),
            base is not str,
        )

    origin = _typing.get_origin(base)
    args = _typing.get_args(base)

    if origin in (list, _typing.List):
        inner = translate(args[0], where=f"{where}[]")
        return Translated(
            f"std::vector<{inner.cpp}>",
            inner.includes | {"vector"},
            inner.contracts,
            False,
        )

    if origin in (tuple, _typing.Tuple):
        first = translate(args[0], where=f"{where}[0]")
        second = translate(args[1], where=f"{where}[1]")
        return Translated(
            f"std::pair<{first.cpp}, {second.cpp}>",
            first.includes | second.includes | {"utility"},
            first.contracts + second.contracts,
            False,
        )

    if is_contract(base):
        return Translated(base.__name__, frozenset(), (base,), False)

    raise PromoteError(
        f"{where}: {getattr(base, '__name__', base)!r} has no C++ spelling. "
        f"@stage should have refused this at decoration time; if it did not, that is a bug "
        f"in the contract validator rather than in this signature."
    )


# ----------------------------------------------------------------------
# Docstrings and `#:` comments
# ----------------------------------------------------------------------
_RST_ROLE = _re.compile(r":(?:attr|class|meth|func|mod|data|obj|exc|ref):`~?([^`]+)`")
#: Only the parameter section is consumed, because it is the only one with a
#: Doxygen command to become. Returns, Raises and Notes stay in the body under
#: their own headings: carrying them across verbatim loses nothing, whereas
#: inventing a mapping would be guessing at what the author meant.
_ARGS_SECTION = _re.compile(r"^(?:Args|Arguments|Parameters):\s*$")
_OTHER_SECTION = _re.compile(r"^(?:Returns|Yields|Raises|Note|Notes|Example|Examples):\s*$")
_ARG_ENTRY = _re.compile(r"^ {1,4}(\*{0,2}\w+)\s*(?:\([^)]*\))?:\s*(.*)$")


def to_doxygen(text: str | None) -> list[str]:
    """Reflow a Python docstring into Doxygen comment body lines.

    Sphinx roles and double-backtick literals become single-backtick literals,
    which is how the hand-written headers spell them and what Doxygen's
    markdown reads. Nothing else is translated: the point is to carry the
    author's words across, not to reformat them.
    """
    if not text:
        return []
    text = _inspect.cleandoc(text)
    text = _RST_ROLE.sub(r"`\1`", text)
    text = _re.sub(r"``([^`]+)``", r"`\1`", text)
    return [line.rstrip() for line in text.split("\n")]


def split_docstring(doc: str | None) -> tuple[str, list[str], dict[str, list[str]]]:
    """``(summary, body lines, per-parameter lines)`` from a Google-style docstring.

    The ``Args:`` section becomes the generated ``@param`` prose, which is the
    only way parameter documentation can reach the C++ side: an annotation
    carries a type and nothing else.
    """
    lines = to_doxygen(doc)
    if not lines:
        return "", [], {}

    summary = lines[0]
    rest = lines[1:]
    while rest and not rest[0]:
        rest.pop(0)

    body: list[str] = []
    params: dict[str, list[str]] = {}
    in_args = False
    current = None
    for line in rest:
        stripped = line.strip()
        if _ARGS_SECTION.match(stripped):
            in_args, current = True, None
            continue
        if _OTHER_SECTION.match(stripped):
            in_args, current = False, None
            body.append(line)
            continue
        if in_args:
            entry = _ARG_ENTRY.match(line)
            if entry:
                current = entry.group(1).lstrip("*")
                params[current] = [entry.group(2).strip()] if entry.group(2).strip() else []
            elif current is not None and stripped:
                params[current].append(stripped)
            elif not stripped:
                # A blank line ends the section; anything after it is prose
                # again rather than a continuation of the last parameter.
                in_args, current = False, None
            continue
        body.append(line)

    while body and not body[-1]:
        body.pop()
    return summary, body, params


def field_comments(cls) -> dict[str, list[str]]:
    """The ``#:`` comments attached to each field, by field name.

    Sphinx's convention, already used by the DLPNO contracts. It survives
    nowhere in the object model, so this reads the source: ``ast`` for where
    each field is declared, then a walk back up through the contiguous ``#:``
    lines above it.
    """
    try:
        source = _textwrap.dedent(_inspect.getsource(cls))
    except (OSError, TypeError):
        return {}
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return {}
    if not tree.body or not isinstance(tree.body[0], _ast.ClassDef):
        return {}
    body = tree.body[0].body
    lines = source.split("\n")

    out: dict[str, list[str]] = {}
    for node in body:
        if not isinstance(node, _ast.AnnAssign) or not isinstance(node.target, _ast.Name):
            continue
        collected = []
        i = node.lineno - 2  # 0-based index of the line above the declaration
        while i >= 0 and lines[i].lstrip().startswith("#:"):
            collected.append(lines[i].lstrip()[2:].strip())
            i -= 1
        if collected:
            joined = " ".join(reversed(collected))
            out[node.target.id] = to_doxygen(joined)
    return out


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------
@_dataclasses.dataclass(frozen=True)
class FieldModel:
    name: str
    cpp: str
    doc: list[str]


@_dataclasses.dataclass(frozen=True)
class ContractModel:
    """One ``@contract`` dataclass, and how the stage set uses it."""

    py_type: type
    name: str
    doc: list[str]
    fields: tuple
    parallel: tuple
    #: True when some stage takes it as an argument, so it needs a caster.
    as_argument: bool
    #: True when some stage returns it, so it needs a ``py::class_``.
    as_return: bool


@_dataclasses.dataclass(frozen=True)
class ParamModel:
    name: str
    cpp: str
    by_value: bool
    doc: list[str]

    @property
    def decl(self) -> str:
        return f"{self.cpp} {self.name}" if self.by_value else f"{self.cpp} const &{self.name}"


@_dataclasses.dataclass(frozen=True)
class StageModel:
    """One promotable stage, as the C++ function it becomes."""

    name: str
    #: CamelCase of the stage name; the header, the source file and nothing else.
    unit: str
    summary: str
    doc: list[str]
    params: tuple
    return_cpp: str
    includes: frozenset
    #: The Python source of the body, for the skeleton's TODO comment.
    python_source: str

    @property
    def entry_point(self) -> str:
        return f"stage_{self.name}"


@_dataclasses.dataclass(frozen=True)
class ModuleModel:
    """Everything the emitters need, and nothing that requires re-introspection."""

    source_module: str
    namespace: str
    module_name: str
    stages: tuple
    contracts: tuple

    @property
    def argument_contracts(self) -> tuple:
        return tuple(c for c in self.contracts if c.as_argument)

    @property
    def return_contracts(self) -> tuple:
        return tuple(c for c in self.contracts if c.as_return)


def _camel(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


def _python_source(fn) -> str:
    try:
        return _textwrap.dedent(_inspect.getsource(fn))
    except (OSError, TypeError):
        return ""


def _contract_model(cls, *, as_argument: bool, as_return: bool) -> ContractModel:
    hints = _resolve_hints(cls)
    comments = field_comments(cls)
    fields = []
    for f in _dataclasses.fields(cls):
        translated = translate(hints[f.name], where=f"{cls.__name__}.{f.name}")
        fields.append(FieldModel(f.name, translated.cpp, comments.get(f.name, [])))
    return ContractModel(
        py_type=cls,
        name=cls.__name__,
        doc=to_doxygen(cls.__doc__),
        fields=tuple(fields),
        parallel=parallel_groups(cls),
        as_argument=as_argument,
        as_return=as_return,
    )


def contract_includes(model: ContractModel) -> frozenset:
    """Headers the aggregate's own fields need."""
    hints = _resolve_hints(model.py_type)
    includes: frozenset = frozenset()
    for f in _dataclasses.fields(model.py_type):
        includes |= translate(hints[f.name], where=f"{model.name}.{f.name}").includes
    return includes


def build_model(
    module,
    *,
    namespace: str | None = None,
    module_name: str | None = None,
) -> ModuleModel:
    """Describe the C++ side of every contracted stage *module* declares.

    Args:
        module: The imported stages module.
        namespace: C++ namespace for the generated code. Defaults to the top
            level of the Python package the stages live in.
        module_name: Name of the compiled pybind module. Defaults to
            ``<namespace>_stages``.

    Raises:
        PromoteError: When the module declares no promotable stage, or when a
            stage's contract cannot be spelled in C++.
    """
    ns = namespace or module.__name__.split(".")[0]
    mod_name = module_name or f"{ns}_stages"

    promotable, skipped = classify_stages(module)
    if not promotable:
        raise PromoteError(
            f"{module.__name__} declares no stage with a contract, so there is nothing to "
            f"generate. " + _skip_summary(skipped)
        )

    stage_models = []
    # Direction is a property of the whole stage set, so both passes run over
    # every stage before any contract model is built.
    argument_contracts: dict[type, None] = {}
    return_contracts: dict[type, None] = {}
    ordered: dict[type, None] = {}

    for st in promotable:
        fn = st.backends["python"]
        hints = _resolve_hints(fn)
        sig = _inspect.signature(fn)
        summary, body, param_docs = split_docstring(fn.__doc__)

        includes: frozenset = frozenset()
        params = []
        for pname in sig.parameters:
            translated = translate(hints[pname], where=f"{st.name}({pname})")
            includes |= translated.includes
            params.append(
                ParamModel(pname, translated.cpp, translated.by_value, param_docs.get(pname, []))
            )
            for c in translated.contracts:
                argument_contracts.setdefault(c, None)
                ordered.setdefault(c, None)

        ret = hints["return"]
        if ret is None or ret is type(None):
            return_cpp = "void"
        else:
            translated = translate(ret, where=f"{st.name}() -> ")
            includes |= translated.includes
            return_cpp = translated.cpp
            for c in translated.contracts:
                return_contracts.setdefault(c, None)
                ordered.setdefault(c, None)

        stage_models.append(
            StageModel(
                name=st.name,
                unit=_camel(st.name),
                summary=summary,
                doc=body,
                params=tuple(params),
                return_cpp=return_cpp,
                includes=includes,
                python_source=_python_source(fn),
            )
        )

    contracts = tuple(
        _contract_model(c, as_argument=c in argument_contracts, as_return=c in return_contracts)
        for c in _dependency_order(ordered)
    )
    return ModuleModel(
        source_module=module.__name__,
        namespace=ns,
        module_name=mod_name,
        stages=tuple(stage_models),
        contracts=contracts,
    )


def _dependency_order(contracts) -> list:
    """Contracts ordered so a nested one is declared before its container."""
    seen: dict[type, None] = {}

    def visit(cls):
        if cls in seen:
            return
        hints = _resolve_hints(cls)
        for f in _dataclasses.fields(cls):
            for nested in translate(hints[f.name], where=f"{cls.__name__}.{f.name}").contracts:
                visit(nested)
        seen.setdefault(cls, None)

    for cls in contracts:
        visit(cls)
    return list(seen)


def classify_stages(module) -> tuple[list, list]:
    """``(promotable, skipped)`` stages the module exposes, in declaration order.

    A skipped entry is ``(stage, reason)``. Both halves are reported, because a
    stage silently absent from the generated module is indistinguishable from
    one the generator failed on.
    """
    promotable, skipped = [], []
    seen = set()
    for value in vars(module).values():
        st = getattr(value, "stage", None)
        if not isinstance(st, Stage) or id(st) in seen:
            continue
        seen.add(id(st))
        if not st.promotable:
            skipped.append((st, "python-only"))
        elif not st.contracted:
            skipped.append((st, "no contract"))
        else:
            promotable.append(st)
    return promotable, skipped


def _skip_summary(skipped) -> str:
    if not skipped:
        return "It declares no stages at all."
    parts = ", ".join(f"{st.name} ({reason})" for st, reason in skipped)
    return (
        f"Skipped: {parts}. A stage states its contract by dropping contract=False from @stage "
        f"and annotating its signature with cross-boundary types."
    )


def load_stages_module(target: str):
    """Import a stages module given a file path or a dotted name.

    A path is resolved to the dotted name its package structure implies, and
    the package root's parent goes on ``sys.path``. Importing the file directly
    would break every relative import in it, and a stages module that imports
    its own contracts module is the normal case rather than the exotic one.
    """
    import importlib
    import pathlib
    import sys

    path = pathlib.Path(target)
    if path.suffix == ".py" or path.exists():
        path = path.resolve()
        if not path.is_file():
            raise PromoteError(f"{target}: not a file")
        parts = [path.stem]
        parent = path.parent
        while (parent / "__init__.py").exists():
            parts.append(parent.name)
            parent = parent.parent
        sys.path.insert(0, str(parent))
        dotted = ".".join(reversed(parts))
    else:
        dotted = target

    try:
        return importlib.import_module(dotted)
    except ImportError as exc:
        raise PromoteError(f"cannot import {dotted!r}: {exc}") from exc
