#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Analyze a method for promotion and scaffold its extraction - once the cut is chosen.

This is the generator half of method/state extraction, and it is built around
the lesson both hand promotions taught: a faithful extractor is the wrong tool.
``pno_transform`` touched 32 fields of ``self``, and extracting them faithfully
would have produced exactly the contract that was rejected as too wide; the
shipped stage takes fifteen parameters because a person chose, field by field,
what crosses, what stays in the Python planner, and what stays in the host-side
finish. So this tool refuses to generate anything until that choice has been
written down.

The workflow has three steps, and the middle one is the point:

1. ``extract mymethod/solver.py::Solver.build_t2`` reports every ``self`` field
   and helper the method touches - a static AST pass, following ``self.helper()``
   calls, so it needs no build and runs on any source - and writes a cut-spec
   template with every entry marked TODO.
2. The developer fills in a *disposition* per field: ``param`` (crosses into
   the stage), ``plan`` (consumed by the Python planning half), ``finish``
   (stays in the host-side finishing half), or ``returns`` (comes back through
   the contract). Helpers get ``plan``, ``stage`` (the body moves into the free
   function) or ``finish``. ``param`` and ``plan`` entries name the stage
   parameters they become, so a field that flattens into three parallel lists
   is three named parameters traceable to their origin.
3. ``extract`` run again emits one scaffold file - contract skeleton, stage
   signature with the parameters in spec order, plan and finish method
   skeletons - plus the narrowing summary: N fields and H helpers in, P
   parameters and R return fields out.

The scaffold is written once and never rewritten, like promote's port skeleton:
the moment it exists it is the developer's file. It deliberately does not
import until its TODO annotations are resolved, because ``@stage`` refuses an
unannotated signature anyway and a scaffold that imports cleanly reads as more
finished than it is.

What this module never does is guess a type or a disposition. The analysis is
evidence, the spec is judgement, and the two are kept in different files on
purpose.
"""

import ast as _ast
import dataclasses as _dataclasses
import pathlib as _pathlib
import textwrap as _textwrap

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # Python < 3.11
    _tomllib = None

__all__ = [
    "ExtractError",
    "Access",
    "FieldInfo",
    "HelperInfo",
    "Analysis",
    "CutSpec",
    "analyze",
    "load_spec",
    "render_report",
    "render_template",
    "render_scaffold",
]


class ExtractError(RuntimeError):
    """The method cannot be analyzed or the cut spec does not add up."""


#: Method names on a field that mutate it in place. ``self.x.append(...)`` is a
#: write for disposition purposes even though the AST says Load, which is the
#: stated lower-bound limitation of the report script this analysis grew from.
_MUTATORS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse",
    "add", "discard", "update", "setdefault", "popitem", "fill", "resize",
})

_FIELD_DISPOSITIONS = ("param", "plan", "finish", "returns")
_HELPER_DISPOSITIONS = ("plan", "stage", "finish")


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
@_dataclasses.dataclass(frozen=True)
class Access:
    """One place a field is touched."""

    line: int
    kind: str  # "read" | "write"
    #: Dotted path below ``self`` when longer than the field itself, as in
    #: ``cut.t_cut_pno``. The spec stays keyed by the bare field; the chain is
    #: what tells the developer a config aggregate is three scalars in disguise.
    chain: str
    #: Helper the access happens inside, or None for the method body itself.
    via: str | None


@_dataclasses.dataclass
class FieldInfo:
    name: str
    accesses: list = _dataclasses.field(default_factory=list)

    @property
    def written(self) -> bool:
        return any(a.kind == "write" for a in self.accesses)

    @property
    def read(self) -> bool:
        return any(a.kind == "read" for a in self.accesses)

    @property
    def chains(self) -> list:
        seen: dict[str, None] = {}
        for a in self.accesses:
            if a.chain != self.name:
                seen.setdefault(a.chain, None)
        return list(seen)

    @property
    def via(self) -> list:
        seen: dict[str, None] = {}
        for a in self.accesses:
            if a.via:
                seen.setdefault(a.via, None)
        return list(seen)

    @property
    def direct(self) -> bool:
        return any(a.via is None for a in self.accesses)

    def via_text(self) -> str:
        if not self.via:
            return ""
        prefix = "also via " if self.direct else "via "
        return prefix + ", ".join(self.via)


@_dataclasses.dataclass
class HelperInfo:
    name: str
    lines: list = _dataclasses.field(default_factory=list)
    #: False when the helper's definition was not found in any given source,
    #: so its own field accesses are invisible to the analysis.
    followed: bool = True


@_dataclasses.dataclass
class Analysis:
    """Everything the method touches, as evidence for the cut."""

    cls: str
    method: str
    fields: dict
    helpers: dict

    @property
    def reads(self) -> list:
        return [f for f in self.fields.values() if not f.written]

    @property
    def writes(self) -> list:
        return [f for f in self.fields.values() if f.written]

    @property
    def unresolved(self) -> list:
        return [h for h in self.helpers.values() if not h.followed]


def _peel(node):
    """``(root, components, subscripted)`` of an attribute chain, or ``None``.

    Walks ``self.cut.t_cut_pno`` or ``self.stores[b].shape`` down to its root
    name, collecting attribute components outermost-last. Anything other than
    attribute or subscript layers (a call in the middle, a literal) means the
    expression is not a plain chain and the caller falls back to descending.
    """
    components = []
    subscripted = False
    while True:
        if isinstance(node, _ast.Attribute):
            components.append(node.attr)
            node = node.value
        elif isinstance(node, _ast.Subscript):
            subscripted = True
            node = node.value
        elif isinstance(node, _ast.Name):
            return node.id, list(reversed(components)), subscripted
        else:
            return None


class _MethodScan(_ast.NodeVisitor):
    """Collect ``self`` field accesses and helper calls in one function body."""

    def __init__(self, methods, properties=frozenset(), via=None):
        self.methods = methods
        self.properties = properties
        self.via = via
        self.accesses = []
        self.helper_calls = []  # (name, line)
        self.property_reads = []  # names whose bodies still get followed

    # -- recording ------------------------------------------------------
    def _record(self, node, components, kind):
        self.accesses.append(
            Access(node.lineno, kind, ".".join(components), self.via)
        )

    def _self_chain(self, node):
        peeled = _peel(node)
        if peeled and peeled[0] == "self" and peeled[1]:
            return peeled[1], peeled[2]
        return None

    # -- writes ---------------------------------------------------------
    def _visit_target(self, target):
        chain = self._self_chain(target)
        if chain is not None:
            self._record(target, chain[0], "write")
            # Subscript indices are still reads; nothing else under a peeled
            # chain can contain one.
            for sub in _ast.walk(target):
                if isinstance(sub, _ast.Subscript):
                    self.visit(sub.slice)
            return
        if isinstance(target, (_ast.Tuple, _ast.List)):
            for elt in target.elts:
                self._visit_target(elt)
            return
        self.visit(target)

    def visit_Assign(self, node):
        for target in node.targets:
            self._visit_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self._visit_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        chain = self._self_chain(node.target)
        if chain is not None:
            self._record(node.target, chain[0], "read")
            self._record(node.target, chain[0], "write")
            for sub in _ast.walk(node.target):
                if isinstance(sub, _ast.Subscript):
                    self.visit(sub.slice)
        else:
            self.visit(node.target)
        self.visit(node.value)

    def visit_Delete(self, node):
        for target in node.targets:
            self._visit_target(target)

    # -- calls and reads ------------------------------------------------
    def visit_Call(self, node):
        chain = self._self_chain(node.func)
        if chain is not None:
            components, subscripted = chain
            if len(components) == 1 and not subscripted \
                    and components[0] in self.methods:
                self.helper_calls.append((components[0], node.lineno))
            elif len(components) == 2 and components[1] in _MUTATORS:
                self._record(node.func, components[:1], "write")
            else:
                self._record(node.func, components, "read")
        else:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node):
        chain = self._self_chain(node)
        if chain is not None:
            components, _ = chain
            if components[0] in self.properties:
                # A property is state wearing a method's clothes: it needs a
                # disposition like any field, and its body still gets followed
                # so the fields it reads show up as evidence.
                self._record(node, components, "read")
                self.property_reads.append(components[0])
            elif len(components) == 1 and components[0] in self.methods:
                # A bound-method reference without a call, handed somewhere
                # else to run. It will execute, so it is a helper, not state.
                self.helper_calls.append((components[0], node.lineno))
            else:
                self._record(node, components, "read")
            return
        self.generic_visit(node)


def _collect_methods(sources):
    """Every method of every class in *sources*, flat, by name.

    Flat because a promotion target routinely calls helpers defined on a base
    class in another file; the caller passes both files and name collisions
    resolve to the last definition, which matches how the report script this
    grew from behaves.
    """
    methods = {}
    classes = {}
    properties = set()
    for source in sources:
        path = _pathlib.Path(source)
        try:
            tree = _ast.parse(path.read_text())
        except (OSError, SyntaxError) as exc:
            raise ExtractError(f"{source}: {exc}") from exc
        for cls in (n for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)):
            for fn in cls.body:
                if isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    methods[fn.name] = fn
                    classes[fn.name] = cls.name
                    if any(_decorator_name(d) in ("property", "cached_property")
                           for d in fn.decorator_list):
                        properties.add(fn.name)
    return methods, classes, properties


def _decorator_name(node) -> str:
    if isinstance(node, _ast.Name):
        return node.id
    if isinstance(node, _ast.Attribute):
        return node.attr
    return ""


def analyze(target: str, also=()) -> Analysis:
    """Everything ``Class.method`` touches, following helpers transitively.

    Args:
        target: ``path/to/file.py::Class.method``.
        also: Extra source files whose classes provide helper definitions,
            for a method whose helpers live on a base class in another file.
    """
    if "::" not in target:
        raise ExtractError(
            f"{target!r}: expected path/to/file.py::Class.method"
        )
    path, _, qual = target.partition("::")
    if "." not in qual:
        raise ExtractError(f"{qual!r}: expected Class.method after '::'")
    cls_name, _, method_name = qual.partition(".")

    methods, classes, properties = _collect_methods([path, *also])
    fn = methods.get(method_name)
    if fn is None or classes.get(method_name) != cls_name:
        near = ", ".join(sorted(n for n, c in classes.items() if c == cls_name)[:8])
        raise ExtractError(
            f"{cls_name}.{method_name} not found in {path}"
            + (f" (that class defines: {near}, ...)" if near else "")
        )

    fields: dict[str, FieldInfo] = {}
    helpers: dict[str, HelperInfo] = {}
    seen = {method_name}
    queue = [(fn, None)]
    while queue:
        node, via = queue.pop(0)
        scan = _MethodScan(methods, properties, via=via)
        scan.visit(node)
        for access in scan.accesses:
            leaf = access.chain.split(".")[0]
            fields.setdefault(leaf, FieldInfo(leaf)).accesses.append(access)
        for name, line in scan.helper_calls:
            helpers.setdefault(name, HelperInfo(name)).lines.append(line)
            if name not in seen:
                seen.add(name)
                queue.append((methods[name], name))
        for name in scan.property_reads:
            if name not in seen:
                seen.add(name)
                queue.append((methods[name], name))

    return Analysis(cls=cls_name, method=method_name, fields=fields, helpers=helpers)


# ----------------------------------------------------------------------
# The cut spec
# ----------------------------------------------------------------------
@_dataclasses.dataclass(frozen=True)
class Entry:
    disposition: str
    params: tuple
    note: str


@_dataclasses.dataclass(frozen=True)
class CutSpec:
    """The judgement half: what crosses, what stays, and under which names."""

    stage_name: str
    contract_name: str
    #: Contract fields born from locals of the method rather than from a
    #: ``self`` field, as ``K_pno`` and the truncation-energy scalars were.
    extra_returns: tuple
    fields: dict
    helpers: dict

    @property
    def params(self) -> list:
        out = []
        for entry in self.fields.values():
            out.extend(entry.params)
        return out

    @property
    def return_fields(self) -> list:
        out = [name for name, e in self.fields.items() if e.disposition == "returns"]
        out.extend(self.extra_returns)
        return out


def _camel(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


def load_spec(path, analysis: Analysis) -> CutSpec:
    """Read and validate a cut spec against what the method actually touches.

    Every complaint is collected and raised at once, because a spec is edited
    as a document and a tool that reveals one problem per run is a tool that
    gets eight runs of feedback it could have given in one.
    """
    if _tomllib is None:
        raise ExtractError("reading a cut spec needs Python >= 3.11 (tomllib)")
    try:
        raw = _tomllib.loads(_pathlib.Path(path).read_text())
    except OSError as exc:
        raise ExtractError(f"{path}: {exc}") from exc
    except _tomllib.TOMLDecodeError as exc:
        raise ExtractError(f"{path}: not valid TOML: {exc}") from exc

    problems = []

    stage_table = raw.get("stage", {})
    stage_name = stage_table.get("name", analysis.method)
    contract_name = stage_table.get("contract", _camel(analysis.method) + "Result")
    extra_returns = tuple(stage_table.get("extra_returns", []))

    def read_entries(section, allowed, params_ok):
        entries = {}
        for name, table in raw.get(section, {}).items():
            if not isinstance(table, dict):
                problems.append(f"[{section}.{name}] is not a table")
                continue
            disposition = table.get("disposition", "TODO")
            if disposition == "TODO":
                problems.append(f"[{section}.{name}] still says TODO")
            elif disposition not in allowed:
                problems.append(
                    f"[{section}.{name}] disposition {disposition!r} is not one of "
                    + ", ".join(allowed)
                )
            params = tuple(table.get("params", ()))
            if params and disposition not in params_ok:
                problems.append(
                    f"[{section}.{name}] declares params but its disposition "
                    f"{disposition!r} does not cross anything into the stage"
                )
            entries[name] = Entry(disposition, params, table.get("note", ""))
        return entries

    fields = read_entries("fields", _FIELD_DISPOSITIONS, ("param", "plan"))
    helpers = read_entries("helpers", _HELPER_DISPOSITIONS, ())

    touched = set(analysis.fields)
    stale = sorted(set(fields) - touched)
    missing = sorted(touched - set(fields))
    for name in stale:
        problems.append(f"[fields.{name}] is not touched by {analysis.method}; stale?")
    for name in missing:
        problems.append(f"{analysis.method} touches self.{name}, which the spec does not mention")

    known_helpers = set(analysis.helpers)
    for name in sorted(set(helpers) - known_helpers):
        problems.append(f"[helpers.{name}] is not called by {analysis.method}; stale?")
    for name in sorted(known_helpers - set(helpers)):
        problems.append(f"{analysis.method} calls self.{name}(), which the spec does not mention")

    for name, entry in fields.items():
        info = analysis.fields.get(name)
        if info is None or entry.disposition == "TODO":
            continue
        if entry.disposition == "returns" and not info.written:
            problems.append(
                f"[fields.{name}] says returns, but the method never writes it"
            )
        if entry.disposition == "param" and not info.read:
            problems.append(
                f"[fields.{name}] says param, but the method only writes it; "
                f"an output cannot cross inward"
            )
        if entry.disposition == "param" and not entry.params:
            entry = fields[name] = Entry(entry.disposition, (name,), entry.note)

    seen_params: dict[str, str] = {}
    for name, entry in fields.items():
        for p in entry.params:
            if p in seen_params:
                problems.append(
                    f"parameter {p!r} is declared by both [fields.{seen_params[p]}] "
                    f"and [fields.{name}]"
                )
            seen_params[p] = name
    for extra in extra_returns:
        if extra in fields and fields[extra].disposition == "returns":
            problems.append(
                f"extra_returns names {extra!r}, which [fields.{extra}] already returns"
            )

    if problems:
        raise ExtractError(
            f"{path} does not add up ({len(problems)} problem(s)):\n  "
            + "\n  ".join(problems)
        )
    return CutSpec(
        stage_name=stage_name,
        contract_name=contract_name,
        extra_returns=extra_returns,
        fields=fields,
        helpers=helpers,
    )


# ----------------------------------------------------------------------
# Rendering: the report, the template, the scaffold
# ----------------------------------------------------------------------
def render_report(analysis: Analysis) -> str:
    lines = [
        f"{analysis.cls}.{analysis.method}: "
        f"{len(analysis.fields)} fields "
        f"({len(analysis.reads)} read-only, {len(analysis.writes)} written), "
        f"{len(analysis.helpers)} helpers",
        "",
        f"  {'field':<28} {'access':<7} {'sites':>5}  detail",
    ]
    ordered = sorted(analysis.fields.values(), key=lambda f: (f.written, f.name))
    for f in ordered:
        detail = []
        if f.chains:
            detail.append(", ".join("." + c.partition(".")[2] for c in f.chains))
        if f.via:
            detail.append(f.via_text())
        access = "write" if f.written else "read"
        lines.append(
            f"  {f.name:<28} {access:<7} {len(f.accesses):>5}  {'; '.join(detail)}"
        )
    if analysis.helpers:
        lines.append("")
        lines.append("  helpers: " + ", ".join(sorted(analysis.helpers)))
    if analysis.unresolved:
        lines.append(
            "  NOT FOLLOWED (definition not in the given sources; pass --also): "
            + ", ".join(sorted(h.name for h in analysis.unresolved))
        )
    lines.append("")
    lines.append(
        "Counts are lower bounds: a mutation through a returned view or a helper's "
        "argument shows as a read."
    )
    return "\n".join(lines)


def render_template(analysis: Analysis) -> str:
    """The cut-spec template, every disposition a TODO.

    The template is the tool's whole opinion: it states the vocabulary and the
    evidence and declines to fill in the judgement.
    """
    out = [
        f"# Cut spec for {analysis.cls}.{analysis.method}, written by",
        "#     python -m einsums.stages extract",
        "# Fill in every disposition; extract refuses to scaffold while a TODO remains.",
        "#",
        "# Field dispositions:",
        "#   param   - crosses into the stage; params = [...] names what it becomes",
        "#             (default: the field's own name)",
        "#   plan    - consumed by the Python planning half; params = [...] for stage",
        "#             parameters the planner derives from it",
        "#   finish  - stays in the host-side finishing half",
        "#   returns - written by the method, comes back as a contract field",
        "# Helper dispositions: plan, stage (body moves into the free function), finish.",
        "",
        "[stage]",
        f'name = "{analysis.method}"',
        f'contract = "{_camel(analysis.method)}Result"',
        "# Contract fields built from locals rather than a self field:",
        "extra_returns = []",
    ]
    ordered = sorted(analysis.fields.values(), key=lambda f: (f.written, f.name))
    for f in ordered:
        evidence = "write" if f.written else "read"
        if f.chains:
            evidence += "; " + ", ".join("." + c.partition(".")[2] for c in f.chains)
        if f.via:
            evidence += "; " + f.via_text()
        out.append("")
        out.append(f"[fields.{f.name}]  # {evidence}")
        out.append('disposition = "TODO"')
    for name in sorted(analysis.helpers):
        helper = analysis.helpers[name]
        out.append("")
        suffix = "" if helper.followed else "  # NOT FOLLOWED: definition not found"
        out.append(f"[helpers.{name}]{suffix}")
        out.append('disposition = "TODO"')
    out.append("")
    return "\n".join(out)


def _wrap_comment(text, indent="    "):
    return _textwrap.wrap(text, width=96, initial_indent=indent + "# ",
                          subsequent_indent=indent + "# ")


def _wrap_absorbs(names):
    text = "Absorbs: " + (", ".join(f"self.{n}" for n in names) or "(nothing)") + "."
    return _textwrap.wrap(text, width=96, initial_indent="    ",
                          subsequent_indent="    ")


def render_scaffold(analysis: Analysis, spec: CutSpec) -> str:
    """The extraction, as one file of skeletons that is the developer's to finish.

    Deliberately does not import until every TODO annotation is resolved:
    ``@stage`` would refuse the unannotated signature anyway, and a scaffold
    that imports cleanly reads as more finished than it is.
    """
    params = spec.params
    plan_fields = [n for n, e in spec.fields.items() if e.disposition == "plan"]
    finish_fields = [n for n, e in spec.fields.items() if e.disposition == "finish"]
    stage_helpers = [n for n, e in spec.helpers.items() if e.disposition == "stage"]

    plan_name = f"plan_{spec.stage_name}"
    finish_name = f"_finish_{spec.stage_name}"

    summary = (
        f"{len(analysis.fields)} fields and {len(analysis.helpers)} helpers of "
        f"{analysis.cls}.{analysis.method} narrowed to {len(params)} parameters "
        f"and {len(spec.return_fields)} return fields"
    )

    out = [
        f'"""Extraction scaffold for ``{analysis.cls}.{analysis.method}``: {spec.stage_name}.',
        "",
        summary + ".",
        "",
        "Written once by ``python -m einsums.stages extract``; this file is yours.",
        "It will not import until every TODO below is a real type, which is",
        "deliberate: the annotations are the contract, and ``@stage`` refuses a",
        "signature without them.",
        '"""',
        "",
        "from dataclasses import dataclass",
        "",
        "from einsums.stages import TensorD, cmp, contract, stage  # noqa: F401",
        "",
        "",
        "@contract",
        "@dataclass(frozen=True)",
        f"class {spec.contract_name}:",
        f'    """TODO: what ``{spec.stage_name}`` produces."""',
        "",
    ]
    for name in spec.return_fields:
        origin = (
            f"from self.{name}" if name in spec.fields else "from a local of the method"
        )
        note = spec.fields[name].note if name in spec.fields else ""
        out.append(f"    #: TODO ({origin})" + (f" - {note}" if note else ""))
        out.append(f"    {name}: TODO = cmp.close()")
    if not spec.return_fields:
        out.append("    pass  # TODO: no returns were dispositioned; is this stage pure effect?")

    out += [
        "",
        "",
        "# TODO: uncomment once every parameter is annotated.",
        "# @stage(eager=True)",
        f"def {spec.stage_name}(",
    ]
    origin_of = {p: name for name, e in spec.fields.items() for p in e.params}
    for p in params:
        origin = origin_of[p]
        entry = spec.fields[origin]
        derived = "" if (entry.disposition == "param" and p == origin) else f" (from self.{origin})"
        out.append(f"    {p},  # TODO: annotate{derived}")
    out += [
        f") -> {spec.contract_name}:",
        f'    """TODO: one line on what ``{spec.stage_name}`` does.',
        "",
        "    Args:",
    ]
    for p in params:
        origin = origin_of[p]
        entry = spec.fields[origin]
        line = f"{p}: TODO."
        if entry.disposition == "plan" or p != origin:
            line += f" From ``self.{origin}``."
        # The origin's note belongs to the family once, not to every member.
        if entry.note and p == entry.params[0]:
            line += f" {entry.note[:1].upper()}{entry.note[1:].rstrip('.')}."
        out.extend(_textwrap.wrap(line, width=96, initial_indent=" " * 8,
                                  subsequent_indent=" " * 12))
    out += [
        '    """',
    ]
    if stage_helpers:
        out.append(f"    # The bodies of {', '.join('self.' + h for h in stage_helpers)} move here.")
    out += [
        f'    raise NotImplementedError("TODO: the numerics of {analysis.method}")',
        "",
        "",
        f"# ---- paste into {analysis.cls} " + "-" * 40,
        "",
        f"def {plan_name}(self):",
        f'    """Plan the {spec.stage_name} stage: everything that stays host-side before it.',
        "",
        *_wrap_absorbs(plan_fields),
        '    """',
    ]
    for name in plan_fields:
        entry = spec.fields[name]
        text = f"self.{name}"
        if entry.params:
            text += " -> " + ", ".join(entry.params)
        if entry.note:
            text += f" ({entry.note})"
        out.extend(_wrap_comment(text))
    out += [
        "    return dict(",
    ]
    for p in params:
        out.append(f"        {p}=TODO,")
    out += [
        "    )",
        "",
        "",
        f"def {finish_name}(self, result):",
        f'    """Finish the {spec.stage_name} stage: everything host-side after it.',
        "",
        *_wrap_absorbs(finish_fields),
        '    """',
    ]
    for name in finish_fields:
        entry = spec.fields[name]
        line = f"    # self.{name}"
        if entry.note:
            line += f": {entry.note}"
        out.append(line)
    out += [
        "    raise NotImplementedError",
        "",
    ]
    return "\n".join(out)


def narrowing_summary(analysis: Analysis, spec: CutSpec) -> str:
    counts: dict[str, int] = {}
    for entry in spec.fields.values():
        counts[entry.disposition] = counts.get(entry.disposition, 0) + 1
    breakdown = ", ".join(
        f"{counts[d]} {d}" for d in _FIELD_DISPOSITIONS if d in counts
    )
    return (
        f"{analysis.cls}.{analysis.method}: {len(analysis.fields)} fields "
        f"({breakdown}) + {len(analysis.helpers)} helpers -> "
        f"{spec.stage_name}({len(spec.params)} parameters) -> "
        f"{spec.contract_name}({len(spec.return_fields)} fields)"
    )
