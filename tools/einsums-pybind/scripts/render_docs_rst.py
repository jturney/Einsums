#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Render the Python API reference (reStructuredText) from einsums-pybind docs JSON.

This is the renderer half of "Option 1" of the custom doc-tool plan: the C++
tool emits a faithful description of the Python-facing surface
(``einsums-pybind --emit-docs-json``); this script turns one or more of those
JSON documents into Sphinx ``.rst`` reference pages, **grouped by Python
submodule** (``einsums``, ``einsums.linalg``, ``einsums.graph``, ...).

The naming authority lives entirely in the C++ tool: every entity already
carries a resolved ``py_name`` and a ``hidden`` flag, so this renderer never
re-derives pybind naming rules. It only formats.

Usage::

    render_docs_rst.py --outdir <dir> module1.docs.json [module2.docs.json ...]
    einsums-pybind --emit-docs-json ... | render_docs_rst.py --outdir <dir> -

One ``.rst`` is written per submodule, plus an ``index.rst`` with a toctree.
The schema this consumes is documented in ``DocsJson.hpp``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 2

LICENSE_HEADER = """..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------
"""

# Directive body indentation. Matches the hand-written reference pages
# (4 spaces) so the generated pages sit alongside them without a style jar.
IND = "    "


def log(msg: str) -> None:
    print(f"render_docs_rst: {msg}", file=sys.stderr)


# ── Loading & grouping ──────────────────────────────────────────────────────


def load_documents(paths: list[str]) -> tuple[str, list[dict]]:
    """Load every docs-JSON document, returning (top_module, documents)."""
    docs = []
    top_module = "einsums"
    for p in paths:
        text = sys.stdin.read() if p == "-" else Path(p).read_text()
        doc = json.loads(text)
        version = doc.get("schema_version")
        if version != SCHEMA_VERSION:
            log(f"warning: {p} has schema_version {version}, expected {SCHEMA_VERSION}")
        # All modules share the same top-level import name; take the first.
        top_module = doc.get("module", top_module)
        docs.append(doc)
    return top_module, docs


def full_module(entity: dict, top: str) -> str:
    """Resolve the dotted Python module an entity belongs to.

    ``submodule`` is null for the top-level module, or a dotted path that may
    already include the top module (``einsums.linalg``) or be relative
    (``linalg``); normalize both forms.
    """
    sub = entity.get("submodule")
    if not sub:
        return top
    if sub == top or sub.startswith(top + "."):
        return sub
    return f"{top}.{sub}"


def group_by_module(top: str, docs: list[dict]) -> dict[str, dict]:
    """Bucket classes/functions/enums from every document by submodule.

    Skips ``hidden`` entities and ``is_external`` ones (captured from another
    module's headers purely for name resolution — the owning module's JSON
    documents them). De-duplicates by ``qualified_name`` so an entity that
    surfaces in more than one document is described only once.
    """
    groups: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()

    def bucket(mod: str) -> dict:
        return groups.setdefault(mod, {"classes": [], "functions": [], "enums": []})

    for doc in docs:
        for kind in ("classes", "functions", "enums"):
            for ent in doc.get(kind, []):
                if ent.get("hidden") or ent.get("is_external"):
                    continue
                key = (kind, ent.get("qualified_name") or ent.get("name", ""))
                if key in seen:
                    continue
                seen.add(key)
                bucket(full_module(ent, top))[kind].append(ent)
    return groups


# ── Type-annotation sanitization ────────────────────────────────────────────
#
# The IR's py_type strings are a best-effort C++→Python translation. Where the
# translator can't resolve a type it hands back the C++ spelling (template
# parameters like ``AType``/``T``, dependent types like
# ``einsums::RuntimeTensorView<T>``, ``typename AType::ValueType``). Those are
# not valid/resolvable Python annotations, so Sphinx's nitpicky mode flags
# them. We post-process every annotation for *display*: keep builtins, typing
# containers, and documented class names; collapse everything unresolvable to
# ``Any``. (The .pyi path deliberately keeps the C++ spelling for pyright; the
# docs want a clean, resolvable annotation instead.)

# Populated in main() from the documented class/enum names so a class
# referenced in a signature renders as itself rather than ``Any``.
KNOWN_TYPES: set[str] = set()

_PY_BUILTIN_TYPES = {
    "int", "float", "str", "bool", "None", "complex", "bytes", "bytearray",
    "object", "Any", "memoryview", "slice", "Iterator", "Iterable", "Sequence",
    "Mapping", "Callable",
}
_PY_CONTAINERS = {"list", "tuple", "dict", "set", "frozenset", "Callable", "Sequence", "Iterable", "Mapping"}

_LEAF_RE = re.compile(r"^([A-Za-z_][\w.]*)\s*\[(.*)\]$", re.S)


def _split_top(s: str, sep: str) -> list[str]:
    """Split on `sep` at bracket/angle/paren depth 0."""
    out, cur, depth = [], [], 0
    for ch in s:
        if ch in "[<(":
            depth += 1
        elif ch in "]>)":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [p.strip() for p in out]


def pythonize(ann: str) -> str:
    """Sanitize a single annotation string into a resolvable Python form."""
    s = (ann or "").strip()
    if not s:
        return "Any"
    # Union: A | B | C
    parts = _split_top(s, "|")
    if len(parts) > 1:
        return " | ".join(pythonize(p) for p in parts)
    # A bracketed list (e.g. Callable's [Args] group): [A, B] -> recurse each
    if s.startswith("[") and s.endswith("]"):
        return "[" + ", ".join(pythonize(a) for a in _split_top(s[1:-1], ",")) + "]"
    # Container head[...]
    m = _LEAF_RE.match(s)
    if m:
        head, args = m.group(1), m.group(2)
        if head.split(".")[-1] in _PY_CONTAINERS or head.startswith("numpy"):
            return f"{head}[" + ", ".join(pythonize(a) for a in _split_top(args, ",")) + "]"
        return "Any"  # unknown templated C++ type
    return _pythonize_leaf(s)


_STD_EXCEPTIONS = {
    "exception", "logic_error", "runtime_error", "invalid_argument", "out_of_range",
    "length_error", "domain_error", "range_error", "overflow_error", "underflow_error",
    "bad_alloc", "bad_cast", "system_error",
}


def _pythonize_leaf(s: str) -> str:
    s = s.replace("typename ", "").strip().rstrip("&* ").strip()
    s = re.sub(r"<.*>$", "", s).strip()      # drop template args
    if "::" in s:
        s = s.split("::")[-1]                # qualified C++ name -> last component
    if s in _STD_EXCEPTIONS:
        return "Exception"
    if s.startswith("numpy") or s in _PY_BUILTIN_TYPES or s in KNOWN_TYPES:
        return s
    return "Any"                              # template params / unresolved -> Any


# ── Signature & doc formatting ──────────────────────────────────────────────


def py_signature(params: list[dict]) -> str:
    """Render a Python parameter list from IR params (py-stub forms)."""
    parts = []
    for p in params:
        name = p.get("name") or "arg"
        ann = pythonize(p.get("py_type") or "")
        piece = f"{name}: {ann}" if ann else name
        default = p.get("default_py") or p.get("default")
        if default:
            piece += f" = {default}"
        parts.append(piece)
    return ", ".join(parts)


def emit_text(out: list[str], text: str, indent: str) -> None:
    """Emit a block of (already reST-ready) text at the given indent."""
    text = (text or "").strip()
    if not text:
        return
    out.append("")
    for line in text.split("\n"):
        out.append(f"{indent}{line}".rstrip())
    out.append("")


def emit_doc(out: list[str], entity: dict, indent: str, *, with_params: bool) -> None:
    """Emit an entity's structured doc as reST under a directive.

    Renders ``brief`` then ``detail``; for callables also renders ``:param:``
    / ``:returns:`` / ``:raises:`` field lists. Falls back to the raw ``doc``
    string when no structured form is present (older JSON, or empty parse).
    """
    ds = entity.get("doc_structured") or {}
    brief = (ds.get("brief") or "").strip()
    detail = (ds.get("detail") or "").strip()
    params = ds.get("params", []) if with_params else []
    returns = (ds.get("returns") or "").strip() if with_params else ""
    throws = ds.get("throws", []) if with_params else []

    if not (brief or detail or params or returns or throws):
        emit_text(out, entity.get("doc", ""), indent)
        return

    out.append("")
    if brief:
        for line in brief.split("\n"):
            out.append(f"{indent}{line}".rstrip())
    if detail:
        out.append("")
        for line in detail.split("\n"):
            out.append(f"{indent}{line}".rstrip())
    if params or returns or throws:
        out.append("")
        for p in params:
            desc = (p.get("description") or "").strip()
            out.append(f"{indent}:param {p['name']}: {desc}".rstrip())
        if returns:
            out.append(f"{indent}:returns: {returns}".rstrip())
        for t in throws:
            desc = (t.get("description") or "").strip()
            out.append(f"{indent}:raises {pythonize(t['name'])}: {desc}".rstrip())
    out.append("")


# ── Per-entity rendering ────────────────────────────────────────────────────


def has_directive(entity: dict, name: str) -> bool:
    return any(d.get("name") == name for d in entity.get("directives", []))


def entity_has_doc(e: dict) -> bool:
    ds = e.get("doc_structured") or {}
    return bool(ds.get("brief") or ds.get("detail") or ds.get("params") or ds.get("returns") or ds.get("throws") or e.get("doc"))


def entity_signature_body(e: dict) -> str:
    """The ``(params) -> return`` tail of a callable's signature."""
    return f"({py_signature(e.get('params', []))}) -> {pythonize(e.get('return_py_type') or 'None')}"


def entity_py_names(e: dict, *, is_ctor: bool = False) -> list[str]:
    """Distinct Python names a callable binds under (overload set)."""
    if is_ctor:
        return ["__init__"]
    names = [o["py_name"] for o in e.get("python_overloads", []) if o.get("py_name")]
    if not names:
        names = [e.get("py_name") or e["name"]]
    return list(dict.fromkeys(names))


def emit_overload_set(out: list[str], base_indent: str, directive: str, name: str, members: list[dict]) -> None:
    """Emit one py-domain directive carrying every overload's signature.

    Grouping overloads (free-function overload sets, multiple constructors,
    overloaded methods) under a single directive — rather than repeating the
    directive — is both correct reST and avoids Sphinx "duplicate object
    description" warnings. Doc text comes from the first member that has one.
    """
    prefix = f"{base_indent}.. py:{directive}:: "
    pad = " " * len(prefix)
    sigs = list(dict.fromkeys(f"{name}{entity_signature_body(m)}" for m in members))
    out.append(prefix + sigs[0])
    for s in sigs[1:]:
        out.append(pad + s)
    doc_member = next((m for m in members if entity_has_doc(m)), members[0])
    emit_doc(out, doc_member, base_indent + IND, with_params=True)
    out.append("")


def render_attribute(out: list[str], py_name: str, py_type: str, entity: dict, writable: bool) -> None:
    out.append(f"{IND}.. py:attribute:: {py_name}")
    if py_type:
        out.append(f"{IND}{IND}:type: {pythonize(py_type)}")
    emit_doc(out, entity, IND * 2, with_params=False)
    if not writable:
        # Plain text, not ``*...*`` — docutils rejects emphasis whose content
        # opens with '(' on its own line. A blank line MUST separate this body
        # line from the ``:type:`` option block above, otherwise docutils reads
        # it as a malformed option ("invalid option block").
        out.append("")
        out.append(f"{IND}{IND}(read-only)")
        out.append("")


def class_py_names(cls: dict) -> list[str]:
    """The Python class name(s) this entity binds under.

    Templated classes contribute one Python class per instantiation
    (``RuntimeTensorF``, ``RuntimeTensorD``, ...); non-templated classes bind
    under their single resolved ``py_name``.
    """
    if cls.get("is_template"):
        names = [i["py_name"] for i in cls.get("instantiations", []) if i.get("py_name")]
        if names:
            return list(dict.fromkeys(names))
    return [cls.get("py_name") or cls["name"]]


def render_class(out: list[str], cls: dict) -> None:
    for cname in class_py_names(cls):
        out.append(f".. py:class:: {cname}")
        emit_doc(out, cls, IND, with_params=False)
        out.append("")
        ctors = cls.get("constructors", [])
        if ctors:
            emit_overload_set(out, IND, "method", "__init__", ctors)
        # Group overloaded methods by Python name; skip hidden and the
        # getter/setter backing methods (exposed via their property).
        method_groups: dict[str, list[dict]] = {}
        for m in cls.get("methods", []):
            if m.get("hidden") or has_directive(m, "getter") or has_directive(m, "setter"):
                continue
            for name in entity_py_names(m):
                method_groups.setdefault(name, []).append(m)
        for name, members in method_groups.items():
            directive = "staticmethod" if members[0].get("is_static") else "method"
            emit_overload_set(out, IND, directive, name, members)
        for p in cls.get("properties", []):
            render_attribute(out, p["py_name"], p.get("py_type", ""), p, p.get("writable", False))
        for f in cls.get("fields", []):
            if f.get("hidden"):
                continue
            render_attribute(out, f.get("py_name") or f["name"], f.get("py_type", ""), f, True)
        for e in cls.get("enums", []):
            render_enum(out, e, nested_indent=IND)
        out.append("")


def render_enum(out: list[str], en: dict, nested_indent: str = "") -> None:
    name = en.get("py_name") or en["name"]
    out.append(f"{nested_indent}.. py:class:: {name}")
    emit_doc(out, en, nested_indent + IND, with_params=False)
    out.append("")
    for v in en.get("enumerators", []):
        out.append(f"{nested_indent}{IND}.. py:attribute:: {v['name']}")
        emit_doc(out, v, nested_indent + IND * 2, with_params=False)
    out.append("")


# ── Page assembly ───────────────────────────────────────────────────────────


def render_page(module: str, group: dict) -> str:
    out: list[str] = [LICENSE_HEADER, ""]
    # Namespace the label (``api_python_einsums_linalg``) so it never
    # collides with the hand-written pages' targets (e.g. ``_einsums``).
    label = "api_python_" + module.replace(".", "_")
    out.append(f".. _{label}:")
    out.append("")
    title = f"``{module}``"
    out.append("=" * len(title))
    out.append(title)
    out.append("=" * len(title))
    out.append("")
    out.append(".. note::")
    out.append(f"{IND}This page is generated from the C++ binding annotations by")
    out.append(f"{IND}``einsums-pybind --emit-docs-json``. Do not edit by hand.")
    out.append("")
    out.append(f".. py:module:: {module}")
    out.append("")

    funcs = sorted(group["functions"], key=lambda f: f.get("py_name") or f["name"])
    classes = sorted(group["classes"], key=lambda c: c.get("py_name") or c["name"])
    enums = sorted(group["enums"], key=lambda e: e.get("py_name") or e["name"])

    if classes:
        out.append("Classes")
        out.append("-------")
        out.append("")
        for c in classes:
            render_class(out, c)
    if funcs:
        out.append("Functions")
        out.append("---------")
        out.append("")
        # Group overloads sharing a Python name (e.g. dot + dot_python both
        # bound as ``dot``) into a single multi-signature directive.
        func_groups: dict[str, list[dict]] = {}
        for f in funcs:
            for name in entity_py_names(f):
                func_groups.setdefault(name, []).append(f)
        for name in sorted(func_groups):
            emit_overload_set(out, "", "function", name, func_groups[name])
    if enums:
        out.append("Enumerations")
        out.append("------------")
        out.append("")
        for e in enums:
            render_enum(out, e)

    return "\n".join(out).rstrip() + "\n"


def render_index(modules: list[str]) -> str:
    out: list[str] = [LICENSE_HEADER, ""]
    out.append(".. _api_python:")
    out.append("")
    title = "Python API Reference"
    out.append("=" * len(title))
    out.append(title)
    out.append("=" * len(title))
    out.append("")
    out.append(".. toctree::")
    out.append(f"{IND}:maxdepth: 2")
    out.append("")
    for m in modules:
        out.append(f"{IND}{m}")
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="docs-JSON files (or '-' for stdin)")
    ap.add_argument("--outdir", required=True, help="directory to write .rst pages into")
    args = ap.parse_args()

    top, docs = load_documents(args.inputs)
    groups = group_by_module(top, docs)
    if not groups:
        log("no entities found; nothing written")
        return 0

    # Record every documented class/enum name (including per-instantiation
    # class names) so the annotation sanitizer keeps real types and only
    # collapses genuinely-unresolvable ones (template params, dependent C++
    # spellings) to Any.
    for g in groups.values():
        for c in g["classes"]:
            KNOWN_TYPES.update(class_py_names(c))
        for e in g["enums"]:
            KNOWN_TYPES.add(e.get("py_name") or e["name"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Top-level module first, then submodules alphabetically.
    modules = sorted(groups, key=lambda m: (m != top, m))
    for module in modules:
        page = render_page(module, groups[module])
        (outdir / f"{module}.rst").write_text(page)
        log(f"wrote {module}.rst ({len(groups[module]['classes'])} classes, "
            f"{len(groups[module]['functions'])} functions, {len(groups[module]['enums'])} enums)")

    (outdir / "index.rst").write_text(render_index(modules))
    log(f"wrote index.rst with {len(modules)} module(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
