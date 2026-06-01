#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Render a C++ API reference page (reStructuredText, Sphinx cpp domain) from
einsums-pybind ``--emit-cpp-docs-json`` output.

This is the renderer half of "Option 2" — replacing Doxygen + Breathe with our
own libclang extraction. Where Breathe forwarded Doxygen's raw signatures to
Sphinx's C++ domain (which choked on export macros and C++23), we emit
``cpp:function``/``cpp:class``/``cpp:enum`` directives with the **clean,
canonical** signatures libclang gives us — so they parse, the cross-references
work, and the warnings disappear.

Usage::

    render_cpp_rst.py --title "Einsums/BLASVendor/Vendor.hpp" \
                      --output Vendor.rst  module.cpp-docs.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LICENSE_HEADER = """..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------
"""

IND = "   "  # cpp-domain directive bodies conventionally indent 3


def log(msg: str) -> None:
    print(f"render_cpp_rst: {msg}", file=sys.stderr)


# ── Signature lowering ──────────────────────────────────────────────────────


def lower_template(params: list[str]) -> str:
    """Render a ``template <...>`` prefix the cpp domain can parse.

    Concept-constrained parameters (``BasicTensorConcept AType``) and other
    forms are lowered to plain ``typename`` — the cpp domain rejects concept
    constraints, so we keep the parameter name and document the real
    constraint in prose.
    """
    if not params:
        return ""
    # Sanitize each name to a bare identifier: clang spells auto-NTTP packs
    # oddly (``chars:auto``), and we only ever emit ``typename <name>``.
    names = []
    for p in params:
        m = re.match(r"[A-Za-z_]\w*", p.strip())
        names.append(m.group(0) if m else "T")
    return "template <" + ", ".join(f"typename {n}" for n in names) + "> "


def declarator(t: str, name: str) -> str:
    """Place a declarator ``name`` into a type ``t`` to form a parseable
    declaration. Handles the C/C++ cases where the name does not simply
    follow the type:
      * array types  — ``char[4]`` + ``magic`` -> ``char magic[4]``
      * array-ref / function-ptr — ``const char (&)[N]`` + ``s`` ->
        ``const char (&s)[N]``; ``T (*)(args)`` + ``f`` -> ``T (*f)(args)``
    """
    t = t.strip()
    if not name:
        return t
    # Array-reference / function-pointer: name goes inside the parentheses.
    m = re.search(r"\((&|\*+)\)", t)
    if m:
        return t[: m.end() - 1] + name + t[m.end() - 1:]
    # Plain C array: the ``[N]...`` suffix follows the NAME, not the type.
    m = re.search(r"((?:\s*\[[^\]]*\])+)\s*$", t)
    if m:
        return f"{t[:m.start()].rstrip()} {name}{m.group(1).replace(' ', '')}"
    # "const float *" + name -> "const float *name"; add a space otherwise.
    sep = "" if t.endswith(("*", "&")) else " "
    return f"{t}{sep}{name}"


def param_decl(p: dict) -> str:
    return declarator((p.get("type") or "").strip(), p.get("name") or "")


def namespace_of(qualified_name: str) -> str:
    """The enclosing namespace of a qualified name (drops the last component)."""
    return "::".join(qualified_name.split("::")[:-1])


def relativize(text: str, ns: str) -> str:
    """Strip the function's own namespace (and ancestors) from qualified type
    names so references resolve in-scope — e.g. inside
    ``einsums::blas::vendor`` a parameter of type ``einsums::blas::int_t``
    becomes ``int_t``. This both resolves the type ref and removes the
    namespace-qualifier cross-reference noise the cpp domain would otherwise
    report for ``einsums`` / ``einsums::blas``.
    """
    parts = ns.split("::") if ns else []
    # Longest prefix first: "a::b::c::", "a::b::", "a::".
    for i in range(len(parts), 0, -1):
        prefix = "::".join(parts[:i]) + "::"
        text = text.replace(prefix, "")
    return text


def _clean_typeparams(s: str) -> str:
    """Replace clang's unsubstituted ``type-parameter-N-M`` spelling (which
    the cpp domain can't parse) with ``auto``."""
    return re.sub(r"type-parameter-\d+-\d+", "auto", s or "")


def function_signature(fn: dict) -> str | None:
    """The cpp-domain signature, or None if it can't be rendered parseably."""
    # The constructor/destructor of a template class arrives named with the
    # class template arguments (``Chunk<Rank>``); strip them so the name is
    # the bare identifier the cpp domain expects inside the class scope.
    name = re.sub(r"<.*>$", "", fn["name"])
    params = ", ".join(param_decl(p) for p in fn.get("params", []))
    tmpl = lower_template(fn.get("template_params", []))
    if fn.get("is_operator") and name == "operator":
        # clang lost the operator symbol (spaceship / hidden friend) — there
        # is nothing parseable to emit.
        return None
    if fn.get("is_operator") and name.startswith("operator "):
        # Conversion operator: ``operator <target>()`` — the return type IS
        # the conversion target; emit no separate return type.
        target = _clean_typeparams((fn.get("return_type") or "").strip()) or _clean_typeparams(name[len("operator "):])
        sig = f"{tmpl}operator {target}({params})"
    elif fn.get("is_constructor") or fn.get("is_destructor"):
        sig = f"{tmpl}{name}({params})"  # no return type
    else:
        ret = _clean_typeparams((fn.get("return_type") or "void").strip())
        sig = f"{tmpl}{ret} {name}({params})"
    return relativize(_clean_typeparams(sig), namespace_of(fn.get("qualified_name", name)))


# ── Doc rendering (cpp-domain field lists) ──────────────────────────────────


def emit_doc(out: list[str], entity: dict, indent: str) -> None:
    ds = entity.get("doc_structured") or {}
    brief = (ds.get("brief") or "").strip()
    detail = (ds.get("detail") or "").strip()
    params = ds.get("params", [])
    tparams = ds.get("tparams", [])
    returns = (ds.get("returns") or "").strip()
    throws = ds.get("throws", [])
    if not (brief or detail or params or returns or throws or tparams):
        return
    out.append("")
    if brief:
        for line in brief.split("\n"):
            out.append(f"{indent}{line}".rstrip())
    if detail:
        out.append("")
        for line in detail.split("\n"):
            out.append(f"{indent}{line}".rstrip())
    if params or tparams or returns or throws:
        out.append("")
        for tp in tparams:
            out.append(f"{indent}:tparam {tp['name']}: {(tp.get('description') or '').strip()}".rstrip())
        for p in params:
            out.append(f"{indent}:param {p['name']}: {(p.get('description') or '').strip()}".rstrip())
        for t in throws:
            out.append(f"{indent}:throws {t['name']}: {(t.get('description') or '').strip()}".rstrip())
        if returns:
            out.append(f"{indent}:returns: {returns}".rstrip())
    out.append("")


def render_function(out: list[str], fn: dict, base: str = "") -> None:
    sig = function_signature(fn)
    if sig is None:
        return
    out.append(f"{base}.. cpp:function:: {sig}")
    emit_doc(out, fn, base + IND)
    out.append("")


def render_typedef(out: list[str], td: dict) -> None:
    ns = namespace_of(td.get("qualified_name", td["name"]))
    name = td["name"]
    underlying = relativize((td.get("underlying_type") or "").strip(), ns)
    tmpl = lower_template(td.get("template_params", []))
    # Implementation-detail-heavy alias underlyings (``decltype(...)``,
    # ``detail::``, SFINAE typename traits) are noise in a reference and
    # confuse the cpp parser — declare just the alias name and let the doc
    # describe it. Keep simple aliases (``int_t = long long int``) verbatim.
    complex_underlying = any(tok in underlying for tok in ("decltype", "detail::", "requires", "typename", "<"))
    decl = f"{tmpl}{name}" if (complex_underlying or not underlying) else f"{tmpl}{name} = {underlying}"
    out.append(f".. cpp:type:: {decl}")
    emit_doc(out, td, IND)
    out.append("")


def render_concept(out: list[str], c: dict) -> None:
    tmpl = lower_template(c.get("template_params", [])) or "template <typename T> "
    out.append(f".. cpp:concept:: {tmpl}{c['name']}")
    emit_doc(out, c, IND)
    out.append("")


def render_enum(out: list[str], en: dict, base: str = "") -> None:
    directive = "cpp:enum-class" if en.get("is_scoped") else "cpp:enum"
    out.append(f"{base}.. {directive}:: {en['name']}")
    emit_doc(out, en, base + IND)
    # A blank line MUST separate the directive line from its enumerator body;
    # emit_doc emits nothing for a doc-less enum, so guarantee it here (else
    # reST folds the first enumerator into the enum's name → parse error).
    out.append("")
    for v in en.get("enumerators", []):
        out.append(f"{base}{IND}.. cpp:enumerator:: {v['name']}")
        out.append("")
    out.append("")


def class_signature(cls: dict) -> str:
    tmpl = lower_template(cls.get("template_params", []))
    return f"{tmpl}{cls['name']}"


def render_class(out: list[str], cls: dict, base: str = "") -> None:
    # Relativize members against the FULL class scope (``einsums::GeneralTensor``),
    # not just the enclosing namespace — so a member signature referencing a
    # sibling nested type (``einsums::GeneralTensor::DeferredAlloc``) renders as
    # the bare ``DeferredAlloc`` and resolves against the nested declaration we
    # emit below, instead of dangling as ``GeneralTensor::DeferredAlloc``.
    scope = cls.get("qualified_name", cls["name"])
    out.append(f"{base}.. cpp:class:: {class_signature(cls)}")
    emit_doc(out, cls, base + IND)
    out.append("")
    # Members nest under the class directive via indentation.
    inner = base + IND
    for ctor in cls.get("constructors", []):
        render_function(out, _relativize_member(ctor, scope), base=inner)
    for m in cls.get("methods", []):
        render_function(out, _relativize_member(m, scope), base=inner)
    for f in cls.get("fields", []):
        ftype = relativize((f.get("type") or "").strip(), scope)
        out.append(f"{inner}.. cpp:member:: {declarator(ftype, f['name'])}")
        emit_doc(out, f, inner + IND)
        out.append("")
    # Nested enums and classes nest one level deeper, becoming
    # ``Parent::Name`` cross-reference targets via the cpp domain's scoping.
    for en in cls.get("enums", []):
        render_enum(out, en, base=inner)
    for nc in cls.get("nested_classes", []):
        if nc.get("is_external"):
            continue
        render_class(out, nc, base=inner)
    out.append("")


def _relativize_member(m: dict, scope: str) -> dict:
    """A member's signature is relativized against its enclosing class scope
    (``einsums::GeneralTensor``) so namespace AND class qualifiers are stripped
    and references resolve in-scope under the ``cpp:class`` directive."""
    m = dict(m)
    m["qualified_name"] = scope + "::" + m.get("name", "")  # function_signature relativizes vs this scope
    return m


# ── Page assembly ───────────────────────────────────────────────────────────


def render_page(title: str, doc: dict, embed: bool = False) -> str:
    """Render the page. ``embed`` omits the license/label/title so the result
    can be ``.. include::``d under a host page that supplies the title (the
    docs build does this so the toctree entry keeps a title)."""
    out: list[str] = []
    if not embed:
        out.append(LICENSE_HEADER)
        out.append("")
        label = "cppapi_" + title.replace("/", "_").replace(".", "_")
        out.append(f".. _{label}:")
        out.append("")
        bar = "=" * len(title)
        out.append(bar)
        out.append(title)
        out.append(bar)
    out.append("")
    out.append(".. note::")
    out.append(f"{IND}Generated from the C++ headers by ``einsums-pybind --emit-cpp-docs-json``.")
    out.append("")

    # Group every entity by its enclosing namespace so each block sets
    # ``.. cpp:namespace::`` and type references resolve in-scope (relative).
    by_ns: dict[str, dict[str, list[dict]]] = {}

    def bucket(ns: str) -> dict[str, list[dict]]:
        return by_ns.setdefault(ns, {"typedefs": [], "concepts": [], "enums": [], "classes": [], "functions": []})

    for td in doc.get("typedefs", []):
        bucket(namespace_of(td.get("qualified_name", td["name"])))["typedefs"].append(td)
    for c in doc.get("concepts", []):
        bucket(namespace_of(c.get("qualified_name", c["name"])))["concepts"].append(c)
    for en in doc.get("enums", []):
        bucket(namespace_of(en.get("qualified_name", en["name"])))["enums"].append(en)
    for cl in doc.get("classes", []):
        if cl.get("is_external"):
            continue
        bucket(namespace_of(cl.get("qualified_name", cl["name"])))["classes"].append(cl)
    for fn in doc.get("functions", []):
        bucket(namespace_of(fn.get("qualified_name", fn["name"])))["functions"].append(fn)

    for ns in sorted(by_ns):
        g = by_ns[ns]
        out.append(f".. cpp:namespace:: {ns or '0'}")
        out.append("")
        for td in sorted(g["typedefs"], key=lambda x: x["name"]):
            render_typedef(out, td)
        for c in sorted(g["concepts"], key=lambda x: x["name"]):
            render_concept(out, c)
        for en in sorted(g["enums"], key=lambda x: x["name"]):
            render_enum(out, en)
        for cl in sorted(g["classes"], key=lambda x: x["name"]):
            render_class(out, cl)
        for fn in sorted(g["functions"], key=lambda x: x["name"]):
            render_function(out, fn)
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="cpp-docs JSON file")
    ap.add_argument("--title", required=True, help="page title (e.g. the header path)")
    ap.add_argument("--output", required=True, help="output .rst path")
    ap.add_argument("--embed", action="store_true", help="omit title/label for ``.. include::``")
    args = ap.parse_args()

    doc = json.loads(Path(args.input).read_text())
    Path(args.output).write_text(render_page(args.title, doc, embed=args.embed))
    log(f"wrote {args.output} ({len(doc.get('functions', []))} functions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
