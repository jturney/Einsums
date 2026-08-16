#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Generate ``einsums/rc.py`` from the C++ option descriptors.

The descriptors in ``libs/Einsums/<Module>/include/Einsums/<Module>/Options.hpp``
are the single place an option's name, type, default, help text, and category
are spelled. This turns them into the Python settings surface, so the three
things that used to be hand-copied - the fields, the ``--help`` listing in the
comments, and the flag spellings - are all one fact again.

Input is apiary's C++ docs JSON for those headers: every documented
namespace-scope variable, with its type and, when it was initialized by a
call, that call's callee and folded arguments. A variable counts as an option
when its initializer names ``cl::config_flag``, ``cl::config_opt``, or
``cl::config_opt_computed``.

Usage::

    generate_rc.py --docs-json <a.cppdocs.json> [...] \
                   --template rc.py.in --output rc.py
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# The factories a descriptor can be declared with, and whether the option they
# declare is a presence-carrying flag.
_FACTORIES = {
    "config_flag": "flag",
    "config_opt": "value",
    "config_opt_computed": "value",
}

# C++ value type -> Python type name. The template arguments arrive
# canonicalized, so ``std::int64_t`` shows up under whichever fundamental type
# it aliases on this platform.
_PY_TYPE = {
    "bool": "bool",
    "double": "float",
    "float": "float",
    "int": "int",
    "long": "int",
    "long long": "int",
    "std::int64_t": "int",
    "int64_t": "int",
    "std::string": "str",
    "std::basic_string<char>": "str",
}

# Options whose Python spelling is richer than their C++ one. ``log:level`` is
# an integer scale in C++ and the ``LogLevel`` enum the template defines here,
# so the field is annotated with the enum and the binding layer reads ``.value``
# off whatever it is given.
_PY_TYPE_OVERRIDE = {
    "einsums:log:level": "LogLevel",
}

_LINE_WIDTH = 100


def die(message: str) -> None:
    print(f"generate_rc: {message}", file=sys.stderr)
    raise SystemExit(1)


def attribute_name(long_name: str) -> str:
    """The Python attribute an option's command-line name maps to.

    Drop a leading ``einsums:``, then turn every ``:`` and ``-`` into an
    underscore. Spelled identically in PyEinsumsMain.cpp's
    ``rc_attribute_name``; the drift test holds the two together.
    """
    rest = long_name[len("einsums:"):] if long_name.startswith("einsums:") else long_name
    return rest.replace(":", "_").replace("-", "_")


def arg_by_name(args: list[dict], name: str) -> dict | None:
    for a in args:
        if a.get("name") == name:
            return a
    return None


def literal(arg: dict | None):
    """The folded value of a call argument, or ``None`` when it has none."""
    if arg is None or arg.get("value_kind") is None:
        return None
    return arg.get("value")


def collect_options(docs: list[dict]) -> list[dict]:
    """Every descriptor in the given documents, in declaration order."""
    options: list[dict] = []
    seen: set[str] = set()

    for doc in docs:
        for var in doc.get("variables", []):
            init = var.get("initializer")
            if not init or init.get("kind") != "call":
                continue
            callee = init.get("callee", "")
            factory = callee.rsplit("::", 1)[-1]
            if factory not in _FACTORIES:
                continue

            args = init.get("args", [])
            name = literal(arg_by_name(args, "name"))
            if not isinstance(name, str) or not name:
                die(f"{var.get('qualified_name')} declares an option whose name is not a literal")
            if name in seen:
                continue
            seen.add(name)

            # The value type: the descriptor's own template argument, which is
            # present whichever factory was used, unlike the factory's.
            type_args = var.get("type_template_args") or init.get("template_args") or []
            cxx_type = type_args[0] if type_args else "bool"
            py_type = _PY_TYPE.get(cxx_type)
            if py_type is None:
                die(f"option {name} has value type {cxx_type!r}, which has no Python spelling")

            computed = factory == "config_opt_computed"
            options.append({
                "attribute": attribute_name(name),
                "name": name,
                "kind": _FACTORIES[factory],
                "type": py_type,
                "cxx_type": cxx_type,
                "help": literal(arg_by_name(args, "help")) or "",
                "category": literal(arg_by_name(args, "category")) or "",
                "value_name": literal(arg_by_name(args, "value_name")) or "",
                # A computed default is produced by a function at registration
                # and cannot be written down, so there is nothing to print.
                "default": None if computed else literal(arg_by_name(args, "default_value")),
                "computed_default": computed,
                "doc": (var.get("doc_structured") or {}).get("brief", "").strip(),
            })

    if not options:
        die("no option descriptors found - check that the Options.hpp headers were parsed")
    return options


def by_category(options: list[dict]) -> list[tuple[str, list[dict]]]:
    """Options grouped under their ``--help`` heading, headings sorted."""
    groups: dict[str, list[dict]] = {}
    for opt in options:
        groups.setdefault(opt["category"], []).append(opt)
    return sorted(groups.items())


def default_text(opt: dict) -> str:
    """How an option's default reads in a comment."""
    if opt["computed_default"]:
        return "computed at startup"
    value = opt["default"]
    if opt["type"] == "bool":
        return repr(bool(value))
    if value is None:
        return "unset"
    return repr(value)


def invocation(opt: dict) -> str:
    """The option as ``--help`` spells it."""
    if opt["kind"] == "flag":
        return f"--{opt['name']}"
    placeholder = opt["value_name"] or "value"
    return f"--{opt['name']} <{placeholder}>"


def render_help(options: list[dict]) -> str:
    """The comment block listing every option, grouped as ``--help`` groups it."""
    invocations = [invocation(o) for o in options]
    column = min(max(len(i) for i in invocations) + 4, 60)

    lines = [
        "# The options this build declares. Rendered from the same descriptors",
        "# the runtime registers, so this cannot fall behind what ``--help`` prints.",
    ]
    for category, opts in by_category(options):
        lines.append("#")
        lines.append(f"# {category}:")
        for opt in opts:
            inv = invocation(opt)
            help_text = opt["help"]
            if opt["kind"] == "flag" and opt["default"]:
                help_text += " (default: true)"
            wrapped = textwrap.wrap(help_text, width=_LINE_WIDTH - column - 4) or [""]
            lines.append(f"#   {inv.ljust(column)}{wrapped[0]}")
            for extra in wrapped[1:]:
                lines.append(f"#   {'' .ljust(column)}{extra}")
    return "\n".join(lines)


def render_fields(options: list[dict]) -> str:
    """The field declarations, one commented block per ``--help`` heading."""
    blocks = []
    for category, opts in by_category(options):
        lines = [f"# {category}"]
        for opt in opts:
            annotation = _PY_TYPE_OVERRIDE.get(opt["name"], opt["type"])
            # The descriptor's own doc comment, which is written for someone
            # reading the code, rather than the help text - that is already in
            # the listing above, and repeating it here would put the same
            # sentence in the file twice.
            for line in textwrap.wrap(opt["doc"] or opt["help"], width=_LINE_WIDTH - 2):
                lines.append(f"# {line}")
            lines.append(f"# Default: {default_text(opt)}")
            lines.append(f"{opt['attribute']}: {annotation} | None = None")
            lines.append("")
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks)


def render_manifest(options: list[dict]) -> str:
    """The option surface as data, for the drift test to compare."""
    lines = [
        "# The descriptors this file was generated from, as data, so a test can hold",
        "# it up against the registry the running library exposes: this side comes",
        "# from parsing the headers and that side from the process, and a mismatch",
        "# is exactly the drift generating this file was meant to end. Private - set",
        "# the fields above instead.",
        "_OPTIONS: tuple[dict[str, object], ...] = (",
    ]
    for opt in options:
        entry = {
            "attribute": opt["attribute"],
            "name": opt["name"],
            "kind": opt["kind"],
            "type": opt["type"],
            "category": opt["category"],
            "default": opt["default"],
            "computed_default": opt["computed_default"],
        }
        lines.append(f"    {entry!r},")
    lines.append(")")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs-json", type=Path, nargs="+", required=True,
                    help="apiary C++ docs JSON for the Options.hpp headers")
    ap.add_argument("--template", type=Path, required=True, help="rc.py.in")
    ap.add_argument("--output", type=Path, required=True, help="the rc.py to write")
    args = ap.parse_args()

    docs = []
    for path in args.docs_json:
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            die(f"cannot read {path}: {exc}")

    options = collect_options(docs)

    text = args.template.read_text(encoding="utf-8")
    for placeholder, rendered in (("@HELP@", render_help(options)),
                                  ("@FIELDS@", render_fields(options)),
                                  ("@MANIFEST@", render_manifest(options))):
        if placeholder not in text:
            die(f"{args.template} has no {placeholder} placeholder")
        text = text.replace(placeholder, rendered)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Written only on change so that a no-op regeneration does not restamp the
    # file and re-trigger everything downstream of it.
    if not args.output.is_file() or args.output.read_text(encoding="utf-8") != text:
        args.output.write_text(text, encoding="utf-8")
    print(f"generate_rc: {len(options)} option(s) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
