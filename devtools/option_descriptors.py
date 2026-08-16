#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""The C++ option descriptors, read out of apiary's C++ docs JSON.

The descriptors in ``libs/Einsums/<Module>/include/Einsums/<Module>/Options.hpp``
are the single place an option's name, type, default, help text, category, and
value placeholder are spelled. Everything derived from that list - the Python
settings surface, the argument reference in the manual - reads it through this
module, so there is one extractor rather than one per consumer.

Input is apiary's C++ docs JSON for those headers: every documented
namespace-scope variable, with its type and, when it was initialized by a call,
that call's callee and folded arguments. A variable counts as an option when
its initializer names ``cl::config_flag``, ``cl::config_opt``, or
``cl::config_opt_computed``.

A generator uses it as::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "devtools"))
    import option_descriptors as od

    options = od.collect_options(od.load_docs(paths))
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The factories a descriptor can be declared with, and whether the option they
# declare is a presence-carrying flag.
FACTORIES = {
    "config_flag": "flag",
    "config_opt": "value",
    "config_opt_computed": "value",
}

# C++ value type -> Python type name. The template arguments arrive
# canonicalized, so ``std::int64_t`` shows up under whichever fundamental type
# it aliases on this platform.
PY_TYPE = {
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


def die(message: str) -> None:
    """Fail the build with a message naming the generator that failed."""
    program = Path(sys.argv[0]).stem or "option_descriptors"
    print(f"{program}: {message}", file=sys.stderr)
    raise SystemExit(1)


def attribute_name(long_name: str) -> str:
    """The Python attribute an option's command-line name maps to.

    Drop a leading ``einsums:``, then turn every ``:`` and ``-`` into an
    underscore. Spelled identically in PyEinsumsMain.cpp's
    ``rc_attribute_name``; the drift test holds the two together.
    """
    rest = long_name[len("einsums:"):] if long_name.startswith("einsums:") else long_name
    return rest.replace(":", "_").replace("-", "_")


def negated_name(long_name: str) -> str:
    """The ``no-`` spelling registration generates for a flag.

    The negation goes on the last segment, where a reader looks for it, so
    ``einsums:profile:report`` pairs with ``einsums:profile:no-report``.
    Mirrors ``cl::derive_negated_name``.
    """
    head, sep, tail = long_name.rpartition(":")
    return f"{head}{sep}no-{tail}"


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


def load_docs(paths) -> list[dict]:
    """Every docs JSON document named, parsed."""
    docs = []
    for path in paths:
        path = Path(path)
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            die(f"cannot read {path}: {exc}")
    return docs


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
            if factory not in FACTORIES:
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
            py_type = PY_TYPE.get(cxx_type)
            if py_type is None:
                die(f"option {name} has value type {cxx_type!r}, which has no Python spelling")

            computed = factory == "config_opt_computed"
            options.append({
                "attribute": attribute_name(name),
                "name": name,
                "kind": FACTORIES[factory],
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
    """The option as ``--help`` spells it.

    The placeholder is the descriptor's own ``value_name``, falling back to the
    ``value`` the help renderer uses when a descriptor names none.
    """
    if opt["kind"] == "flag":
        return f"--{opt['name']}"
    placeholder = opt["value_name"] or "value"
    return f"--{opt['name']} <{placeholder}>"
