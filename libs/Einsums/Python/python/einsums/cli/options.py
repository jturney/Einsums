# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums options``: every runtime option, with its flag, environment variable and default.

Read from the option registry the library itself holds (``cl::registered_options()``),
so it lists exactly what this build accepts. The runtime is not started.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap


def _matches(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "options",
        help="List runtime options (--einsums:* flags, EINSUMS_* variables, einsums.rc fields)",
        description="List the runtime options this build accepts. PATTERN filters by name, category or help text "
        "(a case-insensitive regular expression).",
    )
    p.add_argument("pattern", nargs="?", default="", help="filter, e.g. 'profile' or 'log|debug'")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=run)


def env_var(option: dict) -> str:
    """The environment variable the runtime reads for *option* ('' if it reads none).

    The registry reports it; a ``_core`` from before it did gets the default derivation
    (``einsums:log:level`` -> ``EINSUMS_LOG_LEVEL``), which a descriptor may override.
    """
    if "env" in option:
        return option["env"]
    return "EINSUMS_" + option["key"].upper().replace("-", "_")


def options() -> list[dict]:
    from .. import _core

    found = []
    for opt in _core._registered_options():
        entry = dict(opt)
        entry["flag"] = "--" + opt["name"]
        entry["env"] = env_var(opt)
        entry["rc"] = f"einsums.rc.{opt['attribute']}"
        found.append(entry)
    return sorted(found, key=lambda o: (o["category"], o["name"]))


def run(args: argparse.Namespace) -> int:
    chosen = [
        o for o in options() if any(_matches(o[field] or "", args.pattern) for field in ("name", "category", "help"))
    ]
    if args.json:
        print(json.dumps(chosen, indent=2, default=str))
        return 0
    if not chosen:
        print(f"No option matches {args.pattern!r}.")
        return 1
    category = None
    for o in chosen:
        if o["category"] != category:
            category = o["category"]
            print(f"\n{category}")
        value = f" <{o['value_name'] or o['type']}>" if o["kind"] != "flag" else ""
        default = "computed at run time" if o["computed_default"] else repr(o["default"])
        print(f"  {o['flag']}{value}")
        for line in textwrap.wrap(o["help"], width=90):
            print(f"      {line}")
        extra = f"  (also --{o['negated_name']})" if o["kind"] == "flag" and o.get("negated_name") else ""
        print(f"      default: {default}   env: {o['env'] or '(none)'}   rc: {o['rc']}{extra}")
    return 0
