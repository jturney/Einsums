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

Reading the descriptors is ``devtools/option_descriptors.py``'s job, shared
with the argument reference the manual generates from the same list.

Usage::

    generate_rc.py --docs-json <a.cppdocs.json> [...] \
                   --template rc.py.in --output rc.py
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# libs/Einsums/Python/tools -> the repository root, where the shared extractor
# lives. The generator runs from the build tree with no package installed, so
# the path is relative to this file rather than to a working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "devtools"))

import option_descriptors as od  # noqa: E402

# Options whose Python spelling is richer than their C++ one. ``log:level`` is
# an integer scale in C++ and the ``LogLevel`` enum the template defines here,
# so the field is annotated with the enum and the binding layer reads ``.value``
# off whatever it is given.
_PY_TYPE_OVERRIDE = {
    "einsums:log:level": "LogLevel",
}

_LINE_WIDTH = 100

by_category = od.by_category
default_text = od.default_text
invocation = od.invocation


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

    options = od.collect_options(od.load_docs(args.docs_json))

    text = args.template.read_text(encoding="utf-8")
    for placeholder, rendered in (("@HELP@", render_help(options)),
                                  ("@FIELDS@", render_fields(options)),
                                  ("@MANIFEST@", render_manifest(options))):
        if placeholder not in text:
            od.die(f"{args.template} has no {placeholder} placeholder")
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
