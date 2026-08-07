#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Every einsums namespace block must be opened with EINSUMS_NAMESPACE_BEGIN.

The library lives in ``einsums::inline vN`` so two ABI-incompatible copies in
one process cannot answer for each other. A block written as a bare
``namespace einsums { ... }`` still compiles and still works, because an inline
namespace is transparent to name lookup and unqualified lookup inside
``einsums::vN`` reaches the enclosing ``einsums``. Its symbols simply are not
versioned. That is the whole failure mode: silent, invisible, and exactly what
the sweep existed to remove. Without this check the sweep decays one file at a
time.

Scans EVERY tracked file rather than a ``*.hpp``/``*.cpp`` glob. The sweep
itself used that glob and missed seven files: two generated ``.hpp.in``
templates, an Objective-C++ ``.mm`` backend, and four ``.fstring`` skeletons
that ``create_module_skeleton.py`` stamps into every NEW module. The last group
matters most: had they been left alone, every module created afterwards would
have reintroduced an untagged namespace that nobody typed.

Run with no arguments from anywhere in the repo.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# `namespace einsums {` and `namespace einsums::a::b {`, plus the `{{` spelling
# the Python-formatted .fstring skeletons use.
BARE_OPENER = re.compile(r"^namespace\s+einsums\b[^{]*\{\{?\s*$")

BEGIN = re.compile(r"^EINSUMS_NAMESPACE_BEGIN\(([^)]*)\)\s*$")
END = re.compile(r"^EINSUMS_NAMESPACE_END\(([^)]*)\)\s*$")

# Where the macros themselves live, and the checker's own doc text.
EXEMPT = {
    "libs/Einsums/Config/include/Einsums/Config/Namespace.hpp",
    "devtools/check_namespace_macros.py",
}


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    return [root / f for f in out]


def main() -> int:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    bare: list[str] = []
    mismatched: list[str] = []

    for path in tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in EXEMPT:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue

        counts: dict[str, int] = defaultdict(int)
        for number, line in enumerate(lines, start=1):
            if BARE_OPENER.match(line):
                bare.append(f"{rel}:{number}: {line.strip()}")
            elif m := BEGIN.match(line):
                counts[m.group(1).strip()] += 1
            elif m := END.match(line):
                counts[m.group(1).strip()] -= 1

        for arg, balance in sorted(counts.items()):
            if balance != 0:
                shown = arg or "(no argument)"
                mismatched.append(
                    f"{rel}: {shown} has {balance:+d} unmatched "
                    f"EINSUMS_NAMESPACE_BEGIN/END"
                )

    if bare:
        print("Namespace blocks that bypass EINSUMS_NAMESPACE_BEGIN:\n", file=sys.stderr)
        for entry in bare:
            print(f"  {entry}", file=sys.stderr)
        print(
            "\nThese compile and work, but their symbols are NOT versioned, so a second\n"
            "copy of libEinsums in the same process can interpose on them. Use\n"
            "EINSUMS_NAMESPACE_BEGIN(path) / EINSUMS_NAMESPACE_END(path) from\n"
            "<Einsums/Config/Namespace.hpp>.",
            file=sys.stderr,
        )

    if mismatched:
        print("\nBEGIN/END arguments that do not pair up:\n", file=sys.stderr)
        for entry in mismatched:
            print(f"  {entry}", file=sys.stderr)
        print(
            "\nA DEPTH mismatch is a brace imbalance and fails to compile; an ARGUMENT\n"
            "mismatch compiles fine and leaves the closer describing the wrong block.",
            file=sys.stderr,
        )

    if bare or mismatched:
        return 1

    print("namespace macros: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
