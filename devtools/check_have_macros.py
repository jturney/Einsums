#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Every EINSUMS_HAVE_* macro a preprocessor condition tests must be defined somewhere.

A condition on a macro nothing defines is false in every build, so the code behind
it is dead and the code in its #else runs instead. Nothing reports that: the
compiler never sees the dead branch, and no build configuration can reach it. It
hid three defects at once: complex trsyl threw "not implemented" on every build
because it tested LAPACKE macros no CMake defines, Print's backtraces tested a
cpptrace macro where the real one is EINSUMS_HAVE_BACKTRACES, and Comm advertised
NCCL and RCCL flags that could never be true.

A name counts as defined when it appears
  - on a non-comment line of a tracked CMake file (``einsums_add_config_define``,
    a feature test's ``DEFINITIONS``, and every other way the build emits one),
  - in a ``#define`` or ``#cmakedefine`` in a tracked source, or
  - under a prefix CMake completes from a variable, as in
    ``EINSUMS_HAVE_MALLOC_${...}``.

Scans every tracked file, like the other devtools checks. Run with no arguments
from anywhere in the repo.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

NAME = re.compile(r"\bEINSUMS_HAVE_[A-Z0-9_]+\b")
COMPUTED_PREFIX = re.compile(r"\b(EINSUMS_HAVE_[A-Z0-9_]*)\$\{")
CONDITION = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif|elifdef|elifndef)\b")
DEFINE = re.compile(r"^\s*#\s*(?:define|cmakedefine)\s+(EINSUMS_HAVE_[A-Z0-9_]+)\b")

SOURCE_SUFFIXES = (".hpp", ".cpp", ".h", ".hip", ".cu", ".cuh", ".mm", ".inl", ".in", ".fstring")

EXEMPT = {"devtools/check_have_macros.py"}


def is_cmake(rel: str) -> bool:
    return rel.endswith(".cmake") or rel.endswith("CMakeLists.txt") or rel.endswith(".cmake.in")


def strip_cmake_comment(line: str) -> str:
    # A ``#`` inside a quoted string is not a comment. The config defines never
    # need one, so only unquoted text before the first unquoted ``#`` counts.
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted:
            return line[:i]
    return line


def main() -> int:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()

    defined: set[str] = set()
    prefixes: set[str] = set()
    uses: dict[str, list[str]] = {}

    for rel in tracked:
        if rel in EXEMPT:
            continue
        cmake = is_cmake(rel)
        if not cmake and not rel.endswith(SOURCE_SUFFIXES):
            continue
        try:
            lines = (root / rel).read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue

        for number, line in enumerate(lines, start=1):
            if cmake:
                code = strip_cmake_comment(line)
                defined.update(NAME.findall(code))
                prefixes.update(COMPUTED_PREFIX.findall(code))
                continue
            if m := DEFINE.match(line):
                defined.add(m.group(1))
            elif CONDITION.match(line):
                for name in NAME.findall(line):
                    uses.setdefault(name, []).append(f"{rel}:{number}")

    undefined = {
        name: where
        for name, where in uses.items()
        if name not in defined and not any(name.startswith(p) for p in prefixes)
    }
    if not undefined:
        return 0

    print("Preprocessor conditions test EINSUMS_HAVE_* macros that nothing defines.")
    print("Every build treats them as false, so the code they guard never compiles.")
    print("Define the macro in CMake, fix its spelling, or delete the dead branch:\n")
    for name in sorted(undefined):
        print(f"  {name}")
        for where in undefined[name]:
            print(f"    {where}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
