#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""No version tag may name a release later than the one under development.

``@versionadded{X}`` is rendered verbatim. Nothing compares X against
``EINSUMS_VERSION``, so a tag naming a release that does not exist is not a
build error: the docs build is nitpicky and warnings-are-errors, and it stays
green while publishing "New in version 2.1.0" for a version nobody has cut. The
only detector is a human reading the rendered page, which is how the eight tags
this check was written for were found. A malformed value fails the same way,
silently, so it is reported too.

Tags naming an OLD release are correct and common: most of the tree is
``1.0.0``. The rule is only that a tag may not run ahead of
``EINSUMS_VERSION_MAJOR.MINOR.PATCH`` in the top-level CMakeLists.txt. Bump
those three and the previously-illegal value becomes legal on its own.

Three surfaces carry a version, and all three reach the docs without
validation:

* the Doxygen aliases in C++ doc comments, including the ``desc`` forms;
* ``.. versionadded::`` / ``.. versionchanged::`` written by hand in reST;
* ``added:`` in the docs YAML, which ``generate_arguments_rst.py`` turns into
  the reST directive above.

Scans tracked files, so generated output (``docs/sphinx/user/arguments.rst``)
is excluded for free while its tracked source (``arguments.yaml``) is checked.

Run with no arguments from anywhere in the repo.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

VERSION = re.compile(r"^\d+\.\d+\.\d+$")

# The Doxygen aliases, including the multi-paragraph `desc` spellings.
DOXYGEN = re.compile(r"@version(?:added|changed)(?:desc)?\{([^}]*)\}")

# Hand-written reST. The generated file is untracked and so never reaches here.
REST = re.compile(r"^\s*\.\.\s+version(?:added|changed)::\s*(\S+)")

# `added: "2.0.0"` in the docs YAML that feeds the reST directive. Keyed on the
# value looking like a version rather than on one filename, so a docs YAML
# added later is covered without editing this list.
YAML = re.compile(r"^\s*(?:added|changed):\s*[\"']?(\d[^\"'\s#]*)[\"']?")

SOURCE_SUFFIXES = {".hpp", ".cpp", ".h", ".c", ".mm", ".in", ".fstring", ".rst", ".md"}

# This file names a bad version in its own prose.
EXEMPT = {"devtools/check_version_tags.py"}


def repo_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def current_version(root: Path) -> tuple[int, int, int]:
    """Read EINSUMS_VERSION_MAJOR/MINOR/PATCH out of the top-level CMakeLists.txt."""
    text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"^set\(EINSUMS_VERSION_{field}\s+(\d+)\)", text, re.MULTILINE)
        if not m:
            raise SystemExit(
                f"check_version_tags: EINSUMS_VERSION_{field} not found in CMakeLists.txt"
            )
        parts.append(int(m.group(1)))
    return tuple(parts)  # type: ignore[return-value]


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    return [root / f for f in out]


def main() -> int:
    root = repo_root()
    current = current_version(root)
    shown = ".".join(str(p) for p in current)

    ahead: list[str] = []
    malformed: list[str] = []

    for path in tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in EXEMPT:
            continue
        is_yaml = path.suffix in {".yaml", ".yml"}
        if not is_yaml and path.suffix not in SOURCE_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue

        for number, line in enumerate(lines, start=1):
            found = list(DOXYGEN.findall(line))
            if m := REST.match(line):
                found.append(m.group(1))
            if is_yaml and (m := YAML.match(line)):
                found.append(m.group(1))

            for value in found:
                where = f"{rel}:{number}: {line.strip()}"
                if not VERSION.match(value):
                    malformed.append(where)
                elif tuple(int(p) for p in value.split(".")) > current:
                    ahead.append(where)

    if ahead:
        print(
            f"Version tags naming a release later than {shown}:\n",
            file=sys.stderr,
        )
        for entry in ahead:
            print(f"  {entry}", file=sys.stderr)
        print(
            f"\nThe version under development is {shown} (EINSUMS_VERSION_MAJOR/MINOR/PATCH\n"
            "in CMakeLists.txt). A tag is rendered verbatim and checked by nothing else,\n"
            "so this publishes a release note for a version that does not exist. Tag new\n"
            f"API with {{{shown}}}; do not pre-date it to the next release.",
            file=sys.stderr,
        )

    if malformed:
        print("\nVersion tags that are not MAJOR.MINOR.PATCH:\n", file=sys.stderr)
        for entry in malformed:
            print(f"  {entry}", file=sys.stderr)
        print(
            "\nThe value is substituted into the docs as written, so a typo here reaches\n"
            "the rendered page intact.",
            file=sys.stderr,
        )

    if ahead or malformed:
        return 1

    print(f"version tags: OK (current {shown})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
