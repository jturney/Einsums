#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Run apiary's doc linter over every rule, failing on any finding.

The linter cross-checks each doc comment against the signature it documents
(``@param``/``@tparam`` names, partial parameter lists, ``@return`` on void),
checks that the reST the comment converts to is well formed, and resolves
``[[Type/member]]`` links. A finding is a page that renders wrong or documents
something that is not there, so warnings fail too, as they do for Sphinx.

The C++ reference generator decides at build time which docs JSON files it
writes (one per module, or one per header when a module cannot be parsed
whole), so the inputs may be directories, and each contributes every ``*.json``
directly inside it.

Usage::

    lint_doc_comments.py --lint <apiary_doc_lint.py> <file-or-dir> [...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def expand(inputs: list[str]) -> list[str]:
    files: list[str] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(str(p) for p in sorted(path.glob("*.json")))
        else:
            files.append(str(path))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lint", required=True, help="path to apiary_doc_lint.py")
    ap.add_argument("inputs", nargs="+", help="docs JSON files, or directories of them")
    args = ap.parse_args()

    files = expand(args.inputs)
    if not files:
        print(f"lint_doc_comments: no docs JSON found in {' '.join(args.inputs)}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, args.lint, "--strict", "--check-links", *files])


if __name__ == "__main__":
    sys.exit(main())
