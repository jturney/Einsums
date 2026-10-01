#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""The SIMD headers' instruction-set namespace must name every feature they test.

The SIMD headers compile differently for each set of target features, so they
declare everything inside an inline namespace, ``isa_<tier>_<ext>...``, whose
name Platform.hpp builds from the compiler's feature macros. Two translation
units built for different features then cannot share a definition: the linker
would otherwise keep one copy of, say, ``native_lanes<float>`` and hand it to
both. That only holds while the name changes whenever a definition does, which
is to say while it encodes every feature macro a header tests. A header that
starts testing a new one (a new AVX-512 subset, an SVE macro) without a piece
for it in the tag silently reopens the hazard for that feature.

So this checks, over the SIMD module's headers:
  - every compiler-defined macro (``__X__`` or ``_M_X``) that a preprocessor
    condition tests is named between ``// BEGIN ISA TAG`` and
    ``// END ISA TAG`` in Platform.hpp, apart from the compiler and platform
    identities in IDENTITY, which are the same across one link;
  - every header that opens ``einsums::simd`` also opens the instruction-set
    namespace, except the run-time headers in RUNTIME, which compile the same
    for every feature set and must not test feature macros at all.

Run with no arguments from anywhere in the repo.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HEADERS = Path("libs/Einsums/SIMD/include/Einsums/SIMD")
PLATFORM = HEADERS / "Platform.hpp"

# Headers that describe the machine at run time; they stay in einsums::simd.
RUNTIME = {"RuntimeFeatures.hpp", "RungLadder.hpp", "Options.hpp"}

# Compiler and platform identities: one link is built by one compiler for one
# platform, so they never distinguish two translation units in it.
IDENTITY = {"__clang__", "__GNUC__", "_MSC_VER", "__APPLE__", "__NVCC__", "__CUDACC__"}

CONDITION = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif|elifdef|elifndef)\b(.*)$")
FEATURE = re.compile(r"\b(__[A-Za-z0-9_]+__|_M_[A-Z0-9_]+)\b")
BEGIN_SIMD = re.compile(r"^EINSUMS_NAMESPACE_BEGIN\(simd\)\s*$", re.M)
BEGIN_ISA = re.compile(r"^EINSUMS_SIMD_ISA_NAMESPACE_BEGIN\(\)\s*$", re.M)


def tested_features(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        m = CONDITION.match(line)
        if m:
            # __has_builtin and friends are operators, not feature macros.
            found.update(n for n in FEATURE.findall(m.group(1)) if not n.startswith("__has_"))
    return found


def main() -> int:
    root = Path(
        subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    )
    platform = (root / PLATFORM).read_text(encoding="utf-8")
    try:
        tag = platform[platform.index("// BEGIN ISA TAG") : platform.index("// END ISA TAG")]
    except ValueError:
        print(f"{PLATFORM}: the // BEGIN ISA TAG ... // END ISA TAG markers are missing")
        return 1
    named = set(FEATURE.findall(tag))

    problems: list[str] = []
    for path in sorted((root / HEADERS).glob("*.hpp")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        features = tested_features(text) - IDENTITY
        if path.name in RUNTIME:
            for name in sorted(features):
                problems.append(f"{rel}: tests {name}, but this header is outside the instruction-set namespace")
            continue
        for name in sorted(features - named):
            problems.append(f"{rel}: tests {name}, which the instruction-set tag in {PLATFORM} does not name")
        if BEGIN_SIMD.search(text) and not BEGIN_ISA.search(text):
            problems.append(f"{rel}: opens einsums::simd without EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()")

    if problems:
        print("\n".join(problems))
        print(
            "\nEvery feature the SIMD headers test must change the instruction-set namespace's name;"
            f" add a piece for it between the ISA TAG markers in {PLATFORM}."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
