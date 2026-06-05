# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Make ``import einsums`` work when running ``pytest`` from the repo root.

The compiled Python module lives at ``<repo>/build/lib/einsums/_core.*.so``
after a CMake build. The per-module ComputeGraph tests (and friends) are
registered through ``einsums_add_python_unit_test`` which sets
``PYTHONPATH=${CMAKE_BINARY_DIR}/lib`` for ctest, but tests under
``tests/`` are intended to be runnable directly via ``pytest`` from the
repo root and need the same path injected.

We probe a small list of likely build directories and prepend the first
one that contains an ``einsums`` package. The honored cases are:

  * ``EINSUMS_BUILD_DIR``  — explicit override (set by CI / dev configs).
  * ``./build/lib``        — the conventional out-of-source build location.
  * ``./cmake-build-*/lib``— common IDE-default build directories.

If none of them exist we leave ``sys.path`` alone and let the import
fail with the usual ``ModuleNotFoundError`` — that's a clearer signal
than silently skipping the test.
"""

from __future__ import annotations

import os
import pathlib
import sys


def _candidate_build_libs() -> list[pathlib.Path]:
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    candidates: list[pathlib.Path] = []
    env = os.environ.get("EINSUMS_BUILD_DIR")
    if env:
        candidates.append(pathlib.Path(env) / "lib")
    candidates.append(repo_root / "build" / "lib")
    candidates.extend(sorted((repo_root).glob("cmake-build-*/lib")))
    return candidates


def _einsums_already_importable() -> bool:
    for entry in sys.path:
        if (pathlib.Path(entry) / "einsums" / "__init__.py").is_file():
            return True
    return False


if not _einsums_already_importable():
    for candidate in _candidate_build_libs():
        if (candidate / "einsums" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            break
