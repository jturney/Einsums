# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Generate a conda environment file for building Einsums.

The environment is composed from compiler-agnostic snippets (``snippets/common.yml``,
``snippets/blas/<blas>.yml``, ``snippets/os/<platform>.yml`` and optionally
``snippets/docs.yml``) plus a platform-aware C/C++ toolchain selected here.

The toolchain is defined in this script rather than in the snippets because
conda's compiler package names are platform specific (``gxx_linux-64`` vs
``clangxx_osx-arm64`` ...), so a static snippet cannot express "the latest
clang for whatever OS I'm on". Bump the version pins below to update the
compilers used everywhere (CI and local dev).
"""

from ruamel.yaml import YAML
import os
import platform
import argparse

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# --- Toolchain version pins (single source of truth) ------------------------
# Bump these to move CI + local dev onto newer compilers.
GCC_VERSION = "15"  # gcc/g++ on Linux            -> gcc_linux-64 / gxx_linux-64
CLANG_VERSION = "22"  # clang/clang++ (Linux + macOS)
DPCPP_VERSION = "2026.0.0"  # Intel oneAPI icx/icpx on Linux -> dpcpp_linux-64

# The compiler implied by ``--compiler default`` on each platform.
PLATFORM_DEFAULT_COMPILER = {
    "Linux": "gcc",
    "Darwin": "clang",
    "Windows": "clang",
}


def compiler_packages(compiler, system):
    """Return the conda packages providing the requested C/C++ toolchain.

    Notes:
      * GCC ships its own OpenMP runtime (libgomp); Clang does not, so every
        clang leg pulls ``llvm-openmp`` explicitly. conda's ``_openmp_mutex``
        then pins a single OpenMP runtime, avoiding the libgomp/libomp
        double-load abort. einsums requires OpenMP (find_package(OpenMP REQUIRED)).
      * Intel icx/icpx use ``--gcc-toolchain=$CONDA_PREFIX`` for libstdc++, so
        the intel leg also pulls the pinned GCC for a modern standard library.
    """
    if compiler == "default":
        compiler = PLATFORM_DEFAULT_COMPILER[system]

    if compiler == "gcc":
        if system != "Linux":
            raise SystemExit(f"gcc is only supported on Linux (requested on {system}).")
        return [f"gcc_linux-64={GCC_VERSION}.*", f"gxx_linux-64={GCC_VERSION}.*"]

    if compiler == "clang":
        if system == "Linux":
            return [
                f"clang_linux-64>={CLANG_VERSION}",
                f"clangxx_linux-64>={CLANG_VERSION}",
                "clang-tools",
                "llvm-openmp",
            ]
        if system == "Darwin":
            return [
                f"clang_osx-arm64>={CLANG_VERSION}",
                f"clangxx_osx-arm64>={CLANG_VERSION}",
                "clang-tools",
                "llvm-openmp",
            ]
        if system == "Windows":
            return ["clangdev"]
        raise SystemExit(f"Unknown platform for clang: {system}.")

    if compiler == "intel":
        if system != "Linux":
            raise SystemExit(f"intel (dpcpp) is only supported on Linux (requested on {system}).")
        return [
            f"dpcpp_linux-64={DPCPP_VERSION}",
            f"gcc_linux-64={GCC_VERSION}.*",
            f"gxx_linux-64={GCC_VERSION}.*",
        ]

    raise SystemExit(f"Unknown compiler: {compiler!r}.")


def merge_environment(output_file, system, compiler, blas, docs):
    yaml = YAML()
    merged = {"name": "einsums-dev", "channels": [], "dependencies": []}

    snippets = [
        "snippets/common.yml",
        f"snippets/blas/{blas}.yml",
        f"snippets/os/{system}.yml",
    ]
    if docs:
        snippets.append("snippets/docs.yml")

    for rel in snippets:
        path = os.path.join(DIR_PATH, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = yaml.load(f) or {}
        merged["channels"].extend(data.get("channels") or [])
        merged["dependencies"].extend(data.get("dependencies") or [])

    # Inject the platform-aware toolchain.
    merged["dependencies"].extend(compiler_packages(compiler, system))

    # cpptrace has no Windows package.
    if system == "Windows" and "cpptrace" in merged["dependencies"]:
        merged["dependencies"].remove("cpptrace")

    # Stable de-duplication (preserves first-seen order).
    merged["channels"] = list(dict.fromkeys(merged["channels"]))
    merged["dependencies"] = list(dict.fromkeys(merged["dependencies"]))

    with open(output_file, "w") as f:
        yaml.dump(merged, f)

    print(f"Wrote {output_file}  [{system} / {compiler} / {blas}{' / docs' if docs else ''}]")
    print("  toolchain: " + ", ".join(compiler_packages(compiler, system)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="merge_yml.py",
        description="Creates a conda environment yml file to develop Einsums.",
        usage="""

%(prog)s [OPTIONS] [COMPILER=default] [BLAS=openblas]

Options:
  --docs    Include documentation build dependencies (Sphinx, Doxygen, etc.)""",
    )

    parser.add_argument("--output", help="Name of the output yml file", default="conda.yml")
    parser.add_argument(
        "--docs", help="Install packages needed to build documentation", action="store_true"
    )
    parser.add_argument(
        "compiler",
        nargs="?",
        choices=["default", "gcc", "clang", "intel"],
        default="default",
        help="C/C++ toolchain (default: the platform default -- gcc on Linux, clang on macOS/Windows).",
    )
    parser.add_argument(
        "blas",
        nargs="?",
        choices=["accelerate", "mkl", "openblas"],
        default="openblas",
        help="BLAS library (optional; choices: accelerate, mkl, openblas).",
    )

    args = parser.parse_args()
    system = platform.system()

    # BLAS auto-selection: some (compiler, platform) pairs only make sense with
    # one BLAS, so pin it and tell the user what was chosen.
    if args.compiler == "intel" and args.blas != "mkl":
        print("Intel toolchain selected: forcing BLAS=mkl.")
        args.blas = "mkl"
    if system == "Darwin" and args.blas != "accelerate":
        print("macOS detected: forcing BLAS=accelerate.")
        args.blas = "accelerate"
    if system == "Windows" and args.blas != "mkl":
        print("Windows detected: forcing BLAS=mkl.")
        args.blas = "mkl"

    merge_environment(args.output, system, args.compiler, args.blas, args.docs)
