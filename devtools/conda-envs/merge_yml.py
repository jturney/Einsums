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


def conda_arch_suffix(system, machine=None):
    """Return the conda platform suffix (e.g. ``linux-64``, ``linux-aarch64``).

    conda packages name their per-platform builds with a ``<os>-<arch>`` suffix
    (``gcc_linux-64``, ``clang_osx-arm64`` ...). Hardcoding ``linux-64`` breaks
    when this script runs inside an arm64 Linux container, because the env then
    asks for x86_64 compiler binaries that aren't on PATH inside the arm64 image,
    and CMake's BLAS/LAPACK probes link-fail because there's no compiler to
    drive them. Detect the running architecture and emit the right suffix.

    ``machine`` defaults to ``platform.machine()``; pass an override for tests.
    """
    if machine is None:
        machine = platform.machine()
    machine = machine.lower()

    if system == "Linux":
        if machine in ("x86_64", "amd64"):
            return "linux-64"
        if machine in ("aarch64", "arm64"):
            return "linux-aarch64"
        if machine in ("ppc64le",):
            return "linux-ppc64le"
        raise SystemExit(f"Unsupported Linux arch: {machine!r}.")
    if system == "Darwin":
        if machine in ("x86_64", "amd64"):
            return "osx-64"
        if machine in ("arm64", "aarch64"):
            return "osx-arm64"
        raise SystemExit(f"Unsupported macOS arch: {machine!r}.")
    if system == "Windows":
        # Windows arm64 isn't supported by our toolchains today.
        return "win-64"
    raise SystemExit(f"Unsupported system: {system!r}.")


def compiler_packages(compiler, system):
    """Return the conda packages providing the requested C/C++ toolchain.

    Notes:
      * GCC ships its own OpenMP runtime (libgomp); Clang does not, so every
        clang leg pulls ``llvm-openmp`` explicitly. conda's ``_openmp_mutex``
        then pins a single OpenMP runtime, avoiding the libgomp/libomp
        double-load abort. einsums requires OpenMP (find_package(OpenMP REQUIRED)).
      * Intel icx/icpx use ``--gcc-toolchain=$CONDA_PREFIX`` for libstdc++, so
        the intel leg also pulls the pinned GCC for a modern standard library.
      * The conda arch suffix (``linux-64`` / ``linux-aarch64`` / ...) is
        resolved at runtime via ``conda_arch_suffix`` so this script does the
        right thing inside an arm64 container.
    """
    if compiler == "default":
        compiler = PLATFORM_DEFAULT_COMPILER[system]

    arch = conda_arch_suffix(system)

    if compiler == "gcc":
        if system != "Linux":
            raise SystemExit(f"gcc is only supported on Linux (requested on {system}).")
        return [f"gcc_{arch}={GCC_VERSION}.*", f"gxx_{arch}={GCC_VERSION}.*"]

    if compiler == "clang":
        if system == "Linux":
            return [
                f"clang_{arch}>={CLANG_VERSION}",
                f"clangxx_{arch}>={CLANG_VERSION}",
                "clang-tools",
                "llvm-openmp",
            ]
        if system == "Darwin":
            return [
                f"clang_{arch}>={CLANG_VERSION}",
                f"clangxx_{arch}>={CLANG_VERSION}",
                "clang-tools",
                "llvm-openmp",
            ]
        if system == "Windows":
            return ["clangdev"]
        raise SystemExit(f"Unknown platform for clang: {system}.")

    if compiler == "intel":
        # dpcpp is x86_64-only.
        if system != "Linux" or arch != "linux-64":
            raise SystemExit(
                f"intel (dpcpp) is only supported on Linux x86_64 "
                f"(requested on {system} / {arch})."
            )
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

    # einsums-pybind, the libtooling-based code generator for the Python
    # bindings, is built whenever EINSUMS_BUILD_PYTHON=ON, regardless of which
    # compiler builds the rest of einsums. It does find_package(Clang/LLVM CONFIG)
    # and links clangTooling/clangFormat/.../LLVMSupport, so the env always needs
    # the LLVM/Clang development packages. clangdev pulls clang, clangxx,
    # clang-tools and llvmdev transitively, so it satisfies both the configure-time
    # find_package() calls and the runtime resource headers (lib/clang/<ver>/include).
    # Pinned to the clang toolchain version for ABI consistency on the clang legs.
    # Windows already injects ``clangdev`` as its compiler, so skip the duplicate.
    if system != "Windows":
        merged["dependencies"].append(f"clangdev={CLANG_VERSION}.*")
        merged["dependencies"].append(f"llvmdev={CLANG_VERSION}.*")

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
