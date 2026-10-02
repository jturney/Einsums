# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums info``: what this Einsums is and what it runs on, for a bug report.

Reads the build configuration compiled into ``_core`` without starting the runtime.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Optional packages the tools use, and what each one adds.
OPTIONAL = {
    "numpy": "array interop",
    "textual": "einsums profiler",
    "textual_plotext": "profiler timeline and roofline plots",
    "zeroconf": "profiler server discovery over mDNS",
}

# x86-64 micro-architecture levels by the /proc/cpuinfo flags each requires (psABI).
_X86_LEVELS = (
    ("v4", {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}),
    ("v3", {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "abm"}),
    ("v2", {"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"}),
)


def register(subparsers) -> None:
    p = subparsers.add_parser("info", help="Show the build, the Python environment, and the machine")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=run)


def _cpu() -> dict[str, object]:
    info: dict[str, object] = {"machine": platform.machine(), "logical_cores": os.cpu_count()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    if model := re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M):
        info["model"] = model.group(1).strip()
    if flags := re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.M):
        have = set(flags.group(1).split())
        info["x86_64_level"] = next((level for level, need in _X86_LEVELS if need <= have), "baseline")
    elif sys.platform == "darwin":
        try:
            info["model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    info["simd_override"] = {k: os.environ[k] for k in ("EINSUMS_SIMD_ARCH", "STRIPES_ARCH") if k in os.environ}
    return info


def _linked_libraries(binary: Path) -> list[str]:
    """Shared libraries *binary* resolves to, from ldd or otool (empty if neither is available)."""
    if sys.platform == "darwin" and shutil.which("otool"):
        out = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True).stdout
        return [line.split()[0] for line in out.splitlines()[1:] if line.strip()]
    if shutil.which("ldd"):
        out = subprocess.run(["ldd", str(binary)], capture_output=True, text=True).stdout
        return [m.group(1) for m in re.finditer(r"=>\s+(\S+)", out)]
    return []


def _blas(libraries: list[str]) -> list[str]:
    pattern = re.compile(r"(openblas|mkl|blis|flexiblas|accelerate|essl|armpl|libblas|liblapack)", re.I)
    return sorted({Path(lib).name for lib in libraries if pattern.search(Path(lib).name)})


def collect() -> dict[str, object]:
    from .. import _core

    version = _core._version
    config = {
        m.group(1): m.group(2) == "ON"
        for m in re.finditer(r"^\s*(EINSUMS_WITH_\w+)=(\w+)", version.configuration_string(), re.M)
    }
    full = version.full_build_string()
    git = re.search(r"Git:\s*(\w+)", full)
    core_path = Path(_core.__file__).resolve()
    libraries = _linked_libraries(core_path)
    einsums_lib = next((str(Path(lib).resolve()) for lib in libraries if "libEinsums" in Path(lib).name), None)
    if einsums_lib:  # BLAS hangs off libEinsums, not off _core
        libraries += _linked_libraries(Path(einsums_lib))
    optional = {}
    for module, purpose in OPTIONAL.items():
        if importlib.util.find_spec(module) is None:
            optional[module] = {"version": None, "for": purpose}
            continue
        try:
            found = importlib.metadata.version(module.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            found = "installed"
        optional[module] = {"version": found, "for": purpose}
    return {
        "einsums": {
            "version": version.full_version_as_string(),
            "git_commit": git.group(1) if git else None,
            "build_type": version.build_type(),
            "build_date": version.build_date_time(),
            "package": str(core_path.parent),
            "libEinsums": einsums_lib,
            "blas": _blas(libraries),
            "options": config,
        },
        "python": {"version": platform.python_version(), "executable": sys.executable, "optional": optional},
        "platform": {"system": platform.platform()},
        "cpu": _cpu(),
    }


def run(args: argparse.Namespace) -> int:
    info = collect()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    e, py, cpu = info["einsums"], info["python"], info["cpu"]
    rows = [
        ("Einsums", f"{e['version']} ({e['build_type']}), commit {e['git_commit'] or '?'}, built {e['build_date']}"),
        ("Package", e["package"]),
        ("libEinsums", e["libEinsums"] or "(not found)"),
        ("BLAS", ", ".join(e["blas"]) or "(could not tell)"),
        ("Enabled", " ".join(k.removeprefix("EINSUMS_WITH_") for k, on in e["options"].items() if on) or "-"),
        ("Disabled", " ".join(k.removeprefix("EINSUMS_WITH_") for k, on in e["options"].items() if not on) or "-"),
        ("Python", f"{py['version']} ({py['executable']})"),
        (
            "Optional",
            ", ".join(f"{m} {d['version']}" if d["version"] else f"{m} missing ({d['for']})" for m, d in py["optional"].items()),
        ),
        ("Platform", info["platform"]["system"]),
        ("CPU", f"{cpu.get('model', cpu['machine'])}, {cpu['logical_cores']} logical cores"),
    ]
    if "x86_64_level" in cpu:
        rows.append(("ISA level", f"x86-64-{cpu['x86_64_level']} (the highest SIMD rung this CPU supports)"))
    if cpu["simd_override"]:
        rows.append(("SIMD override", " ".join(f"{k}={v}" for k, v in cpu["simd_override"].items())))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0
