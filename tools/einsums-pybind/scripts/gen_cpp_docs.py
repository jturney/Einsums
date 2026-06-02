#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Generate the C++ API reference pages for selected modules (Option 2).

For each public header of each selected ``lib/module``, runs
``einsums-pybind --emit-cpp-docs-json`` and ``render_cpp_rst.py`` to produce a
cpp-domain reStructuredText page, replacing Breathe's ``autodoxygenfile``.

Compile flags are taken from a representative ``einsums-pybind`` codegen
command already present in the build's ``build.ninja`` (the Tensor module's,
whose transitive include set covers the whole library) — so we don't
re-derive per-module flags here.

Usage::

    gen_cpp_docs.py --source-dir <repo> --build-dir <build> --tool <einsums-pybind> \
                    --out-dir <dir> --modules Einsums/BLASVendor Einsums/Concepts
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"gen_cpp_docs: {msg}", file=sys.stderr)


def universal_flags(build_dir: Path, source_dir: Path) -> list[str]:
    """Compile flags for parsing any module's headers.

    Start from a representative einsums-pybind command in build.ninja (it
    carries the resource-dir / isysroot / -std / system flags libtooling
    needs), then append EVERY module's include dir — both source and
    build-tree (for generated ``Defines.hpp``). The Tensor command alone
    only covers Tensor's transitive deps, so headers of modules Tensor
    doesn't depend on (ComputeGraph, Comm, GPU, ...) wouldn't resolve their
    own includes and would parse to nothing."""
    ninja = (build_dir / "build.ninja").read_text()
    m = re.search(r"einsums-pybind --register-function einsums_pybind_register_Tensor [^\n]*", ninja)
    if not m:
        raise SystemExit("gen_cpp_docs: no Tensor pybind command in build.ninja (need EINSUMS_BUILD_PYTHON)")
    toks = shlex.split(m.group(0))
    flags = toks[toks.index("--"):]
    seen = {f for f in flags if f.startswith("-I")}
    for base in (source_dir / "libs", build_dir / "libs"):
        for inc in sorted(base.glob("*/*/include")):
            flag = f"-I{inc}"
            if flag not in seen:
                seen.add(flag)
                flags.append(flag)
    return flags


def sanitized(relheader: str) -> str:
    return relheader.replace("/", "_").replace(".", "_")


def header_relpath(header: Path) -> str | None:
    """The include-relative path (after ``include/``) — what --source-include
    and Breathe's autodoxygenfile key on."""
    parts = header.parts
    if "include" not in parts:
        return None
    return "/".join(parts[parts.index("include") + 1:])


def collect_template_params(doc: dict, out: set[str]) -> None:
    """Every template-parameter name anywhere in the document. These are
    never cross-reference targets, so the docs build nitpick-ignores them."""
    def from_callable(c: dict) -> None:
        out.update(c.get("template_params", []) or [])

    def from_class(cl: dict) -> None:
        out.update(cl.get("template_params", []) or [])
        for m in cl.get("methods", []) + cl.get("constructors", []):
            from_callable(m)
        for n in cl.get("nested_classes", []):
            from_class(n)

    for cl in doc.get("classes", []):
        from_class(cl)
    for fn in doc.get("functions", []):
        from_callable(fn)
    for td in doc.get("typedefs", []):
        out.update(td.get("template_params", []) or [])
    for c in doc.get("concepts", []):
        out.update(c.get("template_params", []) or [])


def gen_header(tool: str, flags: list[str], header: Path, relheader: str, out_dir: Path,
               tparams: set[str], undoc: set[str] | None = None) -> bool:
    title = relheader
    json_out = out_dir / (sanitized(relheader) + ".json")
    rst_out = out_dir / (sanitized(relheader) + ".rst")
    cmd = [tool, "--emit-cpp-docs-json", "--module", "einsums",
           "--source-include", relheader, str(header), *flags]
    if undoc is not None:
        cmd.insert(2, "--report-undocumented")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if undoc is not None:
        # The tool prints "file:line:col: undocumented <kind> <name>" to stderr
        # (mixed with clang include-trace noise, which we drop). Collect into a
        # shared set so the same entity reported by transitive includes across
        # module runs is deduplicated.
        for ln in res.stderr.splitlines():
            if ": undocumented " in ln:
                undoc.add(ln.strip())
    json_out.write_text(res.stdout)
    if not res.stdout.strip():
        return False
    try:
        collect_template_params(json.loads(res.stdout), tparams)
    except json.JSONDecodeError:
        pass
    render = subprocess.run(
        [sys.executable, str(SCRIPTS / "render_cpp_rst.py"), str(json_out),
         "--title", title, "--output", str(rst_out), "--embed"],
        capture_output=True, text=True)
    if render.returncode != 0:
        log(f"render failed for {relheader}: {render.stderr.strip()[:200]}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--tool", required=True, help="einsums-pybind binary")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modules", nargs="+", default=None,
                    help="lib/module pairs, e.g. Einsums/BLASVendor. When omitted, every "
                         "libs/<lib>/<module>/include directory is discovered automatically "
                         "(handy for a full --report-undocumented sweep without pasting the list).")
    ap.add_argument("--report-undocumented", action="store_true",
                    help="Also collect a deduplicated punch-list of public C++ entities missing a "
                         "doc comment. Prints the sorted list to stdout and writes it to "
                         "<out-dir>/undocumented.txt. Does not change the generated pages.")
    args = ap.parse_args()

    source = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = universal_flags(Path(args.build_dir), source)

    modules = args.modules
    if modules is None:
        # Auto-discover: every libs/<lib>/<module>/include directory.
        modules = sorted(f"{inc.parts[-3]}/{inc.parts[-2]}"
                         for inc in (source / "libs").glob("*/*/include"))
        log(f"auto-discovered {len(modules)} modules under {source / 'libs'}")

    total = 0
    tparams: set[str] = set()
    undoc: set[str] | None = set() if args.report_undocumented else None
    for mod in modules:
        lib, module = mod.split("/", 1)
        inc = source / "libs" / lib / module / "include"
        if not inc.is_dir():
            log(f"skip {mod}: no include dir")
            continue
        for header in sorted(inc.rglob("*.hpp")):
            rel = header_relpath(header)
            if rel is None:
                continue
            if gen_header(args.tool, flags, header, rel, out_dir, tparams, undoc):
                total += 1
        log(f"{mod}: generated pages")
    # The collected template-parameter names — the docs build adds these to
    # nitpick_ignore (they are never cpp cross-reference targets).
    (out_dir / "template_params.txt").write_text("\n".join(sorted(tparams)) + "\n")
    log(f"wrote {total} header pages + {len(tparams)} template-param names into {out_dir}")
    if undoc is not None:
        report = "\n".join(sorted(undoc))
        (out_dir / "undocumented.txt").write_text(report + ("\n" if report else ""))
        if report:
            print(report)
        log(f"{len(undoc)} undocumented public entit{'y' if len(undoc) == 1 else 'ies'} "
            f"(written to {out_dir / 'undocumented.txt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
