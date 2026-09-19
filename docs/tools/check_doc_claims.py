#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Check the prose claims in the documentation against the code they describe.

The generated API reference cannot drift, because apiary derives it from the headers. The
hand-written narrative around it can, and did: an audit found the opening sentence of the manual
claiming the wrong language standard, the Python section claiming an interop that raises
``TypeError``, a pass count off by a factor of four, and an ``einsums::read`` free function that
has never existed in any of the three places it was documented.

What those four have in common is that nothing mechanical looked at them. This script is the
thing that looks. Each check is deliberately narrow: it verifies a claim a machine can settle,
and says nothing about claims it cannot. A check that guesses produces false failures, and a
doc test that cries wolf is one somebody turns off.

Usage::

    check_doc_claims.py --docs-dir docs/sphinx --repo-root . [--cxx-standard 20]
                        [--cpp-index <dir of generated cppdocs rst>]

Exits non-zero, listing every failure with its file and line, if any claim disagrees with the
code. Checks whose inputs are unavailable report SKIP rather than passing quietly.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field


@dataclass
class Result:
    """One check's verdict, with a reason a reader can act on."""

    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""
    failures: list[str] = field(default_factory=list)


def rst_files(docs_dir: str) -> list[str]:
    """Every reStructuredText source under ``docs_dir``, excluding generated trees.

    ``api/`` and ``reference/python`` are written by apiary at build time from the headers, so
    they cannot carry a hand-written claim and checking them would only add noise.
    """
    out = []
    for root, dirs, files in os.walk(docs_dir):
        dirs[:] = [d for d in dirs if d not in {"api", "_build", "generated"}]
        if os.path.join("reference", "python") in root:
            continue
        out.extend(os.path.join(root, f) for f in files if f.endswith(".rst"))
    return sorted(out)


def iter_lines(paths: list[str]):
    """(path, 1-based line number, text) for every line of every file."""
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                yield p, n, line


_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+code-block::\s*(\S+)\s*$")


def iter_code_lines(paths: list[str], languages: set[str]):
    """(path, line number, text) for lines inside ``code-block`` directives of @p languages.

    Restricting a check to code blocks is what keeps it from fighting Sphinx. A name written as
    ``:cpp:func:`einsums::foo``` is already resolved, precisely, by the nitpicky build; a name
    written in prose may be a namespace, a user-side symbol, or an entity behind a build option.
    A name inside a C++ block is code somebody could paste, and that is the claim worth testing.
    """
    languages = {lang.lower() for lang in languages}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        i = 0
        while i < len(lines):
            m = _DIRECTIVE.match(lines[i].rstrip("\n"))
            if not m or m.group(2).lower() not in languages:
                i += 1
                continue
            indent = len(m.group(1))
            i += 1
            # Directive options and the blank line before the body.
            while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(":")):
                i += 1
            while i < len(lines):
                text = lines[i].rstrip("\n")
                if text.strip() and (len(text) - len(text.lstrip())) <= indent:
                    break
                yield path, i + 1, text
                i += 1


_ALIAS = re.compile(r"^\s*using\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_:]*)\s*[<;]")


def collect_alias_templates(root: str) -> dict[str, str]:
    """``einsums::<Alias>`` to the ``einsums::<Target>`` it names, from the public headers.

    Only the forward-declaration headers are scanned, which is where the type aliases the manual
    actually writes are declared. Reading them beats hard-coding the pairs, which would be one
    more thing to keep in step with the code.
    """
    out: dict[str, str] = {}
    for sub in ("Tensor/include/Einsums/Tensor/TensorForward.hpp",):
        path = os.path.join(root, "libs/Einsums", sub)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _ALIAS.match(line)
                if m:
                    target = m.group(2)
                    out[f"einsums::{m.group(1)}"] = target if "::" in target else f"einsums::{target}"
    return out


def rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


# ──────────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────────


def check_cxx_standard(docs: list[str], root: str, standard: str | None) -> Result:
    """The manual must not advertise a language standard the build does not use.

    The claim that started this: the front page opened with "Einsums is a C++23 tensor algebra
    library" while ``EINSUMS_WITH_CXX_STANDARD`` was 20. Prose may still discuss C++23 features
    supplied through the CXX23 compatibility module, so only the "is a C++NN library" shape of
    claim is checked rather than every mention of a standard.
    """
    if not standard:
        return Result("cxx-standard", "skip", "no --cxx-standard given")

    pattern = re.compile(r"is a C\+\+(\d\d) (?:\w+ )*?librar", re.IGNORECASE)
    failures = []
    for path, n, line in iter_lines(docs):
        m = pattern.search(line)
        if m and m.group(1) != standard:
            failures.append(
                f"{rel(path, root)}:{n}: claims a C++{m.group(1)} library, "
                f"but EINSUMS_WITH_CXX_STANDARD is {standard}"
            )
    if failures:
        return Result("cxx-standard", "fail", failures=failures)
    return Result("cxx-standard", "pass", f"no claim disagrees with C++{standard}")


def check_pass_count(docs: list[str], root: str) -> Result:
    """A stated number of optimization passes must match the default pipeline.

    ``build_default_passes()`` is the canonical list; every pass-manager factory is built from
    it. The manual said eleven when the pipeline held forty-five.
    """
    src = os.path.join(root, "libs/Einsums/ComputeGraph/src/Optimizer.cpp")
    if not os.path.isfile(src):
        return Result("pass-count", "skip", f"{rel(src, root)} not found")

    with open(src, encoding="utf-8", errors="replace") as fh:
        body = fh.read()
    start = body.find("build_default_passes")
    actual = len(re.findall(r"list\.push_back\(std::make_shared<passes::", body[start:])) if start >= 0 else 0
    if actual == 0:
        return Result("pass-count", "skip", "could not count passes in build_default_passes()")

    pattern = re.compile(r"\b(\d+)\s+optimization passes\b")
    failures = []
    for path, n, line in iter_lines(docs):
        m = pattern.search(line)
        if m and int(m.group(1)) != actual:
            failures.append(
                f"{rel(path, root)}:{n}: says {m.group(1)} optimization passes, "
                f"but build_default_passes() adds {actual}"
            )
    if failures:
        return Result("pass-count", "fail", failures=failures)
    return Result("pass-count", "pass", f"default pipeline has {actual} passes")


def check_python_names(docs: list[str], root: str) -> Result:
    """Every ``einsums.<dotted.name>`` the manual mentions must actually resolve.

    Import failures are a skip rather than a failure: a build without the Python bindings has
    nothing to check against, and failing there would make the test depend on an unrelated
    option.
    """
    try:
        import importlib

        import einsums
    except Exception as exc:  # noqa: BLE001 - any import problem means "cannot check"
        return Result("python-names", "skip", f"einsums not importable: {type(exc).__name__}")

    def resolves(dotted: str) -> bool:
        parts = dotted.split(".")
        obj = einsums
        for i, part in enumerate(parts[1:], start=2):
            if hasattr(obj, part):
                obj = getattr(obj, part)
                continue
            # A real subpackage may need an explicit import before attribute access works.
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except Exception:  # noqa: BLE001
                return False
        return True

    seen: dict[str, tuple[str, int]] = {}
    for path, n, line in iter_code_lines(docs, {"python", "py"}):
        for m in re.finditer(r"\beinsums(?:\.[A-Za-z_][A-Za-z0-9_]*)+", line):
            seen.setdefault(m.group(0), (path, n))

    failures = [
        f"{rel(p, root)}:{n}: '{name}' does not resolve in the installed einsums module"
        for name, (p, n) in sorted(seen.items())
        if not resolves(name)
    ]
    if failures:
        return Result("python-names", "fail", failures=failures)
    return Result("python-names", "pass", f"{len(seen)} names resolve")


def check_cpp_names(docs: list[str], root: str, cpp_index: str | None) -> Result:
    """Every ``einsums::qualified::name`` must appear in the generated C++ reference.

    The index is apiary's output, one page per documented entity, which is what makes this
    precise enough to be worth running: ``einsums::read`` was documented in three places and has
    no page, while a plain grep of the headers finds ``read`` as a member of ``TensorFile`` and
    concludes it exists.

    Namespaces appear in ``using namespace`` lines and have no page of their own, so they are
    allowed by name rather than by lookup.
    """
    if not cpp_index or not os.path.isdir(cpp_index):
        return Result("cpp-names", "skip", "no --cpp-index given (needs the docs build)")

    pages = set()
    for cur, _dirs, files in os.walk(cpp_index):
        del cur
        for f in files:
            if f.endswith(".rst"):
                pages.add(f[:-4])
    if not pages:
        return Result("cpp-names", "skip", f"no generated pages under {cpp_index}")

    # Names a documented example may legitimately write that are not library entities: the
    # program's own entry point, which the runtime takes as a callback.
    user_side = {"einsums::main"}

    namespaces = {
        "einsums",
        "einsums::compute_graph",
        "einsums::compute_graph::passes",
        "einsums::compute_graph::dispatch",
        "einsums::index",
        "einsums::tensor_algebra",
        "einsums::tensor_algebra::detail",
        "einsums::linear_algebra",
        "einsums::linear_algebra::detail",
        "einsums::detail",
        "einsums::option",
        "einsums::string_util",
        "einsums::blas",
        "einsums::gpu",
        "einsums::profile",
    }

    aliases = collect_alias_templates(root)

    def documented(dotted: str) -> bool:
        if dotted in pages:
            return True
        # A member or nested entity is documented on its parent's page.
        parts = dotted.split(".")
        return any(".".join(parts[:i]) in pages for i in range(len(parts) - 1, 1, -1))

    def known(name: str) -> bool:
        if name in namespaces or name in user_side:
            return True
        # apiary deliberately keeps implementation namespaces out of the public reference, so a
        # detail entity has no page even though it exists and may be worth naming in prose.
        if "::detail::" in name:
            return True
        dotted = name.replace("::", ".")
        if documented(dotted):
            return True
        # An alias template is documented under the template it names: einsums::Tensor is a
        # `using` for einsums::GeneralTensor, and only the latter gets a page. Resolving the
        # alias is what keeps the two most commonly written type names from failing this check.
        target = aliases.get(name)
        return bool(target) and documented(target.replace("::", "."))

    seen: dict[str, tuple[str, int]] = {}
    for path, n, line in iter_code_lines(docs, {"cpp", "c++"}):
        for m in re.finditer(r"\beinsums(?:::[A-Za-z_][A-Za-z0-9_]*)+", line):
            seen.setdefault(m.group(0), (path, n))

    failures = [
        f"{rel(p, root)}:{n}: '{name}' has no entry in the generated C++ reference"
        for name, (p, n) in sorted(seen.items())
        if not known(name)
    ]
    if failures:
        return Result("cpp-names", "fail", failures=failures)
    return Result("cpp-names", "pass", f"{len(seen)} names resolve against {len(pages)} pages")


# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs-dir", required=True, help="root of the reStructuredText sources")
    ap.add_argument("--repo-root", default=".", help="repository root, for resolving sources")
    ap.add_argument("--cxx-standard", default=None, help="EINSUMS_WITH_CXX_STANDARD, e.g. 20")
    ap.add_argument("--cpp-index", default=None, help="directory of generated C++ reference pages")
    args = ap.parse_args()

    if not os.path.isdir(args.docs_dir):
        print(f"error: --docs-dir '{args.docs_dir}' is not a directory", file=sys.stderr)
        return 2

    docs = rst_files(args.docs_dir)
    if not docs:
        print(f"error: no .rst files under '{args.docs_dir}'", file=sys.stderr)
        return 2

    results = [
        check_cxx_standard(docs, args.repo_root, args.cxx_standard),
        check_pass_count(docs, args.repo_root),
        check_python_names(docs, args.repo_root),
        check_cpp_names(docs, args.repo_root, args.cpp_index),
    ]

    print(f"checked {len(docs)} documentation files\n")
    for r in results:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[r.status]
        print(f"  {mark}  {r.name}{f'  ({r.detail})' if r.detail else ''}")
        for f in r.failures:
            print(f"          {f}")

    failed = [r for r in results if r.status == "fail"]
    if failed:
        total = sum(len(r.failures) for r in failed)
        print(f"\n{total} documentation claim(s) disagree with the code.")
        return 1

    print("\nevery checkable claim agrees with the code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
