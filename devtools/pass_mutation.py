#!/usr/bin/env python3
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Mutation testing for the ComputeGraph optimization passes.

Each pass defends itself with a small set of checks: feature guards
(``understands(graph, node)``), buffer identity (``graph.buffer_of(id)``),
span interference (``span_interferes(...)``) and feature tests
(``.covers(NodeFeature::X)``).  A check that no test notices when it is
removed is a check nothing defends.  This tool removes them one at a time,
rebuilds the Python module, runs the pass test suites, and reports every
mutant that survived.

    python devtools/pass_mutation.py list [--pass GLOB] [--operator OP]
    python devtools/pass_mutation.py run  [--pass GLOB] [--operator OP] [--limit N]
                                          [--results FILE] [--ctest REGEX]

Mutants are applied to the working tree one at a time and restored after
each run, including on Ctrl-C.  The restore refuses to overwrite a file that
changed under the tool; it saves the original beside it instead.  Results
are appended to a JSON file keyed by file, line, operator and source text,
so an interrupted sweep resumes where it stopped.  Every build goes through
``ninja-locked.sh``, and nothing else may build or test in the same build
directory while a sweep runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS_SOURCES = ROOT / "libs/Einsums/ComputeGraph/src/Passes"
TEST_DIR = ROOT / "libs/Einsums/ComputeGraph/tests/unit"
DEFAULT_SUITE = (
    "test_pass_audit_reproducers*_python.py",
    "test_fuzz_diff_*_python.py",
    "test_hyp_*_python.py",
)
NINJA_LOCKED = Path.home() / ".claude/hooks/ninja-locked.sh"


@dataclasses.dataclass(frozen=True)
class Mutant:
    file: Path
    line: int
    operator: str
    start: int  # byte offsets into the file text
    end: int
    original: str
    replacement: str

    @property
    def key(self) -> str:
        digest = hashlib.sha1(self.original.encode()).hexdigest()[:10]
        return f"{self.file.relative_to(ROOT)}:{self.line}:{self.operator}:{digest}"


def _matching_paren(text: str, open_index: int) -> int:
    """Index one past the parenthesis that closes text[open_index]."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced parentheses")


# Each operator maps a regex whose match ends at an opening parenthesis (or,
# for covers, at the whole expression) to a replacement for the call.
def _mutants_in(path: Path) -> list[Mutant]:
    text = path.read_text()
    out: list[Mutant] = []

    def add(operator: str, start: int, end: int, replacement: str) -> None:
        line = text.count("\n", 0, start) + 1
        out.append(Mutant(path, line, operator, start, end, text[start:end], replacement))

    # A feature guard that always says "understood": the pass treats every node
    # as one it models.  The declaration `understood_features()` is not a call site.
    for m in re.finditer(r"\bunderstands\(", text):
        before = text[max(0, m.start() - 16) : m.start()]
        if "bool " in before or "::" in before[-2:] or "options." in before:
            continue
        end = _matching_paren(text, m.end() - 1)
        add("understands", m.start(), end, "(true)")

    # Buffer identity dropped: the tensor id stands in for its buffer, as it
    # did before buffer_of existed.
    for m in re.finditer(r"[\w\]\)]+(?:\.|->)buffer_of\(", text):
        end = _matching_paren(text, m.end() - 1)
        add("buffer_of", m.start(), end, "(" + text[m.end() : end - 1] + ")")

    # Interference between the members of a rewrite span is never seen.
    for m in re.finditer(r"\bspan_interferes\(", text):
        if "bool " in text[max(0, m.start() - 16) : m.start()]:
            continue
        end = _matching_paren(text, m.end() - 1)
        add("span_interferes", m.start(), end, "(false)")

    # A feature test that never matches.
    for m in re.finditer(r"(?:\.|->)covers\(NodeFeature::\w+\)", text):
        add("covers", m.start(), m.end(), text[m.start() : m.end()] + " && false")

    return out


PASS_HEADERS = ROOT / "libs/Einsums/ComputeGraph/include/Einsums/ComputeGraph/Passes"
FEATURES_HEADER = ROOT / "libs/Einsums/ComputeGraph/include/Einsums/ComputeGraph/NodeFeatures.hpp"


def _declares_every_feature(stem: str) -> bool:
    """Whether the pass's ``understood_features()`` names every feature there is.

    Its ``understands`` guard then cannot return false, so replacing it with true
    is an equivalent mutant. The guard is not dead weight: a pass that lists the
    features rather than returning ``all()`` stops understanding the next one
    added, and the guard is what declines it. It is just nothing a test can kill.
    """
    header = PASS_HEADERS / f"{stem}.hpp"
    if not header.exists():
        return False
    text = header.read_text()
    m = re.search(r"understood_features\(\)[^{]*\{(.*?)\n\s*\}", text, re.S)
    if m is None:
        return False
    body = m.group(1)
    if "NodeFeatures::all()" in body:
        return True
    enum = re.search(r"enum class NodeFeature[^{]*\{(.*?)\};", FEATURES_HEADER.read_text(), re.S)
    every = set(re.findall(r"^\s*(\w+)\s*=", enum.group(1), re.M)) if enum else set()
    return bool(every) and every <= set(re.findall(r"NodeFeature::(\w+)", body))


def discover(pass_glob: str | None, operators: set[str] | None) -> list[Mutant]:
    files = sorted(PASS_SOURCES.glob("*.cpp"))
    mutants = []
    for path in files:
        if pass_glob and not fnmatch.fnmatch(path.stem, pass_glob):
            continue
        found = [m for m in _mutants_in(path) if not operators or m.operator in operators]
        if _declares_every_feature(path.stem):
            found = [m for m in found if m.operator != "understands"]
        mutants += found
    return mutants


class _Applied:
    """Applies one mutant and guarantees the restore, including on SIGINT/SIGTERM."""

    def __init__(self, mutant: Mutant):
        self.mutant = mutant
        self.original = mutant.file.read_text()
        if self.original[mutant.start : mutant.end] != mutant.original:
            raise RuntimeError(f"{mutant.file} changed since discovery; rerun to rediscover")
        self.mutated = self.original[: mutant.start] + mutant.replacement + self.original[mutant.end :]

    def __enter__(self):
        self._handlers = {s: signal.signal(s, self._on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        self.mutant.file.write_text(self.mutated)
        return self

    def _on_signal(self, signum, frame):
        raise KeyboardInterrupt

    def __exit__(self, *exc):
        path = self.mutant.file
        if path.read_text() == self.mutated:
            path.write_text(self.original)
        else:
            backup = path.with_suffix(path.suffix + ".mutation-orig")
            backup.write_text(self.original)
            print(f"!! {path} changed during the run; NOT restored. Original saved to {backup}", file=sys.stderr)
        for s, h in self._handlers.items():
            signal.signal(s, h)
        return False


def _core_target(build: Path) -> str:
    found = sorted((build / "lib/einsums").glob("_core*.so")) + sorted((build / "lib/einsums").glob("_core*.pyd"))
    if not found:
        sys.exit(f"no built _core module under {build}/lib/einsums; build the Python bindings first")
    return str(found[0].relative_to(build))


def _build(build: Path, targets: list[str]) -> tuple[bool, str]:
    cmd = [str(NINJA_LOCKED) if NINJA_LOCKED.exists() else "ninja", "-C", str(build), *targets]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


def _run_suite(build: Path, suite: list[Path], ctest: str | None, timeout: float, workers: int) -> tuple[str, str]:
    """Returns (outcome, detail): outcome is 'killed', 'survived' or 'timeout'."""
    env = dict(os.environ, PYTHONPATH=str(build / "lib"), EINSUMS_PASS_VERIFY="1")
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"]
    if workers > 1:
        # One file per worker: the firing-floor guards read counters the other
        # trials of their own file accumulate, so a file must not be split.
        cmd += ["-n", str(workers), "--dist", "loadfile"]
        # Split the cores between the workers rather than letting each start a full-width OpenMP team.
        env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or workers) // workers)))
    cmd += [str(f) for f in suite]
    try:
        proc = subprocess.run(cmd, cwd=TEST_DIR, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "pytest exceeded the timeout"
    if proc.returncode != 0:
        failed = [ln for ln in proc.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
        return "killed", (failed[0] if failed else proc.stdout[-600:])
    if ctest:
        try:
            proc = subprocess.run(
                ["ctest", "--test-dir", str(build), "-R", ctest, "--output-on-failure"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "timeout", "ctest exceeded the timeout"
        if proc.returncode != 0:
            failed = [ln.strip() for ln in proc.stdout.splitlines() if "(Failed)" in ln or "SEGFAULT" in ln]
            return "killed", (failed[0] if failed else "ctest failed")
    return "survived", ""


def _suite_files(patterns: list[str]) -> list[Path]:
    """The suite in pattern order: the targeted reproducers first, so a killed
    mutant usually stops (``-x``) within seconds instead of after the fuzzers."""
    files: list[Path] = []
    for pattern in patterns:
        files += [f for f in sorted(TEST_DIR.glob(pattern)) if f not in files]
    return files


def cmd_list(args) -> None:
    mutants = discover(args.pass_glob, set(args.operator) if args.operator else None)
    for m in mutants:
        print(f"{m.file.relative_to(ROOT)}:{m.line}  [{m.operator}]  {' '.join(m.original.split())[:90]}")
    counts: dict[str, int] = {}
    for m in mutants:
        counts[m.operator] = counts.get(m.operator, 0) + 1
    print(f"\n{len(mutants)} mutants: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def cmd_run(args) -> int:
    build = Path(args.build).resolve()
    results_path = Path(args.results)
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    mutants = discover(args.pass_glob, set(args.operator) if args.operator else None)
    pending = [m for m in mutants if m.key not in results]
    if args.limit:
        pending = pending[: args.limit]
    suite = _suite_files(args.suite or list(DEFAULT_SUITE))
    target = _core_target(build)
    print(f"{len(mutants)} mutants, {len(mutants) - len(pending)} already recorded, running {len(pending)}")

    ok, log = _build(build, [target])
    if not ok:
        sys.exit("the unmutated tree does not build:\n" + log)
    baseline, detail = _run_suite(build, suite, args.ctest, args.timeout, args.workers)
    if baseline != "survived":
        sys.exit(f"the suite fails on the unmutated tree ({baseline}): {detail}")

    try:
        for i, m in enumerate(pending, 1):
            started = time.monotonic()
            with _Applied(m):
                built, log = _build(build, [target])
                outcome, detail = _run_suite(build, suite, args.ctest, args.timeout, args.workers) if built else ("stillborn", log[-400:])
            results[m.key] = {
                "file": str(m.file.relative_to(ROOT)),
                "line": m.line,
                "operator": m.operator,
                "original": " ".join(m.original.split()),
                "outcome": outcome,
                "detail": detail,
                "seconds": round(time.monotonic() - started, 1),
            }
            results_path.write_text(json.dumps(results, indent=1, sort_keys=True))
            print(f"[{i}/{len(pending)}] {outcome:9s} {m.file.name}:{m.line} [{m.operator}]")
    finally:
        # Leave the build matching the restored sources.
        _build(build, [target])

    survivors = [r for r in results.values() if r["outcome"] == "survived"]
    tally: dict[str, int] = {}
    for r in results.values():
        tally[r["outcome"]] = tally.get(r["outcome"], 0) + 1
    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    for r in sorted(survivors, key=lambda r: (r["file"], r["line"])):
        print(f"SURVIVED {r['file']}:{r['line']} [{r['operator']}] {r['original'][:90]}")
    return 1 if survivors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "run"):
        p = sub.add_parser(name)
        p.add_argument("--pass", dest="pass_glob", help="glob over pass source stems, e.g. 'Antisymmetrizer*'")
        p.add_argument("--operator", action="append", choices=["understands", "buffer_of", "span_interferes", "covers"])
    run = sub.choices["run"]
    run.add_argument("--build", default=str(ROOT / "build"))
    run.add_argument("--results", default=str(ROOT / "build/pass_mutation_results.json"))
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--suite", action="append", help="test file glob under the ComputeGraph unit tests (repeatable)")
    run.add_argument("--ctest", help="also run the ctest tests matching this regex when pytest passes")
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument(
        "--workers",
        type=int,
        default=4 if importlib.util.find_spec("xdist") else 1,
        help="pytest-xdist worker processes (default 4 when pytest-xdist is installed, else 1)",
    )
    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
