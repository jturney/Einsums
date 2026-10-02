# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Bisect helper: build, run one benchmark, and exit 0 (good) or 1 (bad).

Designed to be used with `git bisect run`:

    git bisect start <bad> <good>
    git bisect run einsums bench bisect \
        --benchmark "blas-gemm N=1024" \
        --threshold-us 500

Exit codes:
    0   — good (metric <= threshold)
    1   — bad (metric > threshold)
    125 — skip (build failed, or the benchmark or metric was not found — git bisect skips the commit)
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from . import runner
from .parser import Result

GOOD, BAD, SKIP = 0, 1, 125


def _pick(results: list[Result], benchmark: str, metric: str | None) -> tuple[float | None, list[str]]:
    """The value of *metric* for *benchmark*, and every metric the benchmark reported."""
    matching = [r for r in results if r.label == benchmark]
    metrics = sorted({r.metric for r in matching})
    if metric is None and len(metrics) == 1:
        metric = metrics[0]
    value = next((r.value_us for r in matching if r.metric == metric), None)
    return value, metrics


def run_bisect(
    *,
    build_dir: Path,
    source_dir: Path,
    benchmark: str,
    metric: str | None,
    threshold_us: float,
    reps: int,
    no_build: bool,
    jobs: int | None = None,
) -> int:
    """Run a single bisect step. Returns 0 (good), 1 (bad) or 125 (skip)."""
    del source_dir  # the build directory is all a step needs
    if not no_build:
        jobs = jobs or runner.default_jobs()
        print(f"[bisect] Building ({jobs} jobs)...")
        result = runner.build_performance_tests(build_dir, jobs, capture=True)
        if result.returncode != 0:
            for line in (result.stderr or result.stdout or "").splitlines()[-10:]:
                print(f"  {line}", file=sys.stderr)
            print("[bisect] Build FAILED — skipping this commit.", file=sys.stderr)
            return SKIP

    tests = runner._discover_perf_tests(build_dir)
    if not tests:
        print("[bisect] No performance test binaries found — skipping.", file=sys.stderr)
        return SKIP

    # The benchmark is usually in one binary; stop at the first that reports it.
    for test in tests:
        test_name, binary = test.name, test.path
        print(f"[bisect] Running {test_name}...")
        run = runner.run_test_binary(binary, timeout=test.timeout)
        if run.status == "timeout":
            print(f"[bisect] {test_name} timed out, trying the next binary.", file=sys.stderr)
            continue
        value, metrics = _pick(run.results, benchmark, metric)
        if not metrics:
            continue
        if value is None:
            wanted = f"metric '{metric}'" if metric else "a metric (it reports several; pass --metric)"
            print(f"[bisect] '{benchmark}' found, but not {wanted}. Reported: {', '.join(metrics)}", file=sys.stderr)
            return SKIP
        name = metric or metrics[0]

        values = [value]
        verdict = "BAD" if value > threshold_us else "GOOD"
        print(f"[bisect] {benchmark} / {name} = {value:.2f} us (threshold {threshold_us:.2f} us) — {verdict}")
        if reps > 1 and verdict == "BAD":
            # One slow run should not condemn a commit: take the median of `reps`.
            print(f"[bisect] Confirming with {reps - 1} more run(s)...")
            for _ in range(reps - 1):
                again, _ = _pick(runner.run_test_binary(binary, timeout=test.timeout).results, benchmark, name)
                if again is not None:
                    values.append(again)
            value = statistics.median(values)
            print(f"[bisect] Median of {len(values)} runs: {value:.2f} us — {'BAD' if value > threshold_us else 'GOOD'}")
        return BAD if value > threshold_us else GOOD

    print(f"[bisect] Benchmark '{benchmark}' not found in any test binary — skipping.", file=sys.stderr)
    return SKIP
