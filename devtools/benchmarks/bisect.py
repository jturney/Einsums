# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Bisect helper: build, run one benchmark, and exit 0 (good) or 1 (bad).

Designed to be used with `git bisect run`:

    git bisect start <bad> <good>
    git bisect run python -m devtools.benchmarks bisect \
        --build-dir build \
        --benchmark "Rank-2 N=64" \
        --metric t_generic \
        --threshold-us 500

Exit codes:
    0   — good (metric <= threshold)
    1   — bad (metric > threshold)
    125 — skip (build failed or benchmark not found — tells git bisect to skip)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from devtools.benchmarks import parser, runner


def run_bisect(
    *,
    build_dir: Path,
    source_dir: Path,
    benchmark: str,
    metric: str,
    threshold_us: float,
    reps: int,
    no_build: bool,
) -> int:
    """Run a single bisect step. Returns exit code (0, 1, or 125)."""

    # --- Build ---
    if not no_build:
        print(f"[bisect] Building...")
        result = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "Tests.Performance"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Print last few lines of build error for diagnostics
            for line in (result.stderr or result.stdout or "").splitlines()[-10:]:
                print(f"  {line}", file=sys.stderr)
            print("[bisect] Build FAILED — skipping this commit.", file=sys.stderr)
            return 125

    # --- Discover the right test binary ---
    tests = runner._discover_perf_tests(build_dir)
    if not tests:
        print("[bisect] No performance test binaries found — skipping.", file=sys.stderr)
        return 125

    # Run each test binary until we find the matching benchmark label.
    # Most of the time only one binary contains the target benchmark,
    # so we stop as soon as we find it.
    for test_name, binary in tests:
        print(f"[bisect] Running {test_name}...")
        try:
            proc = subprocess.run(
                [str(binary)],
                capture_output=True, text=True,
                timeout=600,
            )
            output = proc.stdout
        except subprocess.TimeoutExpired:
            print(f"[bisect] {test_name} timed out, trying next...", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[bisect] {test_name} error: {e}", file=sys.stderr)
            continue

        benchmarks, breakdowns, cache_hits = parser.parse_output(output)

        # Look for matching benchmark
        for b in benchmarks:
            if b.label != benchmark:
                continue

            # Found it — extract the requested metric
            value = _extract_metric(b, metric)
            if value is None:
                print(
                    f"[bisect] Found benchmark '{benchmark}' but metric "
                    f"'{metric}' is not available.",
                    file=sys.stderr,
                )
                print(f"[bisect] Available metrics: {_list_metrics(b)}", file=sys.stderr)
                return 125

            verdict = "BAD" if value > threshold_us else "GOOD"
            print(
                f"[bisect] {benchmark} / {metric} = {value:.2f} us "
                f"(threshold: {threshold_us:.2f} us) — {verdict}"
            )

            if reps > 1 and verdict == "BAD":
                # Re-run to confirm — a single outlier shouldn't condemn a commit.
                # Take the median of `reps` runs to reduce noise.
                print(f"[bisect] Confirming with {reps - 1} more run(s)...")
                values = [value]
                for rep in range(reps - 1):
                    try:
                        proc2 = subprocess.run(
                            [str(binary)],
                            capture_output=True, text=True,
                            timeout=600,
                        )
                        b2_list, _, _ = parser.parse_output(proc2.stdout)
                        for b2 in b2_list:
                            if b2.label == benchmark:
                                v2 = _extract_metric(b2, metric)
                                if v2 is not None:
                                    values.append(v2)
                                break
                    except Exception:
                        pass

                median = sorted(values)[len(values) // 2]
                verdict = "BAD" if median > threshold_us else "GOOD"
                print(
                    f"[bisect] Median of {len(values)} runs: {median:.2f} us — {verdict}"
                )
                return 1 if median > threshold_us else 0

            return 1 if value > threshold_us else 0

    # Benchmark label not found in any test binary
    print(
        f"[bisect] Benchmark '{benchmark}' not found in any test binary — skipping.",
        file=sys.stderr,
    )
    return 125


def _extract_metric(b: parser.BenchmarkResult, metric: str) -> float | None:
    """Extract a named metric from a BenchmarkResult."""
    metric_map = {
        "t_generic": b.t_generic,
        "t_blas_packed": b.t_blas_packed,
        "t_mlir": b.t_mlir,
        "t_einsum": b.t_einsum,
        "t_sort_gemm": b.t_sort_gemm,
    }
    return metric_map.get(metric)


def _list_metrics(b: parser.BenchmarkResult) -> str:
    """List available (non-None) metric names for a BenchmarkResult."""
    names = []
    for name in ("t_generic", "t_blas_packed", "t_mlir", "t_einsum", "t_sort_gemm"):
        if getattr(b, name) is not None:
            names.append(name)
    return ", ".join(names)
