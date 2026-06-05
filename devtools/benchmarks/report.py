# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Format comparison results for console and JSON output."""

from __future__ import annotations

import json
import sys

from devtools.benchmarks.compare import ComparisonReport, ComparisonResult, EnvironmentWarning


def print_report(
    results: list[ComparisonResult],
    *,
    current_commit: str = "",
    current_branch: str = "",
    baseline_branch: str = "main",
    baseline_count: int = 10,
    env_warnings: list[EnvironmentWarning] | None = None,
    current_power: str = "",
    file=None,
) -> int:
    """Print a formatted regression report. Returns count of regressions."""
    if file is None:
        file = sys.stdout

    regressions = [r for r in results if r.verdict == "regression"]
    improvements = [r for r in results if r.verdict == "improvement"]
    unchanged = [r for r in results if r.verdict == "unchanged"]
    insufficient = [r for r in results if r.verdict == "insufficient_data"]

    commit_str = current_commit[:8] if current_commit else "current"
    branch_str = f" ({current_branch})" if current_branch else ""
    power_str = f"  [power: {current_power}]" if current_power else ""

    print(
        f"Performance Comparison: {commit_str}{branch_str}"
        f" vs {baseline_branch} (last {baseline_count} runs)"
        f"{power_str}",
        file=file,
    )
    print("=" * 72, file=file)

    if env_warnings:
        for w in env_warnings:
            severity = "ERROR" if w.severity == "error" else "WARNING"
            print(f"  [{severity}] {w.message}", file=file)
        print(file=file)

    print(file=file)

    if regressions:
        print(f"REGRESSIONS ({len(regressions)}):", file=file)
        for r in sorted(regressions, key=lambda x: -abs(x.pct_change)):
            _print_result(r, file=file)
        print(file=file)

    if improvements:
        print(f"IMPROVEMENTS ({len(improvements)}):", file=file)
        for r in sorted(improvements, key=lambda x: x.pct_change):
            _print_result(r, file=file)
        print(file=file)

    if unchanged:
        print(f"UNCHANGED ({len(unchanged)} metrics)", file=file)
        print(file=file)

    if insufficient:
        print(f"INSUFFICIENT DATA ({len(insufficient)} metrics)", file=file)
        print(file=file)

    print(
        f"Summary: {len(regressions)} regressions, "
        f"{len(improvements)} improvements, "
        f"{len(unchanged)} unchanged",
        file=file,
    )

    return len(regressions)


def _print_result(r: ComparisonResult, file) -> None:
    """Print a single comparison result."""
    z_str = f"z={r.z_score:.2f}" if r.z_score is not None else "n/a"
    print(f"  {r.benchmark_label} / {r.metric_name}:", file=file)
    print(
        f"    Baseline: {r.baseline_median:.2f} "
        f"\u00b1 {r.baseline_mad:.2f} \u00b5s ({r.n_baseline} samples)",
        file=file,
    )
    print(f"    Current:  {r.current_value:.2f} \u00b5s", file=file)
    print(
        f"    Change:   {r.pct_change:+.1f}%  ({z_str})  "
        f"{r.verdict.upper()}",
        file=file,
    )


def to_json(
    results: list[ComparisonResult],
    *,
    current_commit: str = "",
    current_branch: str = "",
) -> str:
    """Serialize comparison results to JSON."""
    data = {
        "commit": current_commit,
        "branch": current_branch,
        "results": [
            {
                "benchmark_label": r.benchmark_label,
                "metric_name": r.metric_name,
                "baseline_median": r.baseline_median,
                "baseline_mad": r.baseline_mad,
                "current_value": r.current_value,
                "z_score": r.z_score,
                "pct_change": r.pct_change,
                "verdict": r.verdict,
                "n_baseline": r.n_baseline,
            }
            for r in results
        ],
        "summary": {
            "regressions": sum(1 for r in results if r.verdict == "regression"),
            "improvements": sum(1 for r in results if r.verdict == "improvement"),
            "unchanged": sum(1 for r in results if r.verdict == "unchanged"),
        },
    }
    return json.dumps(data, indent=2)
