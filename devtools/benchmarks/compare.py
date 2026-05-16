# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Statistical comparison of benchmark runs for regression detection."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from devtools.benchmarks import models


@dataclass
class EnvironmentWarning:
    """Warning about an environment difference between current and baseline."""
    field: str
    current_value: str
    baseline_values: list[str]
    severity: str  # "warning" or "error"
    message: str


@dataclass
class ComparisonResult:
    """Result of comparing one metric against a baseline."""
    benchmark_label: str
    metric_name: str
    baseline_median: float
    baseline_mad: float
    current_value: float
    z_score: float | None
    pct_change: float
    verdict: str  # "regression", "improvement", "unchanged", "insufficient_data", "noisy"
    n_baseline: int
    # Per-rep noise info (from extra_json)
    cv_pct: float | None = None
    gflops: float | None = None


@dataclass
class ComparisonReport:
    """Full comparison report including environment warnings."""
    results: list[ComparisonResult]
    env_warnings: list[EnvironmentWarning]
    current_run_id: int
    baseline_branch: str
    n_baseline_runs: int


def _median(values: list[float]) -> float:
    """Compute the median of a list of values."""
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mad(values: list[float]) -> float:
    """Compute the Median Absolute Deviation."""
    med = _median(values)
    deviations = [abs(v - med) for v in values]
    return _median(deviations)


def _check_environment(
    conn: sqlite3.Connection,
    current_run_id: int,
    baseline_run_ids: list[int],
) -> list[EnvironmentWarning]:
    """Check for environment differences between current and baseline runs."""
    warnings = []
    current = models.get_run_info(conn, current_run_id)
    if not current:
        return warnings

    baseline_infos = [models.get_run_info(conn, rid) for rid in baseline_run_ids]
    baseline_infos = [b for b in baseline_infos if b is not None]
    if not baseline_infos:
        return warnings

    checks = [
        ("power_source", "error",
         "Power source changed: {current} vs baseline {baseline}. "
         "Battery power causes CPU throttling — results are not comparable."),
        ("blas_vendor", "error",
         "BLAS vendor changed: {current} vs baseline {baseline}. "
         "Different BLAS libraries have different performance characteristics."),
        ("compiler", "warning",
         "Compiler changed: {current} vs baseline {baseline}."),
        ("cpu_freq_mhz", "warning",
         "CPU frequency changed: {current} MHz vs baseline {baseline} MHz."),
        ("build_type", "warning",
         "Build type changed: {current} vs baseline {baseline}."),
    ]

    for field_name, severity, msg_template in checks:
        current_val = current.get(field_name)
        baseline_vals = list({str(b.get(field_name)) for b in baseline_infos if b.get(field_name) is not None})

        if current_val is None or not baseline_vals:
            continue

        current_str = str(current_val)
        if current_str not in baseline_vals:
            warnings.append(EnvironmentWarning(
                field=field_name,
                current_value=current_str,
                baseline_values=baseline_vals,
                severity=severity,
                message=msg_template.format(current=current_str, baseline=", ".join(baseline_vals)),
            ))

    return warnings


def compare_run(
    conn: sqlite3.Connection,
    current_run_id: int,
    *,
    baseline_run_ids: list[int] | None = None,
    baseline_branch: str = "main",
    baseline_count: int = 10,
    z_threshold: float = 3.0,
    min_pct_change: float = 5.0,
    noise_cv_threshold: float = 10.0,
) -> ComparisonReport:
    """Compare a run against a baseline distribution.

    Returns a ComparisonReport with per-metric results and environment warnings.
    """
    # Get baseline run IDs if not provided
    if baseline_run_ids is None:
        baseline_run_ids = models.get_baseline_runs(
            conn, branch=baseline_branch, n=baseline_count
        )

    if not baseline_run_ids:
        return ComparisonReport(
            results=[], env_warnings=[],
            current_run_id=current_run_id,
            baseline_branch=baseline_branch,
            n_baseline_runs=0,
        )

    # Check environment differences
    env_warnings = _check_environment(conn, current_run_id, baseline_run_ids)

    # Get current results
    current_results = models.get_results_for_runs(conn, [current_run_id])

    # Get baseline results
    baseline_results = models.get_results_for_runs(conn, baseline_run_ids)

    # Group baseline by (label, metric) -> list of values
    baseline_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in baseline_results:
        key = (r["benchmark_label"], r["metric_name"])
        baseline_by_key[key].append(r["value_us"])

    # Compare each current result against baseline
    comparisons = []
    for r in current_results:
        key = (r["benchmark_label"], r["metric_name"])
        baseline_values = baseline_by_key.get(key, [])
        current_value = r["value_us"]
        n_baseline = len(baseline_values)

        # Extract per-rep stats from extra_json
        cv_pct = None
        gflops = None
        if r.get("extra_json"):
            try:
                extra = json.loads(r["extra_json"])
                cv_pct = extra.get("cv_pct")
                gflops = extra.get("gflops")
            except (json.JSONDecodeError, TypeError):
                pass

        if n_baseline == 0:
            comparisons.append(ComparisonResult(
                benchmark_label=r["benchmark_label"],
                metric_name=r["metric_name"],
                baseline_median=0.0,
                baseline_mad=0.0,
                current_value=current_value,
                z_score=None,
                pct_change=0.0,
                verdict="insufficient_data",
                n_baseline=0,
                cv_pct=cv_pct,
                gflops=gflops,
            ))
            continue

        med = _median(baseline_values)
        mad = _mad(baseline_values)

        # Percentage change
        pct_change = ((current_value - med) / med * 100.0) if med != 0 else 0.0

        # Check if this measurement is too noisy to trust
        if cv_pct is not None and cv_pct > noise_cv_threshold:
            comparisons.append(ComparisonResult(
                benchmark_label=r["benchmark_label"],
                metric_name=r["metric_name"],
                baseline_median=med,
                baseline_mad=mad,
                current_value=current_value,
                z_score=None,
                pct_change=pct_change,
                verdict="noisy",
                n_baseline=n_baseline,
                cv_pct=cv_pct,
                gflops=gflops,
            ))
            continue

        if n_baseline < 5:
            # Insufficient samples for statistical test — use simple threshold
            if abs(pct_change) > 10.0:
                verdict = "regression" if pct_change > 0 else "improvement"
            else:
                verdict = "unchanged"
            comparisons.append(ComparisonResult(
                benchmark_label=r["benchmark_label"],
                metric_name=r["metric_name"],
                baseline_median=med,
                baseline_mad=mad,
                current_value=current_value,
                z_score=None,
                pct_change=pct_change,
                verdict=verdict,
                n_baseline=n_baseline,
                cv_pct=cv_pct,
                gflops=gflops,
            ))
            continue

        # Modified Z-score
        if mad > 0:
            z_score = 0.6745 * (current_value - med) / mad
        else:
            z_score = float("inf") if current_value != med else 0.0

        # Verdict: need both statistical significance AND practical significance
        if z_score > z_threshold and pct_change > min_pct_change:
            verdict = "regression"
        elif z_score < -z_threshold and pct_change < -min_pct_change:
            verdict = "improvement"
        else:
            verdict = "unchanged"

        comparisons.append(ComparisonResult(
            benchmark_label=r["benchmark_label"],
            metric_name=r["metric_name"],
            baseline_median=med,
            baseline_mad=mad,
            current_value=current_value,
            z_score=z_score,
            pct_change=pct_change,
            verdict=verdict,
            n_baseline=n_baseline,
            cv_pct=cv_pct,
            gflops=gflops,
        ))

    return ComparisonReport(
        results=comparisons,
        env_warnings=env_warnings,
        current_run_id=current_run_id,
        baseline_branch=baseline_branch,
        n_baseline_runs=len(baseline_run_ids),
    )
