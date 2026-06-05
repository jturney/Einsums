# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Generate a self-contained HTML report from benchmark data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from devtools.benchmarks import compare, models


def generate_html_report(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    baseline_branch: str = "main",
    output_path: Path | None = None,
) -> str:
    """Generate a self-contained HTML report for a benchmark run.

    Returns the HTML string. If output_path is provided, also writes to file.
    """
    run_info = models.get_run_info(conn, run_id) or {}
    results = models.get_results_for_runs(conn, [run_id])

    # Get comparison data
    cmp_report = compare.compare_run(conn, run_id, baseline_branch=baseline_branch)

    commit = (run_info.get("git_commit") or "")[:8]
    branch = run_info.get("git_branch") or "?"
    timestamp = (run_info.get("timestamp") or "")[:19]
    hostname = run_info.get("hostname") or "?"
    cpu = run_info.get("cpu_model") or "?"
    freq = run_info.get("cpu_freq_mhz")
    power = run_info.get("power_source") or "?"
    blas = run_info.get("blas_vendor") or "?"
    build = run_info.get("build_type") or "?"

    # Group results by benchmark label
    by_label: dict[str, list] = {}
    for r in results:
        by_label.setdefault(r["benchmark_label"], []).append(r)

    # Build comparison lookup
    cmp_lookup: dict[tuple[str, str], compare.ComparisonResult] = {}
    for cr in cmp_report.results:
        cmp_lookup[(cr.benchmark_label, cr.metric_name)] = cr

    # Build HTML
    rows_html = []
    for label in sorted(by_label.keys()):
        for r in by_label[label]:
            cr = cmp_lookup.get((r["benchmark_label"], r["metric_name"]))
            pct = cr.pct_change if cr else 0.0
            verdict = cr.verdict if cr else ""
            baseline = cr.baseline_median if cr else 0.0

            # Extract extra
            gflops = ""
            cv = ""
            warmup = ""
            if r.get("extra_json"):
                try:
                    extra = json.loads(r["extra_json"])
                    if "gflops" in extra:
                        gflops = f"{extra['gflops']:.1f}"
                    if "cv_pct" in extra:
                        cv = f"{extra['cv_pct']:.1f}%"
                    if "warmup_ratio" in extra:
                        warmup = f"{extra['warmup_ratio']:.1f}x"
                except (json.JSONDecodeError, TypeError):
                    pass

            # Color for verdict
            if verdict == "regression":
                row_class = "regression"
            elif verdict == "improvement":
                row_class = "improvement"
            elif verdict == "noisy":
                row_class = "noisy"
            else:
                row_class = ""

            rows_html.append(
                f'<tr class="{row_class}">'
                f'<td>{r["benchmark_label"]}</td>'
                f'<td>{r["metric_name"]}</td>'
                f'<td class="num">{r["value_us"]:.2f}</td>'
                f'<td class="num">{baseline:.2f}</td>'
                f'<td class="num">{pct:+.1f}%</td>'
                f'<td>{verdict}</td>'
                f'<td class="num">{gflops}</td>'
                f'<td class="num">{cv}</td>'
                f'<td class="num">{warmup}</td>'
                f"</tr>"
            )

    # Environment warnings
    warnings_html = ""
    if cmp_report.env_warnings:
        warnings_html = '<div class="warnings"><h3>Environment Warnings</h3><ul>'
        for w in cmp_report.env_warnings:
            sev = "error" if w.severity == "error" else "warning"
            warnings_html += f'<li class="{sev}">{w.message}</li>'
        warnings_html += "</ul></div>"

    # Summary counts
    n_reg = sum(1 for r in cmp_report.results if r.verdict == "regression")
    n_imp = sum(1 for r in cmp_report.results if r.verdict == "improvement")
    n_unc = sum(1 for r in cmp_report.results if r.verdict == "unchanged")
    n_noisy = sum(1 for r in cmp_report.results if r.verdict == "noisy")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Einsums Benchmark Report - Run #{run_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; margin: 2em; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; }}
  h2 {{ color: #7ec8e3; margin-top: 2em; }}
  h3 {{ color: #f0a500; }}
  .meta {{ background: #16213e; padding: 1em; border-radius: 8px; margin-bottom: 1em; }}
  .meta span {{ margin-right: 2em; }}
  .meta .label {{ color: #888; }}
  .summary {{ display: flex; gap: 2em; margin: 1em 0; }}
  .summary .card {{ background: #16213e; padding: 1em 2em; border-radius: 8px; text-align: center; }}
  .summary .card .number {{ font-size: 2em; font-weight: bold; }}
  .summary .reg .number {{ color: #ff4444; }}
  .summary .imp .number {{ color: #44ff44; }}
  .summary .unc .number {{ color: #888; }}
  .summary .noisy .number {{ color: #ffaa00; }}
  .warnings {{ background: #2a1a1a; padding: 1em; border-radius: 8px; margin: 1em 0; }}
  .warnings .error {{ color: #ff4444; }}
  .warnings .warning {{ color: #ffaa00; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th {{ background: #16213e; color: #7ec8e3; padding: 8px 12px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #2a2a4a; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover {{ background: #1a2a4e; }}
  tr.regression {{ background: #2a1a1a; }}
  tr.improvement {{ background: #1a2a1a; }}
  tr.noisy {{ background: #2a2a1a; }}
  .footer {{ margin-top: 3em; color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Einsums Benchmark Report</h1>

<div class="meta">
  <span><span class="label">Run:</span> #{run_id}</span>
  <span><span class="label">Commit:</span> {commit}</span>
  <span><span class="label">Branch:</span> {branch}</span>
  <span><span class="label">Time:</span> {timestamp}</span><br>
  <span><span class="label">Host:</span> {hostname}</span>
  <span><span class="label">CPU:</span> {cpu}{f' @ {freq} MHz' if freq else ''}</span>
  <span><span class="label">Power:</span> {power}</span>
  <span><span class="label">BLAS:</span> {blas}</span>
  <span><span class="label">Build:</span> {build}</span>
</div>

<div class="summary">
  <div class="card reg"><div class="number">{n_reg}</div>Regressions</div>
  <div class="card imp"><div class="number">{n_imp}</div>Improvements</div>
  <div class="card unc"><div class="number">{n_unc}</div>Unchanged</div>
  <div class="card noisy"><div class="number">{n_noisy}</div>Noisy</div>
</div>

{warnings_html}

<h2>Results vs {baseline_branch}</h2>
<table>
<thead>
<tr>
  <th>Benchmark</th><th>Metric</th><th>Current (us)</th><th>Baseline (us)</th>
  <th>Change</th><th>Verdict</th><th>GFLOP/s</th><th>CV</th><th>Warmup</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>

<div class="footer">
  Generated by Einsums Benchmark Tracker &mdash; {len(results)} metrics from {len(by_label)} benchmarks
</div>
</body>
</html>"""

    if output_path:
        output_path.write_text(html)

    return html
