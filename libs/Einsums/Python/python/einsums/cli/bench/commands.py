# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums bench``: run the performance tests and track their results.

Results go to a SQLite database: ``--db``, else ``$EINSUMS_BENCH_DB``, else a per-user
file (see :func:`default_db_path`). One database serves every checkout and worktree; each
run records the commit and branch it measured.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import bisect, compare, models, report, runner


def project_root(start: Path | None = None) -> Path:
    """The top of the git checkout holding *start* (default: the working directory).

    Falls back to *start* itself outside a checkout, so the commands that only read a
    database still work from anywhere.
    """
    start = (start or Path.cwd()).resolve()
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=start, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return start
    return Path(top)


DB_ENV = "EINSUMS_BENCH_DB"


def user_data_dir() -> Path:
    """Where per-user application data goes on this platform, with ``einsums`` appended."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "einsums"


def default_db_path() -> Path:
    """``$EINSUMS_BENCH_DB`` if set, else ``benchmarks.db`` in :func:`user_data_dir`.

    That is ``~/.local/share/einsums/`` on Linux (``$XDG_DATA_HOME`` if set),
    ``~/Library/Application Support/einsums/`` on macOS and ``%LOCALAPPDATA%\\einsums\\``
    on Windows. Outside any checkout, so history survives removing a worktree.
    """
    if env := os.environ.get(DB_ENV):
        return Path(env).expanduser()
    return user_data_dir() / "benchmarks.db"


def _open_db(args: argparse.Namespace):
    return models.init_db(Path(args.db))


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmarks and store results."""
    source_dir = Path(args.source_dir)
    build_dir = Path(args.build_dir)
    db_path = Path(args.db)

    for rep in range(args.reps):
        if args.reps > 1:
            print(f"\n--- Repetition {rep + 1}/{args.reps} ---")
        run_id = runner.run_benchmarks(
            build_dir, source_dir, db_path,
            build=not args.no_build and rep == 0,  # Only build on first rep
            jobs=args.jobs,
            targets=args.targets,
            suite=args.suite,
            notes=args.notes or "",
        )

    print(f"\nLatest run_id: {run_id}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare a run against a baseline."""
    conn = _open_db(args)

    # Determine current run
    if args.run_id:
        current_run_id = args.run_id
    else:
        # Use the most recent run
        runs = models.list_runs(conn, last=1)
        if not runs:
            print("No runs found in database.", file=sys.stderr)
            sys.exit(1)
        current_run_id = runs[0]["run_id"]

    # Determine baseline
    baseline_ids = None
    if args.baseline_run_id:
        baseline_ids = [args.baseline_run_id]

    cmp_report = compare.compare_run(
        conn, current_run_id,
        baseline_run_ids=baseline_ids,
        baseline_branch=args.baseline,
        baseline_count=args.baseline_count,
        z_threshold=args.z_threshold,
        min_pct_change=args.min_pct,
    )
    results = cmp_report.results

    current_info = models.get_run_info(conn, current_run_id) or {}

    if args.json:
        print(report.to_json(
            results,
            current_commit=current_info.get("git_commit", ""),
            current_branch=current_info.get("git_branch", ""),
        ))
    else:
        n_regressions = report.print_report(
            results,
            current_commit=current_info.get("git_commit", ""),
            current_branch=current_info.get("git_branch", ""),
            baseline_branch=args.baseline,
            baseline_count=args.baseline_count,
            env_warnings=cmp_report.env_warnings,
            current_power=current_info.get("power_source", ""),
        )
        if n_regressions > 0:
            baseline_runs = models.get_baseline_runs(
                conn, branch=args.baseline, n=1
            )
            if baseline_runs:
                baseline_info = models.get_run_info(conn, baseline_runs[0])
                if baseline_info:
                    print(f"\nCandidate commits ({baseline_info['git_commit'][:8]}..{current_info.get('git_commit', '')[:8]}):")
                    try:
                        log = subprocess.check_output(
                            ["git", "log", "--oneline",
                             f"{baseline_info['git_commit']}..{current_info.get('git_commit', 'HEAD')}"],
                            text=True,
                        )
                        print(log)
                    except Exception:
                        pass

    conn.close()

    regressions = sum(1 for r in results if r.verdict == "regression")
    if regressions > 0:
        sys.exit(1)


def cmd_list_runs(args: argparse.Namespace) -> None:
    """List recent runs."""
    conn = _open_db(args)
    runs = models.list_runs(conn, branch=args.branch, last=args.last)
    conn.close()

    if args.json:
        print(json.dumps(runs, indent=2, default=str))
        return
    if not runs:
        print("No runs found.")
        return

    print(f"{'ID':>4}  {'Commit':8}  {'Branch':20}  {'Timestamp':25}  {'Host':15}  {'BLAS':15}  {'Power':7}")
    print("-" * 103)
    for r in runs:
        commit = (r["git_commit"] or "")[:8]
        branch = (r["git_branch"] or "")[:20]
        ts = (r["timestamp"] or "")[:25]
        host = (r["hostname"] or "")[:15]
        blas = (r["blas_vendor"] or "")[:15]
        power = (r.get("power_source") or "?")[:7]
        dirty = "*" if r["git_dirty"] else " "
        print(f"{r['run_id']:>4}  {commit}{dirty} {branch:20}  {ts:25}  {host:15}  {blas:15}  {power:7}")


def cmd_tag_baseline(args: argparse.Namespace) -> None:
    """Tag a run as a named baseline."""
    conn = _open_db(args)
    models.tag_baseline(conn, args.run_id, args.name)
    conn.close()
    print(f"Tagged run {args.run_id} as baseline '{args.name}'.")


def cmd_bisect(args: argparse.Namespace) -> None:
    """Run a single bisect step for git bisect run."""
    exit_code = bisect.run_bisect(
        build_dir=Path(args.build_dir),
        source_dir=Path(args.source_dir),
        benchmark=args.benchmark,
        metric=args.metric,
        threshold_us=args.threshold_us,
        reps=args.reps,
        no_build=args.no_build,
        jobs=args.jobs,
    )
    sys.exit(exit_code)


def cmd_export(args: argparse.Namespace) -> None:
    """Export run results as JSON, or as a self-contained HTML report with ``--html``."""
    conn = _open_db(args)
    if args.html:
        from . import html_export

        html_export.generate_html_report(conn, args.run_id, baseline_branch=args.baseline, output_path=Path(args.html))
        conn.close()
        print(f"Wrote {args.html}")
        return
    results = models.get_results_for_runs(conn, [args.run_id])
    run_info = models.get_run_info(conn, args.run_id)
    conn.close()

    data = {"run": run_info, "results": results}
    print(json.dumps(data, indent=2, default=str))


def cmd_show(args: argparse.Namespace) -> None:
    """Show results for a single run."""
    conn = _open_db(args)

    if args.run_id:
        run_id = args.run_id
    else:
        runs = models.list_runs(conn, last=1)
        if not runs:
            print("No runs found.", file=sys.stderr)
            sys.exit(1)
        run_id = runs[0]["run_id"]

    run_info = models.get_run_info(conn, run_id)
    if not run_info:
        print(f"Run {run_id} not found.", file=sys.stderr)
        sys.exit(1)

    results = models.get_results_for_runs(conn, [run_id])
    conn.close()

    # Group by benchmark_label, show metric columns
    by_label: dict[str, dict[str, float]] = {}
    for r in results:
        label = r["benchmark_label"]
        metric = r["metric_name"]
        if args.filter and args.filter.lower() not in label.lower():
            continue
        by_label.setdefault(label, {})[metric] = r["value_us"]

    # Collect all metric columns
    all_metrics = sorted({m for d in by_label.values() for m in d})
    if args.metric:
        all_metrics = [m for m in all_metrics if m in args.metric]
    if args.json:
        table = {label: {m: v for m, v in by_label[label].items() if m in all_metrics} for label in sorted(by_label)}
        print(json.dumps({"run": run_info, "results": table}, indent=2, default=str))
        return
    if not by_label:
        print("No results found (check --filter).")
        return

    commit = (run_info.get("git_commit") or "")[:8]
    print(f"Run #{run_id}  {commit} ({run_info.get('git_branch') or ''})  {(run_info.get('timestamp') or '')[:19]}")
    if run_info.get("notes"):
        print(f"Notes: {run_info['notes']}")
    print()

    # Print table
    label_width = max(len(l) for l in by_label)
    header = f"{'Benchmark':<{label_width}}"
    for m in all_metrics:
        header += f"  {m:>14}"
    print(header)
    print("-" * len(header))

    for label in sorted(by_label):
        row = f"{label:<{label_width}}"
        for m in all_metrics:
            val = by_label[label].get(m)
            if val is not None:
                row += f"  {val:>14.2f}"
            else:
                row += f"  {'':>14}"
        print(row)


def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two runs side by side with speedup ratios."""
    conn = _open_db(args)
    if args.before is None or args.after is None:
        runs = models.list_runs(conn, last=20)
        if len(runs) < 2:
            print("Need at least 2 runs for diff.", file=sys.stderr)
            sys.exit(1)
        # Default: oldest vs newest among recent runs
        args.before = runs[-1]["run_id"] if args.before is None else args.before
        args.after = runs[0]["run_id"] if args.after is None else args.after

    before_info = models.get_run_info(conn, args.before)
    after_info = models.get_run_info(conn, args.after)
    if not before_info or not after_info:
        print("Run not found.", file=sys.stderr)
        sys.exit(1)
    before = {(r["benchmark_label"], r["metric_name"]): r["value_us"] for r in models.get_results_for_runs(conn, [args.before])}
    after = {(r["benchmark_label"], r["metric_name"]): r["value_us"] for r in models.get_results_for_runs(conn, [args.after])}
    conn.close()

    prefix = args.prefix or ""
    keys = sorted(
        (label, metric)
        for label, metric in before.keys() | after.keys()
        if label.startswith(prefix)
        and (args.metric is None or metric == args.metric)
        and (not args.filter or args.filter.lower() in label.lower())
        and metric not in ("binary_size_kb", "build_time_s")
    )
    rows = []
    for label, metric in keys:
        b, a = before.get((label, metric)), after.get((label, metric))
        rows.append(
            {"benchmark": label, "metric": metric, "before_us": b, "after_us": a,
             "speedup": b / a if b is not None and a else None}
        )  # fmt: skip
    rows.sort(key=lambda r: -(r["speedup"] or 0))
    if args.json:
        print(json.dumps({"before": args.before, "after": args.after, "rows": rows}, indent=2))
        return
    if not rows:
        print(f"No matching benchmarks (metric={args.metric or 'any'}, prefix='{prefix}').")
        return

    print(f"Diff: run #{args.before} ({(before_info.get('git_commit') or '')[:8]}) -> "
          f"#{args.after} ({(after_info.get('git_commit') or '')[:8]})")  # fmt: skip
    print(f"Metric: {args.metric or 'all'}  Prefix: '{prefix}'")
    print()
    show_metric = args.metric is None
    names = [r["benchmark"][len(prefix):] + (f" [{r['metric']}]" if show_metric else "") for r in rows]
    width = max(9, *(len(n) for n in names))
    header = f"{'Benchmark':<{width}}  {'Before (us)':>12}  {'After (us)':>12}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))
    for name, r in zip(names, rows):
        b = f"{r['before_us']:>12.1f}" if r["before_us"] is not None else f"{'n/a':>12}"
        a = f"{r['after_us']:>12.1f}" if r["after_us"] is not None else f"{'n/a':>12}"
        sp = r["speedup"]
        sp_str = f"{'n/a':>8}" if sp is None else (f"{sp:>7.2f}x" if sp >= 1.5 or sp <= 0.67 else f"{'~1.0x':>8}")
        print(f"{name:<{width}}  {b}  {a}  {sp_str}")
    speedups = [r["speedup"] for r in rows if r["speedup"] is not None]
    if speedups:
        faster = sum(1 for x in speedups if x >= 1.5)
        slower = sum(1 for x in speedups if x <= 0.67)
        print()
        print(f"{faster} benchmarks faster (>=1.5x), {slower} slower (<=0.67x), {len(speedups) - faster - slower} unchanged")


def _resolve_metric(conn, labels: list[str], metric: str | None) -> str:
    """*metric*, or the one metric *labels* were recorded with; exits listing them if there are several."""
    if metric is not None:
        return metric
    found = sorted({m for label in labels for m in models.metrics_for(conn, label)})
    if len(found) == 1:
        return found[0]
    if not found:
        print(f"No results recorded for {', '.join(labels) or 'that benchmark'}.", file=sys.stderr)
    else:
        print(f"Several metrics recorded ({', '.join(found)}); pick one with --metric.", file=sys.stderr)
    sys.exit(2)


def cmd_trend(args: argparse.Namespace) -> None:
    """Show time series for a benchmark metric."""
    conn = _open_db(args)
    args.metric = _resolve_metric(conn, [args.benchmark], args.metric)
    data = models.get_trend(conn, args.benchmark, args.metric,
                             branch=args.branch, last=args.last)
    conn.close()

    if not data:
        print(f"No data found for '{args.benchmark}' / '{args.metric}' on branch '{args.branch}'.")
        return

    if args.json:
        print(json.dumps(data, indent=2))
        return

    # ASCII table with sparkline
    print(f"Trend: '{args.benchmark}' metric='{args.metric}' branch='{args.branch}'")
    print(f"{'Commit':<10}  {'Date':>19}  {'Time (us)':>12}  {'Sparkline'}")
    print("-" * 65)

    values = [d["value_us"] for d in data]
    vmin, vmax = min(values), max(values)
    spark_chars = " ▁▂▃▄▅▆▇█"

    for d in data:
        commit = (d.get("git_commit") or "")[:8]
        ts = (d.get("timestamp") or "")[:19]
        val = d["value_us"]

        # Sparkline character for this value
        if vmax > vmin:
            idx = int((val - vmin) / (vmax - vmin) * (len(spark_chars) - 1))
        else:
            idx = 4
        spark = spark_chars[idx]

        print(f"{commit:<10}  {ts:>19}  {val:>12.2f}  {spark}")

    print(f"\nRange: {vmin:.2f} — {vmax:.2f} us  ({len(data)} data points)")


def cmd_scaling(args: argparse.Namespace) -> None:
    """Show performance vs N for a benchmark family."""
    conn = _open_db(args)

    run_id = args.run_id
    if run_id is None:
        runs = models.list_runs(conn, last=1)
        if not runs:
            print("No runs found.")
            return
        run_id = runs[0]["run_id"]

    labels = [
        r["benchmark_label"]
        for r in models.get_results_for_runs(conn, [run_id])
        if r["benchmark_label"].startswith(args.benchmark)
    ]
    args.metric = _resolve_metric(conn, labels, args.metric)
    data = models.get_scaling(conn, run_id, args.benchmark, args.metric)
    conn.close()
    for d in data:
        extra = json.loads(d.get("extra_json") or "{}")
        d["gflops"] = extra.get("gflops")
    if args.json:
        print(json.dumps([{"n": d["n"], "value_us": d["value_us"], "gflops": d["gflops"]} for d in data], indent=2))
        return

    if not data:
        print(f"No scaling data for '{args.benchmark}*' / '{args.metric}' in run {run_id}.")
        return

    print(f"Scaling: '{args.benchmark}*' metric='{args.metric}' run #{run_id}")
    print(f"{'N':>8}  {'Time (us)':>12}  {'GFLOP/s':>10}")
    print("-" * 35)

    for d in data:
        n = d["n"]
        gflops = f"{d['gflops']:>10.2f}" if d["gflops"] is not None else f"{'':>10}"
        print(f"{n:>8}  {d['value_us']:>12.2f}  {gflops}")


def cmd_list_tests(args: argparse.Namespace) -> None:
    """List available performance test binaries in the build tree."""
    build_dir = Path(args.build_dir)
    if not build_dir.is_dir():
        print(f"Build directory not found: {build_dir}")
        sys.exit(1)

    tests = runner._discover_perf_tests(build_dir, args.suite)
    if args.json:
        print(json.dumps([{"name": t.name, "path": str(t.path), "suite": t.suite, "timeout": t.timeout} for t in tests], indent=2))
        return
    if not tests:
        print("No performance tests found. Ensure the project is built with EINSUMS_WITH_TESTS_BENCHMARKS=ON.")
        return

    print(f"Found {len(tests)} performance test binaries ({args.suite}):\n")
    for t in tests:
        print(f"  {t.name:<30}  {t.suite or '?':<15}  {t.path}")


def cmd_delete_db(args: argparse.Namespace) -> None:
    """Delete the benchmark database file entirely."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    if not args.yes:
        print(f"This will permanently delete: {db_path}")
        response = input("Are you sure? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled.")
            return

    db_path.unlink()
    print(f"Deleted: {db_path}")



def _resolve_db_then(command):
    """Fill in the defaults that depend on the environment, then run *command*."""

    def run(args: argparse.Namespace) -> None:
        if args.db is None:
            db = default_db_path()
            db.parent.mkdir(parents=True, exist_ok=True)
            args.db = str(db)
        if getattr(args, "source_dir", "unset") is None:
            args.source_dir = str(project_root())
        command(args)

    return run


def register(subparsers) -> None:
    """Add ``bench`` and its subcommands to the top-level ``einsums`` parser."""
    top = subparsers.add_parser(
        "bench",
        help="Run performance tests and track regressions",
        description="Run the Einsums performance tests, store their results, and compare runs.",
    )
    top.add_argument(
        "--db",
        default=None,
        help=f"SQLite database (default: ${DB_ENV}, else {user_data_dir() / 'benchmarks.db'})",
    )
    sub = top.add_subparsers(dest="bench_command", required=True, metavar="COMMAND")

    # --- run ---
    p_run = sub.add_parser("run", help="Run benchmarks and store results")
    p_run.add_argument("--build-dir", default="build", help="CMake build directory (default: build)")
    p_run.add_argument("--source-dir", default=None, help="Source directory (default: the git checkout holding the working directory)")
    p_run.add_argument("--no-build", action="store_true", help="Skip building")
    p_run.add_argument("--jobs", "-j", type=int, help=f"Build parallelism (default: {runner.default_jobs()}, from $CMAKE_BUILD_PARALLEL_LEVEL or half the cores, at most 16)")
    p_run.add_argument("--reps", type=int, default=1, help="Number of repetitions")
    p_run.add_argument("--notes", help="Free-form notes for this run")
    p_run.add_argument("--targets", nargs="+", help="Only run these test targets (by name, whatever their suite)")
    p_run.add_argument(
        "--suite", choices=(*runner.SUITES, "all"), default="contraction",
        help="Which tests to run: contractions and the operations they use (default), the infrastructure around them, or all",
    )
    p_run.set_defaults(func=_resolve_db_then(cmd_run))

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Compare run against baseline")
    p_cmp.add_argument("--run-id", type=int, help="Run to compare (default: latest)")
    p_cmp.add_argument("--baseline-run-id", type=int, help="Specific baseline run")
    p_cmp.add_argument("--baseline", default="main", help="Baseline branch (default: main)")
    p_cmp.add_argument("--baseline-count", type=int, default=10, help="Number of baseline runs")
    p_cmp.add_argument("--z-threshold", type=float, default=3.0, help="Z-score threshold")
    p_cmp.add_argument("--min-pct", type=float, default=5.0, help="Minimum %% change to flag")
    p_cmp.add_argument("--json", action="store_true", help="Output as JSON")
    p_cmp.set_defaults(func=_resolve_db_then(cmd_compare))

    # --- list-runs ---
    p_list = sub.add_parser("list-runs", help="List recent benchmark runs")
    p_list.add_argument("--json", action="store_true", help="Output as JSON")
    p_list.add_argument("--branch", help="Filter by branch")
    p_list.add_argument("--last", type=int, default=20, help="Number of runs to show")
    p_list.set_defaults(func=_resolve_db_then(cmd_list_runs))

    # --- tag-baseline ---
    p_tag = sub.add_parser("tag-baseline", help="Tag a run as a named baseline")
    p_tag.add_argument("--run-id", type=int, required=True)
    p_tag.add_argument("--name", required=True, help="Baseline name")
    p_tag.set_defaults(func=_resolve_db_then(cmd_tag_baseline))

    # --- bisect ---
    p_bis = sub.add_parser(
        "bisect",
        help="Single bisect step for `git bisect run`",
        description=(
            "Build, run one benchmark, and exit 0 (good) or 1 (bad). "
            "Exit 125 if build fails or benchmark not found (git bisect skips the commit)."
        ),
    )
    p_bis.add_argument("--build-dir", default="build", help="CMake build directory (default: build)")
    p_bis.add_argument("--source-dir", default=None, help="Source directory (default: the git checkout holding the working directory)")
    p_bis.add_argument("--benchmark", required=True, help="Benchmark label to check (e.g. 'Rank-2 N=64')")
    p_bis.add_argument("--metric", help="Metric name (default: the benchmark's only metric)")
    p_bis.add_argument("--threshold-us", type=float, required=True, help="Threshold in microseconds; above = bad")
    p_bis.add_argument("--reps", type=int, default=3, help="Repetitions for confirmation (default: 3)")
    p_bis.add_argument("--no-build", action="store_true", help="Skip building (assume already built)")
    p_bis.add_argument("--jobs", "-j", type=int, help="Build parallelism (default as for run)")
    p_bis.set_defaults(func=_resolve_db_then(cmd_bisect))

    # --- export ---
    p_exp = sub.add_parser("export", help="Export run results as JSON or an HTML report")
    p_exp.add_argument("--run-id", type=int, required=True)
    p_exp.add_argument("--html", metavar="FILE", help="Write a self-contained HTML report to FILE instead")
    p_exp.add_argument("--baseline", default="main", help="Baseline branch for the HTML report (default: main)")
    p_exp.set_defaults(func=_resolve_db_then(cmd_export))

    # --- show ---
    p_show = sub.add_parser("show", help="Show results for a single run")
    p_show.add_argument("--json", action="store_true", help="Output as JSON")
    p_show.add_argument("--run-id", type=int, help="Run to show (default: latest)")
    p_show.add_argument("--filter", help="Filter benchmark labels (substring match)")
    p_show.add_argument("--metric", nargs="+", help="Metrics to show (default: all)")
    p_show.set_defaults(func=_resolve_db_then(cmd_show))

    # --- diff ---
    p_diff = sub.add_parser("diff", help="Compare two runs side by side with speedup")
    p_diff.add_argument("--json", action="store_true", help="Output as JSON")
    p_diff.add_argument("--before", type=int, help="Earlier run ID (default: oldest recent)")
    p_diff.add_argument("--after", type=int, help="Later run ID (default: latest)")
    p_diff.add_argument("--metric", help="Metric to compare (default: every metric)")
    p_diff.add_argument("--prefix", help="Only labels starting with this (shown without it)")
    p_diff.add_argument("--filter", help="Additional substring filter on labels")
    p_diff.set_defaults(func=_resolve_db_then(cmd_diff))

    # --- trend ---
    p_trend = sub.add_parser("trend", help="Show time series for a benchmark metric")
    p_trend.add_argument("--benchmark", required=True, help="Benchmark label (exact match)")
    p_trend.add_argument("--metric", help="Metric name (default: the benchmark's only metric)")
    p_trend.add_argument("--branch", default="main", help="Branch to filter (default: main)")
    p_trend.add_argument("--last", type=int, default=50, help="Number of data points")
    p_trend.add_argument("--json", action="store_true", help="Output as JSON")
    p_trend.set_defaults(func=_resolve_db_then(cmd_trend))

    # --- scaling ---
    p_scale = sub.add_parser("scaling", help="Show performance vs N for a benchmark family")
    p_scale.add_argument("--json", action="store_true", help="Output as JSON")
    p_scale.add_argument("--run-id", type=int, help="Run to analyze (default: latest)")
    p_scale.add_argument("--benchmark", required=True, help="Benchmark label prefix (e.g., 'gemm')")
    p_scale.add_argument("--metric", help="Metric name (default: the family's only metric)")
    p_scale.set_defaults(func=_resolve_db_then(cmd_scaling))

    # --- list-tests ---
    p_tests = sub.add_parser("list-tests", help="List available performance test binaries")
    p_tests.add_argument("--json", action="store_true", help="Output as JSON")
    p_tests.add_argument("--build-dir", default="build", help="CMake build directory (default: build)")
    p_tests.add_argument("--suite", choices=(*runner.SUITES, "all"), default="all", help="Only one suite (default: all)")
    p_tests.set_defaults(func=_resolve_db_then(cmd_list_tests))

    # --- delete-db ---
    p_del = sub.add_parser("delete-db", help="Delete the benchmark database entirely")
    p_del.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    p_del.set_defaults(func=_resolve_db_then(cmd_delete_db))
