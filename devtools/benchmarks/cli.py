# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""CLI entry point for the Einsums benchmark regression system."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import bisect, compare, models, report, runner


def _default_db_path() -> Path:
    """Return the default benchmark database path.

    Stored in the project tree at ``<project>/.einsums-studio/benchmarks.db``
    so each checkout has its own benchmark history (and so the data follows
    the source rather than being scattered across user-level config dirs).
    Einsums Studio's C++ benchmark bundle reads from the same path.

    Resolved relative to this file: cli.py lives at
    ``<project>/devtools/benchmarks/cli.py``, so two ``.parent`` hops land
    at ``<project>/devtools`` and one more at ``<project>``.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    base = project_root / ".einsums-studio"
    base.mkdir(parents=True, exist_ok=True)
    return base / "benchmarks.db"


DEFAULT_DB = _default_db_path()


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
            targets=args.targets,
            notes=args.notes or "",
        )

    print(f"\nLatest run_id: {run_id}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare a run against a baseline."""
    conn = models.init_db(Path(args.db))

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
    conn = models.init_db(Path(args.db))
    runs = models.list_runs(conn, branch=args.branch, last=args.last)
    conn.close()

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
    conn = models.init_db(Path(args.db))
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
    )
    sys.exit(exit_code)


def cmd_export(args: argparse.Namespace) -> None:
    """Export run results as JSON."""
    conn = models.init_db(Path(args.db))
    results = models.get_results_for_runs(conn, [args.run_id])
    run_info = models.get_run_info(conn, args.run_id)
    conn.close()

    import json
    data = {"run": run_info, "results": results}
    print(json.dumps(data, indent=2, default=str))


def cmd_show(args: argparse.Namespace) -> None:
    """Show results for a single run."""
    conn = models.init_db(Path(args.db))

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

    commit = (run_info.get("git_commit") or "")[:8]
    branch = run_info.get("git_branch") or ""
    ts = (run_info.get("timestamp") or "")[:19]
    notes = run_info.get("notes") or ""
    print(f"Run #{run_id}  {commit} ({branch})  {ts}")
    if notes:
        print(f"Notes: {notes}")
    print()

    # Group by benchmark_label, show metric columns
    by_label: dict[str, dict[str, float]] = {}
    for r in results:
        label = r["benchmark_label"]
        metric = r["metric_name"]
        if args.filter and args.filter.lower() not in label.lower():
            continue
        by_label.setdefault(label, {})[metric] = r["value_us"]

    if not by_label:
        print("No results found (check --filter).")
        return

    # Collect all metric columns
    all_metrics = sorted({m for d in by_label.values() for m in d})
    if args.metric:
        all_metrics = [m for m in all_metrics if m in args.metric]

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
    conn = models.init_db(Path(args.db))

    # Resolve run IDs
    if args.before is None or args.after is None:
        runs = models.list_runs(conn, last=20)
        if len(runs) < 2 and (args.before is None or args.after is None):
            print("Need at least 2 runs for diff.", file=sys.stderr)
            sys.exit(1)
        # Default: oldest vs newest among recent runs
        if args.before is None:
            args.before = runs[-1]["run_id"]
        if args.after is None:
            args.after = runs[0]["run_id"]

    before_info = models.get_run_info(conn, args.before)
    after_info = models.get_run_info(conn, args.after)
    if not before_info or not after_info:
        print("Run not found.", file=sys.stderr)
        sys.exit(1)

    before_results = models.get_results_for_runs(conn, [args.before])
    after_results = models.get_results_for_runs(conn, [args.after])
    conn.close()

    # Index by (label, metric) -> value
    before_map: dict[tuple[str, str], float] = {}
    for r in before_results:
        before_map[(r["benchmark_label"], r["metric_name"])] = r["value_us"]

    after_map: dict[tuple[str, str], float] = {}
    for r in after_results:
        after_map[(r["benchmark_label"], r["metric_name"])] = r["value_us"]

    # Determine which metric to compare
    metric = args.metric or "t_generic"

    # Determine label filter and prefix filter
    prefix = args.prefix or "einsum-"

    # Collect matching labels
    all_labels = sorted({
        label for (label, m) in (set(before_map) | set(after_map))
        if m == metric and label.startswith(prefix)
    })

    if args.filter:
        all_labels = [l for l in all_labels if args.filter.lower() in l.lower()]

    if not all_labels:
        print(f"No matching benchmarks (metric={metric}, prefix='{prefix}').")
        return

    # Print header
    before_commit = (before_info.get("git_commit") or "")[:8]
    after_commit = (after_info.get("git_commit") or "")[:8]
    print(f"Diff: run #{args.before} ({before_commit}) -> #{args.after} ({after_commit})")
    print(f"Metric: {metric}  Prefix: '{prefix}'")
    print()

    # Strip prefix for display
    display_labels = [(l, l[len(prefix):] if l.startswith(prefix) else l) for l in all_labels]
    name_width = max(len(d) for _, d in display_labels)
    name_width = max(name_width, 9)  # "Benchmark"

    header = f"{'Benchmark':<{name_width}}  {'Before (us)':>12}  {'After (us)':>12}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, display in display_labels:
        before_val = before_map.get((label, metric))
        after_val = after_map.get((label, metric))
        if before_val is not None and after_val is not None and after_val > 0:
            speedup = before_val / after_val
            rows.append((display, before_val, after_val, speedup))
        elif before_val is not None:
            rows.append((display, before_val, None, None))
        elif after_val is not None:
            rows.append((display, None, after_val, None))

    # Sort by speedup descending (biggest wins first)
    rows.sort(key=lambda r: -(r[3] or 0))

    for display, bv, av, sp in rows:
        bv_str = f"{bv:>12.1f}" if bv is not None else f"{'n/a':>12}"
        av_str = f"{av:>12.1f}" if av is not None else f"{'n/a':>12}"
        if sp is not None:
            if sp >= 1.5:
                sp_str = f"{sp:>7.1f}x"
            elif sp <= 0.67:
                sp_str = f"{sp:>7.1f}x"
            else:
                sp_str = f"{'~1.0x':>8}"
        else:
            sp_str = f"{'n/a':>8}"
        print(f"{display:<{name_width}}  {bv_str}  {av_str}  {sp_str}")

    # Summary
    speedups = [sp for _, _, _, sp in rows if sp is not None]
    if speedups:
        big_wins = sum(1 for s in speedups if s >= 1.5)
        regressions = sum(1 for s in speedups if s <= 0.67)
        print()
        print(f"{big_wins} benchmarks faster (>=1.5x), {regressions} slower (<=0.67x), {len(speedups) - big_wins - regressions} unchanged")


def main() -> None:
    """Main CLI entry point."""
    top = argparse.ArgumentParser(
        prog="einsums-bench",
        description="Einsums performance regression detection system",
    )
    top.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database")
    sub = top.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Run benchmarks and store results")
    p_run.add_argument("--build-dir", default="build", required=True, help="CMake build directory")
    p_run.add_argument("--source-dir", default=".", help="Source directory (default: .)")
    p_run.add_argument("--no-build", action="store_true", help="Skip building")
    p_run.add_argument("--reps", type=int, default=1, help="Number of repetitions")
    p_run.add_argument("--notes", help="Free-form notes for this run")
    p_run.add_argument("--targets", nargs="+", help="Only run these test targets (by name)")
    p_run.set_defaults(func=cmd_run)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Compare run against baseline")
    p_cmp.add_argument("--run-id", type=int, help="Run to compare (default: latest)")
    p_cmp.add_argument("--baseline-run-id", type=int, help="Specific baseline run")
    p_cmp.add_argument("--baseline", default="main", help="Baseline branch (default: main)")
    p_cmp.add_argument("--baseline-count", type=int, default=10, help="Number of baseline runs")
    p_cmp.add_argument("--z-threshold", type=float, default=3.0, help="Z-score threshold")
    p_cmp.add_argument("--min-pct", type=float, default=5.0, help="Minimum %% change to flag")
    p_cmp.add_argument("--json", action="store_true", help="Output as JSON")
    p_cmp.set_defaults(func=cmd_compare)

    # --- list-runs ---
    p_list = sub.add_parser("list-runs", help="List recent benchmark runs")
    p_list.add_argument("--branch", help="Filter by branch")
    p_list.add_argument("--last", type=int, default=20, help="Number of runs to show")
    p_list.set_defaults(func=cmd_list_runs)

    # --- tag-baseline ---
    p_tag = sub.add_parser("tag-baseline", help="Tag a run as a named baseline")
    p_tag.add_argument("--run-id", type=int, required=True)
    p_tag.add_argument("--name", required=True, help="Baseline name")
    p_tag.set_defaults(func=cmd_tag_baseline)

    # --- bisect ---
    p_bis = sub.add_parser(
        "bisect",
        help="Single bisect step for `git bisect run`",
        description=(
            "Build, run one benchmark, and exit 0 (good) or 1 (bad). "
            "Exit 125 if build fails or benchmark not found (git bisect skips the commit)."
        ),
    )
    p_bis.add_argument("--build-dir", default="build", required=True, help="CMake build directory")
    p_bis.add_argument("--source-dir", default=".", help="Source directory (default: .)")
    p_bis.add_argument("--benchmark", required=True, help="Benchmark label to check (e.g. 'Rank-2 N=64')")
    p_bis.add_argument("--metric", default="t_generic", help="Metric name (default: t_generic)")
    p_bis.add_argument("--threshold-us", type=float, required=True, help="Threshold in microseconds; above = bad")
    p_bis.add_argument("--reps", type=int, default=3, help="Repetitions for confirmation (default: 3)")
    p_bis.add_argument("--no-build", action="store_true", help="Skip building (assume already built)")
    p_bis.set_defaults(func=cmd_bisect)

    # --- export ---
    p_exp = sub.add_parser("export", help="Export run results as JSON")
    p_exp.add_argument("--run-id", type=int, required=True)
    p_exp.set_defaults(func=cmd_export)

    # --- show ---
    p_show = sub.add_parser("show", help="Show results for a single run")
    p_show.add_argument("--run-id", type=int, help="Run to show (default: latest)")
    p_show.add_argument("--filter", help="Filter benchmark labels (substring match)")
    p_show.add_argument("--metric", nargs="+", help="Metrics to show (default: all)")
    p_show.set_defaults(func=cmd_show)

    # --- diff ---
    p_diff = sub.add_parser("diff", help="Compare two runs side by side with speedup")
    p_diff.add_argument("--before", type=int, help="Earlier run ID (default: oldest recent)")
    p_diff.add_argument("--after", type=int, help="Later run ID (default: latest)")
    p_diff.add_argument("--metric", help="Metric to compare (default: t_generic)")
    p_diff.add_argument("--prefix", help="Label prefix filter (default: 'einsum-')")
    p_diff.add_argument("--filter", help="Additional substring filter on labels")
    p_diff.set_defaults(func=cmd_diff)

    # --- trend ---
    p_trend = sub.add_parser("trend", help="Show time series for a benchmark metric")
    p_trend.add_argument("--benchmark", required=True, help="Benchmark label (exact match)")
    p_trend.add_argument("--metric", default="t_generic", help="Metric name (default: t_generic)")
    p_trend.add_argument("--branch", default="main", help="Branch to filter (default: main)")
    p_trend.add_argument("--last", type=int, default=50, help="Number of data points")
    p_trend.add_argument("--json", action="store_true", help="Output as JSON")
    p_trend.set_defaults(func=cmd_trend)

    # --- scaling ---
    p_scale = sub.add_parser("scaling", help="Show performance vs N for a benchmark family")
    p_scale.add_argument("--run-id", type=int, help="Run to analyze (default: latest)")
    p_scale.add_argument("--benchmark", required=True, help="Benchmark label prefix (e.g., 'gemm')")
    p_scale.add_argument("--metric", default="t_generic", help="Metric name")
    p_scale.set_defaults(func=cmd_scaling)

    # --- list-tests ---
    p_tests = sub.add_parser("list-tests", help="List available performance test binaries")
    p_tests.add_argument("--build-dir", default="build", required=True, help="CMake build directory")
    p_tests.set_defaults(func=cmd_list_tests)

    # --- delete-db ---
    p_del = sub.add_parser("delete-db", help="Delete the benchmark database entirely")
    p_del.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    p_del.set_defaults(func=cmd_delete_db)

    args = top.parse_args()
    args.func(args)


def cmd_trend(args: argparse.Namespace) -> None:
    """Show time series for a benchmark metric."""
    import json as _json

    conn = models.init_db(Path(args.db))
    data = models.get_trend(conn, args.benchmark, args.metric,
                             branch=args.branch, last=args.last)
    conn.close()

    if not data:
        print(f"No data found for '{args.benchmark}' / '{args.metric}' on branch '{args.branch}'.")
        return

    if args.json:
        print(_json.dumps(data, indent=2))
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
    conn = models.init_db(Path(args.db))

    run_id = args.run_id
    if run_id is None:
        runs = models.list_runs(conn, last=1)
        if not runs:
            print("No runs found.")
            return
        run_id = runs[0]["run_id"]

    data = models.get_scaling(conn, run_id, args.benchmark, args.metric)
    conn.close()

    if not data:
        print(f"No scaling data for '{args.benchmark}*' / '{args.metric}' in run {run_id}.")
        return

    print(f"Scaling: '{args.benchmark}*' metric='{args.metric}' run #{run_id}")
    print(f"{'N':>8}  {'Time (us)':>12}  {'GFLOP/s':>10}")
    print("-" * 35)

    for d in data:
        n = d["n"]
        val = d["value_us"]
        # Try to get gflops from extra_json
        gflops = ""
        if d.get("extra_json"):
            import json as _json
            try:
                extra = _json.loads(d["extra_json"])
                if "gflops" in extra:
                    gflops = f"{extra['gflops']:>10.2f}"
            except Exception:
                pass
        if not gflops:
            gflops = f"{'':>10}"
        print(f"{n:>8}  {val:>12.2f}  {gflops}")


def cmd_list_tests(args: argparse.Namespace) -> None:
    """List available performance test binaries in the build tree."""
    build_dir = Path(args.build_dir)
    if not build_dir.is_dir():
        print(f"Build directory not found: {build_dir}")
        sys.exit(1)

    tests = runner._discover_perf_tests(build_dir)
    if not tests:
        print("No performance tests found. Ensure the project is built with EINSUMS_WITH_TESTS_BENCHMARKS=ON.")
        return

    print(f"Found {len(tests)} performance test binaries:\n")
    for name, path in tests:
        print(f"  {name:<30}  {path}")


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


if __name__ == "__main__":
    main()
