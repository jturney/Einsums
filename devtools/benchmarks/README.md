# Einsums Benchmark System

Performance regression detection for the Einsums tensor algebra library.
Results are stored in a shared SQLite database that can be accessed from both
the Python CLI and the C++ ImGui profile viewer.

## Quick Start

```bash
# Run all benchmarks, store results in the database
python -m devtools.benchmarks run --build-dir build

# Compare the latest run against baseline
python -m devtools.benchmarks compare

# View interactive trends and scaling charts
./build/bin/profile_viewer_imgui --db devtools/benchmarks/benchmarks.db
```

## CLI Reference

All commands are invoked via `python -m devtools.benchmarks <command>`.
The `--db` flag is shared across all commands. The default database lives in
the project tree at `<project>/.einsums-studio/benchmarks.db`, the same path
the Einsums Studio benchmark bundle reads from, so each checkout carries
its own history. Pass `--db <path>` to point at a different file.

### `run` -- Execute benchmarks

Builds the project, runs all performance test binaries, collects results via
the profiler server (or falls back to stdout parsing), and stores them in the
database.

```bash
python -m devtools.benchmarks run \
    --build-dir build \
    --source-dir . \
    --reps 3 \
    --notes "baseline for v2.1"

# Skip the build step (already built)
python -m devtools.benchmarks run --build-dir build --no-build
```

Options:
- `--build-dir` (required): CMake build directory
- `--source-dir`: Source tree root (default: `.`)
- `--no-build`: Skip `cmake --build`
- `--reps N`: Run benchmarks N times (builds only on first rep)
- `--notes TEXT`: Free-form notes attached to the run record

### `compare` -- Regression detection

Compares a run against baseline using MAD-based z-score statistics.
Classifies each benchmark as regression, improvement, unchanged, or noisy.

```bash
# Compare latest run against last 10 runs on main
python -m devtools.benchmarks compare

# Compare a specific run
python -m devtools.benchmarks compare --run-id 42

# Against a specific baseline branch with custom thresholds
python -m devtools.benchmarks compare \
    --baseline feature-branch \
    --baseline-count 5 \
    --z-threshold 2.5 \
    --min-pct 3.0

# JSON output (for CI integration)
python -m devtools.benchmarks compare --json
```

Options:
- `--run-id`: Run to compare (default: latest)
- `--baseline-run-id`: Compare against a specific run instead of a branch
- `--baseline`: Baseline branch name (default: `main`)
- `--baseline-count`: Number of baseline runs for statistics (default: 10)
- `--z-threshold`: Z-score threshold for flagging (default: 3.0)
- `--min-pct`: Minimum percent change to flag (default: 5.0)
- `--json`: Output results as JSON

### `show` -- Display run results

```bash
# Show results for the latest run
python -m devtools.benchmarks show

# Show a specific run, filtered
python -m devtools.benchmarks show --run-id 42 --filter "gemm" --metric t_einsum t_generic
```

Options:
- `--run-id`: Run to display (default: latest)
- `--filter`: Substring filter on benchmark labels
- `--metric`: One or more metric names to show (default: all)

### `diff` -- Side-by-side comparison

Compares two runs with speedup ratios.

```bash
python -m devtools.benchmarks diff --before 40 --after 42
python -m devtools.benchmarks diff --metric t_einsum --prefix "einsum-"
```

Options:
- `--before`: Earlier run ID (default: second-latest)
- `--after`: Later run ID (default: latest)
- `--metric`: Metric to compare (default: `t_generic`)
- `--prefix`: Label prefix filter (default: `einsum-`)
- `--filter`: Additional substring filter

### `trend` -- Time series

Shows how a specific benchmark metric changes over time across runs.

```bash
python -m devtools.benchmarks trend \
    --benchmark "blas-gemm N=1024" \
    --metric t_blas \
    --branch main \
    --last 30

# JSON output
python -m devtools.benchmarks trend --benchmark "blas-gemm N=1024" --metric t_blas --json
```

Output includes an ASCII sparkline showing the value trend.

Options:
- `--benchmark` (required): Exact benchmark label
- `--metric`: Metric name (default: `t_generic`)
- `--branch`: Filter by branch (default: `main`)
- `--last`: Number of data points (default: 50)
- `--json`: Output as JSON array

### `scaling` -- Performance vs N

Shows how timing scales with problem size N for a benchmark family.

```bash
python -m devtools.benchmarks scaling \
    --benchmark "blas-gemm" \
    --metric t_blas

# For a specific run
python -m devtools.benchmarks scaling --run-id 42 --benchmark "einsum-gemm" --metric t_einsum
```

Matches all labels starting with the prefix (e.g., `blas-gemm` matches
`blas-gemm N=64`, `blas-gemm N=256`, etc.) and extracts N from the label.

Options:
- `--benchmark` (required): Label prefix
- `--metric`: Metric name (default: `t_generic`)
- `--run-id`: Run to analyze (default: latest)

### `list-runs` -- Browse run history

```bash
python -m devtools.benchmarks list-runs
python -m devtools.benchmarks list-runs --branch main --last 10
```

Options:
- `--branch`: Filter by branch
- `--last`: Number of runs to show (default: 20)

### `tag-baseline` -- Name a baseline

Tags a run for easy reference in comparisons.

```bash
python -m devtools.benchmarks tag-baseline --run-id 42 --name "v2.0-release"
```

### `list-tests` -- Discover available benchmarks

Lists all performance test binaries found in the build tree.

```bash
python -m devtools.benchmarks list-tests --build-dir build
```

Options:
- `--build-dir` (required): CMake build directory

### `bisect` -- Git bisect integration

Single step for `git bisect run`. Builds, runs one benchmark, and exits
0 (good), 1 (bad), or 125 (skip -- build failed or benchmark not found).

```bash
git bisect start HEAD v2.0
git bisect run python -m devtools.benchmarks bisect \
    --build-dir build \
    --benchmark "blas-gemm N=1024" \
    --metric t_blas \
    --threshold-us 500
```

Options:
- `--build-dir` (required): CMake build directory
- `--source-dir`: Source tree root (default: `.`)
- `--benchmark` (required): Benchmark label to check
- `--metric`: Metric name (default: `t_generic`)
- `--threshold-us` (required): Above this = bad (exit 1)
- `--reps`: Repetitions for confirmation (default: 3)
- `--no-build`: Skip building

### `export` -- Export as JSON

```bash
python -m devtools.benchmarks export --run-id 42 > run42.json
```

## ImGui Profile Viewer

The C++ profile viewer includes a Benchmarks panel with interactive
visualization. It connects to the same SQLite database.

```bash
# Launch with default database location
./build/bin/profile_viewer_imgui

# Specify a database path
./build/bin/profile_viewer_imgui --db devtools/benchmarks/benchmarks.db

# Standalone database browser (no live connection)
./build/bin/profile_viewer_imgui --no-mdns --db benchmarks.db
```

The Benchmarks panel has four tabs:

- Live Results: streaming table of benchmark events from a connected process
- Trends: interactive line chart of a metric over time, selectable by label, metric, and branch
- Scaling: log-log plot of N vs time for a benchmark family
- Compare: regression table with color-coded verdicts

When a benchmark process is connected, results are automatically stored in the
database. No manual "record" step is needed.

## Database

The database is SQLite, stored at `<project>/.einsums-studio/benchmarks.db`
by default. Both the Python CLI and the C++ benchmark bundle in Einsums
Studio read and write the same schema.

### Schema

`runs` -- one row per benchmark execution:
- `run_id`, `git_commit`, `git_branch`, `git_dirty`, `timestamp`
- `hostname`, `build_type`, `compiler`, `blas_vendor`, `cpu_model`
- `cpu_freq_mhz`, `power_source`, `hptt_method`, `cmake_options`, `notes`

`results` -- one row per (benchmark, metric) measurement:
- `result_id`, `run_id`, `test_binary`, `benchmark_label`, `metric_name`
- `value_us`, `unit`, `algorithm`, `extra_json`, `annotations`

`baselines` -- named references to runs:
- `baseline_id`, `run_id`, `name`, `created_at`

### Migrations

Schema migrations live in `devtools/benchmarks/migrations/` as numbered
`.sql` files. Both Python and C++ execute the same SQL:

```
devtools/benchmarks/migrations/
    001_initial.sql        # Tables, indexes, cpu_freq_mhz, power_source
    002_hptt_method.sql    # hptt_method column
    003_annotations.sql    # annotations column
```

To add a new migration:
1. Create `devtools/benchmarks/migrations/NNN_description.sql`
2. Bump `SCHEMA_VERSION` in `devtools/benchmarks/models.py`
3. Rebuild the ImGui viewer (so CMake re-embeds the SQL)

## Python Module Structure

| File | Purpose |
|------|---------|
| `__main__.py` | Entry point for `python -m devtools.benchmarks` |
| `cli.py` | Argument parsing and subcommand dispatch |
| `runner.py` | Builds project, runs test binaries, collects results via profiler server |
| `models.py` | SQLite schema, migrations, and query functions |
| `parser.py` | Regex-based stdout parser (fallback when profiler unavailable) |
| `compare.py` | MAD-based z-score regression detection |
| `bisect.py` | Git bisect automation |
| `report.py` | Formatted text output for show/list commands |
| `html_export.py` | HTML report generation |
| `migrations/` | Shared SQL migration files (used by both Python and C++) |
