# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""SQLite database schema, migrations, and access layer for benchmark results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from devtools.benchmarks.parser import BenchmarkResult, BreakdownResult, CacheHitResult, compute_gflops

# ---------------------------------------------------------------------------
# Schema version and migrations
# ---------------------------------------------------------------------------

# Schema version must match the number of .sql files in the migrations/ directory.
# Bump this when adding a new migration file.
SCHEMA_VERSION = 3

# Directory containing the shared .sql migration files (used by both Python and C++).
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _load_migration_sql(version: int) -> str:
    """Load the SQL content for a migration by version number.

    Migration files are named NNN_description.sql and live in the
    migrations/ directory alongside this module.  Both the Python CLI
    and the C++ ImGui viewer execute the same SQL, keeping the schema
    in sync.
    """
    pattern = f"{version:03d}_*.sql"
    matches = sorted(_MIGRATIONS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No migration file matching '{pattern}' in {_MIGRATIONS_DIR}"
        )
    return matches[0].read_text()


# ---------------------------------------------------------------------------
# To add a new migration in the future:
#   1. Create devtools/benchmarks/migrations/NNN_description.sql
#   2. Bump SCHEMA_VERSION at the top of this file
#   3. Rebuild the ImGui viewer (so configure_file re-embeds the SQL)
#   Both Python and C++ will pick up the new migration automatically.
# ---------------------------------------------------------------------------


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, col_type: str,
    default: str | None = None,
) -> None:
    """Add a column to a table if it doesn't already exist."""
    default_clause = f" DEFAULT {default}" if default is not None else ""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from the database.

    Returns 0 if the schema_version table doesn't exist (pre-migration DB
    or fresh DB).
    """
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set the schema version."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    conn.commit()


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run any pending migrations to bring the database up to SCHEMA_VERSION."""
    current = _get_schema_version(conn)

    if current >= SCHEMA_VERSION:
        return  # Already up to date

    # Detect pre-migration databases: if tables exist but no schema_version,
    # we still need to run migrations to add missing columns.
    has_tables = False
    try:
        conn.execute("SELECT 1 FROM runs LIMIT 1")
        has_tables = True
    except sqlite3.OperationalError:
        pass

    if has_tables and current == 0:
        print(f"Migrating existing database to schema version {SCHEMA_VERSION}...")
    elif current == 0:
        pass  # Fresh database, no message needed
    else:
        print(f"Upgrading database from version {current} to {SCHEMA_VERSION}...")

    for version in range(current + 1, SCHEMA_VERSION + 1):
        sql = _load_migration_sql(version)
        # Execute each statement individually so we can catch "duplicate column"
        # errors from ALTER TABLE ADD COLUMN (which aren't idempotent in SQLite).
        for statement in sql.split(";"):
            # Strip comment-only lines from the chunk, then check if anything remains.
            lines = [ln for ln in statement.splitlines() if not ln.strip().startswith("--")]
            cleaned = "\n".join(lines).strip()
            if not cleaned:
                continue
            try:
                conn.execute(cleaned)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        conn.commit()

    _set_schema_version(conn, SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the benchmark database, run migrations, return connection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _run_migrations(conn)
    return conn


def create_run(
    conn: sqlite3.Connection,
    *,
    git_commit: str,
    git_branch: str | None = None,
    git_dirty: bool = False,
    hostname: str | None = None,
    build_type: str | None = None,
    compiler: str | None = None,
    blas_vendor: str | None = None,
    cpu_model: str | None = None,
    cpu_freq_mhz: int | None = None,
    power_source: str | None = None,
    cmake_options: dict | None = None,
    hptt_method: str = "estimate",
    notes: str = "",
) -> int:
    """Insert a new run record and return its run_id."""
    cur = conn.execute(
        """INSERT INTO runs
           (git_commit, git_branch, git_dirty, timestamp, hostname,
            build_type, compiler, blas_vendor, cpu_model, cpu_freq_mhz,
            power_source, cmake_options, hptt_method, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            git_commit,
            git_branch,
            git_dirty,
            datetime.now(timezone.utc).isoformat(),
            hostname,
            build_type,
            compiler,
            blas_vendor,
            cpu_model,
            cpu_freq_mhz,
            power_source,
            json.dumps(cmake_options) if cmake_options else None,
            hptt_method,
            notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def store_results(
    conn: sqlite3.Connection,
    run_id: int,
    test_binary: str,
    benchmarks: list[BenchmarkResult],
    breakdowns: list[BreakdownResult] | None = None,
    cache_hits: list[CacheHitResult] | None = None,
) -> None:
    """Flatten parsed benchmark results into the results table."""
    rows = []

    for b in benchmarks:
        label = f"{b.label} N={b.n}"
        metrics = [
            ("t_generic", b.t_generic, None),
            ("t_blas_packed", b.t_blas_packed, None),
            ("t_mlir", b.t_mlir, None),
            ("t_einsum", b.t_einsum, b.einsum_alg or None),
            ("t_sort_gemm", b.t_sort_gemm, None),
        ]
        extra = {}
        if b.min_us is not None:
            extra["min_us"] = b.min_us
        if b.max_us is not None:
            extra["max_us"] = b.max_us
        if b.stddev_us is not None:
            extra["stddev_us"] = b.stddev_us
        if b.cv_pct is not None:
            extra["cv_pct"] = b.cv_pct
        if b.warmup_us is not None:
            extra["warmup_us"] = b.warmup_us
        if b.warmup_ratio is not None:
            extra["warmup_ratio"] = round(b.warmup_ratio, 2)
        gflops = compute_gflops(b.label, b.n, b.t_generic)
        if gflops is not None:
            extra["gflops"] = round(gflops, 3)
        extra_str = json.dumps(extra) if extra else None

        for metric_name, value, alg in metrics:
            if value is not None:
                rows.append((run_id, test_binary, label, metric_name, value, "us", alg, extra_str))

    if breakdowns:
        for bd in breakdowns:
            label = f"Breakdown N={bd.n}"
            for metric_name, value in [
                ("breakdown_pack_only", bd.t_pack_only),
                ("breakdown_kernel_only", bd.t_kernel_only),
                ("breakdown_total", bd.t_total),
            ]:
                rows.append((run_id, test_binary, label, metric_name, value, "us", None,
                             json.dumps({"mr": bd.mr, "nr": bd.nr, "tiles": bd.tiles})))

    if cache_hits:
        for ch in cache_hits:
            label = f"Cache hit N={ch.n}"
            for metric_name, value in [
                ("cache_t_generic", ch.t_generic),
                ("cache_t_mlir", ch.t_mlir),
                ("cache_t_einsum", ch.t_einsum),
            ]:
                rows.append((run_id, test_binary, label, metric_name, value, "us",
                             ch.einsum_alg if metric_name == "cache_t_einsum" else None, None))

    conn.executemany(
        """INSERT INTO results
           (run_id, test_binary, benchmark_label, metric_name, value_us, unit, algorithm, extra_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def get_baseline_runs(
    conn: sqlite3.Connection,
    *,
    branch: str = "main",
    n: int = 10,
) -> list[int]:
    """Return the run_ids of the last `n` runs on `branch`, newest first.

    Prefers clean (non-dirty) runs, but falls back to including dirty runs
    if no clean runs exist on the branch.
    """
    rows = conn.execute(
        """SELECT run_id FROM runs
           WHERE git_branch = ? AND git_dirty = 0
           ORDER BY timestamp DESC LIMIT ?""",
        (branch, n),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            """SELECT run_id FROM runs
               WHERE git_branch = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (branch, n),
        ).fetchall()
    return [r["run_id"] for r in rows]


def get_results_for_runs(
    conn: sqlite3.Connection,
    run_ids: list[int],
    *,
    benchmark_label: str | None = None,
    metric_name: str | None = None,
) -> list[dict]:
    """Fetch results for given run_ids, optionally filtered."""
    placeholders = ",".join("?" * len(run_ids))
    query = f"SELECT * FROM results WHERE run_id IN ({placeholders})"
    params: list = list(run_ids)

    if benchmark_label:
        query += " AND benchmark_label = ?"
        params.append(benchmark_label)
    if metric_name:
        query += " AND metric_name = ?"
        params.append(metric_name)

    query += " ORDER BY run_id, benchmark_label, metric_name"
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def tag_baseline(conn: sqlite3.Connection, run_id: int, name: str) -> None:
    """Tag a run as a named baseline."""
    conn.execute(
        """INSERT OR REPLACE INTO baselines (run_id, name, created_at)
           VALUES (?, ?, ?)""",
        (run_id, name, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_baseline_by_name(conn: sqlite3.Connection, name: str) -> int | None:
    """Look up a named baseline's run_id."""
    row = conn.execute(
        "SELECT run_id FROM baselines WHERE name = ?", (name,)
    ).fetchone()
    return row["run_id"] if row else None


def get_run_info(conn: sqlite3.Connection, run_id: int) -> dict | None:
    """Get run metadata."""
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(
    conn: sqlite3.Connection,
    *,
    branch: str | None = None,
    last: int = 20,
) -> list[dict]:
    """List recent runs, optionally filtered by branch."""
    if branch:
        rows = conn.execute(
            """SELECT * FROM runs WHERE git_branch = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (branch, last),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (last,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_run(conn: sqlite3.Connection, run_id: int) -> bool:
    """Delete a run and all its results. Returns True if the run existed."""
    row = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return False
    conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM baselines WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    conn.commit()
    return True


def delete_all_runs(conn: sqlite3.Connection) -> int:
    """Delete all runs, results, and baselines. Returns number of runs deleted."""
    count = conn.execute("SELECT COUNT(*) as cnt FROM runs").fetchone()["cnt"]
    conn.execute("DELETE FROM results")
    conn.execute("DELETE FROM baselines")
    conn.execute("DELETE FROM runs")
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Trend and analysis queries
# ---------------------------------------------------------------------------

def get_trend(
    conn: sqlite3.Connection,
    benchmark_label: str,
    metric_name: str,
    *,
    branch: str | None = "main",
    last: int = 50,
) -> list[dict]:
    """Get time series for a benchmark metric across recent runs.

    Returns list of dicts with: run_id, timestamp, git_commit, value_us, annotations.
    Ordered by timestamp ascending (oldest first).
    """
    query = """
        SELECT r.run_id, r.timestamp, r.git_commit, r.git_branch,
               res.value_us, res.algorithm, res.extra_json, res.annotations
        FROM results res
        JOIN runs r ON res.run_id = r.run_id
        WHERE res.benchmark_label = ? AND res.metric_name = ?
    """
    params: list = [benchmark_label, metric_name]
    if branch:
        query += " AND r.git_branch = ?"
        params.append(branch)
    query += " ORDER BY r.timestamp ASC LIMIT ?"
    params.append(last)

    return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_scaling(
    conn: sqlite3.Connection,
    run_id: int,
    benchmark_prefix: str,
    metric_name: str,
) -> list[dict]:
    """Get scaling data (N vs time) for a benchmark family in a single run.

    Expects benchmark labels like "gemm N=64", "gemm N=128", etc.
    Returns list of dicts with: benchmark_label, n, value_us, annotations.
    Ordered by N ascending.
    """
    rows = conn.execute(
        """SELECT benchmark_label, value_us, algorithm, extra_json, annotations
           FROM results
           WHERE run_id = ? AND benchmark_label LIKE ? AND metric_name = ?
           ORDER BY benchmark_label""",
        (run_id, f"{benchmark_prefix}%", metric_name),
    ).fetchall()

    results = []
    import re
    for row in rows:
        d = dict(row)
        # Extract N from label (e.g., "gemm N=256" → 256)
        m = re.search(r"N=(\d+)", d["benchmark_label"])
        d["n"] = int(m.group(1)) if m else 0
        results.append(d)

    return sorted(results, key=lambda x: x["n"])


def get_annotations(
    conn: sqlite3.Connection,
    run_id: int,
    benchmark_label: str,
    metric_name: str,
) -> dict | None:
    """Get annotations for a specific result."""
    row = conn.execute(
        """SELECT annotations FROM results
           WHERE run_id = ? AND benchmark_label = ? AND metric_name = ?""",
        (run_id, benchmark_label, metric_name),
    ).fetchone()
    if row and row["annotations"]:
        return json.loads(row["annotations"])
    return None
