-- Migration 001: Initial schema
-- Creates core tables, indexes, and columns for the benchmark database.

CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    git_commit      TEXT NOT NULL,
    git_branch      TEXT,
    git_dirty       BOOLEAN DEFAULT 0,
    timestamp       TEXT NOT NULL,
    hostname        TEXT,
    build_type      TEXT,
    compiler        TEXT,
    blas_vendor     TEXT,
    cpu_model       TEXT,
    cmake_options   TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    test_binary     TEXT NOT NULL,
    benchmark_label TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    value_us        REAL NOT NULL,
    unit            TEXT DEFAULT 'us',
    algorithm       TEXT,
    extra_json      TEXT
);

CREATE TABLE IF NOT EXISTS baselines (
    baseline_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    name            TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_lookup
    ON results(benchmark_label, metric_name, run_id);

CREATE INDEX IF NOT EXISTS idx_runs_commit
    ON runs(git_commit);

CREATE INDEX IF NOT EXISTS idx_runs_branch_time
    ON runs(git_branch, timestamp);

-- Columns added retroactively to runs (may already exist on upgraded DBs).
-- Both Python and C++ migration runners handle "duplicate column" errors.
ALTER TABLE runs ADD COLUMN cpu_freq_mhz INTEGER;
ALTER TABLE runs ADD COLUMN power_source TEXT;
