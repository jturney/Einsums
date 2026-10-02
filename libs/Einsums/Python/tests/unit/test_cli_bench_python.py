# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``python -m einsums bench``: the result database and the commands that read it.

``bench run`` and ``bench bisect`` build and execute the performance tests, which is too
slow for a unit test; everything after the results are stored is covered here, from a
database filled with parsed sample output.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys

import pytest

from einsums.cli import main
from einsums.cli.bench import commands, models, parser

# The line performance::report prints: with the metric (current), without it (older
# builds, recorded as t_generic), and with the logger's thread prefix.
SAMPLE = """\
[blas-gemm N=256] Time: 300.00 us  min: 290.00  max: 310.00  stddev: 5.00  cv: 1.6%  warmup: 400.00 us (1.3x)  metric: t_blas
[blas-gemm N=512] Time: {t512:.2f} us  min: 2300.00  max: 2500.00  stddev: 50.00  cv: 2.0%  warmup: 3000.00 us (1.2x)
[ tid #  3 ] [blas-axpy N=1024] Time: 12.50 us
"""


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "bench.db"
    conn = models.init_db(path)
    for commit, t512 in (("aaaaaaaa1", 2400.0), ("bbbbbbbb2", 1200.0)):
        run_id = models.create_run(conn, git_commit=commit, git_branch="main", hostname="box")
        models.store_results(conn, run_id, "BenchmarkBLAS", parser.parse_output(SAMPLE.format(t512=t512)))
    conn.close()
    return path


def bench(db, *args):
    return main(["bench", "--db", str(db), *args])


def test_parser_reads_the_printed_line_with_and_without_its_metric():
    results = parser.parse_output(SAMPLE.format(t512=2400.0))
    assert [(r.label, r.n, r.metric) for r in results] == [
        ("blas-gemm N=256", 256, "t_blas"),
        ("blas-gemm N=512", 512, "t_generic"),
        ("blas-axpy N=1024", 1024, "t_generic"),
    ]
    assert results[0].stddev_us == 5.0 and results[2].stddev_us is None
    assert all(r.source == "stdout" for r in results)


def test_server_events_and_printed_lines_store_alike(tmp_path):
    # The two collection paths must agree on label and metric, or a run collected one
    # way will not compare with history collected the other.
    event = {"type": "benchmark_result", "label": "blas-gemm N=256", "metric": "t_blas", "value_us": 300.0,
             "min_us": 290.0, "max_us": 310.0, "stddev_us": 5.0, "warmup_us": 400.0, "reps": 20,
             "annotations": {"algorithm": "dgemm", "gflops": "111.8"}}  # fmt: skip
    printed = parser.parse_output(SAMPLE.format(t512=1.0))[0]
    from_server = parser.Result.from_event(event)
    assert (from_server.label, from_server.metric) == (printed.label, printed.metric)
    conn = models.init_db(tmp_path / "x.db")
    run_id = models.create_run(conn, git_commit="c")
    models.store_results(conn, run_id, "B", [from_server, printed])
    rows = conn.execute("SELECT source, algorithm, extra_json FROM results ORDER BY result_id").fetchall()
    assert [r[0] for r in rows] == ["profiler", "stdout"]
    assert rows[0][1] == "dgemm" and json.loads(rows[0][2])["gflops"] == 111.8
    assert json.loads(rows[1][2])["gflops"] == pytest.approx(2 * 256**3 / 300e3, rel=1e-3)  # estimated from the label


def test_migrations_bring_a_new_database_to_the_current_schema(tmp_path):
    conn = models.init_db(tmp_path / "new.db")
    assert models._get_schema_version(conn) == models.SCHEMA_VERSION
    columns = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    assert {"annotations", "metric_name", "value_us"} <= columns


def test_list_show_and_diff(db, capsys):
    bench(db, "list-runs")
    out = capsys.readouterr().out
    assert "aaaaaaaa" in out and "bbbbbbbb" in out
    bench(db, "show", "--filter", "gemm")
    out = capsys.readouterr().out
    assert "blas-gemm N=512" in out and "blas-axpy" not in out
    bench(db, "diff", "--before", "1", "--after", "2", "--prefix", "blas-")
    out = capsys.readouterr().out
    assert "gemm N=512 [t_generic]" in out and "1 benchmarks faster" in out  # 2400 us -> 1200 us
    bench(db, "diff", "--before", "1", "--after", "2", "--metric", "t_generic", "--json")
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert rows[0]["benchmark"] == "blas-gemm N=512" and rows[0]["speedup"] == pytest.approx(2.0)


def test_trend_and_scaling(db, capsys):
    # One metric recorded for the label, so --metric is not needed.
    bench(db, "trend", "--benchmark", "blas-gemm N=512", "--json")
    assert [p["value_us"] for p in json.loads(capsys.readouterr().out)] == [2400.0, 1200.0]
    # The family has two metrics (t_blas at 256, t_generic at 512): name one.
    with pytest.raises(SystemExit) as exit_info:
        bench(db, "scaling", "--benchmark", "blas-gemm")
    assert exit_info.value.code == 2 and "t_blas, t_generic" in capsys.readouterr().err
    bench(db, "scaling", "--benchmark", "blas-gemm", "--metric", "t_generic", "--json")
    assert [p["n"] for p in json.loads(capsys.readouterr().out)] == [512]


def test_read_commands_speak_json(db, capsys):
    bench(db, "list-runs", "--json")
    assert [r["git_commit"] for r in json.loads(capsys.readouterr().out)] == ["bbbbbbbb2", "aaaaaaaa1"]
    bench(db, "show", "--run-id", "1", "--json")
    shown = json.loads(capsys.readouterr().out)
    assert shown["results"]["blas-gemm N=256"] == {"t_blas": 300.0}


def test_export_json_and_html(db, tmp_path, capsys):
    bench(db, "export", "--run-id", "2")
    data = json.loads(capsys.readouterr().out)
    assert data["run"]["git_commit"] == "bbbbbbbb2" and data["results"]
    html = tmp_path / "report.html"
    bench(db, "export", "--run-id", "2", "--html", str(html))
    assert html.read_text().lstrip().lower().startswith("<!doctype html")


def test_tag_baseline_then_compare(db, capsys):
    bench(db, "tag-baseline", "--run-id", "1", "--name", "before")
    capsys.readouterr()
    bench(db, "compare", "--run-id", "2", "--baseline-run-id", "1", "--json")
    report = json.loads(capsys.readouterr().out)
    assert report["commit"].startswith("bbbbbbbb")
    gemm512 = [r for r in report["results"] if r["benchmark_label"] == "blas-gemm N=512" and r["metric_name"] == "t_generic"]
    assert gemm512 and gemm512[0]["pct_change"] == pytest.approx(-50.0)


def test_delete_db(db):
    bench(db, "delete-db", "--yes")
    assert not db.exists()


def test_default_database_is_per_user_and_env_overrides_it(tmp_path, monkeypatch):
    monkeypatch.delenv(commands.DB_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    default = commands.default_db_path()
    assert default.name == "benchmarks.db" and default.parent.name == "einsums"
    if sys.platform not in ("win32", "darwin"):
        assert default == tmp_path / "xdg" / "einsums" / "benchmarks.db"
    monkeypatch.setenv(commands.DB_ENV, str(tmp_path / "ci.db"))
    assert commands.default_db_path() == tmp_path / "ci.db"


def test_env_database_is_used_and_created(tmp_path, monkeypatch, capsys):
    target = tmp_path / "new" / "dir" / "bench.db"
    monkeypatch.setenv(commands.DB_ENV, str(target))
    main(["bench", "list-runs"])
    assert target.exists() and "No runs found" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_source_dir_defaults_to_the_checkout(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    assert commands.project_root() == tmp_path.resolve()


# ── upgrading existing databases ──────────────────────────────────────────────


def _database_at(path, version):
    """A database built by migrations 1..version, as an older release left it.

    version 0 is a database from before versioning: migration 1's tables without the
    columns it adds retroactively (cpu_freq_mhz, power_source), and no schema_version.
    """
    conn = sqlite3.connect(path)
    if version == 0:
        sql = models._load_migration_sql(1)
        conn.executescript(sql[: sql.index("ALTER TABLE")])
    for v in range(1, version + 1):
        conn.executescript(models._load_migration_sql(v))
    if version > 0:
        models._set_schema_version(conn, version)
    conn.execute("INSERT INTO runs (git_commit, git_branch, timestamp) VALUES ('old', 'main', '2026-01-01')")
    conn.execute(
        "INSERT INTO results (run_id, test_binary, benchmark_label, metric_name, value_us) "
        "VALUES (1, 'BenchmarkBLAS', 'blas-gemm N=64', 't_generic', 12.5)"
    )
    conn.commit()
    conn.close()


@pytest.mark.parametrize("version", range(0, models.SCHEMA_VERSION))
def test_older_databases_upgrade_and_keep_their_rows(tmp_path, version, capsys):
    path = tmp_path / f"v{version}.db"
    _database_at(path, version)
    conn = models.init_db(path)
    assert models._get_schema_version(conn) == models.SCHEMA_VERSION
    assert "Migrating" in capsys.readouterr().out or version > 0
    run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    result_columns = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    assert {"cpu_freq_mhz", "power_source", "hptt_method"} <= run_columns and {"annotations", "source"} <= result_columns
    # Migration 004 marks every older row as read from stdout.
    assert [tuple(r) for r in conn.execute("SELECT source FROM results")] == [("stdout",)]
    assert [tuple(r) for r in conn.execute("SELECT git_commit FROM runs")] == [("old",)]
    assert [tuple(r) for r in conn.execute("SELECT value_us FROM results")] == [(12.5,)]
    # The upgraded database takes new runs, with the columns the later migrations added.
    run_id = models.create_run(conn, git_commit="new", hptt_method="measure", cpu_freq_mhz=3500)
    assert conn.execute("SELECT hptt_method FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0] == "measure"
    conn.close()
    # Opening it again is a no-op.
    assert models._get_schema_version(models.init_db(path)) == models.SCHEMA_VERSION


def test_every_migration_file_is_counted():
    files = sorted(models._MIGRATIONS_DIR.glob("*.sql"))
    assert [int(f.name[:3]) for f in files] == list(range(1, models.SCHEMA_VERSION + 1))


def test_default_jobs_honours_cmake_and_caps_the_rest(monkeypatch):
    from einsums.cli.bench import runner

    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "5")
    assert runner.default_jobs() == 5
    monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL")
    assert 1 <= runner.default_jobs() <= 16


def test_a_sizeless_result_has_the_servers_label():
    # publish_benchmark_result without a size prints "N=0", but the server's event label
    # omits it; the two collection paths must agree.
    [r] = parser.parse_output("[zone-cost empty loop N=0] Time: 0.00 us  min: 0.00  max: 0.00  stddev: 0.00  "
                              "cv: 0.0%  warmup: 0.00 us (0.0x)  metric: t_op")  # fmt: skip
    assert (r.label, r.n, r.metric) == ("zone-cost empty loop", None, "t_op")


def test_discovery_reads_suites_and_timeouts_from_ctest(tmp_path, monkeypatch):
    from einsums.cli.bench import runner

    for name in ("BenchmarkTCB_test", "BenchmarkMemoryPool_test"):
        (tmp_path / name).write_text("")

    def entry(name, labels, timeout=None):
        props = [{"name": "LABELS", "value": labels}] + ([{"name": "TIMEOUT", "value": timeout}] if timeout else [])
        return {"name": f"Tests.Performance.Modules.X.{name}", "command": [str(tmp_path / f"{name}_test")], "properties": props}

    listing = {"tests": [
        entry("BenchmarkTCB", ["PERFORMANCE_ONLY", "SUITE_CONTRACTION"], 3600.0),
        entry("BenchmarkMemoryPool", ["PERFORMANCE_ONLY", "SUITE_INFRASTRUCTURE"]),
        entry("BenchmarkGone", ["PERFORMANCE_ONLY", "SUITE_CONTRACTION"]),  # its binary no longer exists
    ]}  # fmt: skip

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["ctest", "--test-dir"]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(listing), stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    every = runner._discover_perf_tests(tmp_path)
    assert [(t.name, t.suite, t.timeout) for t in every] == [
        ("BenchmarkMemoryPool", "infrastructure", runner.DEFAULT_TIMEOUT),
        ("BenchmarkTCB", "contraction", 3600.0),
    ]
    assert [t.name for t in runner._discover_perf_tests(tmp_path, "contraction")] == ["BenchmarkTCB"]
