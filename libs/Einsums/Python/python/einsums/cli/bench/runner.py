# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Build, run, and store benchmark results."""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import signal
import stat
import subprocess
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path

from . import models, parser
from .parser import Result


def _git_info(source_dir: Path) -> dict:
    """Gather git commit, branch, and dirty status."""

    def _run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=str(source_dir), text=True
        ).strip()

    commit = _run("rev-parse", "HEAD")
    branch = _run("branch", "--show-current") or None
    dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=str(source_dir), capture_output=True
    ).returncode != 0

    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def _system_info(build_dir: Path) -> dict:
    """Gather hostname, CPU model, compiler, build type, and BLAS vendor."""
    info: dict = {
        "hostname": socket.gethostname(),
        "cpu_model": platform.processor() or platform.machine(),
    }

    # Parse CMakeCache.txt for build config
    cache_file = build_dir / "CMakeCache.txt"
    if cache_file.exists():
        cache = cache_file.read_text()

        m = re.search(r"CMAKE_BUILD_TYPE:STRING=(.+)", cache)
        if m:
            info["build_type"] = m.group(1).strip()

        m = re.search(r"CMAKE_CXX_COMPILER:FILEPATH=(.+)", cache)
        if m:
            compiler_path = m.group(1).strip()
            # Try to get version
            try:
                ver = subprocess.check_output(
                    [compiler_path, "--version"], text=True, stderr=subprocess.STDOUT
                ).splitlines()[0]
                info["compiler"] = ver
            except Exception:
                info["compiler"] = compiler_path

        # Try to detect BLAS vendor
        for key in ["BLA_VENDOR", "BLAS_LIBRARIES", "EINSUMS_BLAS_VENDOR"]:
            m = re.search(rf"{key}:\w+=(.+)", cache)
            if m:
                info["blas_vendor"] = m.group(1).strip()
                break

    # CPU frequency (macOS and Linux)
    try:
        if platform.system() == "Darwin":
            # macOS: sysctl
            freq = subprocess.check_output(
                ["sysctl", "-n", "hw.cpufrequency_max"], text=True, stderr=subprocess.DEVNULL,
            ).strip()
            info["cpu_freq_mhz"] = int(freq) // 1_000_000
        else:
            # Linux: read from /proc/cpuinfo or /sys
            cpuinfo = Path("/proc/cpuinfo").read_text()
            m = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
            if m:
                info["cpu_freq_mhz"] = int(float(m.group(1)))
    except Exception:
        pass

    # Power source (macOS and Linux)
    try:
        if platform.system() == "Darwin":
            ps = subprocess.check_output(
                ["pmset", "-g", "ps"], text=True, stderr=subprocess.DEVNULL,
            )
            if "AC Power" in ps:
                info["power_source"] = "AC"
            elif "Battery" in ps:
                info["power_source"] = "Battery"
        else:
            # Linux: check /sys/class/power_supply
            for supply in Path("/sys/class/power_supply").iterdir():
                stype = (supply / "type").read_text().strip()
                if stype == "Mains":
                    online = (supply / "online").read_text().strip()
                    info["power_source"] = "AC" if online == "1" else "Battery"
                    break
    except Exception:
        pass

    # CPU frequency governor (Linux only)
    try:
        if platform.system() == "Linux":
            gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if gov_path.exists():
                info["cpu_governor"] = gov_path.read_text().strip()
    except Exception:
        pass

    return info


def _check_reliability_warnings(sys_info: dict) -> list[str]:
    """Check system state for conditions that affect benchmark reliability."""
    warnings = []

    power = sys_info.get("power_source")
    if power == "Battery":
        warnings.append("Running on BATTERY power — CPU may be throttled. Plug in for reliable results.")

    governor = sys_info.get("cpu_governor")
    if governor and governor != "performance":
        warnings.append(
            f"CPU governor is '{governor}' (not 'performance'). "
            f"Run: sudo cpupower frequency-set -g performance"
        )

    return warnings


SUITES = ("contraction", "infrastructure")
DEFAULT_TIMEOUT = 600.0  # seconds, for a test whose ctest entry sets none


@dataclass
class PerfTest:
    name: str  # "BenchmarkBLAS"
    path: Path
    suite: str = ""  # "contraction" or "infrastructure"; "" when only a scan found it
    timeout: float = DEFAULT_TIMEOUT


def _discover_perf_tests(build_dir: Path, suite: str = "all") -> list[PerfTest]:
    """The performance test binaries the build defines, optionally one suite of them.

    Asks ctest, which lists the PERFORMANCE_ONLY tests with their commands, labels (the
    SUITE_* label einsums_add_performance_test sets) and TIMEOUT, even though they are
    disabled for ordinary runs. Scanning the tree for ``*_test`` files instead also finds
    binaries whose targets no longer exist, which then fail against the current library (the
    SIMD module's benchmark lingered that way after it moved to Stripes). The scan is the
    fallback when ctest is unavailable; it knows no suites, so it serves only ``all``.
    """
    try:
        listing = subprocess.run(
            ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1", "-L", "PERFORMANCE_ONLY"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
        entries = json.loads(listing).get("tests", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        entries = None

    found: dict[str, PerfTest] = {}
    if entries is not None:
        for entry in entries:
            command = entry.get("command") or []
            if not command or not Path(command[0]).is_file():
                continue
            props = {p["name"]: p["value"] for p in entry.get("properties", [])}
            labels = props.get("LABELS", [])
            test_suite = next((label.removeprefix("SUITE_").lower() for label in labels if label.startswith("SUITE_")), "")
            timeout = float(props.get("TIMEOUT") or DEFAULT_TIMEOUT)
            name = entry["name"].rsplit(".", 1)[-1]
            found.setdefault(name, PerfTest(name, Path(command[0]), test_suite, timeout))
    else:
        for candidate in build_dir.rglob("*_test"):
            if candidate.is_file() and not candidate.suffix and "performance" in str(candidate).lower() and _is_executable(candidate):
                name = candidate.stem.removesuffix("_test")
                found.setdefault(name, PerfTest(name, candidate))
    tests = sorted(found.values(), key=lambda t: t.name)
    return tests if suite == "all" else [t for t in tests if t.suite == suite]


def default_jobs() -> int:
    """Build parallelism: ``$CMAKE_BUILD_PARALLEL_LEVEL`` if set, else half the cores, at most 16.

    Ninja's own default is every core, which on a shared machine starves everyone else
    and, with Einsums' heavy translation units, can exhaust memory.
    """
    env = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(16, (os.cpu_count() or 2) // 2))


def build_performance_tests(build_dir: Path, jobs: int, *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "Tests.Performance", "--parallel", str(jobs)],
        capture_output=capture,
        text=True,
        check=False,
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ProfilerCollector:
    """Collects ``benchmark_result`` events from one test binary's profiler server.

    The binary runs with --einsums:profile:wait-for-viewer, so it blocks until this has
    connected; this therefore keeps retrying until it connects or is stopped, rather than
    giving up and leaving the binary waiting.
    """

    def __init__(self, port: int, retry_delay: float = 0.05):
        self.port = port
        self.retry_delay = retry_delay
        self.events: list[dict] = []
        self.connected = False
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        """Wait for the server to close (the binary exited), then return the events."""
        if self._thread and self.connected:
            self._thread.join(timeout=5.0)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        return self.events

    def _collect_loop(self) -> None:
        while not self.connected:
            if self._stop.is_set():
                return
            try:
                self._sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                self._sock.settimeout(1.0)
                self.connected = True
            except OSError:
                _time.sleep(self.retry_delay)
        buf = b""
        while not self._stop.is_set():
            try:
                data = self._sock.recv(1 << 16)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buf += data
            *lines, buf = buf.split(b"\n")
            for line in lines:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if event.get("type") == "benchmark_result":
                    self.events.append(event)


def _is_executable(path: Path) -> bool:
    """Check if a file is executable."""
    try:
        mode = os.stat(path).st_mode
        return bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except OSError:
        return False


@dataclass
class TestRun:
    results: list[Result]
    status: str  # "ok", "failed", "crashed", "timeout", "error"
    message: str = ""


def run_test_binary(binary: Path, *, hptt_method: str = "estimate", timeout: float = 600.0) -> TestRun:
    """Run one performance binary and collect its results.

    The binary gets its own profiler server on a free port, so a viewer attached to
    the default port is undisturbed and two runs cannot cross, and waits for the
    collector before starting, so a test that finishes in a fraction of a second still
    delivers its results. Results come from the server's structured events; the printed
    lines are read only if none arrived (a build without the profiler).
    """
    port = _free_port()
    cmd = [
        str(binary),
        "--einsums:debug:no-attach-debugger",
        "--einsums:profile:server",
        f"--einsums:profile:port={port}",
        "--einsums:profile:wait-for-viewer",
        "--einsums:profile:report=false",
    ]
    if hptt_method != "estimate":
        cmd += ["--einsums:hptt:selection-method", hptt_method]
    collector = ProfilerCollector(port)
    collector.start()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        collector.stop()
        return TestRun([], "timeout", f"exceeded {timeout:.0f} s")
    except OSError as exc:
        collector.stop()
        return TestRun([], "error", str(exc))
    events = collector.stop()
    results = [Result.from_event(e) for e in events] or parser.parse_output(proc.stdout)
    if proc.returncode == 0:
        return TestRun(results, "ok")
    if proc.returncode < 0:
        try:
            name = signal.Signals(-proc.returncode).name
        except ValueError:
            name = f"signal {-proc.returncode}"
        status, message = "crashed", f"killed by {name}"
    else:
        status, message = "failed", f"exit code {proc.returncode}"
    # Catch2 reports a failed assertion on stdout; stderr holds the logger and the server's banner.
    stdout_tail = [f"stdout: {line}" for line in (proc.stdout or "").strip().splitlines()[-12:]]
    stderr_tail = [
        f"stderr: {line}"
        for line in (proc.stderr or "").strip().splitlines()[-6:]
        if line.strip() and not line.startswith("***")  # the server's wait-for-viewer banner
    ]
    return TestRun(results, status, "\n".join([message, *stdout_tail, *stderr_tail]))


def run_benchmarks(
    build_dir: Path,
    source_dir: Path,
    db_path: Path,
    *,
    targets: list[str] | None = None,
    suite: str = "contraction",
    build: bool = True,
    jobs: int | None = None,
    hptt_method: str = "estimate",
    notes: str = "",
) -> int:
    """Run the performance benchmarks and store their results. Returns the run id."""
    conn = models.init_db(db_path)

    build_time_s = None
    if build:
        jobs = jobs or default_jobs()
        print(f"Building performance tests ({jobs} jobs)...")
        t0 = _time.monotonic()
        build_performance_tests(build_dir, jobs)
        build_time_s = _time.monotonic() - t0
        print(f"Build completed in {build_time_s:.1f}s")

    git = _git_info(source_dir)
    sys_info = _system_info(build_dir)
    for warning in _check_reliability_warnings(sys_info):
        print(f"  WARNING: {warning}")

    run_id = models.create_run(
        conn,
        git_commit=git["git_commit"],
        git_branch=git["git_branch"],
        git_dirty=git["git_dirty"],
        hostname=sys_info.get("hostname"),
        build_type=sys_info.get("build_type"),
        compiler=sys_info.get("compiler"),
        blas_vendor=sys_info.get("blas_vendor"),
        cpu_model=sys_info.get("cpu_model"),
        cpu_freq_mhz=sys_info.get("cpu_freq_mhz"),
        power_source=sys_info.get("power_source"),
        hptt_method=hptt_method,
        notes=notes,
    )

    # Named targets are run whatever their suite; otherwise the suite chooses.
    tests = _discover_perf_tests(build_dir, "all" if targets else suite)
    if targets:
        unknown = set(targets) - {t.name for t in tests}
        if unknown:
            print(f"  WARNING: no performance binary named {', '.join(sorted(unknown))}")
        tests = [t for t in tests if t.name in targets]
    if not tests:
        print("No performance tests found. Ensure they are built (EINSUMS_WITH_TESTS_BENCHMARKS=ON).")
    elif not targets:
        print(f"Running the {suite} suite: {len(tests)} test binaries.")

    for test in tests:
        test_name, binary = test.name, test.path
        print(f"Running {test_name}...")
        conn.execute(
            """INSERT INTO results (run_id, test_binary, benchmark_label, metric_name, value_us, unit, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, test_name, f"{test_name} binary", "binary_size_kb", binary.stat().st_size / 1024.0, "KB", "build"),
        )
        conn.commit()

        run = run_test_binary(binary, hptt_method=hptt_method, timeout=test.timeout)
        if run.status != "ok":
            print(f"  {run.status.upper()}: {test_name}: " + run.message.replace("\n", "\n    "))
        if run.results:
            models.store_results(conn, run_id, test_name, run.results)
            via = "the profiler server" if run.results[0].source == "profiler" else "its printed output"
            print(f"  Stored {len(run.results)} results from {via}.")
        elif run.status == "ok":
            print(f"  No results reported by {test_name}.")

    if build_time_s is not None:
        conn.execute(
            """INSERT INTO results (run_id, test_binary, benchmark_label, metric_name, value_us, unit, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, "build", "build", "build_time_s", build_time_s, "s", "build"),
        )
        conn.commit()

    print(f"\nRun {run_id} complete (commit {git['git_commit'][:8]} on {git['git_branch'] or 'detached HEAD'}).")
    conn.close()
    return run_id
