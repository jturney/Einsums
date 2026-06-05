# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Build, run, and store benchmark results."""

from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
import threading
import time as _time
from pathlib import Path

from devtools.benchmarks import models, parser


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


def _discover_perf_tests(build_dir: Path) -> list[tuple[str, Path]]:
    """Discover performance test binaries by scanning the build tree.

    Performance tests are DISABLED in ctest (so they don't run during
    default `ctest`), so we can't rely on `ctest -N -L PERFORMANCE_ONLY`.
    Instead, scan for test binaries under performance test directories.
    """
    binaries: list[tuple[str, Path]] = []
    seen: set[str] = set()

    # Scan for *_test executables under paths containing "performance"
    for candidate in build_dir.rglob("*_test"):
        if not candidate.is_file():
            continue
        # Skip files with extensions (e.g. .cpp, .o) — want bare executables
        if candidate.suffix:
            continue
        # Only include tests from performance directories
        if "performance" not in str(candidate).lower():
            continue
        # Check it's actually executable
        if not _is_executable(candidate):
            continue

        name = candidate.stem  # e.g. "BenchmarkBLAS_test" -> "BenchmarkBLAS_test"
        test_name = name.removesuffix("_test")  # "BenchmarkBLAS"
        if test_name not in seen:
            seen.add(test_name)
            binaries.append((test_name, candidate))

    return sorted(binaries, key=lambda x: x[0])


class ProfilerCollector:
    """Connect to the Einsums profiler server and collect benchmark_result events."""

    def __init__(self, port: int = 19216, max_retries: int = 10, retry_delay: float = 0.5):
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.results: list[dict] = []
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self):
        """Start collecting in a background thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        """Stop collecting and return all received benchmark_result events."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        return self.results

    def _collect_loop(self):
        """Background thread: connect to profiler port and read JSON lines."""
        # Retry connection — benchmark may take a moment to start the profiler server
        for attempt in range(self.max_retries):
            if self._stop.is_set():
                return
            try:
                self._sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                self._sock.settimeout(1.0)  # Non-blocking reads with timeout
                break
            except (ConnectionRefusedError, OSError, TimeoutError):
                _time.sleep(self.retry_delay)
        else:
            return  # Couldn't connect — fall back to stdout parsing

        buf = ""
        while not self._stop.is_set():
            try:
                data = self._sock.recv(65536)
                if not data:
                    break  # Server closed connection
                buf += data.decode("utf-8", errors="replace")

                # Process complete JSON lines
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") == "benchmark_result":
                            self.results.append(event)
                    except json.JSONDecodeError:
                        pass  # Skip non-JSON lines (meta, snapshot, etc.)
            except socket.timeout:
                continue  # Normal — just retry
            except Exception:
                break  # Connection error


def _is_executable(path: Path) -> bool:
    """Check if a file is executable."""
    import os
    import stat
    try:
        mode = os.stat(path).st_mode
        return bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except OSError:
        return False


def run_benchmarks(
    build_dir: Path,
    source_dir: Path,
    db_path: Path,
    *,
    targets: list[str] | None = None,
    build: bool = True,
    hptt_method: str = "estimate",
    notes: str = "",
) -> int:
    """Run all performance benchmarks and store results. Returns run_id."""
    conn = models.init_db(db_path)

    # Optionally build (with timing)
    build_time_s = None
    if build:
        print("Building performance tests...")
        import time as _time
        t0 = _time.monotonic()
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "Tests.Performance"],
            check=False,  # Some targets may not exist
        )
        build_time_s = _time.monotonic() - t0
        print(f"Build completed in {build_time_s:.1f}s")

    # Gather metadata
    git = _git_info(source_dir)
    sys_info = _system_info(build_dir)

    # Reliability warnings
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

    # Discover and run benchmarks
    tests = _discover_perf_tests(build_dir)
    if not tests:
        print("No performance tests found. Ensure they are built.")
        return run_id

    if targets:
        tests = [(name, path) for name, path in tests if name in targets]

    for test_name, binary in tests:
        print(f"Running {test_name}...")

        # Track binary size
        try:
            binary_size = binary.stat().st_size
            conn.execute(
                """INSERT INTO results
                   (run_id, test_binary, benchmark_label, metric_name, value_us, unit)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, test_name, f"{test_name} binary", "binary_size_kb",
                 binary_size / 1024.0, "KB"),
            )
            conn.commit()
        except Exception:
            pass

        # Start profiler collector to receive structured benchmark_result events
        collector = ProfilerCollector()
        collector.start()

        try:
            cmd = [str(binary), "--einsums:debug:no-attach-debugger"]
            if hptt_method != "estimate":
                cmd.extend(["--einsums:hptt:selection-method", hptt_method])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            output = result.stdout  # Only parse stdout, stderr has logging/progress
            if result.returncode != 0:
                sig = -result.returncode if result.returncode < 0 else None
                if sig:
                    import signal
                    sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"
                    print(f"  CRASHED: {test_name} killed by {sig_name} (exit code {result.returncode})")
                else:
                    print(f"  FAILED: {test_name} exited with code {result.returncode}")
                if result.stderr:
                    stderr_lines = result.stderr.strip().splitlines()
                    for line in stderr_lines[-10:]:
                        print(f"    stderr: {line}")
        except subprocess.TimeoutExpired:
            collector.stop()
            print(f"  TIMEOUT: {test_name} exceeded 10 minutes, skipping.")
            continue
        except Exception as e:
            collector.stop()
            print(f"  ERROR running {test_name}: {e}")
            continue

        # Collect structured results from profiler
        profiler_results = collector.stop()

        # Store results: prefer profiler results (structured JSON with annotations),
        # fall back to stdout regex parsing for benchmarks that haven't been updated.
        if profiler_results:
            _store_profiler_results(conn, run_id, test_name, profiler_results)
            print(f"  Stored {len(profiler_results)} results via profiler.")
        else:
            # Fall back to stdout parsing
            benchmarks, breakdowns, cache_hits = parser.parse_output(output)
            if benchmarks or breakdowns or cache_hits:
                models.store_results(
                    conn, run_id, test_name,
                    benchmarks, breakdowns, cache_hits,
                )
                print(f"  Stored {len(benchmarks)} benchmarks, "
                      f"{len(breakdowns)} breakdowns, "
                      f"{len(cache_hits)} cache hits.")
            else:
                print(f"  No parseable results from {test_name}.")

    # Store build time
    if build_time_s is not None:
        conn.execute(
            """INSERT INTO results
               (run_id, test_binary, benchmark_label, metric_name, value_us, unit)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, "build", "build", "build_time_s", build_time_s, "s"),
        )
        conn.commit()

    print(f"\nRun {run_id} complete (commit {git['git_commit'][:8]}"
          f" on {git['git_branch'] or 'detached HEAD'}).")
    conn.close()
    return run_id


def _store_profiler_results(conn, run_id: int, test_name: str, results: list[dict]):
    """Store benchmark_result events from the profiler into the database."""
    for event in results:
        label = event.get("label", "unknown")
        metric = event.get("metric", "t_unknown")
        value_us = event.get("value_us", 0.0)
        annotations = event.get("annotations", {})

        # Build extra_json from timing stats
        extra = {
            "min_us": event.get("min_us"),
            "max_us": event.get("max_us"),
            "stddev_us": event.get("stddev_us"),
            "warmup_us": event.get("warmup_us"),
            "reps": event.get("reps"),
        }

        # Extract known annotation fields
        algorithm = annotations.get("algorithm", "")
        cv_pct = None
        if value_us > 0 and extra.get("stddev_us"):
            cv_pct = extra["stddev_us"] / value_us * 100.0
        extra["cv_pct"] = cv_pct

        # Compute GFLOP/s from annotation if present
        gflops_str = annotations.get("gflops", "")
        if gflops_str:
            try:
                extra["gflops"] = round(float(gflops_str), 3)
            except (ValueError, TypeError):
                pass

        # Warmup ratio
        if value_us > 0 and extra.get("warmup_us"):
            extra["warmup_ratio"] = round(extra["warmup_us"] / value_us, 2)

        # Store annotations as separate JSON
        annotations_json = json.dumps(annotations) if annotations else None

        conn.execute(
            """INSERT INTO results
               (run_id, test_binary, benchmark_label, metric_name, value_us, unit,
                algorithm, extra_json, annotations)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, test_name, label, metric, value_us, "us",
             algorithm or None,
             json.dumps({k: v for k, v in extra.items() if v is not None}),
             annotations_json),
        )
    conn.commit()
