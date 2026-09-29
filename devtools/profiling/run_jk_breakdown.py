#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Measure the J/K breakdown: one Coulomb and one exchange build per strategy.

Runs profile_jk_breakdown (every C, BLAS and Einsums strategy) and
profile_fort_loop (the Fortran loops) back to back, --reps times, and writes
the per-cell median over the repetitions as CSV. Each driver reports the
median of its own --trials calls, so a cell is a median of medians. A third
pass reruns the Einsums driver on one core (OMP_NUM_THREADS=1 and
VECLIB_MAXIMUM_THREADS=1, since Accelerate ignores the OpenMP setting) and keeps
its compile-time einsum row as einsum_1thread.

With --v1-bin (profile_jk_v1, built from devtools/profiling/v1 against an
Einsums 1.x install) each repetition also runs the 1.x einsum rows, threaded and
on one core, as v1_einsum, v1_einsum_permute and v1_einsum_1thread. Their
checksums must match the 2.x driver's: both builds compute the same J and K.
Interleaving the two drivers spreads machine noise over every row instead of
landing it on whichever ran last.

profile_jk_breakdown exits nonzero if any strategy's J or K differs from the
serial loops; this script additionally checks the Fortran checksums against
the ones profile_jk_breakdown prints, so every row of the CSV is a correct
answer.

Build both drivers first (EINSUMS_WITH_PROFILING_DEVTOOLS=ON, and a Fortran
compiler for the Fortran row), then from the repo root:

    python devtools/profiling/run_jk_breakdown.py --bin-dir build/bin \
        -n 100 --trials 20 --reps 3 --out jk_breakdown.csv
"""

import argparse
import datetime
import os
import pathlib
import platform
import re
import statistics
import subprocess
import sys


def run(cmd, env=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.exit(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


def parse(text):
    """Return ({strategy: (j_ms, k_ms)}, checksum (J, K) or None)."""
    rows, checksum = {}, None
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 3 or parts[0] == "strategy":
            continue
        if parts[0] == "checksum":
            checksum = (float(parts[1]), float(parts[2]))
        else:
            rows[parts[0]] = (float(parts[1]), float(parts[2]))
    return rows, checksum


def version():
    """The 2.x build under test: its declared version and the commit, marked dirty when the tree has edits."""
    root = pathlib.Path(__file__).resolve().parents[2]
    declared = "2"
    try:
        text = (root / "CMakeLists.txt").read_text()
        parts = [re.search(rf"set\(EINSUMS_VERSION_{k} \"?([^)\"]*)\"?\)", text) for k in ("MAJOR", "MINOR", "PATCH", "TAG")]
        declared = ".".join(m.group(1) for m in parts[:3]) + (parts[3].group(1) if parts[3] else "")
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True).stdout.strip()
        return f"{declared} ({sha}{'+dirty' if dirty else ''})"
    except (OSError, AttributeError):
        return declared


def cpu_name():
    """The processor's model name, on macOS or Linux; the platform's machine string otherwise."""
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin-dir", type=pathlib.Path, default=pathlib.Path("build/bin"))
    ap.add_argument("-n", type=int, default=100, help="orbitals")
    ap.add_argument("--trials", type=int, default=20, help="timed calls per driver run")
    ap.add_argument("--reps", type=int, default=3, help="interleaved driver runs")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--v1-bin", type=pathlib.Path, default=None, help="profile_jk_v1 built against Einsums 1.x")
    ap.add_argument("--v1-version", default="1.1.5", help="the 1.x release profile_jk_v1 was built against")
    args = ap.parse_args()

    jk = args.bin_dir / "profile_jk_breakdown"
    fort = args.bin_dir / "profile_fort_loop"
    samples = {}
    for rep in range(args.reps):
        p = run([str(jk), "-n", str(args.n), "-t", str(args.trials), "-c", "--einsums:debug:no-attach-debugger"])
        rows, _ = parse(p.stdout)
        _, reference = parse(p.stderr)
        if fort.exists():
            f_rows, f_sum = parse(run([str(fort), "-n", str(args.n), "-t", str(args.trials), "-c"]).stdout)
            for got, want, which in zip(f_sum, reference, "JK"):
                if abs(got - want) > 1e-10 * abs(want):
                    sys.exit(f"fortran {which} checksum {got!r} differs from the reference {want!r}")
            rows.update(f_rows)
        # The compile-time einsum again on one core, for a like-for-like row
        # against the serial hand-written loops. OpenMP is held to one thread,
        # and so is every BLAS that keeps its own count: Accelerate ignores
        # OMP_NUM_THREADS and reads VECLIB_MAXIMUM_THREADS, and OpenBLAS and MKL
        # have theirs. Each variable is inert where its library is absent.
        one = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        rows_1, _ = parse(run([str(jk), "-n", str(args.n), "-t", str(args.trials), "-c", "--einsums:debug:no-attach-debugger"],
                              env=one).stdout)
        rows["einsum_1thread"] = rows_1["einsum"]
        if args.v1_bin is not None:
            v1_cmd = [str(args.v1_bin), "-n", str(args.n), "-t", str(args.trials), "-c"]
            p1 = run(v1_cmd)
            v1_rows, _ = parse(p1.stdout)
            _, v1_sum = parse(p1.stderr)
            for got, want, which in zip(v1_sum, reference, "JK"):
                if abs(got - want) > 1e-10 * abs(want):
                    sys.exit(f"v1 {which} checksum {got!r} differs from the reference {want!r}")
            rows.update(v1_rows)
            v1_one, _ = parse(run(v1_cmd, env=one).stdout)
            rows["v1_einsum_1thread"] = v1_one["v1_einsum"]
        for name, value in rows.items():
            samples.setdefault(name, []).append(value)
        print(f"rep {rep + 1}/{args.reps}: " + "  ".join(f"{k} {j + kk:.1f}" for k, (j, kk) in rows.items()))

    threads = os.environ.get("OMP_NUM_THREADS") or str(os.cpu_count())
    lines = [
        f"# J and K builds at n = {args.n} orbitals, one row per strategy.",
        f"# Measured {datetime.date.today().isoformat()} on {cpu_name()} ({platform.system()} {platform.machine()}),",
        f"# {threads} threads (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}), by devtools/profiling/run_jk_breakdown.py.",
        f"# Einsums {version()} for the unprefixed rows"
        + (f"; Einsums {args.v1_version} for the v1_ rows." if args.v1_bin is not None else "."),
        "# Serial rows: c_loops, fortran_loops, the K half of blas_loop and blas_permute, and the *_1thread rows",
        "# (every OpenMP and BLAS thread count held to 1). Every other row is threaded, and each BLAS gemv",
        "# threads as the BLAS library schedules it.",
        f"# Wall-clock ms. Each driver run reports the median of {args.trials} calls after one warm-up;",
        f"# each cell is the median over {args.reps} interleaved runs, and *_lo/*_hi give their range.",
        "# Every strategy's J and K were checked against the serial loops before being recorded.",
        "# graph_fused captures J and K in one graph; its time is the fused total, stored under j_ms.",
        "strategy,j_ms,k_ms,j_lo,j_hi,k_lo,k_hi",
    ]
    for name, vals in samples.items():
        js, ks = [v[0] for v in vals], [v[1] for v in vals]
        lines.append(f"{name},{statistics.median(js):.3f},{statistics.median(ks):.3f},"
                     f"{min(js):.3f},{max(js):.3f},{min(ks):.3f},{max(ks):.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
