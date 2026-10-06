#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Measure the J/K breakdown over a range of orbital counts.

The sweep form of run_jk_breakdown.py: the same drivers (profile_jk_breakdown
for every C, BLAS and Einsums strategy, profile_fort_loop for the Fortran
loops), the same checksum check of the Fortran row against the serial C loops,
and the same one-core rerun of the compile-time einsum as einsum_1thread, at
each size in --sizes. With --v1-bin (profile_jk_v1, built from
devtools/profiling/v1 against an Einsums 1.x install) each pass also runs the
1.x einsum rows, threaded and on one core, checked against the same checksums.

Repetitions are the outer loop and sizes the inner one, so machine noise is
spread over every size instead of landing on whichever ran last. Each cell is
the median over --reps runs of the median of --trials calls.

The output is one row per (size, strategy) with the J+K total, since
graph_fused only reports the fused total. Speedups are left to the plotting
side: every strategy at a size can be divided into that size's c_loops row.

Build both drivers first (EINSUMS_WITH_PROFILING_DEVTOOLS=ON, and a Fortran
compiler for the Fortran rows), then from the repo root:

    python devtools/profiling/run_jk_sweep.py --bin-dir build/bin \
        --sizes 40 60 80 100 120 140 --trials 10 --reps 3 --out jk_sweep.csv

The TEI is n^4 doubles and the permute strategies copy it: peak memory is
about 2.4 GB at n = 100 and 9.4 GB at n = 140.
"""

import argparse
import datetime
import os
import pathlib
import platform
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_jk_breakdown import cpu_name, parse, run, version  # noqa: E402


def measure(jk, fort, v1, n, trials):
    """One interleaved pass at one size: {strategy: total_ms}, every row checked."""
    flags = ["-n", str(n), "-t", str(trials), "-c"]
    p = run([str(jk), *flags, "--einsums:debug:no-attach-debugger"])
    rows, _ = parse(p.stdout)
    _, reference = parse(p.stderr)
    if fort.exists():
        f_rows, f_sum = parse(run([str(fort), *flags]).stdout)
        for got, want, which in zip(f_sum, reference, "JK"):
            if abs(got - want) > 1e-10 * abs(want):
                sys.exit(f"n={n}: fortran {which} checksum {got!r} differs from the reference {want!r}")
        rows.update(f_rows)
    # As in run_jk_breakdown.py: every OpenMP and BLAS thread count held to one.
    one = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows_1, _ = parse(run([str(jk), *flags, "--einsums:debug:no-attach-debugger"], env=one).stdout)
    rows["einsum_1thread"] = rows_1["einsum"]
    if v1 is not None:
        p1 = run([str(v1), *flags])
        v1_rows, _ = parse(p1.stdout)
        _, v1_sum = parse(p1.stderr)
        for got, want, which in zip(v1_sum, reference, "JK"):
            if abs(got - want) > 1e-10 * abs(want):
                sys.exit(f"n={n}: v1 {which} checksum {got!r} differs from the reference {want!r}")
        rows.update(v1_rows)
        v1_one, _ = parse(run([str(v1), *flags], env=one).stdout)
        rows["v1_einsum_1thread"] = v1_one["v1_einsum"]
    return {name: j + k for name, (j, k) in rows.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin-dir", type=pathlib.Path, default=pathlib.Path("build/bin"))
    ap.add_argument("--sizes", nargs="+", type=int, default=[40, 60, 80, 100, 120, 140], help="orbital counts")
    ap.add_argument("--trials", type=int, default=10, help="timed calls per driver run")
    ap.add_argument("--reps", type=int, default=3, help="interleaved passes over every size")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--v1-bin", type=pathlib.Path, default=None, help="profile_jk_v1 built against Einsums 1.x")
    ap.add_argument("--v1-version", default="1.1.5", help="the 1.x release profile_jk_v1 was built against")
    args = ap.parse_args()

    jk = args.bin_dir / "profile_jk_breakdown"
    fort = args.bin_dir / "profile_fort_loop"
    if not jk.exists():
        sys.exit(f"no {jk}; configure with EINSUMS_WITH_PROFILING_DEVTOOLS=ON and build it")
    if not fort.exists():
        print(f"note: no {fort}, so the sweep has no Fortran rows", file=sys.stderr)

    samples = {}                                 # (n, strategy) -> [total_ms per rep]
    for rep in range(args.reps):
        for n in args.sizes:
            totals = measure(jk, fort, args.v1_bin, n, args.trials)
            for name, ms in totals.items():
                samples.setdefault((n, name), []).append(ms)
            print(f"rep {rep + 1}/{args.reps} n={n:4d}: c_loops {totals['c_loops']:.1f} ms, "
                  f"graph_fused {totals['graph_fused']:.1f} ms", flush=True)

    threads = os.environ.get("OMP_NUM_THREADS") or str(os.cpu_count())
    lines = [
        f"# J + K builds at n = {', '.join(map(str, args.sizes))} orbitals, one row per size and strategy.",
        f"# Measured {datetime.date.today().isoformat()} on {cpu_name()} ({platform.system()} {platform.machine()}),",
        f"# {threads} threads (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}), by devtools/profiling/run_jk_sweep.py.",
        f"# Einsums {version()} for the unprefixed rows"
        + (f"; Einsums {args.v1_version} for the v1_ rows." if args.v1_bin is not None else "."),
        "# Serial rows: c_loops, fortran_loops, the K half of blas_loop and blas_permute, and the *_1thread rows",
        "# (every OpenMP and BLAS thread count held to 1). Every other row is threaded.",
        f"# Wall-clock ms for J + K. Each driver run reports the median of {args.trials} calls after one warm-up;",
        f"# each cell is the median over {args.reps} interleaved passes, and lo/hi give their range.",
        "# Every strategy's J and K were checked against the serial loops before being recorded.",
        "n,strategy,total_ms,lo,hi",
    ]
    for (n, name), vals in sorted(samples.items(), key=lambda kv: kv[0][0]):
        lines.append(f"{n},{name},{statistics.median(vals):.3f},{min(vals):.3f},{max(vals):.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
