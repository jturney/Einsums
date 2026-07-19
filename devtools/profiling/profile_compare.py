# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
"""Generate the "why Einsums" comparison figure from the profile_* drivers.

Runs the compiled drivers in this directory over a range of orbital counts on
the shared Fock-build workload (J/K/G from the two-electron integrals) and
plots the best-effort variant of each approach:

    C loops (OpenMP)       profile_for_loop  - parallel vectorized loops
    permute + BLAS         profile_blas      - sorted TEI, gemv per matrix
    Einsums einsum         profile_einsums   - einsum notation, automatic dispatch

Build the drivers first (they link against an Einsums build or install):

    cmake -S devtools/profiling -B build-profiling -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/build/lib/cmake
    cmake --build build-profiling -j

Then, from the repo root:

    python devtools/profiling/profile_compare.py --bin-dir build-profiling \
        --sizes 20 40 60 80 100 --trials 10 \
        --plot docs/sphinx/_static/index-images/why_einsums.png
"""

import argparse
import pathlib
import platform
import subprocess


def run_driver(exe, norbs, trials):
    """Run one driver at one size in CSV mode; return (baseline_ms, best_ms)."""
    out = subprocess.run([str(exe), "-n", str(norbs), "-e", str(norbs), "-t", str(trials), "-c"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 5 and parts[0] == str(norbs):
            return float(parts[1]) * 1e3, float(parts[3]) * 1e3
    raise RuntimeError(f"no CSV line for n={norbs} in output of {exe}:\n{out}")


def make_plot(sizes, series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    styles = {"C loops (OpenMP)": "#888888", "permute + BLAS": "#d62728", "Einsums einsum": "#1f77b4"}
    for label, times in series.items():
        ax.plot(sizes, times, "o-", color=styles[label], label=label)
    ax.set_xlabel("orbitals")
    ax.set_ylabel("Fock-build time (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")
    print(f"generated on: {platform.platform()}, {platform.processor() or platform.machine()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin-dir", type=pathlib.Path, required=True,
                    help="build directory containing the profile_* executables")
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--plot", type=str, default=None, help="write a PNG to this path")
    args = ap.parse_args()

    drivers = {
        "C loops (OpenMP)": ("profile_for_loop", 1),   # column 1 = best variant (omp)
        "permute + BLAS": ("profile_blas", 1),         # column 1 = sorted TEI + gemv
        "Einsums einsum": ("profile_einsums", 1),      # column 1 = sorted einsum path
    }
    series = {label: [] for label in drivers}
    for n in args.sizes:
        row = [f"n={n:4d}"]
        for label, (name, best_col) in drivers.items():
            exe = args.bin_dir / name
            if not exe.exists():
                raise SystemExit(f"driver not found: {exe} (build devtools/profiling first)")
            baseline, best = run_driver(exe, n, args.trials)
            value = best if best_col == 1 else baseline
            series[label].append(value)
            row.append(f"{label} {value:10.3f} ms")
        print("  ".join(row))

    if args.plot:
        make_plot(args.sizes, series, args.plot)


if __name__ == "__main__":
    main()
