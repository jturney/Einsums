#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Regenerate docs/sphinx/_static/index-images/why_einsums.png.

Runs the profile_strategies driver over a range of orbital counts and plots
the Fock build (G = 2J - K from the TEI and the density) through four
execution strategies:

    hand-fused loops        strong hand code: J and K fused in one OpenMP
                            nest, cache-ordered, unit-stride inner loop
    einsum (eager)          the same math as two einsum calls, automatic
                            dispatch per contraction
    LCCF graph              captured graph + algebraic folding (one
                            contraction over a materialized combination)
    stream-fused graph      captured graph + StreamContractionFusion: one
                            storage-order pass over the TEI feeds both
                            accumulators

Build the driver first (see profile_compare.py for the cmake invocation),
then from the repo root:

    python devtools/profiling/plot_why_einsums.py --bin-dir build-profiling \
        --sizes 32 48 64 80 100 --trials 3 \
        --plot docs/sphinx/_static/index-images/why_einsums.png
"""

import argparse
import pathlib
import platform
import subprocess

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"

# Validated categorical trio + a recessive gray for the hand-code baseline
# (consistent with the packed_gemm_dispatch figure's use of gray).
SERIES = [
    ("hand-fused loops (OpenMP)", "#8a8984"),
    ("einsum (eager)", "#2a78d6"),
    ("LCCF graph (fold)", "#4a3aa7"),
    ("stream-fused graph (now)", "#1baf7a"),
]


def run_driver(exe, n, trials):
    out = subprocess.run([str(exe), "-n", str(n), "-t", str(trials), "-c", "--einsums:debug:no-attach-debugger"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 5 and parts[0] == str(n):
            return [float(p) for p in parts[1:]]
    raise RuntimeError(f"no CSV line for n={n}:\n{out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="build-profiling")
    ap.add_argument("--sizes", nargs="+", type=int, default=[32, 48, 64, 80, 100])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--plot", default="docs/sphinx/_static/index-images/why_einsums.png")
    args = ap.parse_args()

    exe = pathlib.Path(args.bin_dir) / "profile_strategies"
    data = {name: [] for name, _ in SERIES}
    for n in args.sizes:
        row = run_driver(exe, n, args.trials)
        for (name, _), v in zip(SERIES, row):
            data[name].append(v)
        print(f"n={n:4d}  " + "  ".join(f"{name.split(' (')[0]} {v:9.2f} ms" for (name, _), v in zip(SERIES, row)))

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for name, color in SERIES:
        emphasize = name.startswith("stream")
        ax.plot(args.sizes, data[name], "-o", color=color, linewidth=2.4 if emphasize else 2.0,
                markersize=7 if emphasize else 6, label=name, zorder=4 if emphasize else 3)
        # Slight vertical splay keeps end labels of nearby lines apart.
        ax.annotate(f" {data[name][-1]:,.0f}", xy=(args.sizes[-1], data[name][-1]), xytext=(6, -5 if emphasize else 5),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK if emphasize else INK_2, fontweight="bold" if emphasize else "normal")

    ax.set_yscale("log")
    ax.set_xlabel("orbitals", fontsize=10, color=INK_2)
    ax.set_ylabel("Fock-build time, G = 2J − K (ms)", fontsize=10, color=INK_2)
    ax.set_title("One Fock build, four execution strategies — Apple M4", fontsize=12, color=INK, loc="left", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(args.sizes[0] - 2, args.sizes[-1] + 9)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)

    fig.tight_layout()
    out = pathlib.Path(args.plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"generated on: {platform.platform()}, {platform.machine()}")


if __name__ == "__main__":
    main()
