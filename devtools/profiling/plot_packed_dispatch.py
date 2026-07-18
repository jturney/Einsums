#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Regenerate docs/sphinx/_static/index-images/packed_gemm_dispatch.png.

The figure compares einsum()'s scrambled-contraction dispatch before and after
the PackedGemm engine work: Sort+GEMM (TTGT: permute both operands into
canonical order, one vendor GEMM, permute back) against the PackedGemm scatter
engines (SME FMOPA micro-tiles for real types, the 1m method on those same
real tiles for complex types).

Workload: the CCSD-like ring contraction C(abij) = A(aeim) B(ebmj) with
o = 16 and v = 64 (v = 48 for both complex types, so cross-dtype memory
numbers are NOT proportional to sizeof(T): complex<float> is 2*48^2*16^2
elements * 8 B = exactly 9 MiB), best of 7 runs.
The timings below were measured on an Apple M4 (macOS, arm64, Accelerate)
with the benchmark driver in the PR description; re-measure by timing
tensor_algebra::detail::einsum_do_sort_gemm directly against a plain
einsum() call, which now dispatches to PACKED_GEMM for these shapes.

Transient memory is analytic and exact: Sort+GEMM allocates full permuted
copies of A and B per call (2 * v^2 * o^2 * sizeof(T)); the packed engines
use cache-sized thread-local buffers that persist across calls, so their
per-call allocation is zero.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Validated 2-slot categorical palette (dataviz reference palette, light mode).
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"  # Sort+GEMM (before)
AQUA = "#1baf7a"  # PackedGemm (after)
GRAY = "#8a8984"  # generic loops (recessive baseline)
INK = "#0b0b0b"
INK_2 = "#52514e"

DTYPES = ["double", "float", "complex<double> (v=48)", "complex<float> (v=48)"]
GEN_MS = [262.5, 226.8, 125.5, 92.1]
SORT_MS = [13.04, 6.70, 21.24, 9.91]
PACK_MS = [5.65, 1.75, 13.46, 5.16]
ENGINE = ["f64 FMOPA", "f32 FMOPA", "1m · f64 FMOPA", "1m · f32 FMOPA"]

# Per-call transient allocation for the same contractions (MiB, analytic).
ELEMS = [64 * 64 * 16 * 16] * 2 + [48 * 48 * 16 * 16] * 2
SIZEOF = [8, 4, 16, 8]
SORT_MB = [2 * n * s / 2**20 for n, s in zip(ELEMS, SIZEOF)]
PACK_MB = [0.0] * 4


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


def progression_dots(ax):
    """Three engine generations per dtype on a log axis: generic loops ->
    Sort+GEMM -> PackedGemm. Gray marks the recessive baseline (matching the
    'plain loops' gray of the why-Einsums figure); position on a log scale
    carries the magnitude, so every point is direct-labeled."""
    y = range(len(DTYPES))
    for i, (g, s, p) in enumerate(zip(GEN_MS, SORT_MS, PACK_MS)):
        ax.plot([p, g], [i, i], color="#d9d8d4", linewidth=1.6, zorder=1)
        ax.plot([g], [i], "o", color=GRAY, markersize=9, zorder=3, label="generic loops" if i == 0 else None)
        ax.plot([s], [i], "o", color=BLUE, markersize=9, zorder=3, label="Sort+GEMM" if i == 0 else None)
        ax.plot([p], [i], "o", color=AQUA, markersize=10, zorder=4, label="PackedGemm" if i == 0 else None)
        ax.annotate(f"{g:.0f}", xy=(g, i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
        ax.annotate(f"{s:.1f}", xy=(s, i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
        ax.annotate(f"{p:.2f}", xy=(p, i), xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9, color=INK,
                    fontweight="bold")
        if p > 4:  # room to the left of the dot; otherwise stack under the value
            ax.annotate(f"{ENGINE[i]}  ", xy=(p, i), xytext=(-9, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=7.5, color=INK_2)
        else:
            ax.annotate(ENGINE[i], xy=(p, i), xytext=(0, -28), textcoords="offset points", ha="left", fontsize=7.5, color=INK_2)
        ax.text(1.0, i, f"{g / p:.0f}×  ", va="center", ha="right", fontsize=12, color=INK, fontweight="bold",
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(1.0, 700)
    ax.set_yticks(list(y), DTYPES, fontsize=10, color=INK)
    ax.set_ylim(len(DTYPES) - 0.25, -0.75)


def paired_bars(ax, before, after, unit, before_label, after_label, annotate_speedup):
    y = range(len(DTYPES))
    h = 0.34
    ax.barh([i + h / 2 + 0.02 for i in y], before, height=h, color=BLUE, label=before_label)
    ax.barh([i - h / 2 - 0.02 for i in y], after, height=h, color=AQUA, label=after_label)
    ax.set_yticks(list(y), DTYPES, fontsize=10, color=INK)
    ax.invert_yaxis()
    xmax = max(before) * 1.32
    ax.set_xlim(0, xmax)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(b + xmax * 0.012, i + h / 2 + 0.02, f"{b:.1f} {unit}", va="center", fontsize=9, color=INK_2)
        if a > 0:
            atext = f"{a:.2f} {unit}"
            ax.text(a + xmax * 0.012, i - h / 2 - 0.02, atext, va="center", fontsize=9, color=INK, fontweight="bold")
            if annotate_speedup:
                ax.annotate(f"  — {ENGINE[i]}", xy=(a + xmax * 0.012, i - h / 2 - 0.02), xytext=(52, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK_2)
        else:
            ax.text(xmax * 0.012, i - h / 2 - 0.02, "0", va="center", fontsize=10, color=INK, fontweight="bold")
        if annotate_speedup:
            ax.text(xmax * 0.995, i - h / 2 - 0.02, f"{b / a:.1f}×", va="center", ha="right", fontsize=12, color=INK,
                    fontweight="bold")


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2), width_ratios=[1.75, 1.0], dpi=150)
    fig.patch.set_facecolor(SURFACE)

    progression_dots(ax1)
    style_axis(ax1)
    ax1.set_xlabel("time per contraction (ms, log scale, lower is better)", fontsize=9, color=INK_2)
    ax1.set_title("einsum() dispatch time across three engine generations", fontsize=11, color=INK, loc="left", pad=10)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper right", bbox_to_anchor=(0.985, 0.945), frameon=False, fontsize=9,
               labelcolor=INK)

    paired_bars(ax2, SORT_MB, PACK_MB, "MiB", "Sort+GEMM", "PackedGemm", annotate_speedup=False)
    ax2.set_yticks(range(len(DTYPES)), ["", "", "", ""])
    style_axis(ax2)
    ax2.set_xlabel("transient allocation per call (MiB)", fontsize=9, color=INK_2)
    ax2.set_title("operand-sized temporaries", fontsize=11, color=INK, loc="left", pad=10)

    fig.suptitle("Scrambled tensor contraction C(abij) = A(aeim)·B(ebmj)  —  Apple M4, o=16, v=64 (48 complex)",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out = Path(__file__).resolve().parents[2] / "docs" / "sphinx" / "_static" / "index-images" / "packed_gemm_dispatch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
