#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Generate the polynomial coefficients of erf and erfc in Einsums/SIMD/Math.hpp.

Prints the two ``erf_constants`` specializations (double and float) that Math.hpp holds,
ready to replace the ones there. Requires mpmath (``pip install mpmath``); nothing else in
the build does.

The pieces, by |x| (see Math.hpp for how they are combined):

  small  E(s) = erf(x) / x - 1, a polynomial in s = x^2 over [0, 0.25]
  mid    R(x) = erfc(x) e^(x^2) over [0.5, 2], a polynomial in z = x - 1.25
  high   R(x) over [2, 4], a polynomial in z = x - 3
  tail   G(u) = x R(x) with u = 1 / x^2, over [4, top], a polynomial in u rescaled to [-1, 1]

Each is a Chebyshev fit (mpmath.chebyfit) in 60-digit arithmetic, at the degree that brings
its relative error below 2^-57 for double and 2^-27 for float, then rounded to the target
type. The degrees were found by raising each until the error target was met; pass --degrees
to try others. The error that matters is that of the C++ evaluation against a correctly
rounded erf and erfc, which tests/unit/VecErf.cpp measures and bounds.

Usage:  python3 generate_erf_coefficients.py > coefficients.inc
"""

from __future__ import annotations

import argparse
import struct

import mpmath as mp

mp.mp.dps = 60

# Degrees of the small, mid, high and tail polynomials, and the argument past which erfc is
# below the smallest subnormal of the type, where the tail fit ends.
DOUBLE = {"degrees": (8, 19, 19, 14), "top": mp.mpf("27.5")}
FLOAT = {"degrees": (4, 10, 9, 5), "top": mp.mpf("10.25")}


def small_e(s):
    """erf(x) / x - 1 as a function of s = x^2."""
    if s == 0:
        return 2 / mp.sqrt(mp.pi) - 1
    x = mp.sqrt(s)
    return mp.erf(x) / x - 1


def r_of_x(x):
    """erfc(x) e^(x^2)."""
    return mp.erfc(x) * mp.exp(x * x)


def g_of_u(u):
    """x R(x) as a function of u = 1 / x^2; 1 / sqrt(pi) at u = 0."""
    if u == 0:
        return 1 / mp.sqrt(mp.pi)
    x = 1 / mp.sqrt(u)
    return x * r_of_x(x)


def literal(value, kind: str) -> str:
    """value rounded to the C++ type, as an exact hexadecimal literal."""
    if kind == "double":
        return float.hex(float(value))
    rounded = struct.unpack("f", struct.pack("f", float(value)))[0]
    return float.hex(rounded) + "f"


def fit(f, a, b, degree):
    """Chebyshev fit of f over [a, b] of the given degree, coefficients highest power first."""
    coefficients, _ = mp.chebyfit(f, [a, b], degree + 1, error=True)
    return coefficients


def specialization(kind: str, degrees, top) -> str:
    d_small, d_mid, d_high, d_tail = degrees
    small = fit(small_e, 0, mp.mpf("0.25"), d_small)
    mid_center = mp.mpf("1.25")
    mid = fit(lambda z: r_of_x(mid_center + z), mp.mpf("-0.75"), mp.mpf("0.75"), d_mid)
    high_center = mp.mpf(3)
    high = fit(lambda z: r_of_x(high_center + z), -1, 1, d_high)
    u_lo, u_hi = 1 / top**2, mp.mpf(1) / 16
    tail_center, tail_half = (u_lo + u_hi) / 2, (u_hi - u_lo) / 2
    tail = fit(lambda w: g_of_u(tail_center + tail_half * w), -1, 1, d_tail)

    def array(name: str, values) -> str:
        body = ",\n".join(" " * 8 + literal(v, kind) for v in values)
        return f"    static constexpr {kind} {name}[{len(values)}] = {{\n{body}}};\n"

    split = "134217729.0" if kind == "double" else "4097.0f"
    split_power = 27 if kind == "double" else 12
    return (
        f"template <>\nstruct erf_constants<{kind}> {{\n"
        "    // E(s) = erf(x) / x - 1 as a polynomial in s = x^2 over [0, 0.25], highest power first.\n"
        + array("small", small)
        + "    // R(x) = erfc(x) e^(x^2) over [0.5, 2] in z = x - mid_center, exact, highest first.\n"
        + array("mid", mid)
        + "    // R(x) over [2, 4] in z = (x - high_center) / high_half.\n"
        + array("high", high)
        + "    // x R(x) over [4, top] in w = (1 / x^2 - tail_center) / tail_half.\n"
        + array("tail", tail)
        + f"    static constexpr {kind} mid_center = {literal(mid_center, kind)};\n"
        + f"    static constexpr {kind} mid_half = {literal(1, kind)};\n"
        + f"    static constexpr {kind} high_center = {literal(high_center, kind)};\n"
        + f"    static constexpr {kind} high_half = {literal(1, kind)};\n"
        + f"    static constexpr {kind} tail_center = {literal(tail_center, kind)};\n"
        + f"    static constexpr {kind} tail_half = {literal(tail_half, kind)};\n"
        + f"    static constexpr {kind} top = {literal(top, kind)};\n"
        + f"    static constexpr {kind} splitter = {split}; // 2^{split_power} + 1, for an exact square without FMA\n"
        + "};\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degrees-double", type=int, nargs=4, default=DOUBLE["degrees"], metavar="N")
    parser.add_argument("--degrees-float", type=int, nargs=4, default=FLOAT["degrees"], metavar="N")
    args = parser.parse_args()
    print(specialization("double", args.degrees_double, DOUBLE["top"]))
    print(specialization("float", args.degrees_float, FLOAT["top"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
