# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Parse benchmark stdout text into structured results.

Extracted from devtools/profiling/bench_viewer.py for reuse by the
regression detection system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """A single benchmark timing result."""
    label: str
    n: int
    t_generic: float  # us
    t_blas_packed: float | None = None
    t_mlir: float | None = None
    t_einsum: float | None = None
    t_sort_gemm: float | None = None
    einsum_alg: str = ""
    speedup_blas: float | None = None
    speedup_mlir: float | None = None
    speedup_einsum: float | None = None
    speedup_sort_gemm: float | None = None
    # Per-rep statistics (from extended output format)
    min_us: float | None = None
    max_us: float | None = None
    stddev_us: float | None = None
    cv_pct: float | None = None
    warmup_us: float | None = None
    warmup_ratio: float | None = None  # warmup / avg


@dataclass
class BreakdownResult:
    """Packing vs micro-kernel breakdown."""
    n: int
    t_pack_only: float
    t_kernel_only: float
    t_total: float
    mr: int
    nr: int
    tiles: int


@dataclass
class CacheHitResult:
    """Cache-hit overhead measurement."""
    n: int
    t_generic: float
    t_mlir: float
    t_einsum: float
    einsum_alg: str
    speedup_mlir: float
    speedup_einsum: float


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Simple single-timing line with extended stats + warmup:
# [blas-gemm N=256] Time: 345.67 us  min: 340.12  max: 360.45  stddev: 5.67  cv: 1.6%  warmup: 500.12 us (1.4x)
_RE_SIMPLE_EXTENDED = re.compile(
    r"\[(.+?)\s+N=(\d+)\]\s+Time:\s+([\d.]+)\s+.*?s"
    r"\s+min:\s+([\d.]+)"
    r"\s+max:\s+([\d.]+)"
    r"\s+stddev:\s+([\d.]+)"
    r"\s+cv:\s+([\d.]+)%"
    r"\s+warmup:\s+([\d.]+)\s+.*?s"
    r"\s+\(([\d.]+)x\)"
)

# Fallback: simple format without stats
# [blas-gemm N=256] Time: 345.67 us
_RE_SIMPLE = re.compile(
    r"\[(.+?)\s+N=(\d+)\]\s+Time:\s+([\d.]+)\s+.*?s"
)

# Standard line:
# [Rank-2 N=64] Generic: 123.45 us  BLAS packed: 45.67 us  MLIR: 56.78 us  einsum (BLAS-GEMM): 34.56 us  BLAS/gen: 2.70x  MLIR/gen: 2.17x  einsum/gen: 3.57x
_RE_STANDARD = re.compile(
    r"\[(.+?)\s+N=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+BLAS packed:\s+([\d.]+)\s+.*?s"
    r"\s+MLIR:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+BLAS/gen:\s+([\d.]+)x"
    r"\s+MLIR/gen:\s+([\d.]+)x"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Non-GEMM line (only Generic + einsum, no BLAS packed / MLIR columns):
_RE_NONGEMM = re.compile(
    r"\[(.+?)\s+N=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Large-K line (has M, N, K in the label):
_RE_LARGE_K = re.compile(
    r"\[(.+?)\s+M=(\d+)\s+N=(\d+)\s+K=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+BLAS packed:\s+([\d.]+)\s+.*?s"
    r"\s+MLIR:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+BLAS/gen:\s+([\d.]+)x"
    r"\s+MLIR/gen:\s+([\d.]+)x"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Batch line (has B and N in label):
_RE_BATCH = re.compile(
    r"\[(.+?)\s+B=(\d+)\s+N=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Breakdown line:
_RE_BREAKDOWN = re.compile(
    r"\[Breakdown\s+N=(\d+)\]\s+"
    r"pack_only:\s+([\d.]+)\s+.*?s"
    r"\s+kernel_only:\s+([\d.]+)\s+.*?s"
    r"\s+total:\s+([\d.]+)\s+.*?s"
    r"\s+\(MR=(\d+)\s+NR=(\d+)\s+tiles=(\d+)\)"
)

# Sort+GEMM multi-line block:
_RE_SORT_HEADER = re.compile(r"^\[(.+?)\s+N=(\d+)\]\s*$")
_RE_SORT_GENERIC = re.compile(r"^\s*Generic:\s+([\d.]+)\s+.*?s\s*$")
_RE_SORT_MLIR = re.compile(r"^\s*MLIR:\s+([\d.]+)\s+.*?s\s+\(([\d.]+)x\s+vs\s+generic\)")
_RE_SORT_SORTGEMM = re.compile(r"^\s*Sort\+GEMM:\s+([\d.]+)\s+.*?s\s+\(([\d.]+)x\s+vs\s+generic\)")
_RE_SORT_EINSUM = re.compile(r"^\s*einsum\(([^)]+)\):\s+([\d.]+)\s+.*?s\s+\(([\d.]+)x\s+vs\s+generic\)")

# Cache hit line:
_RE_CACHE = re.compile(
    r"\[Cache hit\s+N=(\d+)\]\s+"
    r"Generic OMP:\s+([\d.]+)\s+.*?s"
    r"\s+MLIR direct:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+MLIR/generic:\s+([\d.]+)x"
    r"\s+einsum/generic:\s+([\d.]+)x"
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Einsums' println() prepends "[ tid  #  N ] " to each line when output
# goes through the logging system. Strip this prefix before parsing.
_RE_TID_PREFIX = re.compile(r"^\[\s*tid\s+#\s*\d+\s*\]\s*")


def _strip_tid(line: str) -> str:
    """Remove the '[ tid  #  N ] ' prefix if present."""
    return _RE_TID_PREFIX.sub("", line)


def parse_output(text: str) -> tuple[list[BenchmarkResult], list[BreakdownResult], list[CacheHitResult]]:
    """Parse benchmark stdout text into structured results."""
    benchmarks: list[BenchmarkResult] = []
    breakdowns: list[BreakdownResult] = []
    cache_hits: list[CacheHitResult] = []

    # Strip tid prefixes from all lines before parsing
    lines = [_strip_tid(line) for line in text.splitlines()]
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # --- Multi-line Sort+GEMM block ---
        m_hdr = _RE_SORT_HEADER.match(line)
        if m_hdr:
            label = m_hdr.group(1).strip()
            n = int(m_hdr.group(2))
            t_gen = t_ml = t_sg = t_es = None
            sp_ml = sp_sg = sp_es = None
            es_alg = ""
            j = idx + 1
            while j < len(lines):
                sub = lines[j]
                mg = _RE_SORT_GENERIC.match(sub)
                if mg:
                    t_gen = float(mg.group(1))
                    j += 1
                    continue
                mm = _RE_SORT_MLIR.match(sub)
                if mm:
                    t_ml = float(mm.group(1))
                    sp_ml = float(mm.group(2))
                    j += 1
                    continue
                ms = _RE_SORT_SORTGEMM.match(sub)
                if ms:
                    t_sg = float(ms.group(1))
                    sp_sg = float(ms.group(2))
                    j += 1
                    continue
                me = _RE_SORT_EINSUM.match(sub)
                if me:
                    es_alg = me.group(1)
                    t_es = float(me.group(2))
                    sp_es = float(me.group(3))
                    j += 1
                    continue
                break
            if t_gen is not None:
                benchmarks.append(BenchmarkResult(
                    label=label, n=n, t_generic=t_gen,
                    t_mlir=t_ml, t_sort_gemm=t_sg, t_einsum=t_es,
                    einsum_alg=es_alg, speedup_mlir=sp_ml,
                    speedup_sort_gemm=sp_sg, speedup_einsum=sp_es,
                ))
                idx = j
                continue

        # --- Single-line formats ---

        # Full standard line (Generic + BLAS packed + MLIR + einsum)
        m = _RE_STANDARD.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(), n=int(m.group(2)),
                t_generic=float(m.group(3)), t_blas_packed=float(m.group(4)),
                t_mlir=float(m.group(5)), einsum_alg=m.group(6),
                t_einsum=float(m.group(7)), speedup_blas=float(m.group(8)),
                speedup_mlir=float(m.group(9)), speedup_einsum=float(m.group(10)),
            ))
            idx += 1
            continue

        # Large-K line (M=... N=... K=...)
        m = _RE_LARGE_K.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(), n=int(m.group(3)),
                t_generic=float(m.group(5)), t_blas_packed=float(m.group(6)),
                t_mlir=float(m.group(7)), einsum_alg=m.group(8),
                t_einsum=float(m.group(9)), speedup_blas=float(m.group(10)),
                speedup_mlir=float(m.group(11)), speedup_einsum=float(m.group(12)),
            ))
            idx += 1
            continue

        # Batch line (B=... N=...)
        m = _RE_BATCH.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(), n=int(m.group(3)),
                t_generic=float(m.group(4)), einsum_alg=m.group(5),
                t_einsum=float(m.group(6)), speedup_einsum=float(m.group(7)),
            ))
            idx += 1
            continue

        # Non-GEMM line (Generic + einsum only)
        m = _RE_NONGEMM.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(), n=int(m.group(2)),
                t_generic=float(m.group(3)), einsum_alg=m.group(4),
                t_einsum=float(m.group(5)), speedup_einsum=float(m.group(6)),
            ))
            idx += 1
            continue

        # Simple single-timing line with extended stats + warmup
        m = _RE_SIMPLE_EXTENDED.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(), n=int(m.group(2)),
                t_generic=float(m.group(3)),
                min_us=float(m.group(4)),
                max_us=float(m.group(5)),
                stddev_us=float(m.group(6)),
                cv_pct=float(m.group(7)),
                warmup_us=float(m.group(8)),
                warmup_ratio=float(m.group(9)),
            ))
            idx += 1
            continue

        # Simple single-timing line (fallback without stats)
        m = _RE_SIMPLE.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(), n=int(m.group(2)),
                t_generic=float(m.group(3)),
            ))
            idx += 1
            continue

        m = _RE_BREAKDOWN.search(line)
        if m:
            breakdowns.append(BreakdownResult(
                n=int(m.group(1)), t_pack_only=float(m.group(2)),
                t_kernel_only=float(m.group(3)), t_total=float(m.group(4)),
                mr=int(m.group(5)), nr=int(m.group(6)), tiles=int(m.group(7)),
            ))
            idx += 1
            continue

        m = _RE_CACHE.search(line)
        if m:
            cache_hits.append(CacheHitResult(
                n=int(m.group(1)), t_generic=float(m.group(2)),
                t_mlir=float(m.group(3)), einsum_alg=m.group(4),
                t_einsum=float(m.group(5)), speedup_mlir=float(m.group(6)),
                speedup_einsum=float(m.group(7)),
            ))
            idx += 1
            continue

        idx += 1

    return benchmarks, breakdowns, cache_hits


def estimate_flops(label: str, n: int) -> float | None:
    """Estimate the number of floating-point operations for a benchmark.

    Returns the FLOPs count, or None if the operation is unknown.
    Based on standard computational complexity:
      - dot:    2*N          (N multiplies + N-1 adds)
      - axpy:   2*N          (N multiplies + N adds)
      - scal:   N            (N multiplies)
      - gemv:   2*N*N        (matrix-vector multiply)
      - ger:    2*N*N        (outer product)
      - gemm:   2*N^3        (matrix multiply, 2*M*N*K with M=N=K)
      - syev:   ~(4/3)*N^3   (eigendecomposition)
      - getrf:  ~(2/3)*N^3   (LU factorization)
      - gesvd:  ~11*N^3      (SVD, rough estimate)
      - geqrf:  ~(4/3)*N^3   (QR factorization)
      - einsum-gemm:   2*N^3
      - einsum-rank3:  2*N^4 (three link indices)
      - einsum-hadamard: 2*N^2
      - einsum-trace:    2*N^2
    """
    # Strip blas-/la- prefix for matching
    op = label.lower()
    for prefix in ("blas-", "la-", "einsum-"):
        if op.startswith(prefix):
            op = op[len(prefix):]
            break

    n2 = n * n
    n3 = n * n * n

    flop_map = {
        "dot":      2 * n,
        "axpy":     2 * n,
        "scal":     n,
        "gemv":     2 * n2,
        "ger":      2 * n2,
        "gemm":     2 * n3,
        "gemm-t":   2 * n3,
        "syev":     int(4 / 3 * n3),
        "getrf":    int(2 / 3 * n3),
        "gesvd":    11 * n3,
        "geqrf":    int(4 / 3 * n3),
        "qr":       int(4 / 3 * n3),
        "svd":      11 * n3,
        "rank3":    2 * n * n3,  # N^4
        "hadamard": 2 * n2,
        "trace":    2 * n2,
    }
    return flop_map.get(op)


def compute_gflops(label: str, n: int, time_us: float) -> float | None:
    """Compute GFLOP/s from benchmark label, N, and time in microseconds."""
    flops = estimate_flops(label, n)
    if flops is None or time_us <= 0:
        return None
    return flops / (time_us * 1e3)  # GFLOP/s = FLOPs / (us * 1e3)
