# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Benchmark results from a test's stdout, for when no profiler server reported them.

Every performance test reports through ``performance::publish_benchmark_result``
(``testing/performance/include/Einsums/Performance.hpp``), which sends a structured
``benchmark_result`` event to the profiler server and prints one line::

    [blas-gemm N=256] Time: 345.67 us  min: 340.12  max: 360.45  stddev: 5.67  cv: 1.6%  warmup: 500.12 us (1.4x)  metric: t_blas

The line is read here only when the server path produced nothing (a build without the
profiler). Lines printed before the ``metric:`` field was added carry no metric, and are
recorded as ``t_generic``, the name the database has always used for them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The metric recorded for a line that does not name one.
LEGACY_METRIC = "t_generic"

_RESULT = re.compile(
    r"\[(?P<label>.+?\s+N=\d+)\]\s+Time:\s+(?P<avg>[\d.]+)\s+\S*s"
    r"(?:\s+min:\s+(?P<min>[\d.]+)\s+max:\s+(?P<max>[\d.]+)\s+stddev:\s+(?P<stddev>[\d.]+)"
    r"\s+cv:\s+[\d.]+%\s+warmup:\s+(?P<warmup>[\d.]+)\s+\S*s\s+\([\d.]+x\))?"
    r"(?:\s+metric:\s+(?P<metric>\S+))?"
)

# Einsums' println() prepends "[ tid #  N ] " to a line that goes through the logger.
_TID_PREFIX = re.compile(r"^\[\s*tid\s+#\s*\d+\s*\]\s*")


@dataclass
class Result:
    """One measurement, in the shape of the server's ``benchmark_result`` event."""

    label: str  # "blas-gemm N=256"
    metric: str
    value_us: float
    min_us: float | None = None
    max_us: float | None = None
    stddev_us: float | None = None
    warmup_us: float | None = None
    reps: int | None = None
    annotations: dict[str, str] = field(default_factory=dict)
    source: str = "stdout"  # or "profiler"

    @property
    def n(self) -> int | None:
        match = re.search(r"\bN=(\d+)", self.label)
        return int(match.group(1)) if match else None

    @property
    def base_label(self) -> str:
        return re.sub(r"\s+N=\d+$", "", self.label)

    @classmethod
    def from_event(cls, event: dict) -> Result:
        """A ``benchmark_result`` message from the profiler server."""
        return cls(
            label=event.get("label", "unknown"),
            metric=event.get("metric", LEGACY_METRIC),
            value_us=float(event.get("value_us", 0.0)),
            min_us=event.get("min_us"),
            max_us=event.get("max_us"),
            stddev_us=event.get("stddev_us"),
            warmup_us=event.get("warmup_us"),
            reps=event.get("reps"),
            annotations=dict(event.get("annotations") or {}),
            source="profiler",
        )


def parse_output(text: str) -> list[Result]:
    """Every result line in *text*, in order."""
    results = []
    for raw in text.splitlines():
        match = _RESULT.search(_TID_PREFIX.sub("", raw))
        if match is None:
            continue
        number = lambda key: float(match.group(key)) if match.group(key) is not None else None  # noqa: E731
        label = match.group("label").strip()
        # A result published without a size prints "N=0", but the server's label omits it.
        if label.endswith(" N=0"):
            label = label[: -len(" N=0")]
        results.append(
            Result(
                label=label,
                metric=match.group("metric") or LEGACY_METRIC,
                value_us=float(match.group("avg")),
                min_us=number("min"),
                max_us=number("max"),
                stddev_us=number("stddev"),
                warmup_us=number("warmup"),
            )
        )
    return results


def estimate_flops(label: str, n: int) -> float | None:
    """Floating-point operations of a benchmark named like ``blas-gemm``, at size *n*.

    Standard counts: dot/axpy 2N, scal N, gemv/ger 2N^2, gemm 2N^3, syev and geqrf
    (4/3)N^3, getrf (2/3)N^3, gesvd ~11N^3, rank-3 einsum 2N^4, hadamard/trace 2N^2.
    None for an operation not listed.
    """
    op = label.lower()
    for prefix in ("blas-", "la-", "einsum-"):
        if op.startswith(prefix):
            op = op[len(prefix):]
            break
    n2, n3 = n * n, n * n * n
    return {
        "dot": 2 * n,
        "axpy": 2 * n,
        "scal": n,
        "gemv": 2 * n2,
        "ger": 2 * n2,
        "gemm": 2 * n3,
        "gemm-t": 2 * n3,
        "syev": int(4 / 3 * n3),
        "getrf": int(2 / 3 * n3),
        "gesvd": 11 * n3,
        "geqrf": int(4 / 3 * n3),
        "qr": int(4 / 3 * n3),
        "svd": 11 * n3,
        "rank3": 2 * n * n3,
        "hadamard": 2 * n2,
        "trace": 2 * n2,
    }.get(op)


def compute_gflops(label: str, n: int, time_us: float) -> float | None:
    flops = estimate_flops(label, n)
    if flops is None or time_us <= 0:
        return None
    return flops / (time_us * 1e3)
