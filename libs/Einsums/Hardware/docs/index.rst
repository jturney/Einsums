..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Hardware:

========
Hardware
========

The ``Hardware`` module detects the machine facts that the rest of Einsums
tunes against, and it detects them exactly once. Cache sizes, the native SIMD
width, and the measured cost of an OpenMP parallel region all live behind a
single cached :cpp:func:`~einsums::hardware::cpu_info` call, so every module
that blocks, chunks, or decides whether to fork a thread team is reading the
same numbers.

Why a module for this
=====================

The same facts used to be detected in more than one place, and the copies
disagreed. PackedGemm sized its cache blocking from ``hw.l2cachesize`` while
the ComputeGraph cost model read ``hw.perflevel0.l2cachesize``. On Apple
Silicon the first reports the *efficiency* cluster, so the blocking was derived
from a quarter of the L2 that the cores running the work actually have. That
class of bug does not show up as a wrong answer, only as unexplained slowness,
which is exactly the kind of thing a single detection point prevents.

What belongs here is hardware **facts** and the cheap thresholds derived
directly from them. Algorithm **policy** does not. PackedGemm's micro-kernel
register blocking and its BLIS cache blocking read these numbers, but the
decisions stay with the algorithm.

CpuInfo
=======

.. code-block:: cpp

    #include <Einsums/Hardware/CpuInfo.hpp>

    auto const &hw = einsums::hardware::cpu_info();

    hw.simd_width_f64;      // 2 (SSE2/NEON), 4 (AVX/AVX2), 8 (AVX-512)
    hw.cache.l1;            // bytes
    hw.cache.l2;            // bytes
    hw.cache.l3;            // bytes
    hw.omp_region_cost_ns;  // measured fork/join cost, 0 without OpenMP

Detection runs on first use and the result is cached for the process lifetime.
Every field is populated with a conservative default (32 KB / 256 KB / 8 MB,
two doubles per vector) that survives a failed probe, so callers never have to
check for a zero.

Cache detection is per platform: ``sysctlbyname`` on macOS, the
``/sys/devices/system/cpu/cpu0/cache/index*`` hierarchy on Linux (walked by
``level`` and ``type`` rather than by index number, and skipping instruction
caches). On Apple Silicon the L2 probe deliberately prefers
``hw.perflevel0.l2cachesize``, the performance cluster, and falls back to
``hw.l2cachesize``; an L3 reported as zero falls back to 8 MB.

The SIMD width comes from the compile-time ISA macros, not from CPUID. For the
runtime feature ladder, including the psABI rungs and the ``EINSUMS_SIMD_ARCH``
override, see :ref:`the SIMD module <modules_Einsums_SIMD>`.

The cost of a parallel region
=============================

:cpp:func:`~einsums::hardware::omp_region_cost_ns` is *measured*, not assumed:
an empty ``#pragma omp parallel`` at the default thread count, timed after the
team is warmed so it captures steady-state fork/join rather than one-off thread
creation, best of several trials.

It is not a small number. On a ten-thread machine an empty region costs around
20 microseconds, and it grows with the thread count. A loop needs real work in
it before parallelizing is anything but a loss, which is the whole reason the
two thresholds below exist.

Parallel thresholds
===================

Two derived break-evens answer the question "is this loop big enough to be
worth a thread team?".

:cpp:func:`~einsums::hardware::omp_min_parallel_elements`
    Element count below which an elementwise loop should stay serial.
    Elementwise kernels are bandwidth-bound, so the conversion assumes a
    deliberately conservative one element per nanosecond: the break-even is
    where the loop's own time matches the cost of entering the region. The
    motivating case is a tiled tensor, where an amplitude update over a
    64-element tile was forking a team to divide 64 numbers, thousands of times
    per graph replay.

:cpp:func:`~einsums::hardware::omp_min_parallel_flops`
    Work, in flops, below which parallelizing a contraction is a net loss.
    Scaled from the same measured region cost by a calibration constant of 12
    flops per nanosecond. That constant is not a claim about achieved
    throughput: it is the multiplier that landed on the measured optimum across
    two tiled CCSD residuals. A tiled residual at 50 spin orbitals expands to
    3751 contractions of roughly 295 KFLOP each, and forking for every one of
    them cost 71 ms of that replay's 72 ms of einsum time, against 33 ms with
    the regions declined.

Both are derived from the measured region cost rather than hardcoded, because
that cost varies by an order of magnitude across thread counts and OpenMP
runtimes.

.. note::

    The flops calibration was swept on arm64 (Apple M4 Pro, 10 threads) only.
    x86 wants the same sweep before the constant is trusted there.

Conditional OpenMP pragmas
==========================

Two macros parallelize a loop only when it is worth it:

.. code-block:: cpp

    using einsums::hardware::omp_min_parallel_elements;

    EINSUMS_OMP_PARALLEL_FOR_SIMD_IF(elems >= omp_min_parallel_elements())
    for (size_t i = 0; i < elems; ++i) {
        out[i] = alpha * a[i] + beta * b[i];
    }

Use ``EINSUMS_OMP_PARALLEL_FOR_SIMD_IF`` (or ``EINSUMS_OMP_PARALLEL_FOR_IF``,
the non-vectorized form) instead of the unconditional
``EINSUMS_OMP_PARALLEL_FOR_SIMD`` on any loop whose trip count can be small.
An unconditional region on a short loop costs far more than the loop.

Diagnostic overrides
====================

Three environment variables force the detected values, so a blocking or
threshold change can be A/B tested from **one** binary. Comparing across
rebuilds is not reliable for these benchmarks: they swing by tens of percent
with unrelated machine activity, and only a same-binary comparison holds the
controls steady.

.. envvar:: EINSUMS_CACHE_SIZES

    ``L1,L2,L3`` in bytes. Any field that is absent or ``<= 0`` keeps the
    detected value.

.. envvar:: EINSUMS_OMP_MIN_PARALLEL_ELEMENTS

    Overrides :cpp:func:`~einsums::hardware::omp_min_parallel_elements`. Zero
    restores "always parallelize".

.. envvar:: EINSUMS_PACKED_MIN_PARALLEL_FLOPS

    Overrides :cpp:func:`~einsums::hardware::omp_min_parallel_flops`.

These are diagnostics, not a tuning interface: they are read once, on the first
call, and are not part of the supported configuration surface.

Who reads this
==============

- :ref:`PackedGemm <modules_Einsums_PackedGemm>` derives its ``MC``/``KC``/``NC``
  cache blocking from ``cache.l2`` and ``cache.l3``, and gates its parallel
  loops on ``omp_min_parallel_flops()``.
- :ref:`ComputeGraph <modules_Einsums_ComputeGraph>`'s ``CostModel`` seeds its
  cache hierarchy from ``CpuInfo``, which is what lets ``TiledExpansion`` and
  ``StreamContractionFusion`` size their private accumulators and price a
  per-tile dispatch.
- :ref:`LinearAlgebra <modules_Einsums_LinearAlgebra>`'s elementwise kernels
  (``direct_division``, ``direct_product``, ``norm``) gate their OpenMP regions
  on ``omp_min_parallel_elements()``.

API Reference
=============

- :cpp:struct:`~einsums::hardware::CpuInfo` - the detected facts: SIMD width, cache sizes, OpenMP region cost.
- :cpp:struct:`~einsums::hardware::CacheSizes` - L1/L2/L3 data-cache sizes in bytes.
- :cpp:func:`~einsums::hardware::cpu_info` - the cached singleton; detection happens on first call.
- :cpp:func:`~einsums::hardware::omp_region_cost_ns` - measured cost of entering and leaving a parallel region.
- :cpp:func:`~einsums::hardware::omp_min_parallel_elements` - elementwise parallelization break-even.
- :cpp:func:`~einsums::hardware::omp_min_parallel_flops` - contraction parallelization break-even.

See the :ref:`API reference <modules_Einsums_Hardware_api>` of this module for
the full generated surface.
