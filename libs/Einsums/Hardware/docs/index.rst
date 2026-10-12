..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Hardware:

========
Hardware
========

The ``Hardware`` module detects the machine facts that the rest of Einsums
tunes against, in one place. Cache sizes and the native SIMD width live behind
a single cached :cpp:func:`~einsums::hardware::cpu_info` call, and the cost of
an OpenMP parallel region behind
:cpp:func:`~einsums::hardware::omp_region_cost_ns`, so every module that
blocks, chunks, or decides whether to fork a thread team is reading the same
numbers.

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

    hw.simd_width_f64;      // 2 (SSE2/NEON), 4 (AVX/AVX2), 8 (AVX-512): the rung the
                            // process dispatches to (stripes::selected_arch()), so it
                            // follows the CPU and the --einsums:simd:arch override
    hw.simd_width_f32;      // twice simd_width_f64
    hw.compiled_simd_width_f64; // what the library's own flags were vectorized at
    hw.cache.l1;            // bytes
    hw.cache.l2;            // bytes
    hw.cache.l3;            // bytes
    hw.omp_region_cost_ns;  // fork/join cost at first call, 0 without OpenMP;
                            // omp_region_cost_ns() follows the live team size

Detection runs on first use and the result is cached for the process lifetime.
Every field is populated with a conservative default (32 KB / 256 KB / 8 MB,
two doubles per vector) that survives a failed probe, so callers never have to
check for a zero.

Cache detection is per platform: ``sysctlbyname`` on macOS, the
``/sys/devices/system/cpu/cpu0/cache/index*`` hierarchy on Linux (walked by
``level`` and ``type`` rather than by index number, and skipping instruction
caches). On Apple Silicon the L2 probe takes the larger of
``hw.perflevel0.l2cachesize``, the performance cluster, and
``hw.l2cachesize``; an L3 reported as zero falls back to 8 MB.

The SIMD width comes from the rung the process dispatches to, which CPUID, the
operating system's vector state and :option:`--einsums:simd:arch` decide
together. Only ``compiled_simd_width_f64`` comes from the compile-time ISA
macros. For the runtime feature ladder and its psABI rungs, see
:ref:`SIMD <modules_Einsums_SIMD>`.

The hardware profile
====================

Everything ``einsums tune`` measures on a machine is one JSON file, the
hardware profile. Its header says what it describes: the machine (CPU, core
counts, caches, memory rounded to the GiB, and SIMD rung, hashed into a
16-digit fingerprint), the software (the Einsums version and the BLAS library
that provides ``dgemm``, found from that function's address so that a BLAS
swapped behind a conda environment is noticed) and when it was measured. Each
remaining section belongs to the module that reads it: the OpenMP region cost
to this one, the cost model to ComputeGraph's
:ref:`hardware profiles <computegraph_hardware_profiles>`, and tuned option
values to the option parser. The profile can be found only once the options
are parsed, since they say where to look and which SIMD rung the machine is
using, so the runtime parses again with the profile's options as the
config-file layer, below the environment and the command line. A profile may
set only performance options, and never one that locates profiles; a value the
parser rejects is reported and the profile's options are dropped, never fatal.

:cpp:func:`~einsums::hardware::profile_status` decides which file is in effect,
on first use. A file named by :option:`--einsums:hardware:profile` is used as
given; otherwise ``profiles/<fingerprint>.json`` under
:option:`--einsums:cache-dir`, then the same name under
:option:`--einsums:hardware:site-profiles`. A file applies when it describes
this machine, this BLAS, and an Einsums version that differs at most in the
patch number; one that does not is skipped, and the first reason is kept so
:cpp:func:`~einsums::hardware::profile_reason` can say why a tune had no
effect. The file is read strictly, like the graph IR: an unknown section or
key, or a fingerprint that disagrees with the machine it describes, makes it
invalid rather than half read. Nothing in the library writes a profile.

The cost of a parallel region
=============================

:cpp:func:`~einsums::hardware::omp_region_cost_ns` is resolved, not assumed.
A value pinned with :option:`--einsums:hardware:omp-region-cost-ns` wins.
Otherwise the entry for the current team size is read from the hardware profile
(see below). Failing that, it is measured in process: an empty
``#pragma omp parallel`` at ``omp_get_max_threads()``, timed after the team is
warmed so it captures steady-state fork/join rather than one-off thread
creation, best of five trials. An in-process measurement drifts by tens of
percent between runs; :cpp:func:`~einsums::hardware::region_cost_is_calibrated`
reports which source was used.

The result is memoized per team size rather than frozen at first use, because
the team size can change during a process.

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

Two macros, defined in :ref:`modules_Einsums_Config`, parallelize a loop only
when it is worth it:

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

Options force the detected values, so a blocking or threshold change can be
A/B tested from **one** binary. Comparing across rebuilds is not reliable for
these benchmarks: they swing by tens of percent with unrelated machine
activity, and only a same-binary comparison holds the controls steady. Each
also pins the machine model a test asserts against, on a runner whose own
hardware would give another.

- :option:`--einsums:hardware:l1-cache-size`,
  :option:`--einsums:hardware:l2-cache-size` and
  :option:`--einsums:hardware:l3-cache-size` replace the detected cache sizes.
- :option:`--einsums:hardware:omp-region-cost-ns` pins
  :cpp:func:`~einsums::hardware::omp_region_cost_ns`, ahead of the hardware
  profile.
- :option:`--einsums:hardware:omp-min-parallel-elements` pins
  :cpp:func:`~einsums::hardware::omp_min_parallel_elements`. Zero restores
  "always parallelize".
- :option:`--einsums:hardware:omp-min-parallel-flops` pins
  :cpp:func:`~einsums::hardware::omp_min_parallel_flops`.

The cache sizes are detected once and kept, and the region cost is memoized
per team size, so those options take effect only when they are parsed first:
on the command line or in the environment, before the first contraction.

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
- :cpp:func:`~einsums::hardware::omp_region_cost_ns` - cost of entering and leaving a parallel region at the current team size (pinned, from the hardware profile, or measured).
- :cpp:func:`~einsums::hardware::region_cost_is_calibrated` - whether that cost came from a pin or the hardware profile rather than an in-process measurement.
- :cpp:func:`~einsums::hardware::profile_status` - which hardware profile is in effect, and why one is not; :cpp:func:`~einsums::hardware::profile_state`, :cpp:func:`~einsums::hardware::profile_path` and :cpp:func:`~einsums::hardware::profile_reason` are its parts, for Python.
- :cpp:func:`~einsums::hardware::machine_identity` and :cpp:func:`~einsums::hardware::machine_fingerprint` - the machine a profile names, and the name it is filed under.
- :cpp:func:`~einsums::hardware::default_profile_path` and :cpp:func:`~einsums::hardware::profile_search_path` - where a profile is written, and every place one is looked for.
- :cpp:func:`~einsums::hardware::omp_min_parallel_elements` - elementwise parallelization break-even.
- :cpp:func:`~einsums::hardware::omp_min_parallel_flops` - contraction parallelization break-even.

See the :ref:`API reference <modules_Einsums_Hardware_api>` of this module for
the full generated surface.
