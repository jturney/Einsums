..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_SIMD:

****
SIMD
****

The ``SIMD`` module provides a portable, header-only SIMD abstraction for
vectorized operations. It wraps platform-specific intrinsics behind a clean
C++20 interface that works across:

- x86_64: SSE2, SSSE3, SSE4.1/4.2, AVX, AVX2, and AVX-512.
- ARM: NEON on Apple Silicon and other aarch64 targets.

The module is used internally by the HPTT transpose library and can be used
directly for performance-critical inner loops.

Platform Detection
==================

.. code-block:: cpp

    #include <Einsums/SIMD/Platform.hpp>

    using namespace einsums::simd;

    // Compile-time constants
    static_assert(native_bits == 128 || native_bits == 256 || native_bits == 512);
    static_assert(has_neon || has_sse2);  // At least one must be true

    // Number of elements that fit in a native register
    constexpr size_t float_lanes = native_lanes<float>;   // 4 (SSE), 8 (AVX), 16 (AVX-512)
    constexpr size_t double_lanes = native_lanes<double>;  // 2, 4, or 8

Core Vec Type
=============

``Vec<T>`` wraps a platform SIMD register for type ``T``:

.. code-block:: cpp

    #include <Einsums/SIMD/Vec.hpp>
    #include <Einsums/SIMD/Operations.hpp>

    using namespace einsums::simd;

    // Broadcast a scalar to all lanes
    Vec<float> a = broadcast<float>(3.14f);

    // Load from memory (aligned or unaligned)
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Vec<float> b = loadu<float>(data);

    // Arithmetic
    Vec<float> c = add(a, b);       // or: a + b
    Vec<float> d = mul(a, b);       // or: a * b
    Vec<float> e = fmadd(a, b, c);  // a*b + c (fused multiply-add)

    // Store back to memory
    storeu(data, c);

Arithmetic, Comparisons and Masks
=================================

Beyond ``add``, ``sub``, ``mul`` and ``fmadd``, ``Operations.hpp`` has ``div``,
``sqrt``, ``min``, ``max``, ``abs`` and ``neg`` for ``float`` and ``double``,
and the operators ``/`` and unary ``-``. ``min(a, b)`` is exactly
``a < b ? a : b``, so a NaN on either side, or two zeros of either sign, give
``b`` on every backend. ``abs`` and ``neg`` change only the sign bit.

A comparison returns a mask of the same type: each lane all-ones where it
holds and zero where it does not. The comparisons follow IEEE 754, so every
one involving a NaN is false except ``cmp_ne``.

.. code-block:: cpp

    Vec<float> const too_big = cmp_gt(v, limit);         // lanes where v > limit
    Vec<float> const clamped = select(too_big, limit, v); // limit there, v elsewhere
    if (any(cmp_ne(v, v))) {                              // is any lane a NaN?
        ...
    }

``select(mask, a, b)`` takes ``a`` where the mask is set, for ``float``,
``double`` and the 32- and 64-bit integers. Masks combine with
``bitwise_and``, ``bitwise_or``, ``bitwise_xor`` and ``bitwise_andnot``
(``a & ~b``), and ``any(mask)`` and ``all(mask)`` reduce one to a ``bool``.
Only comparison results are valid masks: the backends read different bits of
a lane, so only all-ones and zero mean the same thing everywhere.

Reductions, Partial Loads and Conversions
=========================================

``Reduce.hpp`` folds a Vec to a scalar with ``reduce_add``, ``reduce_min``
and ``reduce_max``, for ``float``, ``double`` and ``int32_t``. The lanes are
combined in a tree whose shape depends on the width, so a floating-point sum
can differ in its last bits between ISAs and from a sequential loop.

``Partial.hpp`` handles the tail of a loop whose length is not a multiple of
the lane count. ``loadu_partial(p, n)`` reads the first ``n`` elements and
zeroes the rest of the Vec; ``storeu_partial(p, v, n)`` writes the first
``n`` lanes. Neither touches memory past those elements, so a tail that ends
at the end of an allocation is safe:

.. code-block:: cpp

    std::size_t i = 0;
    for (; i + L <= n; i += L) {
        storeu(y + i, fmadd(a, loadu(x + i), loadu(y + i)));
    }
    if (i < n) {
        storeu_partial(y + i, fmadd(a, loadu_partial(x + i, n - i), loadu_partial(y + i, n - i)), n - i);
    }

``Convert.hpp`` converts between ``int32_t`` and ``float``, which have the
same lane count: ``convert<float>(vi)`` rounds to nearest and
``convert<int32_t>(vf)`` truncates toward zero. A NaN or an out-of-range
value gives an unspecified lane.

Examples
========

Each program in ``libs/Einsums/SIMD/examples`` checks its result against a
scalar loop and runs as a test:

- ``PlatformInfo``: the width this build compiled for, and the rungs the CPU
  supports.
- ``Saxpy``: the basic loop shape, with a partial tail.
- ``DotProduct``: independent accumulators and one ``reduce_add``.
- ``ComplexAxpy``: complex arithmetic with ``CVec``.
- ``ColumnSum``: strided ``gather`` against a contiguous traversal.
- ``BlockTranspose``: a matrix transpose in register tiles.
- ``ClampAndSanitize``: comparisons, ``select`` and ``any`` in place of
  branches.
- ``IntegerBits``: shifts, bitwise logic and ``cmp_eq`` counting.
- ``QuantizedGemv``: int8 dot products and ``convert``.
- ``StreamingScale``: ``prefetch``, ``stream_store`` and ``stream_fence``.
- ``RuntimeDispatch``: one kernel compiled per rung and chosen at run time.
- ``GemmMicroKernel``: a register-blocked matrix-multiply kernel.
- ``OperatorsAndFunctions``: the two spellings, and what ``fmadd`` changes.

Shuffle and Transpose
=====================

The ``Shuffle.hpp`` header provides in-register matrix transposes for
micro-kernels:

.. code-block:: cpp

    #include <Einsums/SIMD/Shuffle.hpp>

    // Transpose a 4x4 float matrix stored in 4 Vec<float> registers (SSE/NEON)
    Vec<float> rows[4];
    // ... load rows ...
    transpose_inplace(rows);  // Now rows[i] holds column i

    // On AVX: 8x8 float transpose
    // On AVX-512: 16x16 float transpose

Gather and Scatter
==================

Non-contiguous memory access with optional hardware acceleration:

.. code-block:: cpp

    #include <Einsums/SIMD/Gather.hpp>

    // Gather elements from non-contiguous locations
    int32_t indices[8] = {0, 3, 6, 9, 12, 15, 18, 21};
    float data[22] = { /* ... */ };
    Vec<float> gathered = gather(data, indices);

    // Fixed-stride gather (compile-time optimized)
    Vec<float> strided = gather_fixed<3>(data);  // data[0], data[3], data[6], ...

    // Scatter (write to non-contiguous locations)
    scatter(data, indices, gathered);

Complex Numbers
===============

``ComplexVec.hpp`` provides SIMD operations on interleaved complex data:

.. code-block:: cpp

    #include <Einsums/SIMD/ComplexVec.hpp>

    // Load interleaved complex data: [re0, im0, re1, im1, ...]
    std::complex<float> z[4] = {{1,2}, {3,4}, {5,6}, {7,8}};
    CVec<float> a = complex_loadu(reinterpret_cast<float const*>(z));

    // Complex multiply
    CVec<float> b = complex_broadcast(1.0f, -1.0f);  // (1 - i)
    CVec<float> c = complex_mul(a, b);

    // Conjugate
    CVec<float> conj = conjugate(a);

Prefetch and Streaming
======================

.. code-block:: cpp

    #include <Einsums/SIMD/Prefetch.hpp>

    // Prefetch for read
    prefetch<PrefetchHint::T0>(data);       // Into L1 cache
    prefetch<PrefetchHint::NTA>(data);      // Non-temporal (streaming)

    // Prefetch multiple rows of a matrix
    prefetch_rows<4>(matrix_ptr, stride);

    // Non-temporal store (bypasses cache, useful for write-only patterns)
    stream_store(dst, vec);

Runtime Feature Detection
=========================

Everything above is *compile-time*: the ISA baked into a translation unit by
its compiler flags. ``RuntimeFeatures.hpp`` adds the *runtime* half - what
the CPU executing the process actually supports - so that a single portable
binary can carry kernels for several ISA levels and pick the best one at
startup.

.. code-block:: cpp

    #include <Einsums/SIMD/RuntimeFeatures.hpp>

    using namespace einsums::simd;

    // Cached, thread-safe, detected once per process.
    CpuFeatures const &f = cpu_features();
    if (f.avx512f) { /* ... */ }

    // The dispatch rung for this process: Baseline, V2, V3, V4, or Sme.
    InstructionSet arch = selected_arch();

The x86 rungs follow the psABI micro-architecture levels, which map directly
onto compiler flags:

============  =====================================  =========================
Rung          ISA content                            Compiler flag
============  =====================================  =========================
``Baseline``  x86-64 (SSE2) / aarch64 (NEON)         none (toolchain default)
``V2``        SSE3...SSE4.2, POPCNT, CMPXCHG16B      ``-march=x86-64-v2``
``V3``        adds AVX2, FMA, BMI1/2, F16C           ``-march=x86-64-v3``
``V4``        adds AVX-512 F/BW/CD/DQ/VL             ``-march=x86-64-v4``
============  =====================================  =========================

All AVX-family reports are gated on operating-system state enablement
(OSXSAVE + XCR0, queried with ``xgetbv``), not just CPUID bits: a feature is
reported only if using it will not fault. On aarch64, optional features
(FEAT_FP16, FEAT_BF16, FEAT_I8MM, FEAT_DotProd, FEAT_SVE, FEAT_SVE2, and the
SME family) are detected via ``sysctl`` on macOS and ``getauxval`` on Linux;
NEON itself is the aarch64 baseline.

aarch64 has one optional rung, ``Sme`` (SME2 with FP64 outer products,
compiled with ``-march=armv8.6-a+sme2+sme-f64f64``). Its features do not
nest the way the x86 levels do: Apple M4 has SME but no non-streaming SVE.
So the ladder is not a ranking of enumerator values. ``supports()`` says
whether a machine can run a rung, and ``preference_order()`` lists each
architecture's rungs from most to least preferred (``V4, V3, V2, Baseline``
on x86, ``Sme, Baseline`` on aarch64). The ``Sme`` gate also requires
whatever else the rung's flags switch on for the compiler in use: GCC
before 15 makes ``+sme`` imply ``+sve2``, and a build with such a compiler
only selects ``Sme`` on a core that has SVE2 as well.

Overriding the rung
-------------------

Set :option:`--einsums:simd:arch`, or its environment variable
``EINSUMS_SIMD_ARCH``, to ``baseline``, ``v2``, ``v3``, ``v4``, ``sme``, or
one of the aliases ``sse2``/``sse4.2``/``avx2``/``avx512``/``sme2`` to force
another rung. This is the primary tool for
testing every rung of a dispatch ladder on one machine. An override can only
choose a rung the machine supports. Asking for one it cannot run logs a
warning and takes the next supported rung in the architecture's preference
order, so ``v4`` on an AVX2 machine gives ``v3``. A rung of another
architecture, such as ``v3`` on aarch64, is ignored with a warning. For test
suites, prefer ``einsums_add_simd_rung_tests()``: it wraps each per-rung
registration in the ``simd_rung_guard`` launcher, which turns an unsupported
rung into an honest ctest "Skipped" (exit 77) instead of a silent rerun at
another rung. The value is read once and cached; tests that need to exercise
the resolution logic itself should call ``resolve_arch()`` with explicit
arguments instead of setting the option.

Building a dispatch ladder
--------------------------

A kernel that should use the widest registers the CPU has is written once and
compiled several times, once per rung. ``einsums_add_simd_dispatch_sources``
(``cmake/Einsums_AddSIMDDispatch.cmake``) generates the per-rung translation
units: each includes the implementation file, compiles with that rung's
``-march`` flags, and defines ``EINSUMS_SIMD_ARCH_NS`` to a namespace of its
own. Because the SIMD headers key off compiler-defined macros, the same
source widens ``Vec<T>`` and every operation to each rung's register width.

.. code-block:: cmake

    einsums_add_simd_dispatch_sources(MyKernelRungs IMPL src/KernelImpl.cpp RUNGS baseline v2 v3 v4)
    target_sources(my_target PRIVATE ${MyKernelRungs})
    set_source_files_properties(src/KernelDispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${MyKernelRungs_DEFINITIONS}")

The implementation wraps its entry points in the rung namespace:

.. code-block:: cpp

    // KernelImpl.cpp: compiled once per rung.
    namespace mylib::EINSUMS_SIMD_ARCH_NS {
    void kernel(float const *x, float *y, std::size_t n) { /* Vec<float> code */ }
    }

and one arch-neutral file declares every copy that was built and picks one.
``EINSUMS_SIMD_FOR_EACH_BUILT_RUNG`` names each built namespace,
``EINSUMS_SIMD_LADDER`` expands to the five slots ``select()`` takes (with
``nullptr`` for rungs not built), and ``select()`` returns the slot of the rung
``selected_arch()`` chose:

.. code-block:: cpp

    // KernelDispatch.cpp: compiled once, at the ambient flags.
    namespace mylib {
    #define DECLARE(ns) namespace ns { void kernel(float const *, float *, std::size_t); }
    EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(DECLARE)
    #undef DECLARE

    void kernel(float const *x, float *y, std::size_t n) {
        using Fn = void (*)(float const *, float *, std::size_t);
        static Fn const fn = einsums::simd::select<Fn>(EINSUMS_SIMD_LADDER(kernel));
        fn(x, y, n);
    }
    }

Resolve the pointer once, as the function-local static does: the choice
cannot change while the program runs. The ``RuntimeDispatch`` example is a
complete, buildable version of this, and HPTT and PackedGemm are the library's
own consumers.

The whole mechanism sits behind ``EINSUMS_WITH_SIMD_DISPATCH`` (default ON).
When it is OFF, or when a compile-time pin is in effect (below), the helper
emits a single ``native`` rung compiled at the ambient flags. On aarch64 the
x86 rungs do not exist, so the ladder is that ``native`` rung plus ``sme``
when the caller asks for it.

Interaction with the compile-time pinning options: building with
``EINSUMS_SIMD_NATIVE_ARCH=ON`` or ``EINSUMS_SIMD_TARGET_CPU=<cpu>`` raises
the baseline of *every* SIMD consumer to that target, which makes the binary
non-portable and runtime dispatch pointless; use one approach or the other.

See the :ref:`API reference <modules_Einsums_SIMD_api>` of this module for more details.
