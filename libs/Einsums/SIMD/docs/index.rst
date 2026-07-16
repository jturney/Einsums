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

    // The dispatch rung for this process: Baseline, V2, V3, or V4.
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
(FEAT_FP16, FEAT_BF16, FEAT_I8MM, FEAT_DotProd) are detected via ``sysctl``
on macOS and ``getauxval`` on Linux; NEON itself is the aarch64 baseline.

Overriding the rung
-------------------

Set the ``EINSUMS_SIMD_ARCH`` environment variable (``baseline``, ``v2``,
``v3``, ``v4``, or the aliases ``sse2``/``sse4.2``/``avx2``/``avx512``)
before process start to force a lower rung - the primary tool for testing
every rung of a dispatch ladder on one machine. An override can only lower
the selection; requesting more than the hardware supports logs a warning and
clamps. For test suites, prefer ``einsums_add_simd_rung_tests()``: it wraps
each per-rung registration in the ``simd_rung_guard`` launcher, which turns
an unsupported rung into an honest ctest "Skipped" (exit 77) instead of a
silently clamped rerun. The value is read once and cached; tests that need to exercise the
resolution logic itself should call ``resolve_arch()`` with explicit
arguments instead of mutating the environment.

Building a dispatch ladder
--------------------------

``select()`` picks the best entry point at or below the selected rung,
falling through rungs a module chose not to build:

.. code-block:: cpp

    using KernelFn = void (*)(float const *, float *, std::size_t);

    // One namespace per compiled rung; nullptr for rungs not built.
    static KernelFn const kernel = select<KernelFn>(
        &arch_baseline::kernel,   // required
        nullptr,                  // no dedicated v2 build of this kernel
        &arch_v3::kernel,
        &arch_v4::kernel);

The intended pattern for kernel authors is to compile one implementation
file several times, once per rung, each compilation wrapped in a distinct
namespace and given the matching ``-march`` flag, then bridge the copies
with ``select()`` at the call site. Because the SIMD headers key off
compiler-defined macros (``__AVX2__`` and friends), the same source
automatically widens ``Vec<T>``, ``native_lanes`` and every operation to
each rung's register width - no source changes per rung. CMake helpers that
generate the per-rung translation units live with this module (see
``Einsums_AddSIMDDispatch.cmake``); HPTT is the reference consumer: its
``Transpose.cpp`` compiles once per rung, and the arch-neutral factory in
``TransposeFactory.cpp`` selects a rung at plan creation.

The whole mechanism sits behind ``EINSUMS_WITH_SIMD_DISPATCH`` (default ON).
When it is OFF, on non-x86 targets, or when a compile-time pin is in effect
(below), the helper emits a single ``native`` rung compiled at the ambient
flags - exactly the pre-dispatch behavior.

Interaction with the compile-time pinning options: building with
``EINSUMS_SIMD_NATIVE_ARCH=ON`` or ``EINSUMS_SIMD_TARGET_CPU=<cpu>`` raises
the baseline of *every* SIMD consumer to that target, which makes the binary
non-portable and runtime dispatch pointless; use one approach or the other.

See the :ref:`API reference <modules_Einsums_SIMD_api>` of this module for more details.
