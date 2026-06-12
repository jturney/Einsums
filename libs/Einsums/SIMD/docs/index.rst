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

See the :ref:`API reference <modules_Einsums_SIMD_api>` of this module for more details.
