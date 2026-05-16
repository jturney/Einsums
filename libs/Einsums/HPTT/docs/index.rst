
..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_HPTT:

****
HPTT
****

High-Performance Tensor Transpose library, originally by Paul Springer.
Forked and refactored to use the Einsums SIMD module instead of raw compiler
intrinsics.

See the `original Github repo <https://github.com/springer13/hptt>`_ for the
upstream project. See ``THIRD-PARTY-LICENSES.txt`` for licensing information.

Overview
========

HPTT performs multi-dimensional tensor transpositions using a plan-based
execution model. The key abstraction is the **transpose plan**: a precomputed
strategy for permuting indices of an N-dimensional tensor, including blocking,
threading, and micro-kernel selection.

Supported Types
---------------

- ``float``, ``double`` --- full SIMD-accelerated micro-kernels
- ``std::complex<float>``, ``std::complex<double>`` --- supported with scalar
  micro-kernels (SIMD complex is future work)

How It Works
------------

1. **Plan creation:** ``create_plan()`` analyzes the tensor dimensions, strides,
   and permutation pattern to select an optimal blocking strategy and thread
   decomposition.

2. **Execution:** ``execute()`` runs the plan, performing the transpose using
   SIMD micro-kernels for the inner loops and OpenMP threads for parallelism.

.. code-block:: cpp

    #include <Einsums/HPTT/HPTT.hpp>

    // Transpose a 3D tensor: C[j,k,i] = alpha * A[i,j,k] + beta * C[j,k,i]
    int perm[] = {1, 2, 0};          // Permutation: dimension 0 -> position 2, etc.
    int size[] = {10, 20, 30};        // Dimensions of A
    float alpha = 1.0f, beta = 0.0f;

    auto plan = hptt::create_plan(perm, 3, alpha, A, size, nullptr,
                                   beta, C, nullptr, hptt::ESTIMATE, 4);
    plan->execute();

SIMD Refactoring
================

The HPTT micro-kernels were refactored from raw SSE/AVX/AVX-512/NEON intrinsics
to use the ``einsums::simd`` module. This provides:

- **Portability:** Same code runs on x86_64 (SSE2 through AVX-512) and ARM (NEON)
- **Maintainability:** ~30 lines of ``einsums::simd`` calls replace ~510 lines
  of raw intrinsics per type
- **Performance:** 10--15% improvement on Apple Silicon, substantial improvement
  for complex datatypes

The micro-kernels use:

- ``simd::loadu`` / ``simd::storeu`` for data movement
- ``simd::broadcast`` for alpha/beta scaling
- ``simd::fmadd`` for fused multiply-add
- ``simd::transpose_inplace`` for in-register 4x4/8x8/16x16 transposes
- ``simd::stream_store`` for non-temporal writes (cache-bypass for large transposes)
- ``simd::prefetch`` for prefetching next blocks

See the :ref:`API reference <modules_Einsums_HPTT_api>` of this module for more
details.
