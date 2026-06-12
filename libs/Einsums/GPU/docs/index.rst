..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_GPU:

***
GPU
***

The ``GPU`` module provides a portable abstraction layer for GPU-accelerated
linear algebra. It supports four backends:

- CUDA: NVIDIA GPUs via cuBLAS/cuSOLVER.
- HIP: AMD GPUs via hipBLAS/hipSOLVER.
- MPS: Apple Silicon GPUs via Metal Performance Shaders, auto-detected on macOS.
- Mock: CPU fallback when no GPU is available.

The user writes backend-agnostic code, and the build system selects the
appropriate backend at configure time.

Backends
========

Metal Performance Shaders (MPS)
-------------------------------

On macOS with Apple Silicon, the MPS backend is enabled automatically. It uses
``MPSMatrixMultiplication`` for GEMM and ``MPSMatrixVectorMultiplication`` for
GEMV, running on the M-series GPU cores.

The MPS backend supports the following operations.

+------------------+------------------+-------------------------------------+
| Type             | GEMM             | Notes                               |
+==================+==================+=====================================+
| Float32          | GPU (MPS)        | Full support, all transpose combos  |
+------------------+------------------+-------------------------------------+
| Float16          | GPU (MPS)        | Native FP16 on Apple Silicon        |
+------------------+------------------+-------------------------------------+
| BFloat16         | GPU (via F32)    | Converts BF16->F32, runs MPS sgemm  |
+------------------+------------------+-------------------------------------+
| Float64          | CPU (Accelerate) | No GPU double precision on Apple    |
+------------------+------------------+-------------------------------------+
| Complex<float>   | CPU (Accelerate) | MPS complex GEMM not supported      |
+------------------+------------------+-------------------------------------+
| Complex<double>  | CPU (Accelerate) | No GPU double or complex            |
+------------------+------------------+-------------------------------------+

Apple Silicon shares physical memory between CPU and GPU. The ComputeGraph
inserts explicit H2D/D2H transfer nodes for correctness on all backends, but on
MPS the actual copies are skipped at execution time. The GPU reads tensor data
directly from host memory through zero-copy MTLBuffer wrappers. Final D2H nodes
ensure user-visible results are available after ``execute()`` on discrete GPUs
without relying on implicit flushes.

CUDA and HIP
------------

On systems with NVIDIA or AMD GPUs, the CUDA or HIP backend is selected.
These use discrete device memory with explicit H2D/D2H transfers managed
by the ComputeGraph pipeline.

Mock Backend
------------

When no GPU is available, all ``gpu::`` functions fall back to CPU
implementations (``std::malloc``, ``std::memcpy``, ``blas::vendor::*``).
This allows code using the GPU API to compile and run correctly on any
platform, making it useful for testing and development.

Runtime API
===========

.. code-block:: cpp

    #include <Einsums/GPU/Runtime.hpp>

    // Allocate device memory (MTLBuffer on MPS, cudaMalloc on CUDA)
    void *ptr = einsums::gpu::device_malloc(1024);

    // Copy host -> device
    einsums::gpu::memcpy_host_to_device(ptr, host_data, 1024);

    // Copy device -> host
    einsums::gpu::memcpy_device_to_host(host_data, ptr, 1024);

    // Free
    einsums::gpu::device_free(ptr);

    // Query available device memory
    size_t mem = einsums::gpu::available_device_memory();

BLAS API
========

.. code-block:: cpp

    #include <Einsums/GPU/BLAS.hpp>

    // Float32 GEMM on GPU
    einsums::gpu::blas::gemm<float>('n', 'n', M, N, K,
        1.0f, A, M, B, K, 0.0f, C, M);

    // Float32 GEMV on GPU
    einsums::gpu::blas::gemv<float>('n', M, N,
        1.0f, A, M, x, 1, 0.0f, y, 1);

    // Float16 GEMM (MPS native, CUDA via tensor cores)
    einsums::gpu::blas::hgemm('n', 'n', M, N, K,
        1.0f, A_fp16, M, B_fp16, K, 0.0f, C_fp32, M);

    // BFloat16 GEMM (BF16 inputs, FP32 output)
    einsums::gpu::blas::bfgemm('n', 'n', M, N, K,
        1.0f, A_bf16, M, B_bf16, K, 0.0f, C_fp32, M);

BLAS Level 1 (Element-wise)
----------------------------

These operations execute on device memory, avoiding unnecessary D2H/H2D
transfers when data is already on the GPU from a previous operation:

.. code-block:: cpp

    // Scale: x = alpha * x
    einsums::gpu::blas::scal<float>(n, 2.0f, x, 1);

    // Axpy: y = alpha * x + y
    einsums::gpu::blas::axpy<float>(n, 3.0f, x, 1, y, 1);

    // Axpby: y = alpha * x + beta * y
    einsums::gpu::blas::axpby<float>(n, 1.0f, x, 1, 0.5f, y, 1);

    // Dot product
    float d = einsums::gpu::blas::dot<float>(n, x, 1, y, 1);

    // L2 norm
    float norm = einsums::gpu::blas::nrm2<float>(n, x, 1);

The ComputeGraph automatically dispatches Scale and Axpy nodes through these
functions when the tensor is on GPU, avoiding round-trips to CPU.

Profiler Integration
====================

Save a profiler session to a JSON file that can be loaded in the
EinsumsProfileViewer:

.. code-block:: bash

    ./my_program --einsums:profile:save=session.json

The saved file includes profiling data (call tree, timing, annotations) and
ComputeGraph structure (nodes, edges, GPU placement). Multiple runs append
sessions to the same file for comparison. Load in the viewer:

.. code-block:: bash

    ./EinsumsProfileViewer --load session.json

Platform Detection
==================

.. code-block:: cpp

    #include <Einsums/GPU/Platform.hpp>

    if constexpr (einsums::gpu::has_mps) {
        // Running on Apple Silicon with MPS
    }
    if constexpr (einsums::gpu::has_unified_memory) {
        // No H2D/D2H copies needed
    }
    if constexpr (einsums::gpu::has_fp16_gemm) {
        // FP16 GEMM available (CUDA tensor cores or MPS)
    }

ComputeGraph Integration
========================

The GPU module integrates with the ComputeGraph to provide automatic GPU
offloading. Users write standard CPU tensor code; the graph optimization
passes decide what runs on the GPU:

.. code-block:: cpp

    #include <Einsums/ComputeGraph.hpp>

    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("my_graph");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // Apply GPU optimization passes
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    // Execute. GEMM runs on GPU automatically for large float tensors.
    graph.execute();

See :ref:`tutorial-compute-graph` for the full tutorial.

See the :ref:`API reference <modules_Einsums_GPU_api>` of this module for more details.
