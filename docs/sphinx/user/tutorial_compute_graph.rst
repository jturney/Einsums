..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-compute-graph:

*****************************
Tutorial: Computation Graphs
*****************************

The ``ComputeGraph`` module lets you capture a sequence of tensor operations
into a graph, then execute and replay them. This is useful for iterative
algorithms where the same operations run many times with changing data.

Inspired by CUDA Graphs and PyTorch FX, the Einsums compute graph also supports
optimization passes, parallel execution, control flow, and profiler integration.

Setup
=====

.. code-block:: cpp

    #include <Einsums/ComputeGraph.hpp>
    #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
    #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

    using namespace einsums;
    using namespace einsums::index;
    namespace cg = einsums::compute_graph;

Basic Capture and Execute
=========================

Wrap operations in a ``CaptureGuard`` to record them into a graph:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 10, 5);
    auto B = create_random_tensor<double>("B", 5, 8);
    auto C = create_zero_tensor<double>("C", 10, 8);

    cg::Graph graph("matmul");
    {
        cg::CaptureGuard guard(graph);
        // Operations here are RECORDED, not executed
        cg::einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
    }

    // Execute the recorded operations
    graph.execute();   // C = A * B

    // Change A's data and replay
    A.set_all(1.0);
    graph.execute();   // C = A * B (with new A data)

Graph-Owned Tensors
===================

For intermediate tensors that shouldn't outlive the graph, use
``create_tensor()``:

.. code-block:: cpp

    cg::Graph graph("pipeline");
    auto &tmp = graph.create_zero_tensor<double, 2>("tmp", 10, 10);

    {
        cg::CaptureGuard guard(graph);
        cg::einsum(Indices{i, j}, &tmp, Indices{i, k}, A, Indices{k, j}, B);
        cg::scale(2.0, &tmp);
        cg::axpy(1.0, tmp, &C);
    }

    graph.execute();
    // tmp is destroyed when graph is destroyed

Available Operations
====================

All common tensor operations have graph-aware wrappers in the ``cg::`` namespace:

- **Tensor algebra**: ``einsum``, ``permute``, ``transpose``, ``element_transform``
- **BLAS**: ``gemm``, ``gemv``, ``ger``, ``dot``, ``scale``, ``axpy``, ``axpby``, ``direct_product``
- **LAPACK**: ``syev``, ``svd``, ``qr``, ``gesv``, ``invert``, ``det``, ``pow``

Some returning-form operations (``dot``, ``det``, ``norm``, ``svd``) cannot be
captured because they return new objects. Use them outside capture or use
alternative in-place forms.

Optimization Passes
===================

The graph can be optimized before execution:

.. code-block:: cpp

    // Apply all default optimizations
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

The default pipeline includes 11 passes:

- **ConstantFolding** -- Pre-compute constant subexpressions
- **ScaleAbsorption** -- Absorb ``scale`` into adjacent ``einsum`` prefactors
- **CSE** -- Common subexpression elimination
- **DeadNodeElimination** -- Remove unused computations
- **LoopInvariantHoisting** -- Hoist invariant ops out of loops
- **Reorder** -- Minimize peak memory via node reordering
- **MemoryPlanning** -- Analyze tensor lifetime intervals
- **ChainParenthesization** -- Optimal GEMM chain ordering
- **InplaceOptimization** -- Reuse buffers for in-place operations
- **GEMMBatching** -- Batch independent GEMM calls
- **PermuteFusion** -- Fuse adjacent permute operations

You can also apply a single pass:

.. code-block:: cpp

    auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();
    mem.print_report(std::cout);

Pipeline: Multi-Stage Workflows
================================

For iterative algorithms with convergence checks:

.. code-block:: cpp

    cg::Pipeline pipeline("scf");

    // Stage 1: Setup (runs once)
    pipeline.add_stage("setup", [&]() {
        cg::einsum(Indices{i, j}, &F, Indices{i, k}, H, Indices{k, j}, D);
    });

    // Stage 2: Iteration loop (runs until convergence or max iterations)
    pipeline.add_loop("iterate", 100,
        [&](size_t iter) { return std::abs(energy - energy_old) > 1e-8; },
        [&]() {
            cg::einsum(Indices{i, j}, &F, Indices{i, k}, H, Indices{k, j}, D);
            cg::syev(F, &eigvecs, &eigvals);
            // ... update density, energy ...
        }
    );

    pipeline.execute();

Control Flow
============

Conditionals and loops within a flat graph:

.. code-block:: cpp

    cg::Graph graph("adaptive");
    {
        cg::CaptureGuard guard(graph);

        cg::scale(0.5, &value);

        graph.add_conditional("check",
            [&]() { return value(0) > threshold; },
            [&]() { cg::scale(0.1, &value); },   // then branch
            [&]() { cg::scale(10.0, &value); }    // else branch
        );
    }

Profiling
=========

``execute()`` automatically integrates with the Einsums profiler. Each node
appears as a named region with timing data. Connect the imgui profile viewer
to see a real-time flame graph, and open View > Compute Graph to see the
node DAG.

.. code-block:: cpp

    graph.execute();

    // Print per-node timing
    graph.print_timing_report(std::cout);

    // Export graph as DOT for GraphViz
    std::ofstream f("graph.dot");
    graph.print_dot(f);

Rebinding Tensors
=================

Swap a tensor for a different one without re-capturing:

.. code-block:: cpp

    auto A1 = create_random_tensor<double>("A1", 5, 5);
    auto A2 = create_random_tensor<double>("A2", 5, 5);

    cg::Graph graph("rebind_demo");
    { /* capture with A1 */ }
    graph.execute();  // Uses A1

    graph.rebind(A1, A2);
    graph.execute();  // Now uses A2 (same graph structure)

Blueprints
==========

Pre-built operation sequences for common patterns:

.. code-block:: cpp

    namespace bp = cg::blueprints;

    // Symmetrize: A = 0.5 * (A + A^T)
    bp::symmetrize(&A);

    // Orthogonalization: X = S^{-1/2}
    bp::orthogonalize(&X, S);

    // Matrix exponential via Taylor series
    bp::matrix_exponential(&expA, A, 10);

Blueprints work both inside and outside captured blocks.

TaskPool Integration
====================

The TaskPool module provides fine-grained task parallelism that integrates
directly with ComputeGraph.

Graph-capturable parallel_for
-----------------------------

``cg::parallel_for()`` captures a data-parallel loop as a graph node.
The graph's topological sort ensures it runs before any node that reads
its output tensors:

.. code-block:: cpp

    #include <Einsums/TaskPool/TaskPool.hpp>

    cg::Graph graph("fock_build");
    {
        cg::CaptureGuard guard(graph);

        // Node 1: parallel_for computes integrals (via TaskPool workers)
        cg::parallel_for("integrals", 0, n_pairs,
            [&](size_t pair) { compute_integrals(pair, J, K); },
            &J, &K);  // Declare J and K as outputs

        // Node 2: assemble F = H + 2*J - K (depends on J and K)
        cg::permute(0.0, Indices{i,j}, &F, 1.0, Indices{i,j}, H);
        cg::axpy(2.0, J, &F);
        cg::axpy(-1.0, K, &F);

        // Node 3: compute energy (depends on F)
        cg::parallel_reduce<double>("energy", 0, N*N, &energy,
            []() { return 0.0; },
            [&](size_t idx, double &acc) { acc += D(idx) * F(idx); },
            [](double &g, double const &l) { g += l; },
            &D, &F);
    }

    // Execute: integrals -> assembly -> energy (automatic ordering)
    graph.execute();

    // Replay for next SCF iteration (same graph, different data)
    J.zero(); K.zero(); F.zero();
    graph.execute();

The tensor arguments at the end (``&J, &K`` for parallel_for; ``&D, &F``
for parallel_reduce) declare tensor dependencies for the graph's topological
sort. Without them, the graph cannot know what the body lambda accesses.

DataflowExecutor
----------------

For maximum overlap of independent graph nodes, use ``DataflowExecutor``:

.. code-block:: cpp

    cg::DataflowExecutor df;
    graph.execute(df);  // Independent nodes run concurrently via TaskPool

See the TaskPool documentation for full details on ``submit()``, ``then()``,
``when_all()``, ``dataflow()``, ``parallel_for()``, and ``parallel_reduce()``.

.. _tutorial-compute-graph-gpu:

GPU Offloading
==============

The ComputeGraph can automatically offload operations to the GPU. The user
writes standard CPU code; the optimization passes handle placement, data
transfers, and dispatch.

Requirements
------------

- Use ``float`` tensors (MPS only supports float32 GEMM; double stays on CPU)
- Tensors must be large enough to overcome GPU overhead (default: >64 KB)
- Use ``PassManager::create_default()`` to include GPU passes

.. code-block:: cpp

    // Write normal code with float tensors
    auto A = create_random_tensor<float>("A", 256, 256);
    auto B = create_random_tensor<float>("B", 256, 256);
    auto C = create_zero_tensor<float>("C", 256, 256);

    cg::Graph graph("my_computation");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    }

    // Apply all passes including GPU optimization
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    // Execute --- large float GEMMs run on GPU automatically
    graph.execute();

The default pass manager includes these GPU-related passes:

1. **GPUPlacement** --- decides which nodes run on GPU based on a cost model
   comparing estimated CPU time vs. GPU time + transfer overhead.

2. **TransferInsertion** --- adds HostToDevice / DeviceToHost nodes around GPU
   operations. Skips H2D for "dead" inputs (e.g., ``C`` in ``C = A*B`` when
   ``c_prefactor == 0``). Inserts final D2H for user-visible tensors left on
   device so the graph is self-contained.

3. **TransferElimination** --- removes redundant transfers when consecutive GPU
   nodes share a tensor. Uses Belady's optimal eviction under memory pressure.

4. **GPUDiagnostics** --- logs GPU vs. CPU node counts, transfer bytes, and
   peak device memory.

5. **StreamAssignment** --- tags transfer nodes with stream IDs for future async
   overlap.

GPU Pass Control
----------------

Disable GPU offloading at runtime:

.. code-block:: bash

    ./my_program --einsums:disable-gpu

Disable a specific pass:

.. code-block:: bash

    ./my_program --einsums:pass-disable=GPUPlacement

Or build a custom pass manager without GPU passes:

.. code-block:: cpp

    cg::PassManager pm;
    pm.add<cg::passes::ConstantFolding>();
    pm.add<cg::passes::ScaleAbsorption>();
    pm.add<cg::passes::CSE>();
    // No GPU passes
    graph.apply(pm);

Tuning the Cost Model
---------------------

The ``GPUPlacement`` pass uses a cost model to decide if an operation is worth
offloading. You can tune the parameters:

.. code-block:: cpp

    cg::passes::GPUPlacement placement;
    placement.cpu_throughput_gflops = 50.0;    // Your CPU's peak GFLOP/s
    placement.gpu_throughput_gflops = 5000.0;  // Your GPU's peak GFLOP/s
    placement.pcie_bandwidth_gbs   = 12.0;     // PCIe bandwidth (0 for unified memory)
    placement.gpu_launch_overhead_us = 10.0;   // Kernel launch latency

    cg::PassManager pm;
    pm.add(std::move(placement));
    // ... add other passes ...

Interaction with ConstantFolding
--------------------------------

The ``ConstantFolding`` pass detects tensors that are only ever read and pre-computes their results on CPU at
optimization time. This can prevent GPU placement because the Einsum node gets
replaced with a precomputed constant before ``GPUPlacement`` runs.

This is correct behavior for one-shot computations. In iterative algorithms
where tensor data changes between ``graph.execute()`` calls, ConstantFolding
does not apply because the tensors are mutable.

If you want to force GPU execution for benchmarking or demonstration, build
a custom ``PassManager`` that skips ``ConstantFolding``:

.. code-block:: cpp

    cg::PassManager pm;
    // Skip ConstantFolding — let GPU handle the computation
    pm.add<cg::passes::ScaleAbsorption>();
    pm.add<cg::passes::CSE>();
    pm.add<cg::passes::DeadNodeElimination>();
    pm.add<cg::passes::GPUPlacement>();
    pm.add<cg::passes::TransferInsertion>();
    pm.add<cg::passes::TransferElimination>();
    pm.add<cg::passes::GPUDiagnostics>();
    graph.apply(pm);

In real iterative code, this is not an issue
because the input tensors change between iterations and ConstantFolding cannot
fold them.

Unified Memory (Apple Silicon)
------------------------------

On Apple Silicon, the CPU and GPU share physical memory. The ComputeGraph
detects this via ``gpu::has_unified_memory`` and skips the actual H2D/D2H
``memcpy`` calls at execution time. The GPU reads tensor data directly from
host memory through zero-copy MTLBuffer wrappers.

The transfer nodes still exist in the graph structure, but their execution is a no-op on unified
memory. This means the graph is portable across backends without code changes.

Graph Visualization
-------------------

After applying GPU passes, ``print_dot()`` colors nodes by execution target:

- **Blue** nodes run on GPU
- **Orange** nodes are H2D/D2H transfers
- **White** nodes run on CPU

.. code-block:: cpp

    graph.print_dot(std::cout);  // Pipe to: dot -Tpng > graph.png

The JSON output (``graph.to_json()``) includes ``"target": "GPU"`` and
``"stream_id"`` fields for each node, and ``"residency"`` for each tensor.

Custom Operations and Disk I/O
==============================

Not every operation has a built-in graph wrapper. Use ``cg::custom()`` for
user-defined computations like integral evaluation, and ``cg::read()`` /
``cg::write()`` for disk I/O:

.. code-block:: cpp

    cg::Graph graph("scf_iteration");
    {
        cg::CaptureGuard guard(graph);

        // Load integrals from disk
        cg::read("load ERI", "integrals.h5", "/eri", &ERI, [&]() {
            einsums::read(ERI, h5file);
        });

        // Custom computation: build Fock matrix
        cg::custom("build_fock",
            std::tie(ERI, D),    // inputs
            std::tie(F),         // outputs
            [&]() { build_fock_matrix(ERI, D, F); });

        // Standard einsum
        cg::einsum(0.0, Indices{p,q}, &F_mo, 1.0,
                   Indices{p,i}, C, Indices{i,j}, F, Indices{j,q}, C);

        // Checkpoint to disk
        cg::write("save F_mo", "checkpoint.h5", "/fock_mo", &F_mo, [&]() {
            einsums::write(F_mo, h5file);
        });
    }

The graph tracks dependencies automatically: ``build_fock`` waits for
``load ERI``, the einsum waits for ``build_fock``, and the checkpoint
waits for the einsum. GPU passes will keep ``DiskRead``/``DiskWrite``
on CPU and insert appropriate transfers around GPU-placed operations.

Async I/O — Overlapping Reads with Compute
============================================

For I/O-bound workflows, use ``cg::read_async()`` / ``cg::write_async()``
to overlap disk I/O with independent computation. These accept three lambdas:
``start_fn`` (begins I/O), ``finish_fn`` (waits for completion), and
``sync_fn`` (synchronous fallback).

.. code-block:: cpp

    std::future<void> io_future;

    cg::Graph graph("prefetch_pipeline");
    {
        cg::CaptureGuard guard(graph);

        // Async read: start kicks off background I/O
        cg::read_async("load ERI", "integrals.h5", "/eri", &ERI,
            /*start*/  [&]() { io_future = std::async(std::launch::async,
                           [&]{ einsums::read(ERI, h5file); }); },
            /*finish*/ [&]() { io_future.get(); },
            /*sync*/   [&]() { einsums::read(ERI, h5file); }
        );

        // Independent computation — runs concurrently with the read
        cg::einsum(0.0, Indices{i,j}, &C, 1.0,
                   Indices{i,k}, A, Indices{k,j}, B);

        // Depends on ERI — waits for async read to finish
        cg::custom("build_fock", std::tie(ERI, D), std::tie(F),
            [&]() { build_fock_matrix(ERI, D, F); });
    }

    // IOPrefetch moves DiskRead to position 0, maximizing overlap window
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    // DataflowExecutor enables the overlap
    cg::DataflowExecutor df;
    graph.execute(df);

How it works
------------

1. ``IOPrefetch`` pass moves the ``DiskRead`` to the beginning of the schedule
2. ``DataflowExecutor`` calls ``async_start`` immediately
3. While I/O runs in the background, the executor runs independent compute (A*B)
4. When ``build_fock`` is ready to run, the executor calls ``async_finish``
   which blocks until the read completes
5. ``build_fock`` runs with the loaded data

``SequentialExecutor`` and ``OpenMPExecutor`` call the ``sync_fn`` fallback so that
no overlap occurs. Use ``DataflowExecutor`` for async I/O overlap.

See ``examples/AsyncIO.cpp`` for a complete working example.

Profiler Session Save
=====================

Save the complete profiling session (call tree, timing, annotations, and
ComputeGraph structure) to a JSON file for offline analysis:

.. code-block:: bash

    ./my_program --einsums:profile:save=session.json

The file is compatible with the EinsumsProfileViewer. Multiple runs
append to the same file, creating a multi-session comparison:

.. code-block:: bash

    # Run multiple times — sessions accumulate
    ./program --einsums:profile:save=runs.json
    ./program --einsums:profile:save=runs.json

    # Load all in the viewer
    ./EinsumsProfileViewer --load runs.json

The saved JSON includes:

- Aggregated profiling data (call tree, exclusive/inclusive times, annotations)
- ComputeGraph node structure (kind, target, stream, inputs/outputs, timing)
- Tensor metadata (name, dimensions, dtype, residency)
- Graph edges (data flow between operations)

Runtime Lifecycle
=================

The Einsums runtime automatically calls ``finalize()`` when ``einsums::start()``
returns. The shutdown sequence is:

1. **Profiler session save** --- if ``--einsums:profile:save`` is set
2. **Runtime destructor** --- runs pre-shutdown and shutdown functions
3. **Profiler shutdown** --- drains all events, writes text report if enabled
4. **Module cleanup** --- each module's finalize function runs

No explicit ``finalize()`` call is needed. The ``Runtime`` destructor handles
everything when the ``unique_ptr<Runtime>`` goes out of scope. Calling
``finalize()`` explicitly is safe but redundant.

Workspace and Deferred Allocation
==================================

For distributed-ready code, use ``declare_tensor()`` instead of ``create_*_tensor()``.
Tensors are declared with their shape but no data is allocated until the
``MaterializationPass`` runs during ``apply(pm)``. This enables the
``DistributionPlanningPass`` to decide tensor placement before allocation:

.. code-block:: cpp

    cg::Workspace ws("calculation");

    // No data allocated — just shape + metadata
    auto &eri = ws.declare_tensor<double, 4>("ERI", nao, nao, nao, nao);
    auto &C   = ws.declare_zero_tensor<double, 2>("C", nao, nmo);

    cg::Pipeline scf("scf");
    scf.set_workspace(ws);

    auto &F = scf.declare_zero_tensor<double, 2>("F", nao, nao);

    {
        auto &stage = scf.add_stage("compute");
        cg::CaptureGuard guard(stage);
        cg::einsum(0.0, Indices{i,j}, &F, 1.0, Indices{i,j,k,l}, eri, Indices{k,l}, D);
    }

    auto pm = cg::PassManager::create_default();
    scf.apply(pm);   // Distribution + Materialization happens here
    scf.execute();    // Tensors allocated just-in-time

Three scoping levels:

- **Workspace** — tensors shared across Pipelines (ERI, MO coefficients)
- **Pipeline** — tensors shared across stages (Fock, density)
- **Graph** — single-computation intermediates

See ``ComputeGraph/docs/workspace.rst`` for full details.

Distributed Computing
======================

When ``EINSUMS_WITH_MPI=ON``, the ``Comm`` module provides MPI-based
distributed computing. The user writes ``einsum(C, A, B)`` unchanged;
the ComputeGraph passes handle distribution and communication:

.. code-block:: cpp

    auto pm = cg::PassManager::create_default();
    scf.apply(pm);
    // DistributionPlanning: eri (100GB) → block-distributed across ranks
    //                       F, C (small) → replicated
    // MaterializationPass: each rank allocates its local eri slice + full F, C
    // CommunicationInsertion: allreduce after eri contraction

    scf.execute();
    // Each rank computes its portion. Communication handled automatically.

Without MPI, the mock backend runs everything on a single rank, meaning the
same code works in both serial and distributed modes.

See ``ComputeGraph/docs/distributed.rst`` for full details.

Hardware-Aware Optimization
============================

The ``ContractionPlanning`` pass uses a ``HardwareProfile`` database to make
architecture-specific decisions. The database includes 25+ pre-filled profiles
for common CPUs and GPUs, auto-detected at runtime:

.. code-block:: cpp

    // Automatic detection (runs inside create_default()):
    auto profile = HardwareProfile::detect_default();
    // → "Apple M4 Pro" on your machine

    // Or provide calibrated data:
    auto profile = HardwareProfile::load_json("calibrated.json");

Run the calibration tool for precise measurements:

.. code-block:: bash

    ./calibrate_hardware --output calibrated.json

See ``ComputeGraph/docs/hardware_profiles.rst`` for full details.

What's Next
===========

- :ref:`tutorial-performance` -- Understanding dispatch and profiler usage
- ``GPUOffload.cpp`` in the ComputeGraph examples --- full GPU offloading demo
- ``SCFSimulation.cpp`` --- iterative SCF with custom operations and loops
- ``AsyncIO.cpp`` --- async I/O overlap with DataflowExecutor
- The ``examples/`` directory in the ComputeGraph module contains full,
  compilable examples for each feature.
