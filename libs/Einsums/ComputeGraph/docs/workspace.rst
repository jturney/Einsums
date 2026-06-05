.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

=========================================
Workspace and Deferred Tensor Allocation
=========================================

The Einsums ComputeGraph supports **deferred tensor allocation**: tensors are declared
with their shape and type, but memory is not allocated until optimization passes have
decided where and how to store the data. This enables transparent distributed computing
where the DistributionPlanningPass decides which tensors to partition across MPI ranks
before any memory is allocated.

Scoping Hierarchy
=================

Tensors are owned at three scoping levels:

.. code-block:: text

   Workspace  — cross-computation tensors (ERI, MO coefficients, basis data)
     Pipeline — cross-stage tensors (Fock matrix, density, amplitudes)
       Graph  — single-computation intermediates (temporaries, scratch buffers)

Each level owns its tensors and manages their lifetime. Inner scopes can reference
tensors from outer scopes.

Workspace
---------

A ``Workspace`` holds tensors that persist across multiple Pipelines — for example,
two-electron integrals computed once and reused by SCF, MP2, and CCSD:

.. code-block:: cpp

   cg::Workspace ws("h2o_calculation");

   // Declare workspace-level tensors (no data allocated yet)
   auto &eri = ws.declare_tensor<double, 4>("ERI", nao, nao, nao, nao);
   auto &C   = ws.declare_zero_tensor<double, 2>("C", nao, nmo);

   // Use in SCF pipeline
   cg::Pipeline scf("scf");
   scf.set_workspace(ws);
   // ... add stages using eri and C ...
   scf.apply(pm);
   scf.execute();
   // eri and C survive — reuse in MP2

   // Use in MP2 pipeline
   cg::Pipeline mp2("mp2");
   mp2.set_workspace(ws);
   // ... add stages using eri and C ...

Pipeline
--------

A ``Pipeline`` holds tensors shared across its stages but freed when the pipeline
is destroyed:

.. code-block:: cpp

   cg::Pipeline pipeline("scf");
   auto &F = pipeline.declare_zero_tensor<double, 2>("F", n, n);
   auto &D = pipeline.declare_zero_tensor<double, 2>("D", n, n);
   // F and D are available in setup, loop, and post-processing stages

Graph
-----

A ``Graph`` holds single-computation intermediates:

.. code-block:: cpp

   cg::Graph graph("fock_build");
   auto &tmp = graph.declare_tensor<double, 2>("tmp", n, n);
   // tmp is freed when graph is destroyed

Shell Tensors (Deferred Allocation)
====================================

When you call ``declare_tensor()``, it creates a **shell tensor**: a ``Tensor<T, Rank>``
object with valid dimensions and strides but no data storage. The tensor has a valid
memory address (for graph registration) and ``dim()`` / ``stride()`` work correctly,
but ``data()`` returns an invalid pointer until materialization.

.. code-block:: cpp

   auto &A = ws.declare_tensor<double, 2>("A", 1000, 1000);
   A.dim(0);              // Returns 1000 — works before materialization
   A.is_materialized();   // Returns false
   // A.data() — DO NOT call until after materialization

The ``MaterializationPass`` (included in ``create_default()``) inserts allocation
nodes into the graph that run during ``execute()``, allocating memory just before
each tensor is first used.

declare_tensor Variants
========================

All three scoping levels support the same API:

.. code-block:: cpp

   // No initialization — user fills the tensor manually
   auto &A = scope.declare_tensor<double, 2>("A", rows, cols);

   // Initialize to zero after allocation
   auto &B = scope.declare_zero_tensor<double, 2>("B", rows, cols);

   // Initialize with random values after allocation
   auto &C = scope.declare_random_tensor<double, 2>("C", rows, cols);

   // Initialize with a user-provided fill function (distribution-aware)
   auto &D = graph.declare_tensor_filled<double, 2>("D", Dim<2>{rows, cols},
       [&](auto& T) {
           auto [i0, i1] = T.range(0);  // Global range for this rank
           auto [j0, j1] = T.range(1);
           for (size_t i = i0; i < i1; i++)
               for (size_t j = j0; j < j1; j++)
                   T.global(i, j) = compute(i, j);
       });

The ``declare_tensor_filled`` variant is especially useful for distributed computing.
The fill lambda receives a tensor with ``range(dim)`` and ``global(indices...)``
methods that handle the distribution mapping automatically. See the
:doc:`distributed` documentation for a complete shell-batch ERI example.

The initialization happens as an ``Initialize`` graph node that runs during
``execute()`` after the ``Materialize`` node allocates storage.

How It Works
=============

The lifecycle with deferred allocation:

.. code-block:: text

   1. Declare tensors (shape only, no data)
   2. Capture operations (CaptureGuard scope)
   3. Apply passes:
      a. DistributionPlanning — decide replicate vs distribute per tensor
      b. MaterializationPass — insert Materialize + Initialize nodes
      c. GPU passes, communication passes...
      d. FreeInsertion — insert Free nodes after last consumer of intermediates
      e. MemoryPlanning — analyze peak memory (sees reduced lifetimes)
   4. Execute:
      a. Materialize node runs → allocates storage
      b. Initialize node runs → zeros/fills the tensor
      c. Compute nodes run → einsum, scale, etc.
      d. Free node runs → releases intermediate storage (large tensors only)

On **re-execution** (loops, Pipeline stages), the Materialize node detects
released storage and re-allocates automatically. Small intermediates (< 1 MB)
are kept alive to avoid alloc/free overhead.

Complete Example
=================

.. code-block:: cpp

   namespace cg = einsums::compute_graph;

   cg::Workspace ws("calculation");

   // Declare tensors — no memory allocated yet
   auto &eri = ws.declare_tensor<double, 4>("ERI", nao, nao, nao, nao);
   auto &C   = ws.declare_zero_tensor<double, 2>("C", nao, nmo);

   cg::Pipeline scf("scf");
   scf.set_workspace(ws);

   auto &F = scf.declare_zero_tensor<double, 2>("F", nao, nao);
   auto &D = scf.declare_zero_tensor<double, 2>("D", nao, nao);

   // Stage 1: Setup
   {
       auto &setup = scf.add_stage("setup");
       cg::CaptureGuard guard(setup);
       cg::custom("compute_eri", [&]() { fill_eri(eri); }, &eri);
       cg::custom("init_F", [&]() { F = H; }, &F);
   }

   // Stage 2: SCF loop
   {
       auto &loop = scf.add_loop("scf_iter", 100, convergence_check);
       cg::CaptureGuard guard(loop);
       cg::einsum("ijkl;kl->ij", 0.0, &F, 1.0, eri, D);
       // ... diagonalize, build D, etc.
   }

   // Optimize and execute
   auto pm = cg::PassManager::create_default();
   scf.apply(pm);   // DistributionPlanning + MaterializationPass run here
   scf.execute();    // Tensors allocated just-in-time during execution

Backward Compatibility
=======================

Existing code using ``create_random_tensor()`` and ``create_zero_tensor()`` continues
to work. These functions allocate immediately (not deferred). The ``declare_tensor()``
API is the new path for distributed-ready code.

``Graph::create_tensor()`` and ``Graph::create_zero_tensor()`` also continue to work
for graph-owned intermediates with immediate allocation.
