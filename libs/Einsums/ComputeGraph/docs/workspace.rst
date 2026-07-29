.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _computegraph_workspace:

=========================================
Workspace and Deferred Tensor Allocation
=========================================

The Einsums ComputeGraph supports deferred tensor allocation: tensors are declared
with their shape and type, but memory is not allocated until optimization passes have
decided where and how to store the data. This is what lets ``DistributionPlanning``
decide which tensors to partition across MPI ranks before any memory is allocated,
and what lets ``MemoryPlanning`` place non-overlapping intermediates at planned
offsets in one shared arena.

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

A ``Workspace`` holds tensors that persist across multiple Pipelines, for example
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
   scf.run();
   // eri and C survive — reuse in MP2

   // Use in MP2 pipeline
   cg::Pipeline mp2("mp2");
   mp2.set_workspace(ws);
   // ... add stages using eri and C ...

``ws.size()`` reports how many tensors have been declared, and
``ws.materialize_all()`` allocates and initializes every deferred one at once.
``Pipeline::run()`` calls it for you.

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

A ``Graph`` holds single-computation intermediates. Use ``scratch()``:

.. code-block:: cpp

   cg::Graph graph("fock_build");
   auto &tmp = graph.scratch<double, 2>("tmp", n, n);
   auto &acc = graph.scratch_zero<double, 2>("acc", n, n);

``scratch()`` is the one call that replaces the whole
create/declare/intermediate-flag decision tree, and it is what you want for any
tensor that only exists to carry a value between nodes. A scratch tensor is both:

**deferred**
    No allocation until execution reaches it. ``Materialization``, or the graph's
    own lifecycle nodes, allocate it at the right position, possibly resized to a
    local partition by ``DistributionPlanning`` first.

**intermediate**
    ``FreeInsertion`` reclaims it after its last consumer,
    ``InplaceOptimization`` may merge its storage into a dying input, and
    ``MemoryPlanning``'s arena may host it at a planned offset.

``create_tensor()`` and ``create_zero_tensor()`` still exist and still work, but
they allocate eagerly and hold the memory for the graph's whole lifetime unless
the memory passes intervene. Prefer ``scratch()``.

Shell Tensors (Deferred Allocation)
====================================

When you call ``declare_tensor()`` or ``scratch()``, you get a shell tensor: a
``Tensor<T, Rank>`` object with valid dimensions and strides but no data storage.
The tensor has a valid memory address for graph registration, and ``dim()`` and
``stride()`` work correctly, but ``data()`` returns an invalid pointer until
materialization.

.. code-block:: cpp

   auto &A = ws.declare_tensor<double, 2>("A", 1000, 1000);
   A.dim(0);              // Returns 1000 — works before materialization
   A.is_materialized();   // Returns false
   // A.data() — DO NOT call until after materialization

The ``Materialization`` pass, included in the default pipeline from ``O1``
upwards, inserts allocation nodes into the graph that run during ``execute()``,
allocating memory just before each tensor is first used. It is in ``O1`` because
it is correctness-enabling rather than an optimization: a graph that uses
``declare_tensor()`` cannot execute without it.

Declaration Variants
====================

All three scoping levels support the same compile-time-rank API:

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

Runtime-rank variants
---------------------

The runtime-rank forms take a ``std::vector<size_t>`` of dimensions instead of a
parameter pack, and are the ones exposed to Python:

.. code-block:: cpp

   // Workspace / Pipeline scope
   auto &E = ws.declare_runtime_tensor<double>("E", {nao, nao, nao, nao});
   auto &F = ws.declare_zero_runtime_tensor<double>("F", {nao, nmo});
   auto &G = ws.declare_random_runtime_tensor<double>("G", {nmo, nmo});

   // Graph scope
   auto &t = graph.scratch_runtime<double>("t", {nocc, nvir});
   auto &u = graph.scratch_zero_runtime<double>("u", {nocc, nvir});

.. code-block:: python

   import einsums.graph as cg

   g = cg.Graph("residual")
   tmp = g.scratch("tmp", [nocc, nvir], dtype="float64")
   acc = g.scratch_zero("acc", [nocc, nvir], dtype="float64")

Tiled scratch
-------------

``Graph::declare_zero_tiled_tensor(name, tile_sizes, intermediate)`` declares a
graph-owned :cpp:struct:`~einsums::TiledRuntimeTensor` shell over a tile grid,
with the same deferred lifecycle. The shell starts with no populated tiles, and
ops create them on demand, so Materialize and Initialize are cheap; registering
the handle as deferred is what puts the tensor under the same lifecycle machinery
as dense scratch. See :doc:`tiled` for the details, including why
``MemoryPlanning``'s arena leaves tiled scratch alone.

How It Works
============

The lifecycle with deferred allocation:

.. code-block:: text

   1. Declare tensors (shape only, no data)
   2. Capture operations (CaptureGuard scope)
   3. Apply passes:
      a. TiledExpansion       — lower tiled ops so the tiles are visible below
      b. cleanup + rewrites   — folding, CSE, fusion, hoisting, privatization
      c. ContractionPlanning  — may DECLARE new deferred intermediates
      d. DistributionPlanning — decide replicate vs distribute per tensor
      e. Materialization      — insert Materialize + Initialize nodes
      f. InplaceOptimization  — merge an output buffer into a dying input
      g. FreeInsertion        — insert Free nodes after last consumer
      h. MemoryPlanning       — liveness analysis + place intermediates in the arena
   4. Execute:
      a. Materialize node runs → allocates storage (or binds an arena offset)
      b. Initialize node runs → zeros/fills the tensor
      c. Compute nodes run → einsum, scale, etc.
      d. Free node runs → releases intermediate storage (large tensors only)

``ContractionPlanning`` is in the list for a reason: it rewrites GEMM chains and
declares the intermediates it introduces as deferred, so it has to run before
``DistributionPlanning`` and ``Materialization`` size and allocate them.

On re-execution, as in loops and Pipeline stages, the Materialize node detects
released storage and re-allocates automatically. Small intermediates below 1 MB
are kept alive to avoid alloc/free overhead.

The memory arena
================

``MemoryPlanning`` is the last pass in the default pipeline, and with its default
``apply_arena`` setting it does more than report. Non-overlapping graph-owned
intermediates receive first-fit-decreasing offsets in one shared, 64-byte-aligned
arena per graph, sized to the peak. Two intermediates whose lifetimes do not
overlap then share the same bytes instead of each holding their own allocation:

.. code-block:: cpp

   graph.optimize();
   std::cout << graph.explain();
   //   - MemoryPlanning: peak 184.20 MB of 512.75 MB total
   //       arena: 184.20 MB hosting 31 intermediate(s) (512.75 MB of buffers)

A tensor is arena-placed only if it is a materialized graph-owned intermediate,
not viewed or aliased, with ``materialize_into_fn`` and ``release_fn`` and a
nonzero byte size. Everything else keeps its own allocation.

Pass ``apply_arena=false`` to get the statistics without the placement:

.. code-block:: cpp

   pm.add<cg::passes::MemoryPlanning>(/*apply_arena=*/false);

Complete Example
================

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
   scf.add_stage("setup", [&] {
       cg::custom("compute_eri", [&]() { fill_eri(eri); }, &eri);
       cg::custom("init_F", [&]() { F = H; }, &F);
   });

   // Stage 2: SCF loop
   scf.add_loop("scf_iter", 100, convergence_check, [&] {
       cg::einsum("ij <- ijkl ; kl", &F, eri, D);
       // ... diagonalize, build D, etc.
   });

   // Optimize, materialize the workspace, and execute in one call.
   scf.run();

Backward Compatibility
======================

Existing code using ``create_random_tensor()`` and ``create_zero_tensor()`` continues
to work. These functions allocate immediately, not deferred. ``Graph::create_tensor()``
and ``Graph::create_zero_tensor()`` also continue to work for graph-owned
intermediates with immediate allocation. For new code, ``scratch()`` at graph
scope and ``declare_tensor()`` at pipeline or workspace scope are the paths that
the memory and distribution passes can act on.
