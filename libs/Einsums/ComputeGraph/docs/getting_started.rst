.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

===============
Getting Started
===============

Overview
========

The ComputeGraph module records a sequence of Einsums operations as a directed
acyclic graph, optimizes that graph with built-in passes, and then executes or
replays it. Recording the work once and running it later pays off in three
common situations.

Iterative algorithms such as SCF and coupled cluster run the same sequence of
operations hundreds of times. Capturing the sequence once lets you replay it
without paying the dispatch cost on every iteration.

One-shot optimization applies fusion, common subexpression elimination, and
memory planning to a whole tensor network before it runs. An eager pass cannot
do this because it sees only one operation at a time.

Profiling becomes easier because every node carries profiler annotations that
record tensor dimensions and contraction patterns, so a captured graph is
straightforward to read in a trace.

Every graph-aware operation lives in the ``einsums::compute_graph`` namespace
and mirrors the signature of its eager counterpart in ``tensor_algebra`` and
``linear_algebra``. If you know the eager call, you already know the captured
call.

Basic Workflow
==============

.. code-block:: cpp

   #include <Einsums/ComputeGraph/ComputeGraph.hpp>

   namespace cg = einsums::compute_graph;
   using namespace einsums;
   using namespace einsums::index;

   // 1. Create the input tensors.
   auto A = create_random_tensor<double>("A", 10, 5);
   auto B = create_random_tensor<double>("B", 5, 8);

   // 2. Create a graph and an owned output tensor. This records an Alloc node.
   cg::Graph graph("my_graph");
   auto &C = graph.create_zero_tensor<double, 2>("C", 10, 8);

   // 3. Capture operations inside a guard.
   {
       cg::CaptureGuard guard(graph);
       cg::einsum("ik;kj->ij", &C, A, B);
   }

   // 4. Optimize the graph. This step is optional.
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

   // 5. Execute the graph.
   graph.execute();

   // 6. Replay it. The same operations run again on the same tensors.
   graph.execute();

How Capture Works
=================

Capture happens between the construction and destruction of a ``CaptureGuard``.
While the guard is alive, each call to ``cg::einsum()``, ``cg::scale()``,
``cg::gemm()``, and the other graph-aware operations checks
``CaptureContext::current().is_capturing()`` and then takes one of two paths.

When capture is active, the operation does not run. It creates a ``Node`` that
holds four things: a type-erased ``std::function<void()>`` that closes over the
fully resolved template call, the operation kind such as ``OpKind::Einsum`` or
``OpKind::Scale``, the input and output tensor ids used for dependency
tracking, and the operation metadata such as ``EinsumDescriptor`` or
``ScaleDescriptor``.

When capture is not active, the operation runs immediately and adds no overhead
over the eager path.

When the ``CaptureGuard`` is destroyed, the graph is topologically sorted so
that execution honors the recorded dependencies.

Return-value operations are the one exception. Calls such as ``syev(A)``,
``svd(A)``, and ``qr(A)`` return new tensors, and later captured operations
need those tensors to exist while the graph is still being built. These calls
therefore run eagerly even during capture. They are still recorded as nodes, so
a replay re-runs them.

Graph Inspection
================

.. code-block:: cpp

   // Text summary.
   graph.print_summary(std::cout);
   // Output:
   //   Graph 'my_graph': 1 nodes, 3 tensors
   //     [0] einsum: C[i,j] = A[i,k] * B[k,j] (Einsum)
   //       inputs: A, B
   //       outputs: C

   // GraphViz DOT format. Render it with: dot -Tpng graph.dot -o graph.png
   std::ofstream f("graph.dot");
   graph.print_dot(f);

Parallel Execution
==================

Pass an OpenMP executor to ``execute`` to run independent nodes in parallel.

.. code-block:: cpp

   cg::OpenMPExecutor omp;
   graph.execute(omp);       // Independent nodes run in parallel.

The executor decides which nodes are independent from the recorded data
dependencies, so you do not have to mark parallelism by hand.

String-Based Einsum
===================

You can name indices with a string instead of compile-time index types. Three
notations are accepted.

.. code-block:: cpp

   // Arrow notation.
   cg::einsum("ij <- ik ; kj", &C, A, B);

   // NumPy notation.
   cg::einsum("ik;kj -> ij", &C, A, B);

   // Multi-character indices.
   cg::einsum("mu,nu <- mu,rho ; rho,nu", &C, A, B);

See :doc:`string_einsum` for the full grammar.

Graph Rebind
============

You can reuse a captured graph with different data without capturing it again.

.. code-block:: cpp

   graph.rebind(tensor_id, new_tensor);            // Swap a tensor binding.
   graph.update_prefactors(node_id, c_pf, ab_pf);  // Change the scalar prefactors.

See :doc:`rebind` for the full interface.

Profiled Execution
==================

.. code-block:: cpp

   graph.execute();

Execution wraps each node in a profiler region. The annotations on that region
include the following.

- ``op_kind`` names the operation, for example Einsum, Scale, or Gemm.
- ``c_indices``, ``a_indices``, and ``b_indices`` give the contraction pattern
  for an einsum.
- ``c_prefactor`` and ``ab_prefactor`` give the scalar prefactors.
- ``input.<name>`` and ``output.<name>`` give the tensor dimensions, for
  example 10x8.
- ``estimated_flops`` gives the FLOP count when it has been set.
