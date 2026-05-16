.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

===============
Getting Started
===============

Overview
========

The ComputeGraph module lets you capture a sequence of Einsums operations into a
directed acyclic graph (DAG), optimize it with built-in passes, and then execute or
replay it. This is useful for:

- **Iterative algorithms** (SCF, CC): Replay the same operations hundreds of times
  without re-dispatch overhead.
- **One-shot optimization**: Apply fusion, CSE, and memory planning to a complex
  tensor network before executing it.
- **Profiling**: Get per-operation profiler annotations with tensor dimensions and
  contraction patterns.

All graph-aware operations live in the ``einsums::compute_graph`` namespace and
mirror the signatures of their eager counterparts in ``tensor_algebra`` and
``linear_algebra``.

Basic Workflow
==============

.. code-block:: cpp

   #include <Einsums/ComputeGraph/ComputeGraph.hpp>

   namespace cg = einsums::compute_graph;
   using namespace einsums;
   using namespace einsums::index;

   // 1. Create tensors
   auto A = create_random_tensor<double>("A", 10, 5);
   auto B = create_random_tensor<double>("B", 5, 8);

   // 2. Create a graph with an owned intermediate (inserts Alloc node)
   cg::Graph graph("my_graph");
   auto &C = graph.create_zero_tensor<double, 2>("C", 10, 8);

   // 3. Capture operations
   {
       cg::CaptureGuard guard(graph);
       cg::einsum("ik;kj->ij", &C, A, B);
   }

   // 4. (Optional) Optimize
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

   // 5. Execute
   graph.execute();

   // 6. Replay (for iterative algorithms)
   graph.execute();  // Same operations, same tensors

How Capture Works
=================

During capture (between ``CaptureGuard`` construction and destruction):

1. Each call to ``cg::einsum()``, ``cg::scale()``, ``cg::gemm()``, etc. checks
   ``CaptureContext::current().is_capturing()``.
2. If capturing, the operation is NOT executed. Instead, a ``Node`` is created
   containing:

   - A type-erased lambda (``std::function<void()>``) that captures the fully
     resolved template call
   - The operation kind (``OpKind::Einsum``, ``OpKind::Scale``, etc.)
   - Input/output tensor IDs for dependency tracking
   - Operation metadata (``EinsumDescriptor``, ``ScaleDescriptor``, etc.)

3. If not capturing, the operation executes immediately (zero overhead).

When the ``CaptureGuard`` is destroyed, the graph is topologically sorted.

**Exception**: Return-value operations (``syev(A)``, ``svd(A)``, ``qr(A)``, etc.)
always execute eagerly, even during capture. This is because subsequent captured
operations need the returned tensors to exist. They are recorded as nodes so
that on replay they re-execute.

Graph Inspection
================

.. code-block:: cpp

   // Text summary
   graph.print_summary(std::cout);
   // Output:
   //   Graph 'my_graph': 1 nodes, 3 tensors
   //     [0] einsum: C[i,j] = A[i,k] * B[k,j] (Einsum)
   //       inputs: A, B
   //       outputs: C

   // GraphViz DOT format (render with: dot -Tpng graph.dot -o graph.png)
   std::ofstream f("graph.dot");
   graph.print_dot(f);

Parallel Execution
==================

Use the OpenMP executor for automatic parallelism of independent nodes:

.. code-block:: cpp

   cg::OpenMPExecutor omp;
   graph.execute(omp);       // Independent nodes run in parallel

The executor automatically detects which nodes are independent based on
data dependencies — no user intervention needed.

String-Based Einsum
===================

Instead of compile-time index types, you can use string notation:

.. code-block:: cpp

   // Arrow notation
   cg::einsum("ij <- ik ; kj", &C, A, B);

   // NumPy notation
   cg::einsum("ik;kj -> ij", &C, A, B);

   // Multi-character indices
   cg::einsum("mu,nu <- mu,rho ; rho,nu", &C, A, B);

See :doc:`string_einsum` for full details.

Graph Rebind
============

Reuse a captured graph with different data without re-capturing:

.. code-block:: cpp

   graph.rebind(tensor_id, new_tensor);   // Swap tensor binding
   graph.update_prefactors(node_id, c_pf, ab_pf);  // Change prefactors

See :doc:`rebind` for full details.

Profiled Execution
==================

.. code-block:: cpp

   graph.execute();

This wraps each node in a profiler region with annotations:

- ``op_kind``: "Einsum", "Scale", "Gemm", etc.
- ``c_indices``, ``a_indices``, ``b_indices``: Contraction pattern (for einsum)
- ``c_prefactor``, ``ab_prefactor``: Scalar prefactors
- ``input.<name>``, ``output.<name>``: Tensor dimensions (e.g., "10x8")
- ``estimated_flops``: FLOP count (if set)
