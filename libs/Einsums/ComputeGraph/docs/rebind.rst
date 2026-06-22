.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

=========================
Graph Update / Rebind
=========================

The rebind feature lets you change which tensors a graph operates on and
update scalar prefactors, all without re-capturing the graph.

Tensor Rebind
=============

Capture a graph once, then rebind tensors to process different data:

.. code-block:: cpp

   auto A1 = create_random_tensor<double>("A1", 4, 3);
   auto A2 = create_random_tensor<double>("A2", 4, 3);
   auto B  = create_random_tensor<double>("B", 3, 5);
   auto C  = create_zero_tensor<double>("C", 4, 5);

   cg::Graph graph("rebind_demo");
   { cg::CaptureGuard g(graph); cg::einsum("ik;kj->ij", &C, A1, B); }

   graph.execute();  // Uses A1

   // Rebind to A2 — one line!
   graph.rebind(A1, A2);
   C.zero();
   graph.execute();  // Now uses A2!

Constraints
^^^^^^^^^^^

- New tensor must have the same rank and dimensions as the original
- Type must match (checked at compile time via template)
- Throws ``std::invalid_argument`` if dimensions don't match

Prefactor Update
================

Change einsum scalar prefactors without re-capturing:

.. code-block:: cpp

   cg::Graph graph("prefactor_demo");
   { cg::CaptureGuard g(graph);
     cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
   }

   graph.execute();  // C = A * B

   cg::NodeId einsum_id = graph.nodes()[0].id;
   graph.update_prefactors(einsum_id, 1.0, 2.0);  // Now: C = C + 2*A*B

   graph.execute();  // Accumulates!

Batch Processing with Rebind
=============================

Process multiple datasets with one captured graph:

.. code-block:: cpp

   std::vector<Tensor<double, 2>> inputs = {A1, A2, A3, A4};

   cg::Graph graph("batch");
   { cg::CaptureGuard g(graph); cg::einsum(..., inputs[0], ...); }

   for (size_t i = 1; i < inputs.size(); i++) {
       graph.rebind(inputs[i-1], inputs[i]);  // Swap old for new
       result.zero();
       graph.execute();
       // ... collect result ...
   }

How It Works
============

Under the hood, all operation wrappers capture ``TensorSlot*`` pointers
instead of direct tensor references. A TensorSlot is a stable indirection
that points to the current tensor:

1. During capture: ``cg::einsum()`` creates a ``TensorSlot`` for each tensor
2. The executor lambda dereferences the slot: ``*static_cast<T*>(slot->ptr)``
3. On ``rebind()``: the slot's ``ptr`` is updated to point to the new tensor
4. Next ``execute()``: the lambda sees the new tensor through the slot

For prefactors, einsum operations capture a ``shared_ptr<EinsumParams>``
that can be updated via ``update_prefactors()``.
