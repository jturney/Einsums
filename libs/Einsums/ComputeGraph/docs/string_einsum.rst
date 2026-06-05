.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

====================
String-Based Einsum
====================

The string-based einsum API lets you specify contraction patterns as strings
instead of compile-time index types. This is more concise and enables
Python interoperability.

Notation
========

Two styles are supported (auto-detected):

**Arrow notation** (output on left):

.. code-block:: cpp

   cg::einsum("ij <- ik ; kj", &C, A, B);   // C = A * B

**NumPy notation** (output on right):

.. code-block:: cpp

   cg::einsum("ik;kj -> ij", &C, A, B);     // C = A * B

Rules:

- ``<-`` or ``->`` separates output from inputs
- ``;`` separates the two input operands
- Whitespace is ignored

Index Modes
===========

**Single-character** (default — no commas):

.. code-block:: cpp

   cg::einsum("ij <- ik ; kj", &C, A, B);

Each character is one index: ``i``, ``j``, ``k``.

**Multi-character** (commas present):

.. code-block:: cpp

   cg::einsum("mu,nu <- mu,rho ; rho,nu", &C, A, B);

Commas separate index names. Supports Greek letters, numbered indices, words.

Supported Patterns
==================

The string dispatch handles all common patterns:

.. code-block:: cpp

   // GEMM: matrix × matrix
   cg::einsum("ij <- ik ; kj", &C, A, B);

   // GEMV: matrix × vector
   cg::einsum("i <- ik ; k", &y, A, x);

   // GER: outer product
   cg::einsum("ij <- i ; j", &C, x, y);

   // DOT: scalar output
   cg::einsum(" <- i ; i", &result, x, y);

   // Direct product: element-wise
   cg::einsum("ij <- ij ; ij", &C, A, B);

   // Higher-rank contractions (rank 3+)
   cg::einsum("il <- ijk ; jkl", &C, A, B);
   cg::einsum("ijkl <- ijp ; klp", &C, A, B);

Prefactors
==========

.. code-block:: cpp

   // C = beta * C + alpha * A * B
   cg::einsum("ij <- ik ; kj", beta, &C, alpha, A, B);

   // Default: beta=0, alpha=1
   cg::einsum("ij <- ik ; kj", &C, A, B);

Compile-Time Validation
========================

String literals are validated at compile time via ``EinsumFormatString``:

.. code-block:: cpp

   cg::einsum("ij <- ik ; kj", &C, A, B);   // OK — validated at compile time
   cg::einsum("ij <- ik", &C, A, B);         // COMPILE ERROR: missing ';'

For runtime-constructed strings (e.g., from Python):

.. code-block:: cpp

   std::string spec = build_spec_from_python();
   cg::einsum(cg::EinsumFormatString(spec), &C, A, B);  // Validated at runtime

Mixing with Template-Based Einsum
==================================

Both styles work in the same graph:

.. code-block:: cpp

   cg::Graph graph("mixed");
   { cg::CaptureGuard g(graph);
     cg::einsum("ij <- ik ; kj", &T, A, B);                              // String
     cg::einsum("ik;kj->ij", &C, T, D);    // Template
   }
