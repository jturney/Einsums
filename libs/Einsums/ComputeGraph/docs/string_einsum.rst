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

**Single-character** (the default, used when there are no commas):

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

Permutation Operators
=====================

Coupled-cluster residuals are written with permutation operators, as in

.. math::

   P(ij)\,P(ab) \sum_{kc} t_{ik}^{ac}\,\langle kb\|cj\rangle

A spec may name them directly, prefixing the term (the right-hand side under
either arrow):

.. code-block:: cpp

   cg::einsum("i,j,a,b <- P(ij) P(ab) i,k,a,c ; k,c,j,b", 1.0, &r2, 1.0, t2, W);

``P`` partitions a set of OUTPUT index letters into two or more groups, separated
by ``/``, and expands to one signed term per coset of the Young subgroup:

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Spelling
     - Terms
     - Expansion
   * - ``P(i/j)``
     - 2
     - ``1 - (ij)``
   * - ``P(i/jk)``
     - 3
     - ``1 - (ij) - (ik)``
   * - ``P(ij/k)``
     - 3
     - ``1 - (ik) - (jk)``
   * - ``P(i/j/k)``
     - 6
     - the full antisymmetrizer over ``ijk``
   * - ``P(ij/kl)``
     - 6
     - ``1 - (ik) - (il) - (jk) - (jl) + (ik)(jl)``

Several operators in one spec multiply, so ``P(i/jk) P(a/bc)`` is the nine-term
antisymmetrizer of the triples correction. Each term carries the parity of its
permutation as a sign, and the prefactors apply once to the whole sum rather
than once per term.

In single-character mode ``P(ij)`` is accepted as shorthand for ``P(i/j)``. It is
rejected in multi-character mode, where ``P(mu,nu)`` cannot be told from one
group of two indices, and the groups must be written out as ``P(mu/nu)``.

Applying an operator to a sum
-----------------------------

An operator on an ``einsum`` wraps that one contraction. When the operator
applies to a SUM of contractions, accumulate the sum first and put the operator
on a :cpp:func:`cg::permute <einsums::compute_graph::permute>`:

.. code-block:: cpp

   // r2 += P(ij) P(ab) [ t2 . Wmbej - (t1 (x) t1) . <mb||ej> ]
   cg::einsum("i,j,a,b <- i,m,a,e ; m,b,e,j", 0.0, &tmp, 1.0, t2, Wmbej);
   cg::einsum("i,j,a,b <- i,m,e,a ; m,b,e,j", 1.0, &tmp, -1.0, t1t1, ovvo);
   cg::permute("i,j,a,b <- P(ij) P(ab) i,j,a,b", 1.0, &r2, 1.0, tmp);

Rules
-----

- Every letter an operator names must be an output index, appearing once. A
  contracted index has no output axis to reorder, so naming one is an error
  rather than a no-op.
- Groups within an operator are disjoint, and operators in one spec name
  disjoint letter sets.
- All axes an operator permutes must have equal extent.
- Tiled operands do not support operators and say so.

.. warning::

   A permutation operator is well-defined only on a term that is already
   antisymmetric within each group, which is what makes the notation meaningful
   in the coupled-cluster equations it comes from. A coset of the Young subgroup
   contains permutations of BOTH parities, so on a term without that symmetry the
   result depends on which representative is chosen. Einsums picks the one the
   literature prints, so ``P(i/jk) f`` is ``f(ijk) - f(jik) - f(kji)``.

The simultaneous pair permutation of the closed-shell equations, often written
``P(ia,jb)``, is a different operator: its sign is positive and it is not a coset
expansion. It is not supported, and a spec meaning it must be written out.

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
