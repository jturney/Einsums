..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums_docs_mainpage:

#######################
Einsums's Documentation
#######################

.. toctree::
   :maxdepth: 3
   :hidden:

   User Guide <user/index>
   Installation <building/index>
   API Reference <api/index>
   Developer Guide <contributors/index>
   Release Notes <changelogs/index>

**Version**: |release|

**Useful links**:
`Source Repository <https://github.com/Einsums/Einsums>`_ |
`Issue Tracker <https://github.com/Einsums/Einsums/issues>`_ |
`Discord <https://discord.gg/8GvtkyWZUv>`_ |
:doc:`License <license>`


Einsums is a C++23 tensor algebra library for scientific computing. It provides:

* A multidimensional :code:`Tensor` type with the usual algebraic operations,
  block-sparse and tiled variants, and a disk-backed tensor for out-of-core
  workloads.
* Compile-time contraction pattern analysis. An :code:`einsum` expression
  is statically routed to vendor BLAS, an in-tree BLIS-style packed
  contraction backend (:ref:`PackedGemm <modules_Einsums_PackedGemm>`), or
  a generic loop nest, which is chosen once when compiling your code with no runtime
  dispatch overhead.
* A deferred-execution :ref:`ComputeGraph <modules_Einsums_ComputeGraph>`
  with multi-pass optimization for whole-algorithm rewrites.
* :ref:`Python bindings <modules_Einsums_Python>` auto-generated from the
  C++ headers by an in-tree libclang tool, so the Python surface tracks the
  C++ surface without hand-written glue.

As a short example, the following :code:`einsum` call routes to a single
:code:`dgemm` because the contraction pattern matches a pure GEMM at
compile time:

.. code-block:: C++

   using namespace einsums;                 // Tensor, create_tensor, create_random_tensor
   using namespace einsums::tensor_algebra; // einsum
   using namespace einsums::index;          // i, j, k, Indices

   auto A = create_random_tensor("A", 7, 7);
   auto B = create_random_tensor("B", 7, 7);
   auto C = create_tensor("C", 7, 7);

   einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);


.. grid:: 2

   .. grid-item-card::
      :img-top: _static/index-images/getting_started.svg
      :class-img-top: icon

      Getting Started
      ^^^^^^^^^^^^^^^

      New to Einsums? Check out the Absolute Beginner's Guide. It contains
      an introduction to Einsums' main concepts and links to additional
      tutorials.

      +++

      .. button-ref:: user/absolute_beginners
         :expand:
         :color: primary
         :click-parent:

         To the beginner's guide

   .. grid-item-card::
      :img-top: _static/index-images/user_guide.svg
      :class-img-top: icon

      User's Guide
      ^^^^^^^^^^^^

      The user guide provides in-depth information on the key concepts of
      Einsums with useful background information and explanation.

      +++

      .. button-ref:: user/index
         :expand:
         :color: primary
         :click-parent:

         To the user's guide

   .. grid-item-card::
      :img-top: _static/index-images/api.svg
      :class-img-top: icon

      API Reference
      ^^^^^^^^^^^^^

      The reference for the C++ and Python APIs. Start here when you need
      to know what a particular class does, what symbols a header exposes,
      or which functions the Python package binds.

      +++

      .. button-ref:: api/index
         :expand:
         :color: primary
         :click-parent:

         To the API reference

   .. grid-item-card::
      :img-top: _static/index-images/contributor.svg
      :class-img-top: icon

      Developer Guide
      ^^^^^^^^^^^^^^^

      Want to add to the codebase? The developer guide covers the coding
      and documentation guidelines, the per-module internals, and the
      tooling for improving Einsums.

      +++

      .. button-ref:: contributors/index
         :expand:
         :color: primary
         :click-parent:

         To the developer guide

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`search`
