..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-contractions:

************
Contractions
************

Recipes for writing a contraction and for the index patterns that are easy to get wrong.
:doc:`/user/tutorial_einsum` introduces the notation; this page assumes you have it and answers specific questions.

.. contents:: On this page
    :local:
    :depth: 1

Write the spec
==============

A spec names the indices of each operand and of the result.
The two operands are separated by a semicolon, which is the one piece of syntax that catches people arriving from NumPy.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Operations.hpp>
            #include <Einsums/Tensor/RuntimeTensor.hpp>

            namespace cg = einsums::compute_graph;
            using einsums::RuntimeTensor;

            RuntimeTensor<double> A("A", {m, k});
            RuntimeTensor<double> B("B", {k, n});
            RuntimeTensor<double> C("C", {m, n});

            cg::einsum("ik;kj->ij", &C, A, B);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums

            A = einsums.create_random_tensor("A", [m, k])
            B = einsums.create_random_tensor("B", [k, n])
            C = einsums.zeros([m, n])

            einsums.einsum("ik;kj->ij", C, A, B)

Both languages also accept the arrow-left spelling, ``"ij <- ik ; kj"``, which some people find reads closer to the mathematics.
Whitespace around the separators is ignored.

In C++ the spec is parsed by a ``consteval`` constructor, so a string literal that is not a valid spec fails to compile:

.. code-block:: text

    error: call to consteval function 'EinsumFormatString' is not a constant expression
    note: subexpression not valid in a constant expression
        throw "Invalid einsum format string: must contain exactly one '<-' or '->' and exactly one ';'";

That is the error you get from writing NumPy's comma.
In Python the same mistake is a ``ValueError`` when the call runs.

Scale the result, or accumulate into it
=======================================

The four-argument form overwrites the output.
The six-argument form takes a prefactor for the output and a prefactor for the product, computing
:math:`C = \alpha C + \beta A B`.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            cg::einsum("ik;kj->ij", &C, A, B);              // C = AB
            cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);    // C = C + AB
            cg::einsum("ik;kj->ij", 2.0, &C, 3.0, A, B);    // C = 2C + 3AB

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            einsums.einsum("ik;kj->ij", C, A, B)                        # C = AB
            einsums.einsum("ik;kj->ij", C, A, B, c_pf=1.0, ab_pf=1.0)   # C = C + AB
            einsums.einsum("ik;kj->ij", C, A, B, c_pf=2.0, ab_pf=3.0)   # C = 2C + 3AB

Accumulating is how you build a quantity from several contractions without allocating a temporary for each one.
For complex tensors the prefactors may themselves be complex, and they survive capture and replay without narrowing.

Get a single number out
=======================

A spec with an empty output is not the way to reach a scalar: the string form requires an output of rank one or higher.
Use the dot product, which writes through a pointer.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            double result = 0.0;
            cg::dot(&result, A, B);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            result = einsums.linalg.dot(A, B)

The pointer-writing form is the one to use inside a graph capture.
Returning forms such as ``dot(A, B)`` throw during capture by design, because a captured operation has no value to return until the graph runs.

Repeated indices, traces, and transposes
========================================

A letter repeated within one operand means diagonal access.
``"ii;i->i"`` reads the diagonal of a matrix and multiplies it elementwise by a vector.
A letter that appears in only one input and not in the output is summed over, which is how you write a trace.

Both of these leave the BLAS fast paths and run on the repeat-aware generic loop, which is correct but slower than a contraction that maps onto a matrix multiplication.
That is worth knowing before you put one inside an iteration.

A generalized transpose is ``cg::permute`` rather than an einsum:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            cg::permute("ij->ji", &T, A);        // T = A transposed

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            einsums.permute("ij->ji", T, A)

Einsums does not transpose operands to force a contraction into a BLAS call.
It will use the transposition flags a BLAS call already offers, so ``"ji;kj->ik"`` still reaches a single GEMM, but a pattern that would need a physical permutation runs on the generic algorithm instead.
If you know a permutation would pay for itself, do it explicitly with ``permute`` and contract the result.

Conjugate a complex operand
===========================

Conjugation is a flag on the call rather than something spelled in the spec.
It conjugates elements only, so a conjugate transpose is the flag combined with transposed index placement.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            using cd = std::complex<double>;
            cg::einsum("ik;kj->ij", cd(0, 0), &C, cd(1, 0), Z, I,
                       /*conj_a=*/true, /*conj_b=*/false);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            einsums.einsum("ik;kj->ij", C, Z, I, conj_a=True)

For real dtypes the flags are accepted and do nothing, so a routine written once for both does not need two code paths.

Write into a tensor you are also reading
========================================

An output that overlaps an input is rejected unless the aliased operand's index list is identical to the output's.
The identical case is a pure elementwise update in place, which is well defined; anything else would read elements the same call has already overwritten.

.. code-block:: cpp

    cg::einsum("ij;ij->ij", &D, D, B);     // allowed: elementwise, in place
    cg::einsum("ik;kj->ij", &A, A, B);     // throws std::invalid_argument

The check fires only on provable overlap, meaning both regions are contiguous or the base pointers are identical.
Two disjoint column-major slices of one parent tensor interleave in memory without sharing any element, so they are not rejected and must not be.

The two input operands may always share a tensor.
``cg::einsum("ik;kj->ij", &C, A, A)`` is fine.

Contract tensors with a zero-length dimension
=============================================

A zero extent is valid input, not an error to guard against.
An empty contraction contributes nothing, and the output is still scaled by its own prefactor exactly once:

.. code-block:: cpp

    auto Z  = /* 4 x 0 */;
    auto Z2 = /* 0 x 4 */;
    C.set_all(7.0);
    cg::einsum("ik;kj->ij", 2.0, &C, 1.0, Z, Z2);
    // C is now filled with 14.0, not 7.0 and not 0.0

This matters most when extents come from a screening step that can legitimately select nothing.
Code that special-cases the empty result usually gets the prefactor wrong.

Find out which kernel ran
=========================

``einsum`` chooses between a BLAS call, the packed-GEMM backend, and a generic loop.
:cpp:func:`~einsums::compute_graph::dispatch::last_dispatch_route` names the route the most recent string contraction took on this thread.

.. code-block:: cpp

    #include <Einsums/ComputeGraph/StringDispatch.hpp>

    cg::einsum("ik;kj->ij", &C, A, B);
    std::printf("%s\n", cg::dispatch::last_dispatch_route());   // gemm_direct_runtime

These are the routes for the common shapes, as measured with :cpp:type:`einsums::RuntimeTensor` operands:

.. list-table::
    :header-rows: 1
    :widths: 30 40 30

    * - Spec
      - Route
      - What ran
    * - ``ik;kj->ij``
      - ``gemm_direct_runtime``
      - one GEMM
    * - ``ji;kj->ik``
      - ``gemm_direct_runtime``
      - one GEMM, using its transpose flags
    * - ``ij;j->i``
      - ``gemv_mat_vec_runtime``
      - one GEMV
    * - ``j;ij->i``
      - ``gemv_vec_mat_runtime``
      - one GEMV, transposed
    * - ``i;j->ij``
      - ``ger_runtime``
      - rank-1 update
    * - ``ij;ij->ij``
      - ``direct_product_runtime``
      - elementwise
    * - ``ii;i->i``
      - ``generic_loop_repeated_indices``
      - repeat-aware loop
    * - ``ijkl;klmn->ijmn``
      - ``packed_gemm``
      - cache-blocked packed contraction

Statically ranked operands produce the same names without the ``_runtime`` suffix.
Contractions with a zero extent report ``empty_input_scale_only`` or ``empty_output_noop``.

Two cautions.
The value is thread-local and is written only by string contractions, so reading it after a call that is not one, such as ``cg::dot``, returns whatever the previous contraction left behind.
And it exists for test introspection rather than for steering execution: assert on it in a test that means to pin a fast path, and do not branch on it in a calculation.

For where the time actually goes, rather than which kernel was picked, see :doc:`/user/tutorial_performance`.

Next
====

- :ref:`howto-views` for contracting sub-blocks without copying them.
- :ref:`howto-graphs` for capturing a sequence of contractions and optimizing it as a unit.
- :doc:`/user/tutorial_best_practices` for structuring code so the time goes to arithmetic.
