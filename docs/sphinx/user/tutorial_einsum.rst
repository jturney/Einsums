..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-einsum:

******************************
Tutorial: Einstein Summation
******************************

``einsum`` is the reason this library exists. You write a contraction the way the equation writes
it, by naming the indices, and Einsums picks the kernel.

Setup
=====

.. code-block:: cpp

    #include <Einsums/ComputeGraph/Operations.hpp>
    #include <Einsums/Tensor/RuntimeTensor.hpp>
    #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
    #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

    namespace cg = einsums::compute_graph;
    using namespace einsums;

Index Notation
==============

A contraction is written as a spec: the indices of each operand, and the indices of the result.

.. code-block:: text

    "ik;kj->ij"
     ^^ ^^  ^^
     A  B   C

The two operands are separated by a semicolon, not by the comma NumPy uses. An index appearing in
both operands and not in the result is summed over. One appearing in the result is kept.

``"ij <- ik ; kj"`` is the same spec written the other way round, if that reads closer to the
mathematics you are transcribing. In C++ the spec is checked while your code compiles, so a
malformed one is a compile error rather than a surprise at run time.

Matrix Multiplication
=====================

:math:`C_{ij} = \sum_k A_{ik} B_{kj}`

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", {7, 7});
    auto B = create_random_tensor<double>("B", {7, 7});
    auto C = create_zero_tensor<double>("C", {7, 7});

    cg::einsum("ik;kj->ij", &C, A, B);

That call reaches a single vendor ``dgemm``. Nothing is copied and nothing is packed.

The output is an argument rather than a return value. That is what lets a contraction accumulate
into a result you already have, and what lets a captured graph plan its memory, since the
destination exists before the operation is recorded.

Dot Product
===========

A spec cannot produce a scalar: the string form needs an output of rank one or higher. For a
single number, use the dot product, which writes through a pointer:

.. code-block:: cpp

    auto u = create_random_tensor<double>("u", {100});
    auto v = create_random_tensor<double>("v", {100});

    double result = 0.0;
    cg::dot(&result, u, v);

The pointer-writing form is also the one to use inside a capture, where an operation has no value
to return until the graph runs.

Outer Product
=============

:math:`C_{ij} = u_i v_j`

.. code-block:: cpp

    auto C = create_zero_tensor<double>("C", {100, 100});
    cg::einsum("i;j->ij", &C, u, v);

No index is shared, so nothing is summed and the result is a rank-1 update.

Transpose
=========

A generalized transpose is ``permute`` rather than an einsum:

.. code-block:: cpp

    auto A  = create_random_tensor<double>("A", {5, 8});
    auto At = create_zero_tensor<double>("At", {8, 5});

    cg::permute("ij->ji", &At, A);

Einsums will not permute operands to force a contraction onto a faster kernel. It uses the
transposition flags a BLAS call already offers, so ``"ki;jk->ij"`` still reaches one ``GEMM``,
but a pattern needing a physical rearrangement runs on the generic loop instead. If you know a
permutation would pay, write it and contract the result.

Scaling with Prefactors
=======================

The six-argument form takes a prefactor for the output and one for the product, computing
:math:`C = \alpha C + \beta A B`:

.. code-block:: cpp

    cg::einsum("ik;kj->ij", &C, A, B);              // C = AB
    cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);    // C = C + AB
    cg::einsum("ik;kj->ij", 0.0, &C, 0.5, A, B);    // C = 0.5 AB

Accumulating is how you build a quantity from several contractions without a temporary for each
one, which is most of what a correlated method does.

Higher-Rank Contractions
========================

Rank is not special. :math:`C_{ijmn} = \sum_{kl} A_{ijkl} B_{klmn}`:

.. code-block:: cpp

    auto T = create_random_tensor<double>("T", {4, 4, 4, 4});
    auto U = create_random_tensor<double>("U", {4, 4, 4, 4});
    auto W = create_zero_tensor<double>("W", {4, 4, 4, 4});

    cg::einsum("ijkl;klmn->ijmn", &W, T, U);

This one does not map onto a plain ``GEMM``, so it runs on the packed contraction backend, which
blocks and packs the operands for cache reuse rather than falling back to loops. You write the
same call either way.

An index repeated **within one operand** is a diagonal rather than a contraction, and diagonals
are the one common pattern with no matrix-multiplication form. ``"ii;i->i"`` runs a loop. That is
worth knowing before you put one inside an iteration.

.. _tutorial-einsum-graph:

The Same Contractions, Captured
===============================

Every call above ran immediately. The same ``cg::einsum`` inside a capture is recorded instead,
and the recording is what the optimizer works on:

.. code-block:: cpp

    #include <Einsums/ComputeGraph/Graph.hpp>

    cg::Graph graph("two steps");

    auto &tmp = graph.create_runtime_tensor<double>("tmp", {7, 7});      // scratch
    auto &out = graph.create_runtime_tensor<double>("out", {7, 7},
                                                    /*intermediate=*/false);  // a result

    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &out, tmp, A);
    }

    graph.optimize();

    for (int iter = 0; iter < 100; ++iter) {
        graph.execute();      // no re-dispatch, no re-analysis
    }

Two things to notice. The function is the same one, so nothing about how you write a contraction
changes; capture is a property of where the call happens, not of which call it is. And ``tmp``
is scratch while ``out`` is a result, which the graph cannot work out for itself: a graph-owned
tensor you read after ``execute()`` needs ``intermediate=false``, or the optimizer removes the
contraction that fills it.

Capturing is worth it when the same operations repeat, which is every iterative method. A single
contraction gains nothing from being recorded.

Dispatch and Performance
========================

Einsums selects, in order: a vendor BLAS call when the contraction already is one, a permutation
followed by BLAS when the shape is right and the index order is not, the packed contraction
backend for higher-rank patterns with a valid decomposition, and a generic loop for the rest.

You do not choose. You can find out what was chosen: every ``einsum`` zone in a profile carries a
``dispatch`` annotation naming its route, which :doc:`tutorial_performance` covers along with the
measured route for every common shape.

What's Next
===========

- :ref:`tutorial-views` for contracting sub-blocks without copying them
- :ref:`tutorial-linalg` for decompositions and solves
- :ref:`tutorial-compute-graph` for control flow, pipelines and what the passes do
- :ref:`howto-contractions` for the recipes: diagonals, traces, conjugation, zero extents, aliasing
