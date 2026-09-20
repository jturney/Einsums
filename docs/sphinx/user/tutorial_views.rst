..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-views:

*************************************
Tutorial: Views and Slicing
*************************************

A view is a non-owning window into an existing tensor. Views let you work with sub-blocks, rows
and columns without copying anything, which is what makes splitting a matrix into orbital blocks
free rather than a set of allocations.

Setup
=====

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Operations.hpp>
            #include <Einsums/Tensor/RuntimeTensor.hpp>
            #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
            #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

            namespace cg = einsums::compute_graph;
            using namespace einsums;

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import numpy as np
            import einsums

Creating Views with Range
=========================

Take a half-open interval of a dimension. C++ spells it with
:cpp:struct:`einsums::Range`; Python uses ordinary slice syntax.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = create_random_tensor<double>("A", {10, 10});

            // Top-left 3x3 sub-block
            auto block = A(Range{0, 3}, Range{0, 3});
            // no data is copied

            // Writing through the view writes through to A
            block(0, 0) = 999.0;
            // A(0, 0) is now 999.0

            // A single integer drops that dimension, so a row view is rank 1
            auto row = A(5, All);      // 10 elements
            auto col = A(All, 3);      // 10 elements

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [10, 10])

            # Top-left 3x3 sub-block
            block = A[0:3, 0:3]
            # no data is copied

            # Writing through the view writes through to A
            block[0, 0] = 999.0
            # A[0, 0] is now 999.0

            # A single integer drops that dimension, so a row view is rank 1
            row = A[5, :]      # 10 elements
            col = A[:, 3]      # 10 elements

Writing through a view writes through to the parent. That is the point of it, and it is the
thing to remember when handing one to a routine that writes its output.

Occupied and Virtual Blocks
===========================

This is the slicing pattern most quantum chemistry code is made of:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            size_t n_occ  = 5;
            size_t n_virt = 15;
            size_t n_orbs = n_occ + n_virt;

            auto F = create_random_tensor<double>("Fock", {n_orbs, n_orbs});

            auto Foo = F(Range{0, n_occ},      Range{0, n_occ});
            auto Fov = F(Range{0, n_occ},      Range{n_occ, n_orbs});
            auto Fvv = F(Range{n_occ, n_orbs}, Range{n_occ, n_orbs});

            // Fov(i, a) is F(i, n_occ + a)

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            n_occ, n_virt = 5, 15
            n_orbs = n_occ + n_virt

            F = einsums.create_random_tensor("Fock", [n_orbs, n_orbs])

            Foo = F[:n_occ, :n_occ]
            Fov = F[:n_occ, n_occ:]
            Fvv = F[n_occ:, n_occ:]

            # Fov[i, a] is F[i, n_occ + a]

Forming these costs nothing. The alternative, copying each block into its own tensor, costs
three allocations and a full traversal, and then leaves you maintaining the copies.

Views as Operands
=================

A view is an ordinary operand. Contracting one needs no copy and no special call:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto C  = create_random_tensor<double>("C", {20, 20});
            auto Co = C(Range{0, 20}, Range{0, 5});    // 20 x 5  occupied MOs
            auto Cv = C(Range{0, 20}, Range{5, 20});   // 20 x 15 virtual MOs

            auto AO_ints = create_random_tensor<double>("ints", {20, 20});
            auto MO_ints = create_zero_tensor<double>("MO_ints", {5, 15});

            // MO_ints_ia = Co^T * AO_ints * Cv, through one intermediate
            auto tmp = create_zero_tensor<double>("tmp", {5, 20});
            cg::einsum("ki;kj->ij", &tmp, Co, AO_ints);
            cg::einsum("ik;kj->ij", &MO_ints, tmp, Cv);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            C  = einsums.create_random_tensor("C", [20, 20])
            Co = C[:, 0:5]     # 20 x 5  occupied MOs
            Cv = C[:, 5:20]    # 20 x 15 virtual MOs

            AO_ints = einsums.create_random_tensor("ints", [20, 20])
            MO_ints = einsums.zeros([5, 15], name="MO_ints")

            # MO_ints_ia = Co^T * AO_ints * Cv, through one intermediate
            tmp = einsums.zeros([5, 20], name="tmp")
            einsums.einsum("ki;kj->ij", tmp, Co, AO_ints)
            einsums.einsum("ik;kj->ij", MO_ints, tmp, Cv)

Strides
=======

Tensors are column-major, so the **first** index varies fastest and a run along it is contiguous:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = create_random_tensor<double>("A", {10, 10});
            // A.stride(0) == 1, A.stride(1) == 10

            auto block = A(Range{2, 5}, Range{3, 7});
            // block.dim(0) == 3, block.dim(1) == 4
            // block.stride(0) == 1, block.stride(1) == 10   (inherited from A)

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [10, 10])
            # A.stride(0) == 1, A.stride(1) == 10

            block = A[2:5, 3:7]
            # block.dim(0) == 3, block.dim(1) == 4
            # block.stride(0) == 1, block.stride(1) == 10   (inherited from A)

A view keeps its parent's strides, which is exactly why it needs no copy: the elements are where
they always were, and the view only changes where the walk starts and how far it goes. Strides
are counted in elements rather than bytes, and Einsums and BLAS both handle non-unit strides.

Two disjoint views of one tensor may be the two inputs of a contraction, and a view may be the
output while another is an input. What is rejected is an output that provably overlaps an input
under a different index list; see :ref:`howto-contractions`.

View Lifetime
=============

This is the second place the two languages differ rather than merely spell things differently.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. warning::

           A view holds a pointer into the tensor it came from. The tensor must outlive every
           view into it. Using a view after its parent is destroyed is undefined behaviour,
           exactly as a raw pointer would be.

        .. code-block:: cpp

            // Do not do this
            auto make_bad_view() {
                auto A = create_random_tensor<double>("A", {5, 5});
                return A(Range{0, 3}, Range{0, 3});   // A dies at the return
            }

        ``data()`` on a view returns a pointer to the view's own first element, already offset
        into the parent, which is what you want when handing one to a BLAS call and a trap if you
        then apply the offset a second time.

    .. tab-item:: Python
        :sync: python

        A view keeps its parent alive, so it cannot dangle. Returning one from a function is
        fine, and the tensor it refers to stays reachable for as long as the view does.

        .. code-block:: python

            def make_view():
                local = einsums.create_random_tensor("local", [5, 5])
                return local[0:3, 0:3]

            v = make_view()
            v[0, 0]        # fine: the parent is still referenced

        What still applies is that a view writes through. Handing one to an operation that writes
        its output changes the tensor underneath.

Views in a Captured Graph
=========================

Views are first class in a capture. Form one before the capture and contract it inside, and the
graph records the aliasing relationship rather than a bare pointer, which is what lets the
optimizer resolve a write through a view back to the buffer it belongs to:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Graph.hpp>

            auto F   = create_random_tensor<double>("F", {n_orbs, n_orbs});
            auto T   = create_random_tensor<double>("T", {n_occ, n_virt});
            auto Fov = F(Range{0, n_occ}, Range{n_occ, n_orbs});

            cg::Graph graph("blocks");
            auto     &out = graph.create_runtime_tensor<double>("out", {n_occ, n_occ},
                                                                /*intermediate=*/false);
            {
                cg::CaptureGuard guard(graph);
                cg::einsum("ia;ja->ij", &out, Fov, T);
            }

            graph.optimize();
            graph.execute();

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums.graph as cg

            F   = einsums.create_random_tensor("F", [n_orbs, n_orbs])
            T   = einsums.create_random_tensor("T", [n_occ, n_virt])
            Fov = F[:n_occ, n_occ:]

            graph = cg.Graph("blocks")
            out = graph.create_tensor("out", [n_occ, n_occ], intermediate=False)

            with cg.capture(graph):
                einsums.einsum("ia;ja->ij", out, Fov, T)

            graph.optimize()
            graph.execute()

The one thing to avoid is slicing a tensor whose storage the graph may move. Bind or materialize
first, then take the view.

What's Next
===========

- :ref:`tutorial-linalg` for decompositions and solves
- :ref:`tutorial-compute-graph` for what capture buys across a whole calculation
- :ref:`howto-views` for the recipes, including the aliasing rule in full
