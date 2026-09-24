..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _absolute_beginners:

******************************************
Einsums: The Absolute Basics for Beginners
******************************************

.. sectionauthor:: Justin M. Turney

Welcome to the absolute beginner's guide to Einsums! If you have comments or
suggestions, please do not hesitate to `reach out <https://discord.gg/8GvtkyWZUv>`_!

Welcome to Einsums!
===================

Einsums is an open-source C++ library. It provides :cpp:any:`~einsums::Tensor`, a homogeneous n-dimensional
tensor object, with methods to efficiently operate on it. Einsums can be used to perform
a wide variety of mathematical operations on tensors.

.. _installing:

Installing Einsums
==================

You have two choices for installing Einsums, compiling from source directly or by using conda.
Compiling from source allows you to enable/disable certain features and to include debugging
options into your code.

If you wish to compile from source, visit
:ref:`Building from source <building-from-source>`.

If you have Python, you can install Einsums with::

    conda install einsums

How to include Einsums
======================

To access Einsums within your C++ code, you'll need to know the locations of the headers and library.
If you are using CMake use can use the :code:`find_package` function. Then with the
:code:`target_link_libraries` function you can link against Einsums.

For example, in your CMakeLists.txt file you can have lines similar to the following:

.. parsed-literal::
    find_package(Einsums \ |version| \ CONFIG)

    add_executable(sample main.cpp)
    target_link_libraries(samples Einsums::Einsums)

Then in your main.cpp you can have something like

.. code-block:: c++

    #include <Einsums/Tensor.hpp> // Provides Tensor

    int main() {

        auto A = einsums::Tensor("A", 3, 3); // Tensor<double, 2>

        return 0;
    }

Einsums is also compatible with Python through Pybind 11. To use it, simply use :code:`import einsums`. Much of
the C++ code is exported under the ``einsums._core`` module, with some extra utilities in other modules.

Reading the example code
========================

If you are not already comfortable with reading tutorials that contain a lot code,
you might not know how to interpret a code block that looks
like this

.. code-block:: C++

    auto A = einsums::create_random_tensor(6);
    auto B = einsums::Tensor{std::move(A), -1, 6};
    B.dims();  // --> Dims{1, 6 }

If you are not familiar with this style, it's very easy to understand.
If you do not see ``-->``, you're looking at the input, the code that
you would type. Everything that is a comment and has ``-->`` in front of it is potential
output, or a representation of what you should expect.  The lines with
``-->`` should not be copied into your code and will cause a compile error
if types or pasted into your code.

Setting up a program
====================

In C++ the library has to be brought up before you use it and shut down afterwards. From Python
importing does that for you.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/Runtime.hpp>

            namespace einsums {
            int main() {
                // Your code here.

                finalize();
                return EXIT_SUCCESS;
            }
            } // end namespace einsums

            int main(int argc, char **argv) {
                // This call takes responsibility of initializing Einsums and then calling
                // your passed main function.
                return einsums::start(einsums::main, argc, argv);
            }

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums

            # That is the whole of it. The runtime comes up on first use and shuts
            # down with the interpreter.

        Nothing else is required, and there is no finalize to call. Note that the
        profiler's session export is written during shutdown, so a program that ends by
        other means may not produce one; see :doc:`tutorial_performance`.

In C++ you can alternatively do this by hand. Wrap your code in OpenMP directives when you do,
otherwise the threading environment is not set up properly and every call pays to establish it
again:

.. code:: C++

    int main(int argc, char **argv) {
    #pragma omp parallel
    {
    #   pragma omp single
        {
            einsums::initialize(argc, argv);

            // Your code here.

            einsums::finalize();
        }
    }
        return 0; // This needs to be outside. You can't return from within a parallel block.
    }


How to create a Tensor
======================

The tensor type to reach for is :cpp:type:`einsums::RuntimeTensor`. Pass a name and the size of
each dimension:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            einsums::RuntimeTensor<double> A{"A", {2, 2}};

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.zeros([2, 2], name="A")

The name is not decoration. It is how this tensor is identified in profiler output and in the
reports the optimizer prints, so a meaningful one pays for itself the first time you read either.

The number of dimensions is the length of that list, and it is carried by the value rather than
by the type. One function can therefore take a matrix and a rank-4 tensor without being a
template, which is why this is the type the Python bindings and the
:doc:`ComputeGraph <tutorial_compute_graph>` are built around.

Specifying your data type
-------------------------

The element type is the template argument, and there is no default: write it out.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            einsums::RuntimeTensor<double> A{"A", {2, 2}};
            einsums::RuntimeTensor<float>  B{"B", {2, 2}};

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.zeros([2, 2], name="A")                     # float64 by default
            B = einsums.zeros([2, 2], dtype="float32", name="B")

Einsums also supports complex numbers.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            einsums::RuntimeTensor<std::complex<float>>  C{"C", {2, 2}};
            einsums::RuntimeTensor<std::complex<double>> D{"D", {2, 2}};

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            C = einsums.zeros([2, 2], dtype="complex64",  name="C")
            D = einsums.zeros([2, 2], dtype="complex128", name="D")

The supported types are floating point and complex floating point. Integers work for some
operations; arbitrary objects are not supported.

.. note::

   Einsums also has :cpp:type:`einsums::Tensor`, whose rank is part of its type, spelled
   ``Tensor<double, 2>``. It is the right choice for eager code whose ranks are fixed when you
   write it, and a few routines take only that form. Everything in these tutorials works with
   ``RuntimeTensor``, and the two interoperate.

Different Tensor Layouts
------------------------

Einsums also provides several different tensor layouts. For a tensor that only has elements along
a block diagonal, there is the :cpp:any:`~einsums::BlockTensor`. When a tensor is blockwise sparse,
but has blocks that are not on the diagonal, or have axes of varying dimensions, there is the
:cpp:any:`~einsums::TiledTensor`, which can be viewed by a :cpp:any:`~einsums::TiledTensorView`.

Different Tensor Storage
------------------------

Einsums intends to provide tensors that are compatible with GPU and CPU operations, as well as tensors stored on disk.
These are intended to be drop-in replacements, though there may be some variability in the interfaces for these tensors.
The disk tensor class is :cpp:any:`~einsums::DiskTensor`, which can be viewed by a :cpp:any:`~einsums::DiskView`.
GPU storage is selected through the allocator: the :cpp:any:`~einsums::GPUTensor` and
:cpp:any:`~einsums::RuntimeGPUTensor` aliases store their data on the device. 

Basic Tensor operations
=======================

There are several basic things we can do with tensors. We can fill tensors with values, perform in-place arithmetic operations, and more.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            einsums::RuntimeTensor<double> A{"A", {10, 10}};
            auto B = einsums::create_random_tensor<double>("B", {10, 10});

            // Filling values
            A = B;           // Copy the values from B.
            A.zero();        // Every element becomes 0.
            A.set_all(0.3);  // Every element becomes 0.3.
            A = 0.3;         // Same as above.

            // In-place arithmetic with a scalar, element-wise
            A += 2;
            A -= 2;
            A *= 2;
            A /= 2;

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.zeros([10, 10], name="A")
            B = einsums.create_random_tensor("B", [10, 10])

            # Filling values
            A = einsums.array(B)   # copy the values from B (plain `A = B` would rebind)
            A.zero()               # every element becomes 0
            A.set_all(0.3)         # every element becomes 0.3

            # In-place arithmetic with a scalar, element-wise
            A += 2
            A -= 2
            A *= 2
            A /= 2

Element-wise arithmetic between two tensors is written as a contraction rather than as an
operator:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            namespace cg = einsums::compute_graph;

            cg::einsum("ij;ij->ij", &A, A, B);   // A = A * B, element by element

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            einsums.einsum("ij;ij->ij", A, A, B)   # A = A * B, element by element

That is not a detour. A contraction spec says which indices line up, and once tensors have more
than two dimensions that is the question an operator cannot answer. :ref:`tutorial-einsum` is
where this goes next, and it is the heart of the library.

Indexing and slicing
--------------------

Index a tensor with the function call syntax, one argument per dimension:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = einsums::create_random_tensor<double>("A", {3, 3});

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    printf("%lf ", A(i, j));
                }
            }

            // Negative indices wrap around, as in Python.
            assert(A(-1, -1) == A(2, 2));

            // The same syntax assigns.
            A(2, 2) = 10;

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [3, 3])

            for i in range(3):
                for j in range(3):
                    print(A[i, j], end=" ")

            # Negative indices wrap around
            assert A[-1, -1] == A[2, 2]

            # The same syntax assigns
            A[2, 2] = 10

Element access is for inspecting and for setting up small things. A loop over ``operator()``
pays an index computation per element, so it is not how bulk work gets done: that is what the
contractions in :ref:`tutorial-einsum` are for, and they reach BLAS.

Slicing uses the same call syntax. Fewer arguments than the rank are allowed, and a
:cpp:struct:`einsums::Range` selects part of a dimension. A range is half-open, so
``Range{0, 2}`` is two entries:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = einsums::create_random_tensor<double>("A", {3, 3});

            auto View1 = A(Range{0, 2}, All);          // first two rows
            auto View2 = A(2, All);                    // row 2, as a rank-1 view
            auto View3 = A(All, 2);                    // column 2, as a rank-1 view
            auto View4 = A(Range{1, 3}, Range{0, 2});  // a 2x2 block

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [3, 3])

            View1 = A[0:2, :]     # first two rows
            View2 = A[2, :]       # row 2, as a rank-1 view
            View3 = A[:, 2]       # column 2, as a rank-1 view
            View4 = A[1:3, 0:2]   # a 2x2 block

None of these copy. A view writes through to the tensor it came from, and it is only valid while
that tensor is. :ref:`tutorial-views` goes into both.

Shape and size of a Tensor
--------------------------

The dimensions of a tensor can be accessed using the :code:`dim` and :code:`dims` methods. The first lets you specify the axis, while
the second gives all dimensions in a container. To get the size of a tensor, use the :code:`size` method.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            einsums::RuntimeTensor<double> A{"A", {3, 4, 5}};

            assert(A.size() == 3 * 4 * 5);
            assert(A.dim(0) == 3);
            assert(A.rank() == 3);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.zeros([3, 4, 5], name="A")

            assert A.size == 3 * 4 * 5   # a property in Python
            assert A.dim(0) == 3         # a method
            assert A.rank() == 3         # a method

Reshaping a Tensor
------------------

A statically ranked :cpp:type:`einsums::Tensor` has a constructor that reinterprets an existing
tensor's data under new dimensions. It takes the source by rvalue and a name for the result, and
the source is left empty, so pass it with ``std::move`` and do not use it afterwards. The data is
reinterpreted rather than rearranged.

.. code:: C++

    einsums::Tensor<double, 3> A{"A", 3, 4, 5};
    einsums::Tensor<double, 3> B{std::move(A), "B", 2, 3, 10};
    // B is 2x3x10; A is no longer usable.

A negative extent is a wildcard, and the constructor works out what it has to be for the sizes to
match:

.. code:: C++

    einsums::Tensor<double, 3> C{"C", 3, 4, 5};
    einsums::Tensor<double, 2> D{std::move(C), "D", 10, -1};
    // D is 10x6

Converting a 1D Tensor into a 2D Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same constructor changes the rank:

.. code:: C++

    einsums::Tensor<double, 1> A{"A", 30};
    einsums::Tensor<double, 2> B{std::move(A), "B", -1, 10};
    // B is 3x10


More advanced Tensor operations
===============================

We can do more complicated things with tensors as well. For instance, we can perform linear algebra with some tensors, tensor contractions,
transpositions, element mapping, and more. Here are some useful things we can do.

Permuting elements
------------------

To permute the axes of a tensor, you can use the ``cg::permute`` function. This takes an input tensor and an output tensor,
and it permutes the input tensor, scales it, scales the output tensor, then adds them together.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            using namespace einsums;
            namespace cg = einsums::compute_graph;

            auto A = create_random_tensor<double>("A", {3, 4, 5});
            auto B = create_random_tensor<double>("B", {5, 4, 3});
            RuntimeTensor<double> C{"C", {5, 4, 3}};

            C = B;   // so there is something to add to

            // C = 1 * C + 0.5 * A permuted from (k, j, i) to (i, j, k)
            cg::permute("ijk <- kji", 1.0, &C, 0.5, A);

            for (size_t i = 0; i < 5; i++) {
                for (size_t j = 0; j < 4; j++) {
                    for (size_t k = 0; k < 3; k++) {
                        assert(C(i, j, k) == B(i, j, k) + 0.5 * A(k, j, i));
                    }
                }
            }

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [3, 4, 5])
            B = einsums.create_random_tensor("B", [5, 4, 3])
            C = einsums.array(B)   # so there is something to add to

            # C = 1 * C + 0.5 * A permuted from (k, j, i) to (i, j, k)
            einsums.permute("ijk <- kji", C, A, c_pf=1.0, a_pf=0.5)

            expect = np.asarray(B) + 0.5 * np.asarray(A).transpose(2, 1, 0)
            assert np.allclose(np.asarray(C), expect)

Linear Algebra
--------------

Most procedures provided by LAPACK and BLAS are available to use with tensors. Here are some common examples.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            using namespace einsums;

            auto A = create_random_tensor<double>("A", {10, 10});
            auto B = create_random_tensor<double>("B", {10, 10});
            auto C = create_random_tensor<double>("C", {10, 10});

            auto u = create_random_tensor<double>("u", {10});
            auto v = create_random_tensor<double>("v", {10});

            // gemm computes C = alpha * op(A) * op(B) + beta * C. Whether each input is
            // transposed is a template parameter; the prefactors are arguments.
            linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

            // Dot products. This one does not conjugate the first argument.
            auto val = linear_algebra::dot(u, v);
            // This one does. Since u and v are real here, the two agree.
            auto val2 = linear_algebra::true_dot(u, v);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            from einsums import linalg

            A = einsums.create_random_tensor("A", [10, 10])
            B = einsums.create_random_tensor("B", [10, 10])
            C = einsums.create_random_tensor("C", [10, 10])

            u = einsums.create_random_tensor("u", [10])
            v = einsums.create_random_tensor("v", [10])

            # gemm computes C = alpha * op(A) * op(B) + beta * C. Transposition is a
            # keyword argument here rather than a template parameter.
            linalg.gemm(1.0, A, B, 0.0, C)

            # Dot product, which returns rather than writing through a pointer.
            val = linalg.dot(u, v)

        ``true_dot``, the conjugating form, has no Python binding; for complex data use
        ``linalg.dotc``.

A few routines want a statically ranked :cpp:type:`einsums::Tensor` rather than a
``RuntimeTensor``, because they are written against a compile-time rank of two. General
eigendecomposition is one, and its eigenvalues and eigenvectors are complex even when the input
is real:

.. code:: C++

    auto          M = create_random_tensor<double>("M", 10, 10);
    Tensor<cd, 1> evals{"evals", 10};
    Tensor<cd, 2> rvecs{"rvecs", 10, 10};
    Tensor<cd, 2> lvecs{"lvecs", 10, 10};

    // Both sets of eigenvectors.
    linear_algebra::geev(&M, &evals, &lvecs, &rvecs);

    // Pass a null pointer for a set you do not want computed.
    auto M2 = create_random_tensor<double>("M2", 10, 10);
    linear_algebra::geev(&M2, &evals, nullptr, &rvecs);

``geev`` overwrites the matrix it is given, which is why each call above gets its own copy.

Tensor Contractions
-------------------

This is what Einsums was made for! We can do any operation that looks like :math:`C_{ijk\cdots} = \alpha C_{ijk\cdots} + \beta A_{abc\cdots} B_{def\cdots}`.
Here's an example for something like :math:`C_{ijk} = A_{ik}B_{kj}`.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            using namespace einsums;
            namespace cg = einsums::compute_graph;

            auto A = create_random_tensor<double>("A", {10, 10});
            auto B = create_random_tensor<double>("B", {10, 10});
            auto C = create_zero_tensor<double>("C", {10, 10, 10});

            cg::einsum("ik;kj->ijk", &C, A, B);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [10, 10])
            B = einsums.create_random_tensor("B", [10, 10])
            C = einsums.zeros([10, 10, 10], name="C")

            einsums.einsum("ik;kj->ijk", C, A, B)

If we do something that can become a BLAS call, then it will normally become a BLAS call. Currently, index permutations are not
performed, so calls can only be optimized when the indices exactly match the pattern for a BLAS call. This will change in the future,
as permuting indices can seriously improve performance.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            namespace cg = einsums::compute_graph;

            auto A = einsums::create_random_tensor<double>("A", {10, 10});
            auto B = einsums::create_random_tensor<double>("B", {10, 10});

            // A single number comes out through dot, which writes through a pointer.
            // This one reaches a BLAS dot product.
            double val = 0.0;
            cg::dot(&val, A, B);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [10, 10])
            B = einsums.create_random_tensor("B", [10, 10])

            # A single number comes out through dot, which returns in Python.
            val = einsums.linalg.dot(A, B)

The second of those two used to be written with B's indices reversed, which is a different
quantity and lands on the generic loop, because reaching a BLAS call would need a physical
permutation that Einsums does not insert on your behalf. If you want it, write the permutation
and contract the result. :ref:`tutorial-performance` tabulates which shapes take which kernel,
measured rather than predicted.
.. _absolute-beginners-graph:

Putting it together, then capturing it
======================================

Here is a whole program using what this page covered. It builds two matrices, contracts them,
scales the result and prints a number:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Operations.hpp>
            #include <Einsums/Print.hpp>
            #include <Einsums/Runtime.hpp>
            #include <Einsums/Tensor/RuntimeTensor.hpp>
            #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
            #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

            namespace cg = einsums::compute_graph;

            namespace einsums {
            int main() {
                auto A = create_random_tensor<double>("A", {64, 64});
                auto B = create_random_tensor<double>("B", {64, 64});
                auto C = create_zero_tensor<double>("C", {64, 64});

                cg::einsum("ik;kj->ij", &C, A, B);
                cg::scale(0.5, &C);

                println("C(0, 0) = {}", C(0, 0));
                return EXIT_SUCCESS;
            }
            } // namespace einsums

            int main(int argc, char **argv) {
                return einsums::start(einsums::main, argc, argv);
            }

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums
            from einsums import linalg

            A = einsums.create_random_tensor("A", [64, 64])
            B = einsums.create_random_tensor("B", [64, 64])
            C = einsums.zeros([64, 64], name="C")

            einsums.einsum("ik;kj->ij", C, A, B)
            linalg.scale(0.5, C)

            print(f"C(0, 0) = {C[0, 0]}")

Every call there runs the moment it is reached. That is the right thing for a program that does
the work once, and it is the wrong thing for one that repeats it, because each call is analysed
and dispatched again every time and nothing can be optimized across the pair.

The same work, recorded once and replayed, looks like this:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Graph.hpp>

            namespace einsums {
            int main() {
                auto A = create_random_tensor<double>("A", {64, 64});
                auto B = create_random_tensor<double>("B", {64, 64});

                cg::Graph graph("scaled product");

                // intermediate = false: a result this code reads, not working space.
                auto &C = graph.create_runtime_tensor<double>("C", {64, 64},
                                                              /*intermediate=*/false);

                {
                    cg::CaptureGuard guard(graph);
                    cg::einsum("ik;kj->ij", &C, A, B);
                    cg::scale(0.5, &C);
                }

                graph.optimize();

                for (int iter = 0; iter < 100; ++iter) {
                    graph.execute();
                }

                println("C(0, 0) = {}", C(0, 0));
                return EXIT_SUCCESS;
            }
            } // namespace einsums

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums
            import einsums.graph as cg
            from einsums import linalg

            A = einsums.create_random_tensor("A", [64, 64])
            B = einsums.create_random_tensor("B", [64, 64])

            graph = cg.Graph("scaled product")

            # intermediate=False: a result this code reads, not working space.
            C = graph.create_tensor("C", [64, 64], intermediate=False)

            with cg.capture(graph):
                einsums.einsum("ik;kj->ij", C, A, B)
                linalg.scale(0.5, C)

            graph.optimize()

            for _ in range(100):
                graph.execute()

            print(f"C(0, 0) = {C[0, 0]}")

Three things changed and nothing else did. The operations are inside a ``CaptureGuard``, so they
are recorded rather than run. ``optimize()`` rewrites the recording. And ``execute()`` replays it,
as many times as you like, paying the analysis once.

The tensors the graph creates for you need one piece of information you have to supply:
``intermediate=false`` says ``C`` is an answer rather than scratch. Without it the optimizer will
notice that nothing inside the graph reads ``C``, remove the contraction that fills it, and leave
you reading whatever the buffer held. It is the one mistake on this page that produces no error
message, which is why it is worth meeting now.

Capture when the work repeats. A single contraction gains nothing from being recorded, and an
iteration gains most of what the library has to offer.

Where to go next
================

- :ref:`tutorial-tensors` for the tensor type in more detail
- :ref:`tutorial-einsum` for contractions, which is the heart of the library
- :ref:`tutorial-views` for working on part of a tensor without copying it
- :ref:`tutorial-compute-graph` for what capture makes possible across a whole calculation
- :ref:`howto` when you know what you want and need the specific way to do it
