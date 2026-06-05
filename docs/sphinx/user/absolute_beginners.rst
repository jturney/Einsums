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

Einsums is an open-source C++ library. It provides :cpp:class:`Tensor`, a homogeneous n-dimensional
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
If you see do not see ``-->``, you're looking at the **input**, or the code that
you would type. Everything that is comment and has ``-->`` in front of it is potential
**output** or a representation of what you should expect.  The lines with
``-->`` should not be copied into your code and will cause a compile error
if types or pasted into your code.

Setting up a program
====================

To create a program using Einsums, you must initialize the library before you do anything with Einsums,
and finalize it after you finish using Einsums.

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

Alternatively, you can do things manually. Make sure you wrap your code in OpenMP directives,
otherwise the threading environment won't be set up properly and you will notice a significant
slow down as each call will need to set it up again.:

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

To create an Einsums Tensor, you can use the constructors for the tensor class
:cpp:class:`Tensor`.

All you need to do to create a basic tensor is pass a name for the tensor and the
dimensionality of each index.

.. code:: C++

    auto A = einsums::Tensor{"A", 2, 2};  // --> einsums::Tensor<2, double>

In this example, we are using the C++ ``auto`` to simplify the type signature. We can
write the data type explicitly if we want to.

.. code:: C++

    // Full explicit data type
    einsums::Tensor<2, double> A = einsums::Tensor{"A", 2, 2};

    // The default underlying type of a tensor is `double`
    einsums::Tensor<2> B = einsums::Tensor{"B", 2, 2};

    // Allow the compiler to determine things.
    auto C = einsums::Tensor{"C", 2, 2};


Specifying your data type
-------------------------

While the default data type is double-precision floating point (``double``), you
can explicitly specify which data type you want use.

.. code:: C++

    auto B = einsums::Tensor<float>{"B", 2, 2};

Einsums also supports the use of complex numbers.

.. code:: C++

    auto D = einsums::Tensor<std::complex<float>>{"D", 2, 2};

The only supported data types are floating point and complex floating point. Integers and arbitrary objects are not supported.

Different Tensor Layouts
------------------------

Einsums also provides several different tensor layouts. For a tensor that only has elements along
a block diagonal, there is the :cpp:class:`BlockTensor`. When a tensor is blockwise sparse,
but has blocks that are not on the diagonal, or have axes of varying dimensions, there is the
:cpp:class:`TiledTensor`, which can be viewed by a :cpp:class:`TiledTensorView`.

Different Tensor Storage
------------------------

Einsums intends to provide tensors that are compatible with GPU and CPU operations, as well as tensors stored on disk.
These are intended to be drop-in replacements, though there may be some variability in the interfaces for these tensors.
The disk tensor class is :cpp:class:`DiskTensor`, which can be viewed by a :cpp:class:`DiskView`.
For GPU tensors, there are :cpp:class:`DeviceTensor` and :cpp:class:`DeviceTensorView`, as well as
:cpp:class:`BlockDeviceTensor`, :cpp:class:`TiledDeviceTensor`, and :cpp:class:`TiledDeviceTensorView`. 

Basic Tensor operations
=======================

There are several basic things we can do with tensors. We can fill tensors with values, perform in-place arithmetic operations, and more.

.. code:: C++

    Tensor<double, 2> A{"A", 10, 10};
    auto B = create_random_tensor("B", 10, 10);

    // Filling values
    A = B; // Fill A with the values from B.
    A.zero(); // Fills with zero.
    A.set_all(0.3); // Sets every value to 0.3.
    A = 0.3; // Same as above.

    // In-place arithmetic
    // We can use tensors. These will be done element-wise.
    A += B;
    A -= B;
    A *= B;
    A /= B;

    // We can also use scalars. These will be done element-wise.
    A += 2;
    A -= 2;
    A *= 2;
    A /= 2;

    // For some kinds of tensors, we can also do some
    // arbitrary element-wise arithmetic.
    A = 1 / (2 * B + 1) * (B + B * B);

Indexing and slicing
--------------------

There are two ways to index into a tensor. The first is the function call syntax. This must be provided by a tensor class for a tensor to
be interpreted as a tensor. The other way is using the :code:`subscript` method, which is only provided by some tensor classes.
The function call operator will handle things such as negative indices, and may do some bounds checking. The :code:`subscript` method,
if provided, does none of this, and will simply treat the arguments as correct. This means that the :code:`subscript` method is much faster
than the function call syntax, but it is much more limited in its capabilities.

.. code:: C++

    auto A = create_random_tensor("A", 3, 3);

    // Function call syntax. Can be slow for large tensors.
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            printf("%lf", A(i, j));
        }
    }

    // Equivalent to the one before, but with the subscript method. Much faster.
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            printf("%lf", A.subscript(i, j));
        }
    }

    // Negative indices will wrap around like in Python.
    assert(A(-1, -1) == A(2, 2));

    // Passing negative indices to the subscript method produces undefined behavior.
    assert(A.subscript(-1, -1) != A.subscript(2, 2))

    // You can also use these to assign elements.
    A(2, 2) = 10;
    A.subscript(2, 2) = 10;

Tensors can also be sliced. This is done using the function call syntax. The number of arguments passed is allowed to be less than the rank,
and ranges can also be passed for slicing.

.. code:: C++

    auto A = create_random_tensor("A", 3, 3);

    // Get the first two rows of the tensor.
    TensorView View1 = A(Range{0, 1}, All);    // TensorView<double, 2>

    // Get the last row of the tensor.
    TensorView View2 = A(2);    // TensorView<double, 1>
    // Get the last column of the tensor.
    TensorView View3 = A(All, 2);    // TensorView<double, 1>

    // Get a 2x2 block from the tensor.
    TensorView View4 = A(Range{1, 2}, Range{0, 1});    // TensorView<double, 2>

Shape and size of a Tensor
--------------------------

The dimensions of a tensor can be accessed using the :code:`dim` and :code:`dims` methods. The first lets you specify the axis, while
the second gives all dimensions in a container. To get the size of a tensor, use the :code:`size` method.

.. code:: C++

    TensorA{"A", 3, 4, 5};

    assert(A.size() == 3 * 4 * 5);
    assert(A.dim(0) == 3);
    assert(A.dim(1) == 4);
    assert(A.dim(2) == 5);

    auto dims = A.dims();

    assert(dims[0] == 3);
    assert(dims[1] == 4);
    assert(dims[2] == 5);

Reshaping a Tensor
------------------

A tensor constructor is provided for reshaping a tensor. Note that the tensor passed in will be invalidated at the end of the call,
so further operations can cause undefined behavior. The underlying data is not modified, simple reinterpreted or moved.

.. code:: C++

    Tensor A{"A", 3, 4, 5};
    Tensor B{A, 2, 3, 10}; // Reshape A to have new dimensions.
                           // A is no longer valid after this call.

    Tensor C{B, 10, -1};   // Reshape B to have a new rank and
                           // new dimensions. The -1 will be replaced with a
                           // number - 6 in this case - so that the size
                           // of the input and output are the same.

A negative index will be treated as a wildcard, and the constructor will figure out what it should be instead to make the
sizes correct.

Converting a 1D Tensor into a 2D Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This can be used to convert a 1D tensor into a 2D tensor.

.. code:: C++

    Tensor A{"A", 30};
    Tensor B{A, -1, 10}; // Make A into a 2D tensor.
                         // The -1 will be replaced with a
                         // number - 3 in this case - so that
                         // the size of the output matches the input.


More advanced Tensor operations
===============================

We can do more complicated things with tensors as well. For instance, we can perform linear algebra with some tensors, tensor contractions,
transpositions, element mapping, and more. Here are some useful things we can do.

Permuting elements
------------------

To permute the axes of a tensor, you can use the :cpp:func:`~einsums::permute` function. This takes an input tensor and an output tensor,
and it permutes the input tensor, scales it, scales the output tensor, then adds them together.

.. code:: C++

    using namespace einsums;

    auto A = create_random_tensor("A", 3, 4, 5);
    auto B = create_random_tensor("B", 5, 4, 3);
    Tensor C{"C", 5, 4, 3};

    // Copy B into C for testing.
    C = B;

    tensor_algebra::permute(1, Indices{index::i, index::j, index::k}, &C,
                          0.5, Indices{index::k, index::j, index::i}, A);

    for(size_t i = 0; i <5; i++) {
        for(size_t j = 0; j < 4; j++) {
            for(size_t k = 0; k < 3; k++) {
                assert(C(i, j, k) = B(i, j, k) + 0.5 * A(k, j, i));
            }
        }
    }

Linear Algebra
--------------

Most procedures provided by LAPACK and BLAS are available to use with tensors. Here are some common examples.

.. code:: C++

    using namespace einsums;

    Tensor A = create_random_tensor("A", 10, 10);
    Tensor B = create_random_tensor("B", 10, 10);
    Tensor C = create_random_tensor("C", 10, 10);

    Tensor u = create_random_tensor("u", 10);
    Tensor v = create_random_tensor("v", 10);
    Tensor<std::complex<double>> evals{"evals", 10};

    // gemm is available. Whether to transpose the inputs is
    // passed as template parameters.
    linear_algebra::gemm<false, false>(A, B, &C);

    // We can also do eigendecomposition. The following computes
    // both the left and right eigenvectors.
    linear_algebra::geev(&A, &evals, &B, &C);

    // Whereas this only computes the right eigenvectors.
    linear_algebra::geev(&A, &evals, nullptr, &C);

    // And dot products. This one does not conjugate the first argument.
    auto val = linear_algebra::dot(u, v);
    // This one does. Since u and v are real, these are actually the same.
    auto val2 = linear_algebra::true_dot(u, v);

Tensor Contractions
-------------------

This is what Einsums was made for! We can do any operation that looks like :math:`C_{ijk\cdots} = \alpha C_{ijk\cdots} + \beta A_{abc\cdots} B_{def\cdots}`.
Here's an example for something like :math:`C_{ijk} = A_{ik}B_{kj}`.

.. code:: C++

    using namespace einsums;

    auto A = create_random_tensor("A", 10, 10);
    auto B = create_random_tensor("B", 10, 10);
    auto C = create_random_tensor("C", 10, 10, 10);

    tensor_algebra::einsum(Indices{index::i, index::j, index::k}, &C, 
                           Indices{index::i, index::k}, A,
                           Indices{index::k, index::j}, B);

If we do something that can become a BLAS call, then it will normally become a BLAS call. Currently, index permutations are not
performed, so calls can only be optimized when the indices exactly match the pattern for a BLAS call. This will change in the future,
as permuting indices can seriously improve performance.

.. code:: C++

    using namespace einsums;

    auto A = create_random_tensor("A", 10, 10);
    auto B = create_random_tensor("B", 10, 10);
    double val;

    // This will optimize to a dot product BLAS call. When the output should be
    // a zero-rank tensor, a scalar may be used in its place.
    // That way, you don't have to deal with a zero-rank tensor.
    tensor_algebra::einsum(Indices{}, &val, 
        Indices{index::i, index::j}, A,
        Indices{index::i, index::j}, B);

    // This will not optimize to a BLAS call,
    // since Einsums can't currently permute indices.
    tensor_algebra::einsum(Indices{}, &val, 
        Indices{index::i, index::j}, A,
        Indices{index::j, index::i}, B);