..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Errors:

======
Errors
======

This module contains symbols used for error handling. This includes exception classes, and macros for throwing
exceptions with location information. 

See the :ref:`API reference <modules_Einsums_Errors_api>` of this module for more
details.

Error Reference
---------------

If all goes well, Einsums should never throw an error. Of course, this is wishful thinking. A reference as to
what each kind of error means is provided for users who wish to debug their code, or at least be able to handle
the errors as they come up. For the most part, the kinds of errors thrown by a function should be documented, though
this is still a work in progress.

.. cpp:class:: template<typename ErrorClass, int ErrorCode> CodedError : ErrorClass

    This is a wrapper for an error. It acts just like an error of type :code:`ErrorClass`, but it comes with an extra
    integer error code. This way, a function can throw multiple errors of the same class, but they can be separated if
    needed by specifying the error code. These errors can then be caught either as an error of type :code:`ErrorClass`,
    or you can specify which error you want to handle as :code:`CodedError<ErrorClass, ErrorCode>`.

    As an example, in some specializations of the :code:`gemm` call, multiple :cpp:class:`tensor_compat_error` s can be
    thrown. For the :cpp:class:`TiledTensor` version, for instance, a :cpp:class:`tensor_compat_error` can be thrown if 
    either the output tensor's grid doesn't match what the input tensors require, or if the inner input tensor dimension's
    grid doesn't match what is required. If you wanted to catch both of these at once, you can use something like the following.

    .. code:: C++

        TiledTensor<double, 2> A, B, C;
        try {
            gemm<false, false>(1.0, A, B, 0.0, &C);
        } catch(tensor_compat_error &exc) {
            // Handle both errors.
        }
    
    Or you can handle each individually, like here.

    .. code:: C++

        try {
            gemm<false, false>(1.0, A, B, 0.0, &C);
        } catch(CodedError<tensor_compat_error, 0> &exc) {
            // Output doesn't have a compatible grid, so fix that.
        } catch(CodedError<tensor_compat_error, 1> &exc) {
            // Input doesn't have a compatible grid, so fix that.
        }

    :tparam ErrorClass: The kind of error the object wraps.
    :tparam ErrorCode: The identifier for the error.

.. cpp:class:: dimension_error

    Indicates that the dimensions of some tensor arguments are not compatible with the given operation.
    For instance, you can only take the determinant of a square matrix, so passing in a matrix that is not
    square to :cpp:func:`~einsums::linear_algebra::det` will result in this error.

.. cpp:class:: tensor_compat_error

    Indicates that two or more tensors are not compatible with each other for the requested operation.
    For instance, matrix multiplication is only allowed for matrices of certain dimensions, where for some
    natural numbers :math:`n, m, k`, the only allowed contraction is of the form 
    :math:`n\times k, k\times m \rightarrow n\times m`. If you were to pass a 3-by-2 matrix as the first
    matrix argument, a 1-by-4 matrix as the second matrix argument, and a 3-by-5 matrix as the third matrix
    argument to :cpp:func:`~einsums::linear_algebra::gemm`, this error will be thrown because the inner dimensions of the input arguments
    don't match (:math:`2 \ne 1`), and the outer dimensions of the input matrices and the dimensions of the output
    matrix don't match either (:math:`3 = 3`, but :math:`4 \ne 5`).

.. cpp:class:: num_argument_error

    Indicates that a function that can receive a variable number of arguments did not receive the right number
    of arguments. This is especially used in the :cpp:class:`RuntimeTensor` subscript functions, where the 
    number of indices needed is not known at compile time, so compile-time checks can't be made. This 
    exception has two specializations for too many and not enough arguments. They will be discussed further below.
    This exception is also thrown when a function takes an array of values that are treated as individual
    arguments, but the array is too small or too big.

.. cpp:class:: not_enough_args

    Indicates that a function that can receive a variable number of arguments did not receive enough arguments to work.

.. cpp:class:: too_many_args

    Indicates that a function that can receive a variable number of arguments received too many arguments to work.

.. cpp:class:: access_denied

    This exception is mosly only thrown by the HDF5 compatibility code. In that case, it is thrown when an illegal
    action is attempted on a file object, such as writing read-only data, or accessing a file without the required
    permissions.

.. cpp:class:: todo_error

    This exception, along with the :cpp:class:`not_implemented` exception, indicates that the action you requested
    is not yet implemented. If you get this error, come tell us 
    `on our discussion page <https://github.com/Einsums/Einsums/discussions>`_ or `our Discord server <https://discord.gg/8GvtkyWZUv>`_, and we will try to focus some energy
    to filling it out. If you are an experienced C++ programmer, we would appreciate your assistance if you think you
    have a solution.

.. cpp:class:: not_implemented

    This exception indicates that an action you requested is not implemented. This may be because the feature
    is not yet ready, or it may be that the specific combination of parameters is not acceptable. The message
    provided should give more information. If you absolutely need that set of features, come tell us
    `on our discussion page <https://github.com/Einsums/Einsums/discussions>`_ or `our Discord server <https://discord.gg/8GvtkyWZUv>`_, and we will try to work it out.
    If you are an experienced C++ programmer, we would appreciate your assistance if you think you have a solution.

.. cpp:class:: bad_logic

    This means the same thing as :code:`std::logic_error`. However, since :code:`std::logic_error` is the base class
    for so many errors, this specialization is provided so that you can catch specific exceptions by reference, and not
    have it match to all exceptions derived from :code:`std::logic_error`.

.. cpp:class:: uninitialized_error

    Indicates that the code is handling uninitialized data. This is usually thrown when Einsums was not initialized.

.. cpp:class:: system_error

    Indicates that an error occured when making a system call, or some sort of system utility failed. For instance,
    it can be thrown when trying to find a file that doesn't exist.

.. cpp:class:: enum_error

    Indicates that an invalid enum value was passed to a function.

.. cpp:class:: rank_error

    Indicates that a tensor argument had an invalid or incompatible rank.

This module also contains exception classes for every HIP, hipBLAS, and hipSOLVER status code. To get the name of a HIP
exception, take the :code:`hip` off of the beginning of the status code. For instance :code:`hipSuccess`
becomes :code:`Success`. To get the hipBLAS or hipSOLVER exception, remove the :code:`HIPBLAS_STATUS` or
:code:`HIPSOLVER_STATUS`, then convert the :code:`UPPERCASE_WITH_UNDERSCORES` into :code:`CamelCase` and 
put either :code:`blas` or :code:`solver` at the beginning. For instance :code:`HIP_BLAS_STATUS_SUCCESS`
becomes :code:`blasSuccess`. Since there are so many of these, they will not be documented here. The documentation
can be found in the :ref:`API reference <modules_Einsums_Errors_api>`.