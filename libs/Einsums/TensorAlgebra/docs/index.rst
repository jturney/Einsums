..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_TensorAlgebra:

=============
TensorAlgebra
=============

Contains tensor contractions.

See the :ref:`API reference <modules_Einsums_TensorAlgebra_api>` of this module for more
details.

.. cpp:namespace:: einsums

----------
Public API
----------

.. cpp:function:: template <TensorConcept AType, TensorConcept BType, typename CType, typename U, typename... CIndices, typename... AIndices, typename... BIndices> void einsum(U const C_prefactor, std::tuple<CIndices...> const & C_inds, CType *C, U const UAB_prefactor, std::tuple<AIndices...> const & A_inds, AType const &A, std::tuple<BIndices...> const & B_inds, BType const &B)

    This is the einsum call. It is the reason this package exists. It computes the tensor contraction represented by 
    :math:`C_{abc\cdots} = \alpha C_{abc\cdots} + \beta A_{def\cdots}B_{ghi\cdots}`. The indices must be known at compile
    time, as well as the ranks of the tensors. The :code:`C` tensor may also be a scalar if the index tuple is empty.
    The tensor parameters may be any combination of smart pointers. Also, the prefactors may be left off. If the first
    prefactor is left off, it will default to zero. If the second is left off, it will default to one. Most combinations
    of kinds of tensors are accepted. However, for best results, avoid using :cpp:type:`~einsums::tensor_base::FunctionTensor`,
    :cpp:type:`RuntimeTensor`, or :cpp:type:`ArithmeticTensor`, as these can't be used with LAPACK or BLAS calls.

    This function will analyze the indices that it is given to determine if it can be turned into a BLAS call.
    As of the current version, it will not perform any major transpositions to force it into a BLAS call. The
    only transpositions it will do are the ones that can be specified as parameters to those BLAS calls. For 
    instance, if it can call :code:`gemm` for a matrix multiplication, it will determine if it needs to
    tell :code:`gemm` to transpose the arguments or not. It can not conjugate these yet, only transpose.
    It will not, however, try to swap the indices of a tensor around to coerce it into a :code:`gemm` call
    if it is not seen immediately. This is up to the user to perform. We have plans to make this a feature
    in the future, though, so feel free to write your code as if it did do this transposition (we are always
    looking for help, if you feel inclined to make this a reality).

.. cpp:function:: template <TensorConcept AType, TensorConcept CType, typename... CIndices, typename... AIndices, typename U> void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor, std::tuple<AIndices...> const &A_indices, AType const &A)

    Permute the entries of a tensor using the indices to guide it. The indices must be the same in the first and second index specifications,
    though obviously the orders can be different. The prefactors allow you to do something like
    :math:`C_{abc\cdots} = \alpha C_{abc\cdots} + \beta A_{cba\cdots}`, where the original value of the output tensor is
    scaled and added back into the output.

    .. note::

        This function uses HPTT to perform the tensor transpositions. However, HPTT does not work with :cpp:class:`TensorView`\ s.
        You may see slowdowns if you use this with :cpp:class:`TensorView`\ s, though we are trying to improve this. The best bet
        will probably be to copy the tensor view to a :cpp:type:`Tensor` first, then permute the elements.
    
    .. versionchanged:: 1.0.3

        This function now works with :cpp:class:`TensorView`\ s.

.. cpp:function:: template<typename... Args> sort(Args &&... args)

    This is the old version of :cpp:func:`~einsums::permute`. It may be used for something completely different in the future.

    .. deprecated:: 1.0.0

        The name "sort" is confusing since we are not sorting values, just permuting. As such,
        we will be removing this in the future.

    .. versionremoved:: 1.1.0