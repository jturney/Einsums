..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Tensor:

======
Tensor
======

This module contains code for the tensor types used in Einsums.

See the :ref:`API reference <modules_Einsums_Tensor_api>` of this module for more
details.

----------------
Public Reference
----------------

.. cpp:class:: template<typename T, size_t Rank> Tensor

    Represents a tensor. The data is stored contiguously

.. cpp:class:: template<typename T, size_t Rank> TensorView

    Holds a view of a :cpp:class:`Tensor`, which may have a different rank and different dimensions.

.. cpp:class:: template<typename T, size_t Rank> DeviceTensor

    A tensor that makes data available to the GPU. These should be used when more permanent occupancy is desired.
    If you want to map/copy a tensor into GPU memory temporarily, but have it live in core most of the time, consider using 
    a :cpp:class::`DeviceTensorView` instead.

.. cpp:class:: template<typename T, size_t Rank> DeviceTensorView

    Holds a view of data that is available to the GPU. This may be a view of a :cpp:class:`DeviceTensor`, or it may
    be a mapping of a :cpp:class:`Tensor` or :cpp:class:`TensorView`.

.. cpp:class:: template<typename T, size_t Rank> DiskTensor

    A tensor whose data is stored on disk.

.. cpp:class:: template<typename T, size_t ViewRank, size_t Rank> DiskView

    A view of a :cpp:class:`DiskTensor`.

.. cpp:class:: template<typename T, size_t Rank> BlockTensor

    A tensor that has square blocks of entries along its main diagonal. The rank must be at least 2, since
    rank-1 and rank-0 tensors don't have diagonals.

.. cpp:class:: template<typename T, size_t Rank> BlockDeviceTensor

    Similar to :cpp:class:`BlockTensor`, but the data is available to the GPU.

.. cpp:class:: template<typename T, size_t Rank> TiledTensor

    A tensor that can be split up into a grid of smaller tensors. The assumption is that
    most of these smaller tensors are rigorously zero, and so are not stored. This is similar
    to a :cpp:class:`BlockTensor`, but the tiles do not have to lie on the diagonal,
    and the grid can be arbitrary on all dimensions, rather than needing to be the same across
    all dimensions.

.. cpp:class:: template<typename T, size_t Rank> TiledTensorView

    Conceptually, this is a view of a :cpp:class:`TiledTensor`. Practically,
    this stores views of each of the tiles, allowing different slices to be taken from each.

.. cpp:class:: template<typename T, size_t Rank> TiledDeviceTensor

    Same as :cpp:class:`TiledTensor`, but available to the GPU.

.. cpp:class:: template<typename T, size_t Rank> TiledDeviceTensorView

    Same as :cpp:class:`TiledTensorView`, but available to the GPU.

.. cpp:class:: template<typename T, size_t Rank> tensor_base::FunctionTensor

    This is the base class for tensors which pass indices onto a different function. Users may wish to 
    use these, so this is really the only base class that is here in the public API.

.. cpp:class:: template<typename T, size_t Rank> FuncPointerTensor

    This is one specialization of the :cpp:class:`tensor_base::FunctionTensor` that wraps a
    function pointer, and passes arguments to that function pointer.

.. cpp:class:: template<typename T, size_t Rank> FunctionTensorView

    Applies an offset to the arguments passed to a function tensor. In essence, acting like a view.

.. cpp:class:: template<typename T> KroneckerDelta

    This is an example implementation of a function tensor that evaluates the Kronecker delta. Some may find it useful,
    so it is provided in the public API, rather than just being an example.

.. cpp:class:: template<typename T> RuntimeTensor

    This is a convenience class for interacting with the Python module. It will never work with many Einsums calls. Instead,
    it should be converted into a :cpp:class:`TensorView` so that the rank can be coerced at compile time.

.. cpp:class:: template<typename T> RuntimeTensorView

    This is a convenience class for interacting with the Python module. It will never work with many Einsums calls. Instead,
    it should be converted into a :cpp:class:`TensorView` so that the rank can be coerced at compile time.

Symmetry Metadata
-----------------

Every tensor can optionally carry a ``SymmetryDescriptor`` describing
invariants such as ``T(i,j) = T(j,i)``. The descriptor is metadata only, so
storage stays dense. It is consumed by the rank-2 BLAS dispatch, which
promotes ``gemm`` to ``symm`` or ``hemm``, and by the ComputeGraph
``SymmetryPropagation`` pass.

Attach a descriptor with ``tensor.set_symmetry(desc)``, read it with
``tensor.symmetry()``, and clear it with ``tensor.clear_symmetry()``. Enforce
a descriptor with ``symmetrize(tensor)`` and verify one with
``check_symmetry(tensor, tolerance)``. Both functions are declared in
``Einsums/Tensor/SymmetryOps.hpp``.

See the ComputeGraph module's ``symmetry`` page for the full guide, which
covers named factories for common patterns, propagation rules, and design
notes.