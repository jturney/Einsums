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

- :cpp:type:`~einsums::Tensor` - the standard in-core tensor; the data is stored contiguously.
- :cpp:class:`~einsums::TensorView` - a view of a tensor, which may have a different rank and different dimensions.
- :cpp:class:`~einsums::DiskTensor` - a tensor whose data is stored on disk.
- :cpp:class:`~einsums::DiskView` - a view of a :cpp:class:`~einsums::DiskTensor`.
- :cpp:class:`~einsums::BlockTensor` - a tensor with square blocks of entries along its main diagonal.
- :cpp:class:`~einsums::TiledTensor` - a tile-wise sparse tensor; tiles that are rigorously zero are not stored.
- :cpp:class:`~einsums::TiledTensorView` - a view of a :cpp:class:`~einsums::TiledTensor`, holding a view of each tile.
- :cpp:class:`~einsums::tensor_base::FunctionTensor` - optional public base class for tensors that pass indices on to a function.
- :cpp:class:`~einsums::FuncPointerTensor` - a function tensor that wraps a function pointer.
- :cpp:class:`~einsums::FunctionTensorView` - acts as a view of a function tensor by applying an offset to the indices.
- :cpp:class:`~einsums::KroneckerDelta` - a function tensor that evaluates the Kronecker delta; usable in einsum calls.
- :cpp:type:`~einsums::RuntimeTensor` - a runtime-rank tensor, mostly for interacting with the Python module.
- :cpp:class:`~einsums::RuntimeTensorView` - a runtime-rank view, mostly for interacting with the Python module.

GPU-resident storage is no longer a separate family of device tensor classes.
It is provided through allocator-based aliases such as
:cpp:type:`~einsums::GPUTensor` and :cpp:type:`~einsums::RuntimeGPUTensor`.

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