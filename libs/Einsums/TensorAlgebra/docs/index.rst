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

----------
Public API
----------

- :cpp:func:`~einsums::tensor_algebra::einsum` - the einsum call; computes a tensor contraction, lowering to a BLAS call when the index analysis allows it.
- :cpp:func:`~einsums::tensor_algebra::permute` - permutes the entries of a tensor into an output tensor, with prefactors for scaling the original output back in.
