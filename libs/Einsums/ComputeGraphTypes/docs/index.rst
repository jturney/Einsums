..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_ComputeGraphTypes:

#################
ComputeGraphTypes
#################

``ComputeGraphTypes`` is a lightweight, dependency-free module that
declares the data types the :ref:`ComputeGraph <modules_Einsums_ComputeGraph>`
uses to describe nodes, tensors, and contractions. It is a separate
module so that code which only needs to describe a graph, such as
the dispatcher, the distribution descriptors, or third-party tooling, can
include the type definitions without pulling in the full ComputeGraph
implementation.

Contents
========

The headers in this module:

- ``Descriptors.hpp``: the inert operation descriptors a node can carry,
  such as ``BatchedGemmDescriptor``, ``GroupedBatchedGemmDescriptor``,
  ``AllocDescriptor``, ``TransferDescriptor``, ``DiskIODescriptor`` and
  ``CommDescriptor``. Descriptors that need a type from further up (einsum,
  permute, loop, view, and so on) live in ``Einsums/ComputeGraph/Node.hpp``.
- ``Enums.hpp``: ``OpKind`` for graph nodes, and ``Target``, ``AllocState``,
  ``TensorOwnership``, ``InitKind`` and ``Residency``.
- ``EnumNames.hpp``: one name table per enum, used both to print and to
  parse the spellings.
- ``GraphData.hpp``: the serialized form of a graph snapshot (tensors,
  nodes, edges) used for profiling and visualization.
- ``Ids.hpp``: the identifier types ``NodeId`` and ``TensorId``.
- ``Spaces.hpp``: index-space annotations (``IndexSpace``, ``SpaceId``,
  ``GrowthClass``) and the ``SpaceRegistry`` that holds them.

The parsed einsum specification is not part of this module; it is
``Einsums/ComputeGraph/EinsumSpec.hpp``.

Design rule
===========

This module is header-only and depends only on ``Einsums_Config`` and the
C++ standard library, plus ``Einsums/Python/Annotations.hpp`` (the
header-only apiary markers for its Python bindings).
That keeps the type declarations cheap to include from any other module
and avoids circular dependencies in the build graph. Implementation logic
that operates on these types lives in the
:ref:`ComputeGraph <modules_Einsums_ComputeGraph>` module proper.

See the :ref:`API reference <modules_Einsums_ComputeGraphTypes_api>` of
this module for the full set of types.
