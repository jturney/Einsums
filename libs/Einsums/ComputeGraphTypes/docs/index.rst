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
uses to describe nodes, tensors, and contractions. It exists as a separate
module so that code which only needs to *describe* a graph (for example,
the dispatcher, distribution descriptors, or third-party tooling) can
include the type definitions without pulling in the full ComputeGraph
implementation.

Contents
========

The headers in this module:

- ``Descriptors.hpp`` — tensor-shape descriptors used by graph nodes
  (dimensions, strides, layout flags).
- ``EinsumSpec.hpp`` — a structured representation of an einsum
  expression (indices, sizes, batch axes) consumed by the dispatcher and
  the optimization passes.
- ``Enums.hpp`` — kind tags for graph nodes, distribution axes, executors,
  and related categorical types.
- ``GraphData.hpp`` — the raw fields each graph node carries (operands,
  metadata, pass-attached annotations).
- ``Ids.hpp`` — opaque identifier types (``TensorId``, ``NodeId``,
  ``GraphId``) used throughout the graph layer.

Design rule
===========

This module has **no runtime dependencies** beyond the C++ standard library.
That keeps the type declarations cheap to include from any other module
and avoids circular dependencies in the build graph. Implementation logic
that operates on these types lives in the
:ref:`ComputeGraph <modules_Einsums_ComputeGraph>` module proper.

See the :ref:`API reference <modules_Einsums_ComputeGraphTypes_api>` of
this module for the full set of types.
