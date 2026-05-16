.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

=============
API Reference
=============

All classes and functions are in the ``einsums::compute_graph`` namespace.
Optimization passes are in ``einsums::compute_graph::passes``.

Core Classes
============

Graph
-----

.. doxygenclass:: einsums::compute_graph::Graph
   :members:

Pipeline
--------

.. doxygenclass:: einsums::compute_graph::Pipeline
   :members:

CaptureGuard
-------------

.. doxygenstruct:: einsums::compute_graph::CaptureGuard
   :members:

CaptureContext
--------------

.. doxygenclass:: einsums::compute_graph::CaptureContext
   :members:

Execution
=========

Executor
--------

.. doxygenclass:: einsums::compute_graph::Executor
   :members:

SequentialExecutor
-------------------

.. doxygenclass:: einsums::compute_graph::SequentialExecutor
   :members:

OpenMPExecutor
--------------

.. doxygenclass:: einsums::compute_graph::OpenMPExecutor
   :members:

DependencyInfo
--------------

.. doxygenstruct:: einsums::compute_graph::DependencyInfo
   :members:

Data Types
==========

TensorHandle
-------------

.. doxygenstruct:: einsums::compute_graph::TensorHandle
   :members:

TensorSlot
----------

.. doxygenstruct:: einsums::compute_graph::TensorSlot
   :members:

EinsumParams
------------

.. doxygenstruct:: einsums::compute_graph::EinsumParams
   :members:

Node
----

.. doxygenstruct:: einsums::compute_graph::Node
   :members:

OpKind
------

.. doxygenenum:: einsums::compute_graph::OpKind

EinsumDescriptor
-----------------

.. doxygenstruct:: einsums::compute_graph::EinsumDescriptor
   :members:

ScaleDescriptor
----------------

.. doxygenstruct:: einsums::compute_graph::ScaleDescriptor
   :members:

PermuteDescriptor
------------------

.. doxygenstruct:: einsums::compute_graph::PermuteDescriptor
   :members:

ConditionalDescriptor
----------------------

.. doxygenstruct:: einsums::compute_graph::ConditionalDescriptor
   :members:

LoopDescriptor
--------------

.. doxygenstruct:: einsums::compute_graph::LoopDescriptor
   :members:

AllocDescriptor
----------------

.. doxygenstruct:: einsums::compute_graph::AllocDescriptor
   :members:

LoopNode (Pipeline)
-------------------

.. doxygenstruct:: einsums::compute_graph::LoopNode
   :members:

String Einsum
=============

ParsedEinsumSpec
-----------------

.. doxygenstruct:: einsums::compute_graph::ParsedEinsumSpec
   :members:

EinsumFormatString
-------------------

.. doxygenstruct:: einsums::compute_graph::EinsumFormatString
   :members:

Type Aliases
============

.. doxygentypedef:: einsums::compute_graph::TensorId
.. doxygentypedef:: einsums::compute_graph::NodeId
.. doxygentypedef:: einsums::compute_graph::LoopCondition
.. doxygentypedef:: einsums::compute_graph::OpData

Optimization Passes
===================

OptimizerPass (base class)
---------------------------

.. doxygenclass:: einsums::compute_graph::OptimizerPass
   :members:

CSE
---

.. doxygenclass:: einsums::compute_graph::passes::CSE
   :members:

Reorder
-------

.. doxygenclass:: einsums::compute_graph::passes::Reorder
   :members:

MemoryPlanning
--------------

.. doxygenclass:: einsums::compute_graph::passes::MemoryPlanning
   :members:

ChainParenthesization
----------------------

.. doxygenclass:: einsums::compute_graph::passes::ChainParenthesization
   :members:

ConstantFolding
----------------

.. doxygenclass:: einsums::compute_graph::passes::ConstantFolding
   :members:

DeadNodeElimination
--------------------

.. doxygenclass:: einsums::compute_graph::passes::DeadNodeElimination
   :members:

ScaleAbsorption
----------------

.. doxygenclass:: einsums::compute_graph::passes::ScaleAbsorption
   :members:

LoopInvariantHoisting
----------------------

.. doxygenclass:: einsums::compute_graph::passes::LoopInvariantHoisting
   :members:

InplaceOptimization
--------------------

.. doxygenclass:: einsums::compute_graph::passes::InplaceOptimization
   :members:

GEMMBatching
-------------

.. doxygenclass:: einsums::compute_graph::passes::GEMMBatching
   :members:

PermuteFusion
--------------

.. doxygenclass:: einsums::compute_graph::passes::PermuteFusion
   :members:

SymmetryPropagation
--------------------

.. doxygenclass:: einsums::compute_graph::passes::SymmetryPropagation
   :members:

See :doc:`symmetry` for the user-facing guide.

Free Functions
==============

.. doxygenfunction:: einsums::compute_graph::make_handle
.. doxygenfunction:: einsums::compute_graph::make_scalar_handle
.. doxygenfunction:: einsums::compute_graph::op_kind_name
.. doxygenfunction:: einsums::compute_graph::parse_einsum_spec
