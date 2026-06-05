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

:cpp:class:`einsums::compute_graph::Graph`

Pipeline
--------

:cpp:class:`einsums::compute_graph::Pipeline`

CaptureGuard
-------------

:cpp:class:`einsums::compute_graph::CaptureGuard`

CaptureContext
--------------

:cpp:class:`einsums::compute_graph::CaptureContext`

Execution
=========

Executor
--------

:cpp:class:`einsums::compute_graph::Executor`

SequentialExecutor
-------------------

:cpp:class:`einsums::compute_graph::SequentialExecutor`

OpenMPExecutor
--------------

:cpp:class:`einsums::compute_graph::OpenMPExecutor`

DependencyInfo
--------------

:cpp:class:`einsums::compute_graph::DependencyInfo`

Data Types
==========

TensorHandle
-------------

:cpp:class:`einsums::compute_graph::TensorHandle`

TensorSlot
----------

:cpp:class:`einsums::compute_graph::TensorSlot`

EinsumParams
------------

:cpp:class:`einsums::compute_graph::EinsumParams`

Node
----

:cpp:class:`einsums::compute_graph::Node`

OpKind
------

:cpp:enum:`einsums::compute_graph::OpKind`

EinsumDescriptor
-----------------

:cpp:class:`einsums::compute_graph::EinsumDescriptor`

ScaleDescriptor
----------------

:cpp:class:`einsums::compute_graph::ScaleDescriptor`

PermuteDescriptor
------------------

:cpp:class:`einsums::compute_graph::PermuteDescriptor`

ConditionalDescriptor
----------------------

:cpp:class:`einsums::compute_graph::ConditionalDescriptor`

LoopDescriptor
--------------

:cpp:class:`einsums::compute_graph::LoopDescriptor`

AllocDescriptor
----------------

:cpp:class:`einsums::compute_graph::AllocDescriptor`

LoopNode (Pipeline)
-------------------

:cpp:class:`einsums::compute_graph::LoopNode`

String Einsum
=============

ParsedEinsumSpec
-----------------

:cpp:class:`einsums::compute_graph::ParsedEinsumSpec`

EinsumFormatString
-------------------

:cpp:class:`einsums::compute_graph::EinsumFormatString`

Type Aliases
============

:cpp:type:`einsums::compute_graph::TensorId`
:cpp:type:`einsums::compute_graph::NodeId`
:cpp:type:`einsums::compute_graph::LoopCondition`
:cpp:type:`einsums::compute_graph::OpData`

Optimization Passes
===================

OptimizerPass (base class)
---------------------------

:cpp:class:`einsums::compute_graph::OptimizerPass`

CSE
---

:cpp:class:`einsums::compute_graph::passes::CSE`

Reorder
-------

:cpp:class:`einsums::compute_graph::passes::Reorder`

MemoryPlanning
--------------

:cpp:class:`einsums::compute_graph::passes::MemoryPlanning`

ChainParenthesization
----------------------

:cpp:class:`einsums::compute_graph::passes::ChainParenthesization`

ConstantFolding
----------------

:cpp:class:`einsums::compute_graph::passes::ConstantFolding`

DeadNodeElimination
--------------------

:cpp:class:`einsums::compute_graph::passes::DeadNodeElimination`

ScaleAbsorption
----------------

:cpp:class:`einsums::compute_graph::passes::ScaleAbsorption`

LoopInvariantHoisting
----------------------

:cpp:class:`einsums::compute_graph::passes::LoopInvariantHoisting`

InplaceOptimization
--------------------

:cpp:class:`einsums::compute_graph::passes::InplaceOptimization`

GEMMBatching
-------------

:cpp:class:`einsums::compute_graph::passes::GEMMBatching`

PermuteFusion
--------------

:cpp:class:`einsums::compute_graph::passes::PermuteFusion`

SymmetryPropagation
--------------------

:cpp:class:`einsums::compute_graph::passes::SymmetryPropagation`

See :doc:`symmetry` for the user-facing guide.

Free Functions
==============

:cpp:func:`einsums::compute_graph::make_handle`
:cpp:func:`einsums::compute_graph::make_scalar_handle`
:cpp:func:`einsums::compute_graph::op_kind_name`
:cpp:func:`einsums::compute_graph::parse_einsum_spec`
