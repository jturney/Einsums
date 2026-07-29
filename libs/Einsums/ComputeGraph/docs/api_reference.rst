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

Workspace
---------

:cpp:class:`einsums::compute_graph::Workspace`

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

DataflowExecutor
----------------

:cpp:class:`einsums::compute_graph::DataflowExecutor`

MPIExecutor
-----------

:cpp:class:`einsums::compute_graph::MPIExecutor`

DependencyInfo
--------------

:cpp:struct:`einsums::compute_graph::DependencyInfo`

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

See :ref:`the pass catalog <computegraph_optimization_passes>` for what each one
matches, what it reports, and where it sits in the default pipeline.

Infrastructure
--------------

- :cpp:class:`einsums::compute_graph::OptimizerPass` - the base class every pass derives from.
- :cpp:class:`einsums::compute_graph::PassManager` - ordered pass list; ``create_default()``, ``create_for(OptLevel)``, ``explain()``.
- :cpp:enum:`einsums::compute_graph::OptLevel` - ``O0`` / ``O1`` / ``O2``.

Lowering and cleanup
--------------------

- :cpp:class:`einsums::compute_graph::passes::TiledExpansion`
- :cpp:class:`einsums::compute_graph::passes::ConstantFolding`
- :cpp:class:`einsums::compute_graph::passes::ScaleAbsorption`
- :cpp:class:`einsums::compute_graph::passes::PermuteFusion`
- :cpp:class:`einsums::compute_graph::passes::CSE`
- :cpp:class:`einsums::compute_graph::passes::DeadNodeElimination`
- :cpp:class:`einsums::compute_graph::passes::ElementWiseFusion`

Algebraic rewrites
------------------

- :cpp:class:`einsums::compute_graph::passes::SymmetrizedAccumulation`
- :cpp:class:`einsums::compute_graph::passes::LinearCombinationContractionFolding`
- :cpp:class:`einsums::compute_graph::passes::StreamContractionFusion`
- :cpp:class:`einsums::compute_graph::passes::DistributiveFactoring` - opt-in.

Scheduling and planning
-----------------------

- :cpp:class:`einsums::compute_graph::passes::LoopInvariantHoisting`
- :cpp:class:`einsums::compute_graph::passes::ScratchPrivatization`
- :cpp:class:`einsums::compute_graph::passes::ContractionPlanning`
- :cpp:class:`einsums::compute_graph::passes::GEMMBatching`
- :cpp:class:`einsums::compute_graph::passes::Reorder`
- :cpp:class:`einsums::compute_graph::passes::IOPrefetch`

Memory and symmetry
-------------------

- :cpp:class:`einsums::compute_graph::passes::DistributionPlanning` - always runs; a no-op on one rank.
- :cpp:class:`einsums::compute_graph::passes::Materialization`
- :cpp:class:`einsums::compute_graph::passes::SymmetryPropagation` - see :doc:`symmetry` for the user-facing guide.
- :cpp:class:`einsums::compute_graph::passes::InplaceOptimization`
- :cpp:class:`einsums::compute_graph::passes::FreeInsertion`
- :cpp:class:`einsums::compute_graph::passes::MemoryPlanning`

GPU (GPU builds only)
---------------------

- :cpp:class:`einsums::compute_graph::passes::GPUPlacement`
- :cpp:class:`einsums::compute_graph::passes::TransferInsertion`
- :cpp:class:`einsums::compute_graph::passes::TransferElimination`
- :cpp:class:`einsums::compute_graph::passes::GPUDiagnostics`
- :cpp:class:`einsums::compute_graph::passes::StreamAssignment`

Distributed (MPI builds only)
-----------------------------

- :cpp:class:`einsums::compute_graph::passes::InputSlicing`
- :cpp:class:`einsums::compute_graph::passes::SUMMAExpansion`
- :cpp:class:`einsums::compute_graph::passes::CommunicationInsertion`
- :cpp:class:`einsums::compute_graph::passes::CommunicationElimination`
- :cpp:class:`einsums::compute_graph::passes::CommunicationScheduling`

Free Functions
==============

:cpp:func:`einsums::compute_graph::make_handle`
:cpp:func:`einsums::compute_graph::make_scalar_handle`
:cpp:func:`einsums::compute_graph::op_kind_name`
:cpp:func:`einsums::compute_graph::parse_einsum_spec`
