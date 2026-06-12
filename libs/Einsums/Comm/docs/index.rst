..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_Comm:

####
Comm
####

The ``Comm`` module is Einsums' distributed-computing layer. It provides MPI
communication primitives behind a thin abstraction so the same Einsums code
runs on a single process through a mock backend or across many ranks through
Open MPI or MPICH.

The module is the foundation for distributed tensor support in the
:ref:`ComputeGraph <modules_Einsums_ComputeGraph>` and exposes:

- A 2D process grid with row and column sub-communicators.
- Per-dimension distribution descriptors that partition tensor axes
  across ranks with balanced blocking.
- Blocking and non-blocking collectives (allreduce, broadcast, scatter,
  allgather) usable directly or via ComputeGraph passes.

ProcessGrid
===========

``ProcessGrid`` is a 2D :math:`P_r \times P_c` factorization of the world
communicator. The factorization defaults to near-square. Row and column
sub-communicators are exposed for distribution-aware kernels such as
SUMMA-style GEMM.

.. code-block:: cpp

    #include <Einsums/Comm/ProcessGrid.hpp>

    using namespace einsums::comm;

    // Default grid, auto-factored near-square on first access
    auto &grid = ProcessGrid::default_grid();

    int row    = grid.my_row();
    int col    = grid.my_col();
    int p_rows = grid.rows();
    int p_cols = grid.cols();

    // Sub-communicators
    Communicator const &row_comm = grid.row_comm();
    Communicator const &col_comm = grid.col_comm();

DistributionDescriptor
======================

``DistributionDescriptor`` describes how each axis of a tensor is partitioned
across the process grid. Every axis takes one ``GridAxis`` value: ``None``
leaves the axis replicated, ``Row`` splits it along the grid's row dimension,
and ``Col`` splits it along the column dimension.

.. code-block:: cpp

    #include <Einsums/Comm/DistributionDescriptor.hpp>

    using namespace einsums::comm;

    // 2D tensor with rows split across grid rows, columns replicated
    DistributionDescriptor d;
    d.dim_to_axis = {GridAxis::Row, GridAxis::None};
    d.global_dims = {1024, 512};
    d.grid        = &grid;

    // Local dimensions for this rank
    auto local = d.local_dims_for(world_rank());

Collectives
===========

Both blocking and non-blocking versions are available:

.. code-block:: cpp

    #include <Einsums/Comm/Collectives.hpp>

    using namespace einsums::comm;

    // Blocking allreduce on a Tensor
    Tensor<double, 2> A = create_random_tensor("A", 100, 100);
    allreduce(A, ReduceOp::Sum);

    // Non-blocking allreduce + wait, useful for overlap with compute
    auto handle = iallreduce(A, ReduceOp::Sum);
    // ... do other work ...
    wait(handle);

ComputeGraph integration
========================

The :ref:`ComputeGraph <modules_Einsums_ComputeGraph>` ships several passes that
operate on this layer:

- ``DistributionPlanning`` classifies einsum indices and assigns each axis a
  distribution (target indices on A go to Row, target indices on B to Col,
  shared indices to None or balanced).
- ``Materialization`` resizes deferred tensors to local partitions using
  ``DistributionDescriptor::local_dims_for()``.
- ``InputSlicing`` creates rank-local views of pre-allocated inputs.
- ``SUMMAExpansion`` rewrites einsum into broadcast+GEMM loops on square
  grids.
- ``CommunicationInsertion`` inserts ``allreduce`` for replicated outputs from
  distributed inputs.
- ``CommunicationScheduling`` splits ``allreduce`` into async
  ``iallreduce`` + ``wait`` for overlap with compute.

Build configuration
===================

The MPI backend is opt-in. Build with::

    cmake -S . -B build -DEINSUMS_WITH_MPI=ON

Without ``EINSUMS_WITH_MPI``, ``Comm`` falls back to a single-process mock
backend that exercises the same API, useful for development and CI on
machines without an MPI installation.

See the :ref:`API reference <modules_Einsums_Comm_api>` of this module for
the full set of symbols.
