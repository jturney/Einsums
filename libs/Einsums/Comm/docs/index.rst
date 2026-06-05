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
runs on a single process (using a mock backend) or across many ranks (using
Open MPI or MPICH).

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
communicator. The factorization defaults to near-square; row and column
sub-communicators are exposed for distribution-aware kernels (for example,
SUMMA-style GEMM).

.. code-block:: cpp

    #include <Einsums/Comm/ProcessGrid.hpp>

    using namespace einsums::comm;

    // World grid (auto-factored near-square)
    auto const &grid = ProcessGrid::world();

    int rank      = grid.rank();
    int row       = grid.row();
    int col       = grid.col();
    int p_rows    = grid.p_rows();
    int p_cols    = grid.p_cols();

    // Sub-communicators
    Communicator const &row_comm = grid.row_comm();
    Communicator const &col_comm = grid.col_comm();

DistributionDescriptor
======================

``DistributionDescriptor`` describes how each axis of a tensor is partitioned
across the process grid. Each axis is one of ``None`` (replicated),
``Row`` (split along the grid's row dimension), or ``Col`` (split along
the column dimension).

.. code-block:: cpp

    #include <Einsums/Comm/DistributionDescriptor.hpp>

    using namespace einsums::comm;

    // 2D tensor with rows split across grid rows, columns replicated
    DistributionDescriptor d{Distribution::Row, Distribution::None};

    // Local dimensions for THIS rank, given a global shape
    auto local = d.local_dims_for({1024, 512}, grid);

Collectives
===========

Both blocking and non-blocking versions are available:

.. code-block:: cpp

    #include <Einsums/Comm/Collectives.hpp>

    using namespace einsums::comm;

    // Blocking allreduce on a Tensor
    Tensor<double, 2> A = create_random_tensor("A", 100, 100);
    allreduce(A, ReduceOp::Sum);

    // Non-blocking allreduce + wait — useful for overlap with compute
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
