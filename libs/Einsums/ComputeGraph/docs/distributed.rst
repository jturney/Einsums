.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

=======================
Distributed Computing
=======================

Einsums supports transparent distributed computing across MPI ranks. The user
writes ``einsum(C, A, B)`` unchanged; the ComputeGraph passes handle tensor
distribution, communication, and local computation automatically.

Comm Module
===========

The ``Comm`` module (``libs/Einsums/Comm/``) provides the communication layer:

- **Always compiles**: mock backend (serial stubs) when MPI is unavailable
- **Runtime dispatch**: MPI for host tensors; NCCL support planned for GPU
- **Platform detection**: ``comm::has_mpi``, ``comm::has_nccl``, ``comm::is_mock``

.. code-block:: cpp

   #include <Einsums/Comm/Runtime.hpp>
   #include <Einsums/Comm/Collectives.hpp>

   // Automatically initialized/finalized via runtime hooks
   int rank  = comm::world_rank();   // 0 on mock
   int nproc = comm::world_size();   // 1 on mock

   // Collectives use std::span and return expected<void, CommError>
   std::vector<double> data = {1.0, 2.0, 3.0};
   (void)comm::allreduce_inplace<double>(data, comm::ReduceOp::Sum);
   (void)comm::broadcast<double>(data, /*root=*/0);
   (void)comm::barrier();

CMake Configuration
-------------------

.. code-block:: bash

   cmake -DEINSUMS_WITH_MPI=ON ...

When ``EINSUMS_WITH_MPI=OFF`` (default), the mock backend compiles and all
distributed code runs serially on a single rank.

Process Grid
=============

The ``ProcessGrid`` arranges P MPI ranks into a 2D grid (Pr × Pc):

.. code-block:: cpp

   #include <Einsums/Comm/ProcessGrid.hpp>

   auto &grid = comm::ProcessGrid::default_grid();
   // 4 ranks → 2×2 grid, 6 ranks → 3×2 grid, 7 ranks → 7×1 grid

   int my_row = grid.my_row();  // Row coordinate [0, Pr)
   int my_col = grid.my_col();  // Column coordinate [0, Pc)

   // Sub-communicators for SUMMA broadcasts
   auto &row_comm = grid.row_comm();  // All ranks in same row
   auto &col_comm = grid.col_comm();  // All ranks in same column

The grid dimensions are auto-computed to be near-square (minimizes \|Pr - Pc\|).

Distribution Strategy
======================

The ``DistributionPlanning`` pass assigns each tensor dimension to a grid axis:

- **GridAxis::Row**: distributed across Pr grid rows
- **GridAxis::Col**: distributed across Pc grid columns
- **GridAxis::None**: replicated, with full data on each rank

Index classification for ``C[i,j] = A[i,k] * B[k,j]``:

========== ============ ======== =====================
Index      Role         Axis     Meaning
========== ============ ======== =====================
``i``      target_a     Row      In C and A, not B
``j``      target_b     Col      In C and B, not A
``k``      link         None     Contraction (in A,B, not C)
========== ============ ======== =====================

For batch indices, those present in all three tensors, the pass assigns them to
the grid axis with fewer dimensions so far, a load-balancing heuristic.

**Balanced blocking**: Elements are distributed evenly. For N=41 across P=4
ranks the split is {11, 11, 10, 9} rather than {11, 11, 11, 8}, so the maximum
imbalance is 1.

Distribution Patterns
======================

Outer-Product (zero communication)
------------------------------------

When target indices are distributed and link indices are not:

.. code-block:: text

   C[i,j] → [Row, Col] = (M/Pr, N/Pc)
   A[i,k] → [Row, None] = (M/Pr, K)
   B[k,j] → [None, Col] = (K, N/Pc)

Each rank computes its local partition independently. No communication needed.
Works for any grid shape.

SUMMA (broadcast communication)
----------------------------------

On square grids (Pr = Pc), link indices are also distributed:

.. code-block:: text

   C[i,j] → [Row, Col] = (M/Pr, N/Pc)
   A[i,k] → [Row, Col] = (M/Pr, K/Pc)
   B[k,j] → [Row, Col] = (K/Pr, N/Pc)

The SUMMA loop iterates Pc panels, broadcasting A along rows and B along columns:

.. code-block:: text

   for p = 0..Pc-1:
       A_panel = broadcast(A_local if my_col==p, row_comm)
       B_panel = broadcast(B_local if my_row==p, col_comm)
       C_local += A_panel * B_panel

Higher-Rank Tensors
--------------------

Higher-rank contractions fold multiple indices onto the same grid axis:

.. code-block:: text

   C[n,i,j] = A[n,i,k] * B[n,j,k]
   → n (shared/batch) → Row, i (target_a) → Row, j (target_b) → Col
   → C[Row,Row,Col] = (n/Pr, i/Pr, j/Pc)

Automatic Input Slicing
========================

Pre-allocated tensors (not deferred) are automatically sliced by the
``InputSlicing`` pass. Each rank gets a temporary view of its local partition:

.. code-block:: cpp

   auto A = create_random_tensor<double>("A", M, K);  // Full, pre-allocated
   auto &C = graph.declare_zero_tensor<double, 2>("C", M, N);  // Deferred

   {
       cg::CaptureGuard guard(graph);
       cg::einsum("ik;kj->ij", &C, A, B);
   }
   // InputSlicing detects that A shares index i with distributed C
   // → automatically creates a view of A's local rows for each rank

This also works for distributed permute (cross-axis redistribution):

.. code-block:: cpp

   cg::permute("ji <- ij", 0.0, &C, 1.0, A);
   // C[j,i] = A[i,j] — j→Row and i→Col cross the grid axes
   // InputSlicing slices A along both dimensions for each rank

Chain Conflict Resolution
==========================

When an intermediate tensor's distribution conflicts with its usage in a
subsequent contraction, the pass detects and resolves it:

.. code-block:: text

   T[i,j] = A[i,k] * B[k,j]   → T gets [Row, Col]
   C[i,j] = T[i,k] * D[k,j]   → T's j-dim (Col) is now link index k!

``DistributionPlanning`` detects that T's Col dimension becomes a link index and
downgrades it to None: T gets [Row, None] instead. This prevents incorrect
partial results.

Filling Distributed Tensors
============================

``declare_tensor_filled`` declares a deferred tensor with a user-provided fill
function. The fill lambda is called after materialization and distribution, and
it receives a tensor with ``range()`` and ``global()`` methods for
distribution-aware access.

.. code-block:: cpp

   auto &T = graph.declare_tensor_filled<double, 2>("T", Dim<2>{M, N},
       [&](auto& T) {
           auto [i0, i1] = T.range(0);  // Global index range for this rank
           auto [j0, j1] = T.range(1);
           for (size_t i = i0; i < i1; i++)
               for (size_t j = j0; j < j1; j++)
                   T.global(i, j) = compute(i, j);  // Global indices, auto-mapped
       });

Key methods on the tensor
^^^^^^^^^^^^^^^^^^^^^^^^^

``T.range(dim)``
   Returns ``{start, end}``, the global index range this rank owns along
   dimension ``dim``. For a non-distributed dimension this is ``{0, full_size}``.
   Use it to skip irrelevant work, such as shell batches outside the local range.

``T.global(indices...)``
   Access an element using global indices. Automatically subtracts the local
   offset. For non-distributed tensors, equivalent to ``T(indices...)``.

``T.global_dim(dim)``
   Returns the global (pre-partition) size along dimension ``dim``.

The same code works on 1 rank (serial) and N ranks (distributed). The user
only adds ``range()`` checks to skip irrelevant batches.

Chemistry Example: Shell-Batch ERI Fill
-----------------------------------------

Integral engines compute in shell quartets. Use ``range()`` to skip shells
outside this rank's partition:

.. code-block:: cpp

   auto &eri = graph.declare_tensor_filled<double, 4>("ERI",
       Dim<4>{nao, nao, nao, nao},
       [&](auto& T) {
           auto [p0, p1] = T.range(0);
           auto [q0, q1] = T.range(1);
           auto [r0, r1] = T.range(2);
           auto [s0, s1] = T.range(3);

           for (int P = 0; P < nshell; P++) {
               if (shell_end[P] <= p0 || shell_start[P] >= p1) continue;
               for (int Q = 0; Q < nshell; Q++) {
                   if (shell_end[Q] <= q0 || shell_start[Q] >= q1) continue;
                   for (int R = 0; R < nshell; R++) {
                       if (shell_end[R] <= r0 || shell_start[R] >= r1) continue;
                       for (int S = 0; S < nshell; S++) {
                           if (shell_end[S] <= s0 || shell_start[S] >= s1) continue;

                           auto buf = engine.compute(P, Q, R, S);
                           // Copy shell block using global indices
                           for (size_t p = shell_start[P]; p < shell_end[P] && p < p1; p++)
                               for (size_t q = ...) // similar bounds
                                   T.global(p, q, r, s) = buf[...];
                       }
                   }
               }
           }
       });

   // Then contract: half[p,q,r,a] = ERI[p,q,r,s] * C[s,a]
   auto &half = graph.declare_zero_tensor<double, 4>("half", nao, nao, nao, nmo);
   {
       cg::CaptureGuard guard(graph);
       cg::einsum("pqrs;sa->pqra", &half, eri, C);
   }

Each rank computes only its local shell quartets. The einsum on the distributed
ERI tensor is automatically parallelized by the pass infrastructure.

Parallelizing the Fill Lambda
-------------------------------

The fill lambda runs on the host CPU in a single thread by default. Since it
executes during ``graph.execute()`` where the full runtime is available, you can
parallelize it with OpenMP or the einsums TaskPool.

``T.global()`` is thread-safe: each shell pair writes to non-overlapping
regions, so no locks are needed.

OpenMP (simplest)
^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   auto &eri = graph.declare_tensor_filled<double, 4>("ERI",
       Dim<4>{nao, nao, nao, nao},
       [&](auto& T) {
           auto [p0, p1] = T.range(0);
           auto [q0, q1] = T.range(1);
           auto [r0, r1] = T.range(2);
           auto [s0, s1] = T.range(3);

           #pragma omp parallel for collapse(2) schedule(dynamic)
           for (int P = 0; P < nshell; P++) {
               for (int Q = 0; Q <= P; Q++) {
                   if (shell_end[P] <= p0 || shell_start[P] >= p1) continue;
                   if (shell_end[Q] <= q0 || shell_start[Q] >= q1) continue;

                   // Each thread needs its own engine (not thread-safe)
                   auto& engine = engines.local();

                   for (int R = 0; R <= P; R++) {
                       if (shell_end[R] <= r0 || shell_start[R] >= r1) continue;
                       for (int S = 0; S <= (P == R ? Q : R); S++) {
                           if (shell_end[S] <= s0 || shell_start[S] >= s1) continue;

                           auto buf = engine.compute(P, Q, R, S);
                           // T.global() is thread-safe for non-overlapping writes
                           copy_shell_block(T, buf, P, Q, R, S);
                       }
                   }
               }
           }
       });

TaskPool (integrates with einsums task infrastructure)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include <Einsums/TaskPool/TaskPool.hpp>

   auto &eri = graph.declare_tensor_filled<double, 4>("ERI",
       Dim<4>{nao, nao, nao, nao},
       [&](auto& T) {
           auto [p0, p1] = T.range(0);
           auto [q0, q1] = T.range(1);

           auto& pool = task_pool::TaskPool::get_singleton();
           std::vector<task_pool::TaskHandle<void>> handles;

           // Submit one task per significant shell pair (P, Q)
           for (int P = 0; P < nshell; P++) {
               if (shell_end[P] <= p0 || shell_start[P] >= p1) continue;
               for (int Q = 0; Q <= P; Q++) {
                   if (shell_end[Q] <= q0 || shell_start[Q] >= q1) continue;

                   handles.push_back(pool.submit(
                       fmt::format("eri_{}_{}", P, Q),
                       [&T, P, Q, /* capture needed state */]() {
                           auto& engine = engines.local();
                           // Compute all (R, S) for this (P, Q)
                           for (int R = 0; R <= P; R++)
                               for (int S = 0; S <= (P == R ? Q : R); S++) {
                                   auto buf = engine.compute(P, Q, R, S);
                                   copy_shell_block(T, buf, P, Q, R, S);
                               }
                       }));
               }
           }

           // Wait for all tasks
           for (auto& h : handles) h.wait();
       });

The OpenMP version is simpler. The TaskPool version gives more control (task
naming for profiling, priority, and potential dataflow dependencies between tasks).

Communication Passes
=====================

``CommunicationInsertion``
   Inserts allreduce after any compute node with distributed inputs and
   replicated outputs. Evaluates all compute nodes (einsum, scale, axpy,
   permute), not just einsums.

``CommunicationElimination``
   Removes redundant allreduces (e.g., back-to-back allreduce on same tensor).

``CommunicationScheduling``
   Converts synchronous allreduce to async (iallreduce + wait) for overlapping
   communication with computation via DataflowExecutor.

**Ordering**: Allreduce is inserted immediately after the producing node.
Element-wise operations (scale, axpy) captured after the einsum execute AFTER
the allreduce, operating on the globally-reduced result.

Capture-Aware Dot and Norm
===========================

Graph-aware versions that write to pre-allocated scalars:

.. code-block:: cpp

   double result = 0.0;
   {
       cg::CaptureGuard guard(graph);
       cg::dot(&result, A, B);    // Records into graph
       cg::norm(&nrm, linear_algebra::Norm::Frobenius, A);
   }
   // CommunicationInsertion adds allreduce for scalar results
   // from distributed inputs

Complete Example
=================

See ``examples/DistributedGEMM.cpp`` for a complete working example.

.. code-block:: bash

   cmake --build build --target CG_DistributedGEMM
   mpirun -np 4 ./build/bin/CG_DistributedGEMM

.. code-block:: cpp

   namespace cg = einsums::compute_graph;

   // Deferred output — passes handle distribution automatically
   cg::Graph graph("distributed_gemm");
   auto &C = graph.declare_zero_tensor<double, 2>("C", M, N);

   {
       cg::CaptureGuard guard(graph);
       cg::einsum("ik;kj->ij", &C, A, B);
   }

   auto pm = cg::PassManager::create_default();
   graph.apply(pm);
   // DistributionPlanning: C → [Row, Col] on 2×2 grid
   // Materialization: C_local = (M/2, N/2)
   // InputSlicing: A sliced along rows, B sliced along cols
   // No allreduce needed (outer-product)

   graph.execute();
   // Each rank computed its (M/2, N/2) partition of C
