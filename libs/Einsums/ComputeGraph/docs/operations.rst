.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

==========================
Graph-Aware Operations
==========================

All graph-aware operation wrappers live in the ``einsums::compute_graph`` namespace.
They mirror the signatures of their eager counterparts. When capture is active, they
record a node; when not capturing, they delegate directly to the real function.

TensorAlgebra Operations
========================

einsum
------

.. code-block:: cpp

   // With explicit prefactors: C = c_pf * C + ab_pf * A * B
   cg::einsum("ik;kj->ij", c_pf, &C, ab_pf, A, B);

   // With default prefactors (c_pf=0, ab_pf=1): C = A * B
   cg::einsum("ik;kj->ij", &C, A, B);

   // With conjugation
   cg::einsum<true, false>("ik;kj->ij", &C, A, B);

Records an ``OpKind::Einsum`` node with ``EinsumDescriptor`` metadata containing
the contraction pattern, prefactors, and conjugation flags.

permute
-------

.. code-block:: cpp

   // C = beta * C + alpha * permute(A)
   cg::permute("ji <- ij", beta, &C, alpha, A);

transpose
---------

.. code-block:: cpp

   cg::transpose(&C, A);

element_transform
-----------------

.. code-block:: cpp

   cg::element_transform(&C, [](double x) { return x * x; });

LinearAlgebra - BLAS Level
==========================

scale
-----

.. code-block:: cpp

   cg::scale(2.5, &A);   // A *= 2.5

shift
-----

.. code-block:: cpp

   cg::shift(3.0, &A);   // A += 3.0 (adds a scalar to every element)

The additive complement of ``scale``. It is a tight compiled loop rather than a
Python-callback ``element_transform``, so it backs the Python numpy-style
scalar ``+`` and ``-`` operators, such as ``A + c`` and ``A += c``. Records an
opaque ``OpKind::Custom`` node when capturing.

gemm
----

.. code-block:: cpp

   // C = alpha * op(A) * op(B) + beta * C
   cg::gemm<false, false>(alpha, A, B, beta, &C);   // TransA=N, TransB=N
   cg::gemm<true, false>(alpha, A, B, beta, &C);    // TransA=T, TransB=N

gemv
----

.. code-block:: cpp

   // y = alpha * op(A) * z + beta * y
   cg::gemv<false>(alpha, A, z, beta, &y);

ger
---

.. code-block:: cpp

   // A += alpha * X * Y^T
   cg::ger(alpha, X, Y, &A);

axpy / axpby
-------------

.. code-block:: cpp

   cg::axpy(alpha, X, &Y);            // Y += alpha * X
   cg::axpby(alpha, X, beta, &Y);     // Y = alpha * X + beta * Y

dot
---

.. code-block:: cpp

   // Eager (outside capture): returns the scalar inner product.
   auto result = cg::dot(A, B);   // the scalar-return form throws during capture

   // Recorded (inside capture): writes into a pre-allocated [1] tensor at
   // execute time — the graph-resident form (also accepts cg::view operands).
   cg::dot(&result_tensor, A, B);

norm
----

.. code-block:: cpp

   auto n = cg::norm(linear_algebra::Norm::Frobenius, A);

trace
-----

.. code-block:: cpp

   // Eager (outside capture): returns the diagonal sum.
   auto t = cg::trace(A);

   // Recorded (inside capture): writes the result into a pre-allocated scalar
   // at execute time. Required form for graph-resident traces — like cg::dot
   // / cg::norm, the eager scalar-return form throws during capture.
   double t = 0.0;
   cg::trace(&t, A);          // A must be square
   cg::trace(&t, slice);      // also works on cg::view results

Records a label-only ``OpKind::Trace`` node. The executor body is just a
diagonal-sum loop, with no special pass treatment.

sum / max
---------

.. code-block:: cpp

   // Reduce every element to a scalar, written into result->data()[0].
   // Like dot/trace: recorded into the graph when capturing, eager otherwise.
   cg::sum(&result, A);   // result[0] = Σ A_i   (any dtype)
   cg::max(&result, A);   // result[0] = max A_i (real dtypes only)

``result`` is a pre-allocated rank-1 ``[1]`` tensor, a graph-native scalar
handle, as for the scalar-writing ``cg::dot`` and ``cg::trace``. Both are
stride-correct, so they reduce slice and transpose views, not just contiguous
storage. They back the Python ``A.sum()`` / ``A.mean()`` / ``A.max()``
methods. ``max`` is real-only, since complex has no natural ordering, so use
``cg::norm(MAXABS, A)`` for the largest magnitude. Each records an opaque
``OpKind::Custom`` node when capturing.

direct_product
--------------

.. code-block:: cpp

   cg::direct_product(alpha, A, B, beta, &C);   // C = alpha * (A ⊙ B) + beta * C

symm_gemm
----------

.. code-block:: cpp

   cg::symm_gemm<false, false>(A, B, &C);   // C = B^T * A * B

LinearAlgebra - LAPACK Level
=============================

These operations always execute eagerly during capture, because they produce
output tensors that subsequent operations need. They are still recorded as nodes
for replay.

syev
----

.. code-block:: cpp

   // In-place form: A → eigenvectors, W → eigenvalues
   cg::syev(&A, &W);

   // Returning form: returns (eigenvectors, eigenvalues)
   auto [evecs, evals] = cg::syev(A);

heev
----

.. code-block:: cpp

   cg::heev(&A, &W);   // Hermitian eigendecomposition (complex types)

gesv
----

.. code-block:: cpp

   int status = cg::gesv(&A, &B);   // Solve AX = B

invert
------

.. code-block:: cpp

   cg::invert(&A);   // In-place matrix inverse

svd / svd_dd / truncated_svd
------------------------------

.. code-block:: cpp

   auto [U, S, Vt] = cg::svd(A);
   auto [U2, S2, Vt2] = cg::svd_dd(A);
   auto [U3, S3, Vt3] = cg::truncated_svd(A, k);

qr
--

.. code-block:: cpp

   auto [Q, R] = cg::qr(A);

pow
---

.. code-block:: cpp

   auto A_half = cg::pow(A, 0.5);   // Matrix square root

det
---

.. code-block:: cpp

   auto d = cg::det(A);   // Matrix determinant

Custom Operations
=================

custom
------

Record a user-defined operation with typed tensor inputs/outputs for
dependency tracking:

.. code-block:: cpp

   cg::custom("compute_ERI", {}, {&ERI}, [&]() {
       compute_two_electron_integrals(basis, ERI);
   });

   // Full form with explicit input/output tuples:
   cg::custom("fock_build",
       std::make_tuple(std::cref(D), std::cref(ERI)),
       std::make_tuple(std::ref(F)),
       [&]() { build_fock_matrix(D, ERI, F); });

Disk I/O Operations
===================

read / write (synchronous)
--------------------------

.. code-block:: cpp

   cg::read("load integrals", "integrals.h5", "/eri", &ERI, [&]() {
       einsums::read(ERI, "integrals.h5", "/eri");
   });

   cg::write("checkpoint F", "checkpoint.h5", "/fock", &F, [&]() {
       einsums::write(F, "checkpoint.h5", "/fock");
   });

The graph tracks input/output dependencies so that downstream operations
wait for the read to complete, and the write waits for its input tensor
to be computed.

read_async / write_async (asynchronous)
----------------------------------------

For I/O-compute overlap, use the async variants. They accept three lambdas:

- **start_fn**: Begins the I/O operation (should return quickly)
- **finish_fn**: Waits for the I/O to complete and finalizes the tensor
- **sync_fn**: Synchronous fallback for Sequential/OpenMP executors

.. code-block:: cpp

   std::future<void> io_future;

   cg::read_async("load ERI", "integrals.h5", "/eri", &ERI,
       /*start*/  [&]() { io_future = std::async(std::launch::async,
                      [&] { load_from_disk(ERI); }); },
       /*finish*/ [&]() { io_future.get(); },
       /*sync*/   [&]() { load_from_disk(ERI); }
   );

How async I/O works with executors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **SequentialExecutor / OpenMPExecutor**: Call ``sync_fn`` directly (no overlap).
- **DataflowExecutor**: Calls ``start_fn`` as soon as predecessors complete,
  then submits ``finish_fn`` as a separate task. Independent compute nodes
  can execute between start and finish, overlapping I/O with computation.

Combine with IOPrefetch for maximum overlap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``IOPrefetch`` pass moves ``DiskRead`` nodes to the earliest legal position
in the schedule, maximizing the window between ``async_start`` and the first
consumer.

.. code-block:: cpp

   auto pm = cg::PassManager::create_default();  // Includes IOPrefetch
   graph.apply(pm);

   cg::DataflowExecutor df;
   graph.execute(df);  // Async reads overlap with independent compute

Deferred Tensor Operations
============================

declare_tensor / declare_zero_tensor
--------------------------------------

Declare tensors with deferred allocation on Workspace, Pipeline, or Graph:

.. code-block:: cpp

   cg::Workspace ws("calc");
   auto &A = ws.declare_tensor<double, 2>("A", 1000, 1000);
   auto &B = ws.declare_zero_tensor<double, 2>("B", 500, 500);

   cg::Pipeline pipeline("scf");
   auto &F = pipeline.declare_zero_tensor<double, 2>("F", n, n);

   cg::Graph graph("work");
   auto &tmp = graph.declare_tensor<double, 2>("tmp", m, n);

Memory is allocated by the ``MaterializationPass`` during ``apply(pm)``.
See :doc:`workspace` for the full lifecycle.

Internal Node Types
====================

These ``OpKind`` values are inserted by passes, not directly by users:

- ``Materialize``: allocates storage for a deferred tensor
- ``Initialize``: fills tensor with zeros or random values after materialization
- ``HostToDevice`` / ``DeviceToHost``: GPU data transfers
- ``Allreduce`` / ``Broadcast`` / ``Allgather`` / ``Scatter`` / ``Barrier``: MPI collectives
