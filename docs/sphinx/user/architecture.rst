..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _architecture:

#####################
Architecture Overview
#####################

This page is for readers who want to know how Einsums is put together, how
the modules depend on each other, what happens when you call :code:`einsum`,
and where the :ref:`ComputeGraph <modules_Einsums_ComputeGraph>` and the
Python bindings sit relative to the rest. If you just want to use Einsums,
start with the :ref:`Absolute Beginner's Guide <absolute_beginners>` and the
tutorials instead.

Layered design
==============

Einsums is organized as a stack of modules, each with a single
responsibility and an explicit set of dependencies. The build refuses
cycles: every module declares its dependencies in CMake, and adding a new
one means following the established direction of the arrows below.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────┐
    │   ComputeGraph     (deferred IR, optimization passes,         │
    │                     executors, distributed expansion)         │
    └────────────────────────────┬─────────────────────────────────┘
                                 │
    ┌────────────────────────────▼─────────────────────────────────┐
    │   TensorAlgebra     (einsum dispatcher, contraction backends) │
    └────────────────────────────┬─────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────────┐
            │                    │                        │
    ┌───────▼─────────┐ ┌────────▼──────────┐ ┌───────────▼───────┐
    │  LinearAlgebra  │ │   PackedGemm      │ │   (generic loop   │
    │  (gemm, syev,   │ │  (BLIS-style      │ │    fallback,      │
    │   invert, ...)  │ │   pack-and-tile)  │ │    inlined)       │
    └───────┬─────────┘ └────────┬──────────┘ └───────────────────┘
            │                    │
    ┌───────▼─────────┐ ┌────────▼──────────┐
    │      BLAS       │ │                   │
    │   (Einsums-     │ │   (links vendor   │
    │   level API)    │ │    BLAS directly) │
    └───────┬─────────┘ │                   │
            │           │                   │
    ┌───────▼─────────┐ │                   │
    │   BLASBase      │ │                   │
    │   (types, ABI)  │ │                   │
    └───────┬─────────┘ │                   │
            │           │                   │
    ┌───────▼───────────▼──────────────────▼─────────────────────┐
    │            BLASVendor                                       │
    │            (MKL / OpenBLAS / Accelerate)                    │
    └─────────────────────────────────────────────────────────────┘

The leaf modules wrap the moving
parts of the hardware target.
Everything above them is portable C++23 that targets the Einsums-level
abstractions, not the vendor primitives directly.

The dispatch flow
=================

Einsums' headline feature is that an :code:`einsum` call is routed at
compile time to the most specialized backend that can handle the
contraction's index pattern. There is no runtime "which kernel?" branch on
the hot path. The dispatch is settled when the template instantiates.

When you write

.. code-block:: cpp

   einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

the dispatcher (in :code:`TensorAlgebra/Backends/Dispatch.hpp`) does the
following at instantiation time:

1. Extracts the index letters from each operand (``i, j`` on ``C``,
   ``i, k`` on ``A``, ``k, j`` on ``B``).
2. Classifies them into groups: target-only-in-``A`` (the **M** axes),
   target-only-in-``B`` (the **N** axes), shared link (the **K** axes),
   and shared target (**batch** axes).
3. Tries each available backend in order of specialization:

   * Vendor BLAS if the pattern matches a pure ``gemm``,
     ``gemv``, ``ger``, ``syrk``, etc.
   * :ref:`PackedGemm <modules_Einsums_PackedGemm>` for arbitrary-rank
     contractions that don't fit a stock BLAS call. Packs ``A`` and
     ``B`` into BLIS-style tiles and calls vendor ``gemm`` per tile.
   * A generic nested loop as the last resort.

4. The matching backend is instantiated for this specific contraction
   shape. No virtual dispatch, no runtime conditionals. The compiler emits the
   specialized call.

For the example above the dispatcher sees ``ij = ik * kj`` and matches a
pure ``dgemm``. The emitted code is one ``cblas_dgemm`` call plus the
strided-data setup with no extra abstraction overhead.

When the dispatcher can't match a stock BLAS shape — say,
``ijl = ik * kjl`` — it falls back to PackedGemm. PackedGemm's
:code:`PackingPlan` records the strides of each ``M / N / K / batch``
group, packs the inputs into contiguous ``MC × KC`` and ``KC × NC``
tiles tuned to the local cache hierarchy, and calls vendor ``gemm`` per
tile. The user wrote one expression, which was then optimized away by the
compiler before runtime.

If you want to see the dispatch decision for a specific call, raise the
log level to INFO (``--einsums:log-level 2``) and PackedGemm prints
either the chosen plan or the reason it rejected the contraction.

The ComputeGraph layer
======================

Above the eager-dispatch layer sits the
:ref:`ComputeGraph <modules_Einsums_ComputeGraph>`. This is an optional
deferred-execution intermediate representation. Capture a sequence of operations into a
:cpp:class:`einsums::compute_graph::Graph`, run optimization passes over
it, then execute. The model is the same idea as CUDA Graphs or PyTorch FX,
but the passes are tensor-algebra aware and integrate with the
:ref:`distribution layer <modules_Einsums_Comm>`.

Two execution modes coexist:

* **Eager**: every call dispatches and runs immediately. Simple, perfect
  for ad-hoc work and tutorials. This is what every example in the
  beginner's guide uses.

* **Deferred (graph)**: calls inside a ``capture`` block become graph
  nodes instead of running. After capture, pass managers can rewrite the
  graph. This includes fusing adjacent operations, eliminating dead nodes, creating shared common
  intermediates, planning memory reuse, folding linear combinations of
  contractions, creating partitions for distributed execution, and scheduling
  communication. Then :code:`g.execute()` walks the optimized DAG with
  whichever executor (sequential, OMP, dataflow / TaskPool) you pick.

The passes that ship today include common-subexpression elimination,
dead-node elimination, reordering for register reuse, memory planning,
chain-parenthesization, scale absorption, element-wise fusion,
loop-invariant hoisting, linear-combination contraction folding,
plus the distributed-execution passes mentioned below.

You don't have to use the graph. But if you're running a coupled-cluster
inner loop, an SCF cycle, or any iteration that touches the same handful
of tensors hundreds of times, capturing once and executing many times
gets you whole-algorithm rewrites for free that no per-call dispatcher
could see.

Tensor types
============

Einsums has several tensor types, each with a different tradeoff between
flexibility and compile-time information.

* :cpp:type:`einsums::Tensor` is the workhorse: rank fixed at compile
  time, scalar type fixed at compile time, owns its data, dense and
  contiguous (column-major by default). When you write
  ``Tensor<double, 2> A("A", 100, 100)`` the compiler knows the rank
  and data type, so all the dispatch decisions above resolve statically.

* :cpp:class:`einsums::TensorView` is a non-owning window onto another
  tensor with explicit strides. Use it to grab a sub-block, a column,
  a strided slice, or to alias the same buffer with a different shape.

* ``RuntimeTensor`` carries its rank and dtype as runtime values. Use it
  when those can't be known at compile time. For example, the
  :ref:`Python bindings <modules_Einsums_Python>` always go through
  ``RuntimeTensor`` because Python doesn't have C++ templates.

* :cpp:class:`einsums::BlockTensor` is a block-diagonal sparse variant, and
  :cpp:class:`einsums::TiledTensor` is a tiled variant for cache-friendly
  access on operations that walk the matrix in blocks. The Python
  surface currently supports the dense types.

* :cpp:class:`einsums::DiskTensor` (in
  :ref:`TensorIO <modules_Einsums_TensorIO>`) stores its data in HDF5 format on a mass storage device
  rather than RAM. Use it for working sets that are too large to live in memory,
  combined with :code:`tensor_io::Slab` to schedule slab-by-slab reads
  and writes through the ComputeGraph.

Most kernels are written against compile-time-typed ``Tensor``. The
runtime-typed path is a thin shim that does the type erasure once and
then routes into the same kernels.

Python bindings
===============

Einsums' Python surface is generated, not hand-written. A purpose-built
libclang tool (``einsums-pybind``) walks the annotated C++ headers and
emits pybind11 translation units plus ``.pyi`` stubs. The native
extension ends up at ``${BUILD}/lib/einsums/_core.cpython-*.so``.
Pure Python wrappers in ``libs/Einsums/Python/python/einsums/`` add
ergonomics on top.

The consequence of this is that, when a C++ symbol is added with the right annotation, the
Python binding will be created automatically on the next build. No hand-written
glue to forget to update. The bindings always track the C++ surface.

See the :ref:`Python module documentation <modules_Einsums_Python>` for
the call surface and the codegen annotation reference.

Distributed computing
=====================

The :ref:`Comm <modules_Einsums_Comm>` module is the foundation for
running Einsums across MPI ranks. The model is:

* A ``ProcessGrid`` factors the world communicator into a 2D
  :math:`P_r \times P_c` grid with row and column sub-communicators.
* A ``DistributionDescriptor`` says, per tensor axis, whether the axis
  is replicated, split along the grid's row dimension, or split along
  the column dimension.
* :ref:`ComputeGraph <modules_Einsums_ComputeGraph>` passes do the actual
  partitioning and communication scheduling:

  * ``DistributionPlanning`` assigns each einsum's axes to grid axes
    (target indices on ``A`` go to Row, target indices on ``B`` to
    Col, shared link to None or balanced reduction).
  * ``Materialization`` resizes deferred tensors to per-rank slices.
  * ``InputSlicing`` creates rank-local views of pre-allocated inputs.
  * ``SUMMAExpansion`` rewrites einsums on square grids into the
    standard broadcast + GEMM loop.
  * ``CommunicationInsertion`` adds allreduce after distributed
    contractions when needed.
  * ``CommunicationScheduling`` splits blocking allreduce into async
    ``iallreduce`` + ``wait`` so communication overlaps compute.

For development and CI on machines without MPI, ``Comm`` falls back to a
single-process mock backend that exposes the same API, so the passes
above can be exercised without a real MPI installation.

GPU layer
=========

The :ref:`GPU <modules_Einsums_GPU>` module is a thin abstraction over the
device-side primitives, including memory allocation, streams, copies, and device-side
BLAS. The current state is "infrastructure in place, validation on real
hardware pending". The production GPU path will land alongside the planned
Ozaki mixed-precision tile-GEMM work. The same
:ref:`PackedGemm <modules_Einsums_PackedGemm>` plan abstraction is intended
to drive the GPU backend once the kernel side is implemented.

Build configuration
===================

A handful of CMake options gate the optional layers:

* ``EINSUMS_BUILD_PYTHON`` builds the libtooling codegen, generates the
  pybind translation units, and emits ``_core.*.so``.
* ``EINSUMS_WITH_MPI`` replaces ``Comm``'s mock backend with a real
  Open MPI or MPICH integration.
* ``EINSUMS_WITH_CUDA`` / ``EINSUMS_WITH_HIP`` enable the GPU
  backends (pending validation work).
* ``EINSUMS_WITH_PROFILER`` enables the
  :ref:`Profile <modules_Einsums_Profile>` instrumentation hooks.
* ``EINSUMS_WITH_SANITIZERS=address,leak,undefined`` /
  ``EINSUMS_WITH_SANITIZERS=thread`` builds with the corresponding
  Clang/GCC sanitizers.

See the per-module documentation for the option each module reacts to.

Where to go from here
=====================

* For end-to-end usage, jump to the tutorials:
  :ref:`tutorial-einsum`, :ref:`tutorial-compute-graph`,
  :ref:`tutorial-linalg`, :ref:`tutorial-views`.
* For a tour of the public API by module, see the
  :ref:`Developer's Guide <modules_overview>`.
* For performance trade-offs (when each backend fires, how to read a
  PackedGemm log message), see :ref:`tutorial-performance`.
