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
2. Classifies them into groups: the ``M`` axes appear only in the target and
   ``A``, the ``N`` axes appear only in the target and ``B``, the ``K`` axes
   are shared links, and the batch axes are shared targets.
3. Tries each available backend in order of specialization:

   * Vendor BLAS if the pattern matches a pure ``gemm``,
     ``gemv``, ``ger``, ``syrk``, etc.
   * :ref:`PackedGemm <modules_Einsums_PackedGemm>` for arbitrary-rank
     contractions that don't fit a stock BLAS call. It either hands the
     whole contraction to a vendor ``gemm`` whose strides happen to fit,
     or runs its own BLIS-style packed loops over a register-blocked
     micro-kernel. It may also decline, in which case the generic loop
     takes the contraction.
   * A generic nested loop as the last resort.

4. The matching backend is instantiated for this specific contraction
   shape. No virtual dispatch, no runtime conditionals. The compiler emits the
   specialized call.

For the example above the dispatcher sees ``ij = ik * kj`` and matches a
pure ``dgemm``. The emitted code is one ``cblas_dgemm`` call plus the
strided-data setup with no extra abstraction overhead.

When the dispatcher can't match a stock BLAS shape, for example
``ijl = ik * kjl``, it falls back to PackedGemm. The user wrote one
expression, and the choice to route it here was made by the compiler; what
happens next is decided at run time, from the plan and the machine.

Planning
--------

:code:`try_packed_gemm` builds a :code:`ContractionSpec` from the index
letters and derives a :code:`PackingPlan`, which records the ``M / N / K
/ batch`` dimension groups and each group's stride in each tensor. Plans
are memoized in a process-wide :code:`PackingPlanCache` keyed by a
:code:`ContractionKey`; a repeated call on a graph node can go further and
carry a :code:`ContractionSite`, which caches the resolved plan on the node
so a hit skips the key build and the cache lookup entirely.

PackedGemm can also decline, and a decline is memoized like any other
outcome so later identical calls turn away immediately. The cases:

* the packing topology doesn't fit the contraction pattern at all;
* an outer-product-shaped contraction below roughly 4096 ``M × N``
  elements, where the generic loop measures faster;
* a scatter-layout shape where the caller has a TTGT (Sort+GEMM) fallback
  and this CPU's kernel does not beat it, which includes every batched
  scatter shape.

Execution
---------

Once a plan is accepted, :code:`blis_contraction` runs it by one of four
routes, named by :code:`packed_gemm::last_contraction_route()`:

``gemm_batch``
    A batched shape whose per-slice strides map onto a stock GEMM. Builds
    the pointer arrays and makes a single :code:`blas::gemm_batch` call.

``flatten_gemm``
    Multi-``K`` with a non-scatter ``C``. Flattens ``A`` and ``B`` into
    contiguous ``M × K`` and ``K × N`` buffers and calls vendor ``gemm``.

``single_k_gemm``
    A single-``K`` slice whose strides already describe a GEMM. Called
    directly, with no packing at all.

``packed``
    The engine's own loops. Packs ``A`` and ``B`` into cache-blocked
    panels, ``MC × KC`` and ``KC × NC``, sized by :code:`compute_blocking`
    from the detected L1, L2, and L3 (one column of packed ``A`` in L1, the
    ``A`` panel in half the L2, the ``B`` panel in half the L3), then
    contracts each block one of two ways depending on the resolved kernel's
    :code:`MicroKernelShape::block_gemm`: either the rung's own ``MR × NR``
    register-blocked micro-kernel, or one vendor ``gemm`` per cache block
    into a contiguous temporary followed by a scatter into ``C``. The
    second wins where the vendor library reaches matrix hardware the tile
    kernel cannot, such as Accelerate's AMX; the first wins where the
    tile kernel is the better path, such as the SME rung's FMOPA tiles.

The first three routes hand the whole contraction to the vendor. That is
usually what you want, but not always: a caller holding a node-scoped
thread width has its vendor calls clamped to one thread, so those routes
would run the node serially while the packed loops would fork from the
same ICV the width raised. :code:`blis_contraction` checks
:code:`blas::vendor_call_is_fenced()` and prefers the packed loops when it
is set. ``gemm_batch`` is exempt, since its vendor entry point is Einsums'
own OpenMP loop over serial GEMMs and carries no fence.

Micro-kernels and SIMD rungs
----------------------------

The micro-kernel bodies are compiled once per instruction-set rung by
:code:`einsums_add_simd_dispatch_sources()`, each copy in its own
namespace, and the rung is chosen at run time in
:code:`MicroKernelDispatch.cpp` by walking the ladder (baseline or native,
V2, V3, V4, SME) down from :code:`simd::selected_arch()`. The result is
cached per element type. :code:`micro_kernel_entry<T>()` and
:code:`micro_kernel_shape<T>()` resolve through the *same* ladder, which is
what keeps the packing geometry matched to the kernel that will consume
it: the NEON and AVX rungs pack to :code:`cpu_config()`'s vector blocking,
while the SME rung packs to ZA-tile blocking and raises ``KC`` so its
accumulators hold the ``C`` block across the whole ``K`` loop.

Complex types reuse the real kernels rather than needing their own. On the
tile path that is Van Zee's 1m method, where ``A`` packs in expanded ``1e``
form and ``B`` packs real and imaginary parts as adjacent ``K`` rows so the
real kernel writes interleaved-complex output directly. On the block-GEMM
path it is the 3m (Karatsuba) method, three real GEMMs per block instead
of one complex GEMM, which is 25% fewer flops and reaches real-only matrix
hardware, at the cost of mildly weaker error bounds than conventional
complex multiplication.

Inspecting a decision
---------------------

Raising the log level to INFO (``--einsums:log:level 2``) reports the
declines described above, not the accepted plans. To see what actually ran,
ask for it directly. Every eager :code:`einsum` overload takes an optional
trailing :code:`detail::AlgorithmChoice *`, which is written with the
backend that took the call: ``GEMM``, ``GEMV``, ``GER``, ``DOT``,
``DIRECT``, ``PACKED_GEMM``, ``SORT_GEMM``, or ``GENERIC``.

.. code-block:: cpp

   using einsums::tensor_algebra::detail::AlgorithmChoice;

   AlgorithmChoice chosen{};
   einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B, &chosen);
   // chosen == AlgorithmChoice::GEMM

One level down, :code:`packed_gemm::last_contraction_route()` names which
of the four routes above PackedGemm took, and for string-spec einsums
:code:`compute_graph::dispatch::last_dispatch_route()` names the route the
last :code:`string_einsum` selected. Both are thread-local and exist for
test introspection, not for steering execution; the test suite asserts on
them so that a silent fall back to the generic loop cannot pass unnoticed.

Every decision is also annotated into the profile, under ``packed_gemm_skip``
for a decline and ``packed_gemm_path`` for an accepted plan.

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

* In eager mode, every call dispatches and runs immediately. It is simple and
  well suited to ad-hoc work and tutorials, and it is what every example in the
  beginner's guide uses.

* In deferred mode, calls inside a ``capture`` block become graph
  nodes instead of running. After capture, pass managers can rewrite the
  graph. This includes fusing adjacent operations, eliminating dead nodes, creating shared common
  intermediates, planning memory reuse, folding linear combinations of
  contractions, creating partitions for distributed execution, and scheduling
  communication. Then :code:`g.execute()` walks the optimized DAG with
  whichever executor you pick: sequential, OMP, dataflow, or TaskPool.

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
  contiguous, and column-major by default. When you write
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
* ``EINSUMS_PYBIND_NUM_TU`` splits every module's generated binding across
  that many translation units, overriding the per-module count a module
  declares with ``PYBIND_NUM_TU``.
  Only ComputeGraph declares one, because only its binding is large enough
  for the memory saving to outweigh re-parsing the header set once per shard.
  Raise it on a build whose memory allowance is tighter than a developer
  machine's; set it to ``1`` to turn sharding off.
* ``EINSUMS_WITH_MPI`` replaces ``Comm``'s mock backend with a real
  Open MPI or MPICH integration.
* ``EINSUMS_WITH_CUDA`` / ``EINSUMS_WITH_HIP`` enable the GPU
  backends, which are pending validation work.
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
  :ref:`library modules overview <modules_overview>`.
* For performance trade-offs (when each backend fires, how to read a
  PackedGemm log message), see :ref:`tutorial-performance`.
