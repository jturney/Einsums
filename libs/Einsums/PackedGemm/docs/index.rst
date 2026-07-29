..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_PackedGemm:

##########
PackedGemm
##########

The ``PackedGemm`` module is Einsums' in-tree BLIS-style packed contraction
backend for arbitrary-rank tensor contractions. It sits between
:ref:`vendor BLAS <modules_Einsums_BLAS>`, which only handles matrix-matrix
multiplies plus one batch index, and the generic loop-nest fallback.

Why a packed backend
====================

Tensor contractions in chemistry codes do not all map directly onto ``gemm``.
A rank-3 ``ijk,kl->ijl`` looks like a strided matrix multiply where one
dimension wraps in a way ``cblas_dgemm`` cannot express, and the naive fallback
is a nested loop with no SIMD, no cache blocking, and no vendor BLAS
acceleration.

The classical alternative is TTGT, "transpose-transpose-GEMM-transpose": permute
each operand into a matrix, call ``gemm``, permute the result back. That works,
but it materializes up to three full tensor-sized copies. PackedGemm fills the
gap between the two: it reads the operands where they lie, packs only the cache
tiles it is about to multiply, and never materializes a whole transposed
operand.

The M / N / K / batch decomposition
====================================

For a contraction generalised from
:math:`C[m,n] = \beta\,C[m,n] + \alpha \sum_k A[m,k]\,B[k,n]` to arbitrary
rank, every index falls into exactly one of four groups:

``M`` dimensions
    Target indices appearing in ``A`` but not ``B``.
``N`` dimensions
    Target indices appearing in ``B`` but not ``A``.
``K`` dimensions
    Link indices, appearing in both ``A`` and ``B`` but not in ``C``.
Batch dimensions
    Target indices appearing in **both** ``A`` and ``B``.

Multi-M, multi-N and multi-K are all supported: the group sizes multiply into
``M_total``, ``N_total`` and ``K_total``. A :cpp:struct:`~einsums::packed_gemm::PackingPlan`
records the sizes and per-tensor strides of each group, in each tensor that
sees them, plus the batch descriptors.

Two structural details make the machinery uniform:

- **Synthetic unit dims.** A GEMV-shaped contraction has no M or no N group; an
  outer product has no K group. Rather than special-casing them,
  ``compute_packing_topology`` synthesizes a unit dim (size 1, stride 0) into
  the empty group so the whole GEMM path applies unchanged. The plan flags this
  so the direct-BLAS fast paths, where a zero stride would produce an invalid
  leading dimension, know to stand down.
- **Dim coalescing.** Adjacent dims within a group that tile contiguously in
  every tensor that sees them are merged into one. This is what turns many
  multi-M/N contractions, which would otherwise need the slow scatter path,
  into plain single-M/N GEMMs.

When PackedGemm fires
=====================

The dispatcher in :ref:`TensorAlgebra <modules_Einsums_TensorAlgebra>` tries
PackedGemm for every contraction that did not match a pure BLAS shape.
:cpp:func:`~einsums::packed_gemm::try_packed_gemm` returns ``true`` when the
plan was valid and the contraction was executed, and ``false`` when the caller
should fall back.

A plan is invalid, and the call declines, when:

- The output is a scalar (``c_indices`` is empty).
- ``a_indices`` or ``b_indices`` repeat a letter, which is a diagonal/Hadamard
  access rather than a contraction.
- The element types of ``A`` and ``B`` disagree, or are not one of ``float``,
  ``double``, ``std::complex<float>``, ``std::complex<double>``.

There is also a *policy* decline. ``allow_scatter=false`` tells
``try_packed_gemm`` to give up on contractions that remain multi-M/N after
coalescing, instead of taking the per-tile scatter path. Callers pass ``false``
when they have a faster fallback (the compile-time einsum dispatch falls back to
Sort+GEMM) and ``true`` when their only alternative is a generic loop (the
ComputeGraph runtime string dispatch). A micro-kernel rung whose scatter path
actually beats Sort+GEMM, which the SME rung does by a measured 2.3x, advertises
that in its :cpp:struct:`~einsums::packed_gemm::MicroKernelShape` so callers
take the scatter path anyway.

Declines are logged at INFO, so ``--einsums:log:level 2`` shows why a given
contraction was not accelerated. Every attempt also emits a
``packed_gemm_skip`` profiler annotation naming the reason.

Execution routes
================

Within a valid plan, ``blis_contraction`` picks the cheapest route the shape
allows, from fastest to most general:

**gemm_batch fast path**
    Single-M, single-N, single-K with stride-compatible batch slices. The
    pointer arrays are precomputed and one ``blas::gemm_batch`` call covers
    every batch at once.

**Multi-K flatten + GEMM**
    Single-M, single-N with several link dims. ``A`` and ``B`` are flattened
    into contiguous ``M*K`` and ``K*N`` buffers, then one BLAS ``gemm`` runs on
    them. The flatten uses :ref:`HPTT <modules_Einsums_HPTT>`, so it is
    cache-blocked and vectorized; a scalar gather loop is the fallback where
    HPTT is unavailable or the source is non-contiguous. When both operands are
    already in a usable layout the path is fully **zero-copy**: one ``gemm``,
    no packing at all.

**BLIS-style tiled packing**
    The general route. ``A`` is packed ``MC x KC`` and ``B`` is packed
    ``KC x NC`` into contiguous panels, and the resolved micro-kernel or a
    vendor ``gemm`` runs per tile, accumulating into ``C``.

**Scatter**
    Multi-M/N outputs, and single-M/N layouts where neither output dim is
    unit-stride (batched ``C`` with a stride-1 batch index, strided views, the
    synthetic unit dims above). Results are written back element by element.

Leading dimensions are clamped up to the BLAS minimum on every route. For a
transposed or size-1 output axis the natural stride can collapse below the row
count BLAS requires (``"nm <- mkq ; kqn"`` with ``n = 1`` gives a C stride of 1
against an M larger than that), which BLAS rejects. The clamp is a no-op
whenever the real stride already meets the bound.

Cache-aware blocking
====================

:cpp:func:`~einsums::packed_gemm::compute_blocking` derives
:math:`MC \times KC \times NC` at runtime from the cache hierarchy that
:ref:`Hardware <modules_Einsums_Hardware>` detected, for the element size passed
in, so ``double`` (8 bytes) and ``std::complex<double>`` (16 bytes) get
different blocks from the same code:

``KC``
    The K tile. Sized so one packed column of A (``MR * KC``) fits in roughly
    L1, floored at 64 and rounded down to a multiple of 8.
``MC``
    The M tile. Sized so the whole packed A panel (``MC * KC``) fits in half of
    L2, rounded down to a multiple of ``MR``.
``NC``
    The N tile. Sized so the packed B panel (``KC * NC``) fits in half of L3,
    rounded down to a multiple of ``NR``.
``NR``
    The N register block, fixed at 6, which LLVM fully unrolls.

``MR``, the M register block, is ``2 * VL``: two full vector registers of
``C`` accumulators per column, where ``VL`` is the native vector length in
doubles.

.. note::

    A micro-kernel rung may override ``KC``. The SME rung raises it, because
    its ZA tile accumulators hold the C block across the whole K loop, so a
    deeper K block means C is read-modify-written and the tiles extracted once
    rather than once per cache-sized K slice. ``MC`` shrinks to compensate so
    the packed A panel stays L2-sized.

The micro-kernel ladder
=======================

The tile kernel is compiled once per :ref:`SIMD dispatch <modules_Einsums_SIMD>`
rung, each copy in its own namespace, and selected at runtime from the highest
rung the CPU actually supports. PackedGemm builds the ``baseline``, ``v2``,
``v3``, ``v4`` and ``sme`` rungs. :cpp:func:`~einsums::packed_gemm::micro_kernel_entry`
resolves the entry point once per contraction, and callers hoist it out of the
tile loops; element types without a per-rung build fall back to an
ambient-flags header instantiation.

The rung owns the packing geometry, not just the arithmetic, which is why
``MicroKernelShape`` is queried alongside the kernel. NEON and AVX rungs use
``cpu_config()``'s vector blocking (``MR = 2*VL``, ``NR = 6``); the SME rung
uses ZA-tile blocking (``MR = NR = 2 *`` streaming vector length in doubles,
16x16 on an Apple M4). Panels must be packed in the geometry of the kernel that
will read them.

Complex arithmetic
------------------

Two rung-selectable schemes replace the naive complex GEMM:

**1m (Van Zee)**
    ``A`` packs in the expanded 1e form (:math:`[[a_r, -a_i], [a_i, a_r]]` per
    element) and ``B`` packs real and imaginary parts as adjacent K rows (1r),
    so the **real** tile kernel of the underlying real type computes
    interleaved-complex output directly. The working extents double
    (:math:`M_h = 2M`, :math:`K_h = 2K`). Measured 1.74x over Sort+GEMM for
    ``complex<double>`` on the M4 SME rung.

**3m (Karatsuba)**
    On the block-GEMM path: three real vendor GEMMs per block
    (Re*Re, Im*Im, (Re+Im)*(Re+Im)) instead of one complex GEMM. That is 25%
    fewer flops, and the real GEMMs reach matrix hardware, such as Accelerate's
    AMX, that complex arithmetic may not.

.. warning::

    3m mildly weakens the error bounds relative to conventional complex
    multiplication (Higham's 3m analysis). A rung enables it only where the win
    has been measured.

Choosing the scatter engine
---------------------------

On the scatter path a rung picks between one vendor GEMM per cache block into a
contiguous temporary followed by a scatter, which wins when the vendor library
reaches hardware the tile kernel cannot (Accelerate's AMX on M1-M3), and its own
``MR x NR`` tile kernel, which wins when the rung has FMOPA micro-tiles (SME).

Parallelism gate
================

The BLIS loops are OpenMP-parallel, but only above a work threshold.
``CpuConfig::min_parallel_flops``, taken from
:cpp:func:`einsums::hardware::omp_min_parallel_flops`, is the flop count below
which forking a team costs more than the contraction. This is not a
micro-optimization: a tiled CCSD residual at 50 spin orbitals expands to 3751
contractions of roughly 295 KFLOP each, and forking for every one of them cost
71 ms of that replay's 72 ms of einsum time, against 33 ms with the regions
declined. See :ref:`Hardware <modules_Einsums_Hardware>` for how the threshold
is derived and how to override it for an A/B measurement.

PackingPlanCache
================

Plans are deterministic functions of the contraction's index pattern,
dimensions, and strides, so they cache cleanly.
:cpp:class:`~einsums::packed_gemm::PackingPlanCache` keys on a
:cpp:struct:`~einsums::packed_gemm::ContractionKey`, an exact-match digest of
the call shape, behind a shared mutex. Only the *planning* is memoized, since
the pack itself depends on the data and runs every time.

CPU configuration
=================

:cpp:func:`~einsums::packed_gemm::cpu_config` returns the cached
:cpp:struct:`~einsums::packed_gemm::CpuConfig` for the local machine: ``VL``,
``MR``, ``NR``, the three cache sizes, the measured OpenMP region cost, and the
parallelism threshold. The hardware **facts** in it come from
:ref:`Einsums_Hardware <modules_Einsums_Hardware>`, the single detector shared
with the ComputeGraph cost model. What stays here is kernel **policy**: the
``MR``/``NR`` register blocking and the BLIS cache blocking are tuning
decisions that happen to read hardware numbers.

The resolved configuration is logged at INFO on first use:

.. code-block:: text

    [info] cpu_config: VL=2, MR=4, NR=6, L1=128K, L2=16384K, L3=8192K,
           omp_region=19.84us, min_parallel_flops=238080

Future direction: GPU
=====================

The ``PackingPlan`` abstraction is backend-agnostic by design; only the pack
kernel and the tile-GEMM emitter need swapping for a GPU backend. See the
project roadmap for the Ozaki mixed-precision integration plan.

API Reference
=============

- :cpp:func:`~einsums::packed_gemm::try_packed_gemm` - the entry point; returns false when the caller should fall back.
- :cpp:struct:`~einsums::packed_gemm::ContractionSpec` - the raw index lists plus the derived target/link/loop spaces.
- :cpp:struct:`~einsums::packed_gemm::ContractionKey` - exact-match digest of a call shape, the plan-cache key.
- :cpp:struct:`~einsums::packed_gemm::PackingPlan` - the M/N/K/batch decomposition with per-tensor sizes and strides.
- :cpp:func:`~einsums::packed_gemm::compute_packing_topology` - build the plan from a key; sets ``valid``.
- :cpp:class:`~einsums::packed_gemm::PackingPlanCache` - thread-safe plan memoization.
- :cpp:func:`~einsums::packed_gemm::compute_blocking` - cache-derived ``MC``/``KC``/``NC``/``NR`` for an element size.
- :cpp:struct:`~einsums::packed_gemm::CpuConfig` / :cpp:func:`~einsums::packed_gemm::cpu_config` - vector and cache configuration.
- :cpp:func:`~einsums::packed_gemm::micro_kernel_entry` - resolve the tile kernel for the running CPU's rung.

See the :ref:`API reference <modules_Einsums_PackedGemm_api>` of this module for
the full generated surface.
