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
:ref:`vendor BLAS <modules_Einsums_BLAS>` (which only handles matrix-matrix
multiplies + one batch index) and the generic loop-nest fallback.

Why a packed backend
====================

Tensor contractions in chemistry codes don't all map directly to ``gemm``.
A rank-3 ``ijk,kl->ijl`` looks like a strided matrix multiply where one
dimension wraps in a way that ``cblas_dgemm`` can't express. The naive
fallback is a triply-nested loop with no SIMD, no cache blocking, and
no vendor BLAS acceleration.

PackedGemm fills that gap. For every contraction Einsums detects at
compile time, the dispatcher classifies indices into:

- **M dimensions** — target indices that appear in ``A`` but not ``B``
- **N dimensions** — target indices that appear in ``B`` but not ``A``
- **K dimensions** — link indices that appear in both ``A`` and ``B``
- **Batch dimensions** — target indices that appear in both ``A`` and ``B``

A ``PackingPlan`` records the strides and sizes of each group. The
runtime then packs ``A`` and ``B`` into contiguous tiles
(``MC × KC`` for A and ``KC × NC`` for B), calls vendor ``gemm`` per
tile, and accumulates back into ``C``.

When PackedGemm fires
=====================

The dispatcher in :ref:`TensorAlgebra <modules_Einsums_TensorAlgebra>` tries
PackedGemm unconditionally for every contraction that didn't match a pure
BLAS shape. ``try_packed_gemm()`` returns ``true`` when the plan is valid
and the contraction was executed; ``false`` falls through to the generic
algorithm.

Activation criteria (handled internally):

- The contraction's M/N/K decomposition produces non-empty groups.
- Sizes are large enough that pack overhead is worth it.
- ``ValueType`` is one of the supported scalar types.

Cache-aware blocking
====================

Block sizes :math:`MC \times KC \times NC` are tuned at runtime from the
detected CPU cache hierarchy. ``compute_blocking(elem_size)`` returns:

- ``KC`` — K tile size, sized so one A panel fits in ~half of L2.
- ``MC`` — M tile size, sized so one A panel fits in ~half of L2.
- ``NC`` — N tile size, sized so one B panel fits in ~half of L3.
- ``NR`` — N register-block (6, fully unrolled by LLVM).

The values automatically adapt to the element width:
``sizeof(double) = 8`` vs ``sizeof(std::complex<double>) = 16``.

PackingPlanCache
================

Plans are deterministic functions of the contraction's index pattern,
dimensions, and strides — so they cache cleanly. ``PackingPlanCache``
keys on a ``ContractionKey`` (an exact-match digest of the call shape)
and reuses plans across repeated calls. The pack itself runs every time;
only the planning is memoized.

CPU configuration
=================

``cpu_config()`` returns the cached ``CpuConfig`` describing the SIMD
register width and per-core cache sizes for the local machine. PackedGemm
uses it during planning. The register blocks (``MR``, ``NR``) and tile
sizes derive from these values.

Future direction: GPU
=====================

The same ``PackingPlan`` abstraction is intended to drive a GPU backend
once NVIDIA hardware is available — the planning code is backend-agnostic;
only the pack kernel and tile-GEMM emitter need swapping. See the project
roadmap for the Ozaki mixed-precision integration plan.

See the :ref:`API reference <modules_Einsums_PackedGemm_api>` of this module
for the public surface (``try_packed_gemm``, ``PackingPlan``,
``ContractionKey``, ``CpuConfig``).
