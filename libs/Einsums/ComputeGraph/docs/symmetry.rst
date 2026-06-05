..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

=========================
Symmetry-Aware Tensors
=========================

Einsums supports declaring tensor symmetries as first-class metadata. A
``SymmetryDescriptor`` attached to a tensor tells the library "this tensor
satisfies the invariant ``T(i,j,...) = ±T(permutation(i,j,...))``." The
information is consumed in three places:

1. **Rank-2 BLAS dispatch** — ``gemm()`` on a symmetric tensor routes to
   ``symm`` (or ``hemm`` for Hermitian complex matrices), which does half
   the work of the general kernel.

2. **ComputeGraph propagation** — the ``SymmetryPropagation`` pass walks
   a captured graph and tags intermediates whose symmetry can be proven
   from their inputs. Downstream ``graph.execute()`` calls then pick up
   the BLAS dispatch above automatically.

3. **Symmetry-exploiting algorithms** — analytic passes (memory planning,
   CSE, inplace detection) can read the descriptor to make better
   decisions in the future. Today they're mostly informational.

.. contents::
   :local:

Declaring Symmetry
==================

Every ``Tensor`` can optionally carry a ``SymmetryDescriptor``. Build one
from a named factory and attach it with ``set_symmetry``:

.. code-block:: cpp

   #include <Einsums/Tensor/Tensor.hpp>
   #include <Einsums/TensorBase/SymmetryDescriptor.hpp>

   auto F = create_zero_tensor<double>("F", N, N);
   F.set_symmetry(einsums::SymmetryDescriptor::symmetric_pair(0, 1));

The named factories cover the patterns chemistry codes reach for most
often:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Factory
     - Invariant
   * - ``symmetric_pair(a, b)``
     - ``T(..,a,..,b,..) = T(..,b,..,a,..)``
   * - ``antisymmetric_pair(a, b)``
     - ``T(..,a,..,b,..) = -T(..,b,..,a,..)``
   * - ``hermitian_pair(a, b)``
     - ``T(..,a,..,b,..) = conj(T(..,b,..,a,..))``
   * - ``anti_hermitian_pair(a, b)``
     - ``T(..,a,..,b,..) = -conj(T(..,b,..,a,..))``
   * - ``eri_8fold()``
     - Real two-electron-integral 8-fold: ``(μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν)`` and permutations
   * - ``eri_4fold()``
     - Complex ERI 4-fold (inner-pair swaps, no bra/ket swap)
   * - ``ccsd_t2()``
     - CCSD T2 amplitudes: antisym in virtual pair (0,1) and occupied pair (2,3)

For uncommon patterns, build a descriptor directly from generators:

.. code-block:: cpp

   SymmetryDescriptor desc;
   desc.add(SymmetryOp::swap(0, 1, /*sign=*/+1));
   desc.add(SymmetryOp::swap(2, 3, /*sign=*/-1));
   desc.add(SymmetryOp::group_swap({0,1}, {2,3}, /*sign=*/+1));

The descriptor stores the **generators** of the invariance group, not the
full group — ERIs' 8-fold symmetry fits in three generators; the 48-element
group is recomputed on demand when needed.

.. note::

   The descriptor is **metadata only** — attaching it doesn't rearrange
   the stored data or reduce memory. Storage stays dense; the benefit
   comes from BLAS dispatch and graph-level reasoning that trust the
   declared invariant.

Enforcing and Verifying
=======================

Symmetry declarations are load-bearing: downstream kernels assume the
invariant actually holds. Two helpers make that safe:

``symmetrize(T)`` mutates ``T`` in place so it exactly satisfies its
declared descriptor. For symmetric: averages each ``(T[i,j], T[j,i])``
pair. For antisymmetric: subtract-and-halve, zeroing the diagonal. For
Hermitian: averages with the conjugate partner and enforces real
diagonals.

.. code-block:: cpp

   auto A = create_random_tensor<double>("A", N, N);
   A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
   symmetrize(A);                // make it actually symmetric

``check_symmetry(T, tolerance)`` returns ``true`` iff the stored data
satisfies the declared descriptor within ``tolerance`` (defaults to the
descriptor's own tolerance, ``1e-12``). Use this for debugging or in
defensive asserts:

.. code-block:: cpp

   REQUIRE(check_symmetry(A));   // should be true after symmetrize()

Both functions are rank-N generic. They visit every index pair, apply each
generator's permutation, and either average (``symmetrize``) or compare
(``check_symmetry``).

Rank-2 BLAS Dispatch
====================

Once a rank-2 tensor carries a matching descriptor, ``linear_algebra::gemm``
routes to a specialized BLAS kernel:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Input
     - Kernel picked
   * - Symmetric A, any trans
     - ``symm(side='L')``
   * - Symmetric B, any trans
     - ``symm(side='R')``
   * - Hermitian A, trans ``'n'`` or ``'c'``
     - ``hemm(side='L')``
   * - Hermitian B, trans ``'n'`` or ``'c'``
     - ``hemm(side='R')``
   * - Anything else
     - Falls through to ``gemm``

For Hermitian matrices, ``'t'`` (transpose without conjugation) has no
shortcut and falls through. Both row-major and column-major storage work
— the dispatch adjusts argument order to match BLAS's column-major
semantics.

The win is the usual ~2× from computing only one triangle of the
symmetric product, plus the cache-friendly layout ``symm`` uses. The
dispatch happens at the user-facing ``gemm()`` entry; callers don't need
to know which kernel fires.

Hermitian-only coverage today: ``symm``/``hemm``/``syrk``/``syev``/
``potrf`` and friends exist on CUDA and HIP, falling back to general
``gemm`` on MPS. Packed storage (``spsv``/``spev``) is deferred — dense
storage with a flag is used throughout.

ComputeGraph Propagation
========================

If you capture a graph that computes, say, ``C = Aᵀ·A`` as an
intermediate and then uses ``C`` in subsequent operations, the
``SymmetryPropagation`` pass recognizes that ``C`` must be symmetric and
tags it for you — the downstream ``gemm(C, …)`` then takes the ``symm``
fast path automatically.

.. code-block:: cpp

   cg::Graph graph("example");
   auto &C = graph.create_zero_tensor<double, 2>("C", N, N);
   auto &D = graph.create_zero_tensor<double, 2>("D", N, N);
   {
       cg::CaptureGuard g(graph);
       cg::einsum("ki;kj->ij", &C, A, A);   // AᵀA — propagation tags C symmetric
       cg::einsum("ij;jk->ik", &D, C, B);    // D = C·B — hits symm dispatch
   }
   graph.apply(cg::PassManager::create_default());
   graph.execute();

The pass is registered in ``PassManager::create_default()``, right after
``Materialization`` (so all tensors exist) and before the GPU placement
block. No user action is needed beyond using the default pipeline.

**Rules** the pass currently applies:

- **Scale**: ``C = α·A`` inherits A's descriptor.
- **Axpy / Axpby**: ``C = α·A + β·B`` where A and B share a descriptor
  inherits that descriptor.
- **Rank-2 self-contraction**: ``C = einsum("ki;kj->ij", A, A)`` (or any
  rank-2 × rank-2 → rank-2 einsum with exactly one link index and the
  same tensor in both slots) yields a symmetric ``C``. For complex
  tensors with one operand conjugated, ``C`` is Hermitian.
- **Permute of rank-2 symmetric**: a ``permute`` with β=0 acting on a
  symmetric tensor produces a symmetric output.

Rules are conservative — only cases that provably hold are tagged.
Everything else is left untagged and falls through to general dispatch.

**Scope**: propagation only tags graph-owned intermediates (created by
``graph.create_tensor<T>()``). User-owned tensors passed into the graph
are never mutated.

Higher-Rank Coverage
====================

The Phase 2 BLAS dispatch covers rank 2 exhaustively. Higher-rank
symmetries are declared and verified today (the descriptor type and
``symmetrize`` / ``check_symmetry`` are rank-N generic), but vendor BLAS
has no direct support for them — no ``spmm`` for 4-index symmetric, no
batched ``syrk``. Two paths are available when needed:

1. **Dense storage + custom kernels** — write the specialized loop by
   hand (what production CC codes do for 8-fold ERI contractions).
2. **Expand-on-demand** — call general ``gemm`` / ``gemm_batch`` on the
   dense storage and accept paying for the redundant work.

Packed storage for big higher-rank tensors (compact ERI, antisym T2) is
future work — the memory savings are real (~8× for 8-fold ERI) but
require either custom kernels or per-access canonicalization.

Python API
==========

The same descriptor API is exposed through pybind11 on
``RuntimeTensor``:

.. code-block:: python

   import core
   A = core.RuntimeTensorD("A", [N, N])
   A.set_symmetry(core.SymmetryDescriptor.symmetric_pair(0, 1))
   assert A.has_symmetry()
   desc = A.symmetry()      # or None
   A.clear_symmetry()

Factories: ``symmetric_pair``, ``antisymmetric_pair``, ``hermitian_pair``,
``anti_hermitian_pair``, ``eri_8fold``, ``eri_4fold``, ``ccsd_t2``.
``symmetrize()`` and ``check_symmetry()`` are C++-only today (they're
rank-templated); a Python wrapper is a straightforward follow-up when
needed.

Design Notes
============

**Independent copy semantics.** The descriptor is stored as a
``std::unique_ptr`` so tensor copies produce independent descriptors.
Modifying one tensor's symmetry never silently affects an aliased tensor.
8 bytes overhead per tensor when no descriptor is attached.

**Floating-point drift.** Operations on a "symmetric" tensor can produce
slight numerical asymmetry (``A + Aᵀ`` drifts by machine ε). The
descriptor's ``tolerance`` field controls how much drift
``check_symmetry`` accepts; the default ``1e-12`` is generous enough for
double-precision BLAS workloads.

**Descriptor as source of truth.** BLAS dispatch and ComputeGraph
passes trust the declared descriptor — they do not re-verify at every
call. If you attach a symmetric descriptor to a non-symmetric matrix,
you'll get silently wrong results. Use ``symmetrize()`` on ingest when in
doubt.

**Views don't inherit symmetry.** ``TensorView`` always reports
``symmetry() == nullptr``. Reasoning about which slice of a symmetric
tensor is itself symmetric is future work; for now views fall through to
general dispatch.

See Also
========

- :doc:`optimization_passes` — catalog of every pass in the default pipeline.
- ``libs/Einsums/TensorBase/include/Einsums/TensorBase/SymmetryDescriptor.hpp``
  — the descriptor type with inline documentation on each factory.
- ``libs/Einsums/Tensor/include/Einsums/Tensor/SymmetryOps.hpp`` —
  ``symmetrize`` and ``check_symmetry``.
- ``libs/Einsums/LinearAlgebra/include/Einsums/LinearAlgebra/SymmetryDispatch.hpp``
  — the gemm → symm/hemm dispatcher.
- ``libs/Einsums/ComputeGraph/src/Passes/SymmetryPropagation.cpp`` —
  the propagation pass's rule implementations.
- ``libs/Einsums/Tensor/tests/unit/Symmetry.cpp`` — descriptor and
  symmetrize/check_symmetry test coverage.
- ``libs/Einsums/LinearAlgebra/tests/unit/symm_dispatch.cpp`` — BLAS
  dispatch correctness parity.
- ``libs/Einsums/ComputeGraph/tests/unit/Pass_SymmetryPropagation.cpp``
  — propagation rule coverage.
