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

1. **Rank-2 BLAS dispatch**: ``gemm()`` on a symmetric tensor routes to
   ``symm``, or ``hemm`` for Hermitian complex matrices, which does half
   the work of the general kernel.

2. **ComputeGraph propagation**: the ``SymmetryPropagation`` pass walks
   a captured graph and tags intermediates whose symmetry can be proven
   from their inputs. Graph execution does not use the ``symm`` dispatch
   yet: captured Gemm and Einsum nodes run on the general kernels.

3. **Symmetry-exploiting passes**: ``AntisymmetryInference``,
   ``AntisymmetrizerFolding``, and ``LayoutAssignment`` read the descriptor
   to make their decisions, and ``AxisTiling`` computes only half of a
   contraction whose result it can prove pair-symmetric.

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

The descriptor stores the generators of the invariance group, not the
full group. The 8-element ERI group fits in three generators, and consumers
apply the generators directly; nothing builds the full group.

.. note::

   The descriptor is metadata only. Attaching it does not rearrange
   the stored data or reduce memory. Storage stays dense, and the benefit
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
shortcut and falls through. Both row-major and column-major storage work;
the dispatch adjusts argument order to match BLAS's column-major
semantics.

The win is the usual factor of about 2 from computing only one triangle of
the symmetric product, plus the cache-friendly layout ``symm`` uses. The
dispatch happens at the user-facing ``gemm()`` entry, so callers do not need
to know which kernel fires.

Coverage today: the ``symm`` / ``hemm`` dispatch is CPU vendor BLAS only. The
GPU backends have no ``symm``, ``hemm``, ``syrk``, or ``potrf``; their solver
set is ``syev``, ``heev``, ``gesv``, ``getrf``, ``getri``, and ``gesvd``. Packed
storage (``spsv``, ``spev``) is deferred; dense storage with a flag is used
throughout.

ComputeGraph Propagation
========================

If you capture a graph that computes, say, ``C = Aᵀ·A`` as an
intermediate and then uses ``C`` in subsequent operations, the
``SymmetryPropagation`` pass recognizes that ``C`` must be symmetric and
tags it for you. The tag is visible to the passes that read descriptors; the
graph's own Gemm and Einsum nodes still run the general kernels, so today the
``symm`` fast path is reached only through the eager ``gemm()`` entry points.

.. code-block:: cpp

   cg::Graph graph("example");
   auto &C = graph.create_zero_tensor<double, 2>("C", N, N);
   auto &D = graph.create_zero_tensor<double, 2>("D", N, N);
   {
       cg::CaptureGuard g(graph);
       cg::einsum("ki;kj->ij", &C, A, A);   // AᵀA — propagation tags C symmetric
       cg::einsum("ij;jk->ik", &D, C, B);    // D = C·B, general kernel
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

Rules are conservative: only cases that provably hold are tagged.
Everything else is left untagged and falls through to general dispatch.

**Scope**: propagation only tags graph-owned intermediates, those created by
``graph.create_tensor<T>()``. User-owned tensors passed into the graph
are never mutated.

Higher-Rank Coverage
====================

The BLAS dispatch covers rank 2 exhaustively. Higher-rank
symmetries are declared and verified today, since the descriptor type and
``symmetrize`` / ``check_symmetry`` are rank-N generic, but vendor BLAS
has no direct support for them: no ``spmm`` for 4-index symmetric, no
batched ``syrk``. Two paths are available when needed:

1. **Dense storage with custom kernels**: write the specialized loop by
   hand, as production CC codes do for 8-fold ERI contractions.
2. **Expand-on-demand**: call general ``gemm`` or ``gemm_batch`` on the
   dense storage and accept paying for the redundant work.

Packed storage for big higher-rank tensors such as compact ERI or antisym T2
is future work. The memory savings are real, roughly 8 times for 8-fold ERI,
but require either custom kernels or per-access canonicalization.

Computing Half a Pair-Symmetric Contraction
===========================================

The closed-shell CCSD amplitude is unchanged when the two electrons trade
places, ``t2[i,j,a,b] = t2[j,i,b,a]``: a swap of the axis pair ``(0, 2)`` with
the pair ``(1, 3)``. Declared on the tensor, it lets ``AxisTiling`` prove that a
contraction's result has the same symmetry, slice that result over two
exchangeable axes, compute only the blocks on and above the diagonal, and write
each off-diagonal block's mirror from it.

.. code-block:: python

   t2.set_symmetry(einsums.SymmetryDescriptor([einsums.SymmetryOp.group_swap([0, 2], [1, 3])]))

The triangle needs all of the following, and the report names the first one
that fails:

- Two sliced axes of the same extent and block, in a region that writes one of
  the caller's tensors and neither reduces nor sums into a tensor
  (``axis-tiling.triangle-shape``).
- A contribution provably unchanged when the two axes are exchanged, from the
  declared symmetries of the operands (``axis-tiling.not-exchangeable``).
- An output that is not backed by storage, since a mirrored block lands
  outside the box the block streams (``axis-tiling.triangle-backed``).
- A block of two or more that fits beside the scratch block the mirror is
  computed into (``axis-tiling.triangle-fit``).
- When the region has nothing to stream and would otherwise run as captured, a
  cost-model price for the triangle below the price of running it whole
  (``axis-tiling.triangle-price``).

A declaration is a promise the triangle relies on, so the tiling checks it: a
node before the loop runs ``check_symmetry`` on the declared tensor at every
replay, and throws if its values no longer have the declared symmetry.

Whether the triangle pays depends on how much of the iteration the
contraction is, and on the block. A small block runs more contractions and
packs their operands more often; a large one computes more of the diagonal
twice. ``AxisTiling`` prices each candidate block as the contraction engine
would run its tiles, from the hardware profile ``einsums tune`` writes or from
a built-in table, and takes the cheapest. On benzene in cc-pVDZ, where the
particle ladder dominates, the triangle took a CCSD iteration to 0.69 to 0.72
of the time without it. On water in cc-pVTZ it picks blocks of 7 x 7 and makes
an iteration 18% faster, and 4% faster with the integrals in files, where
reading them takes much of the iteration whatever the ladder does. :ref:`tutorial-ccsd` shows the
declaration and those measurements.

A symmetry declared on a tensor the graph declared, such as one backed by a
file, may be set after the declaration or after the capture: every pass
pipeline reads the declarations again before its first pass.

Python API
==========

``RuntimeTensor`` has ``set_symmetry``, ``has_symmetry`` and
``clear_symmetry``. ``einsums.SymmetryDescriptor`` is built from a list of
``einsums.SymmetryOp`` generators (``swap``, ``group_swap``, ``identity``) or
from the same named factories as in C++ (``symmetric_pair``, ``eri_8fold``,
``ccsd_t2`` and the rest). There is no getter for the descriptor itself. The
propagation pass is ``einsums.graph.SymmetryPropagation``, which reports
``num_inferred``.

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
passes trust the declared descriptor and do not re-verify at every
call; the one exception is the triangle above, which checks it at every
replay. If you attach a symmetric descriptor to a non-symmetric matrix,
you will get silently wrong results. Use ``symmetrize()`` on ingest when in
doubt.

**Views don't inherit symmetry.** ``TensorView`` always reports
``symmetry() == nullptr``. Reasoning about which slice of a symmetric
tensor is itself symmetric is future work; for now views fall through to
general dispatch.

See Also
========

- :doc:`optimization_passes`: catalog of every pass in the default pipeline.
- ``libs/Einsums/TensorBase/include/Einsums/TensorBase/SymmetryDescriptor.hpp``:
  the descriptor type with inline documentation on each factory.
- ``libs/Einsums/Tensor/include/Einsums/Tensor/SymmetryOps.hpp``:
  ``symmetrize`` and ``check_symmetry``.
- ``libs/Einsums/LinearAlgebra/include/Einsums/LinearAlgebra/SymmetryDispatch.hpp``:
  the gemm to symm/hemm dispatcher.
- ``libs/Einsums/ComputeGraph/src/Passes/SymmetryPropagation.cpp``:
  the propagation pass's rule implementations.
- ``libs/Einsums/Tensor/tests/unit/Symmetry.cpp``: descriptor and
  symmetrize/check_symmetry test coverage.
- ``libs/Einsums/LinearAlgebra/tests/unit/symm_dispatch.cpp``: BLAS
  dispatch correctness parity.
- ``libs/Einsums/ComputeGraph/tests/unit/Pass_SymmetryPropagation.cpp``:
  propagation rule coverage.
