.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

==================================
Views, Aliasing, and Pipeline Parameters
==================================

The ``cg::view()`` operation records a non-owning slice of a tensor inside a
graph. Like Einsums' ``TensorView``, it shares storage with the parent — no
copy. Slice bounds can be statically known, runtime-resolvable from a named
:cpp:class:`Pipeline parameter <einsums::compute_graph::ParamTable>`, or
arbitrary callbacks evaluated at execute time.

This page also covers :cpp:func:`cg::write_param` for graph-driven parameter
updates between iterations.

When to use ``cg::view``
========================

Use a graph view when you want a slice of a tensor to participate in the
graph's optimization passes — particularly:

- The slice's lifetime should be tied to the parent's, but tracked
  independently for scheduling.
- Slice bounds change between iterations of an SCF/CG/CCSD loop (occupation
  number, current rank, search-space size, …) without rebuilding the graph.
- Multiple downstream ops consume the slice, and you want the dependency
  scheduler to see the alias relationship — so a write to the parent and a
  read through the slice can't be reordered or parallelized.

If the slice is a constant cut that never changes between executions, you can
also build a plain ``TensorView`` outside the graph and pass it as input. The
graph won't know about the aliasing, but the simpler form may be sufficient.

Basic example
=============

Slicing a constant range — the first 2 rows of a 4×4 matrix:

.. code-block:: cpp

   namespace cg = einsums::compute_graph;
   using namespace einsums::index;

   Tensor<double, 2> A("A", 4, 4);
   Tensor<double, 2> dst("dst", 2, 4);

   cg::Pipeline pipe("example");
   {
       auto &stage = pipe.add_stage("s");
       cg::CaptureGuard g(stage);

       // slice = A[0:2, :]   (rank-preserving)
       auto &slice = cg::view<double, 2>(A,
           cg::ViewAxis::range(0, 2),
           cg::ViewAxis::full());

       cg::permute("ij <- ij", 0.0, &dst, 1.0, slice);
   }
   pipe.execute();

The returned reference is a graph-owned ``TensorView<double, 2>`` you can pass
to any ``cg::*`` operation that accepts a tensor input. The view is rebuilt
every time the View node executes — its data pointer always reflects the
parent's current backing buffer plus the resolved slice offsets.

Dynamic slice bounds via Pipeline parameters
============================================

The HF SCF case: ``C_occ = C[:, 0:n_occ]``, where ``n_occ`` is determined
once at runtime (per molecule) and never changes during the loop:

.. code-block:: cpp

   cg::Pipeline pipe("scf");
   pipe.set_param("n_occ", n_electrons / 2);

   {
       auto &iter = pipe.add_loop("scf_iter", 200, /* condition */);
       cg::CaptureGuard g(iter);

       // ... fock build, syev populates C ...

       // C_occ aliases C[:, 0:n_occ] — re-resolved each iteration.
       auto &C_occ = cg::view<double, 2>(C,
           cg::ViewAxis::full(),
           cg::ViewAxis::range(0, "n_occ"));

       // D = C_occ * C_occᵀ
       cg::gemm(/*transA=*/false, /*transB=*/true, 1.0, C_occ, C_occ, 0.0, &D);
   }
   pipe.execute();

A ``BoundExpr`` constructed from a string literal (``"n_occ"``) becomes a
:cpp:class:`Param <einsums::compute_graph::BoundExpr>` reference. The View
executor calls ``params.get("n_occ")`` each iteration; updating ``n_occ`` via
``pipe.set_param("n_occ", new_value)`` between executions takes effect on the
next execute.

Mid-iteration parameter updates: ``cg::write_param``
=====================================================

When the parameter must change *during* the loop — Maximum Overlap Method
(MOM), level shifting, integer-occupation excited-state SCF — use
:cpp:func:`cg::write_param`:

.. code-block:: cpp

   {
       auto &iter = pipe.add_loop("scf_iter", 200, condition);
       cg::CaptureGuard g(iter);

       // ... compute new occupation index from MO energies ...
       int64_t n_occ_new = 0;
       cg::custom("compute_mom_n_occ", /*inputs=*/{C_id, eps_id}, /*outputs=*/{},
                  [&]{ n_occ_new = compute_mom(C, eps); });

       // Push the new value into the param table — explicit dataflow edge:
       //   compute_mom_n_occ → write_param → C_occ view
       cg::write_param("n_occ", n_occ_new);

       auto &C_occ = cg::view<double, 2>(C,
           cg::ViewAxis::full(),
           cg::ViewAxis::range(0, "n_occ"));

       // ... downstream ops using C_occ ...
   }

Two forms:

- **Scalar source** (``cg::write_param(name, int64_var)``) — the graph
  captures ``&int64_var`` and reads its current value at execute time. If
  some upstream graph op writes to that same scalar address (registered via
  ``ctx.get_or_register_scalar(&int64_var, …)``), the dependency edge from
  producer to ``write_param`` is automatic, so the scheduler correctly
  orders the producer before any downstream View that reads ``Param(name)``.
- **Callback source** (``cg::write_param(name, std::function<int64_t()>)``) —
  the callback is invoked at execute time. Use this when the value comes
  from external (non-graph) state, e.g. a ``LoopCondition`` lambda that
  recomputes ``n_occ`` from C++ state outside the graph. No dataflow edge
  is created.

Aliasing semantics
==================

The output of ``cg::view`` aliases its parent — they share storage. Writing
to the slice mutates the parent in place; freeing the parent invalidates the
slice. The graph passes are aware of this:

- **FreeInsertion** never inserts ``Free`` for an aliased tensor (the parent
  owns the storage). It also extends the parent's last-use to cover any
  reads/writes through aliases, so the parent isn't freed prematurely.
- **DeadNodeElimination** treats a View node as live if any consumer reads
  the slice, otherwise prunes it.
- **Topological sort** folds reads/writes through an alias as touching the
  parent. ``GEMM(C_occ, …)`` and ``Syev(C, …)`` recorded in the same body
  are correctly ordered as dependent (not parallelizable), even though they
  reference different ``TensorId``\s at the surface.

What ``BoundExpr`` is, and how passes treat it
==============================================

A :cpp:class:`BoundExpr <einsums::compute_graph::BoundExpr>` is a tagged
union over three forms:

============== ================================== ===================================
Variant        Meaning                            Resolution
============== ================================== ===================================
``Const(v)``   Compile-time integer literal       Returns ``v``
``Param(n)``   Lookup into the active ParamTable  ``params.get(n)``; throws if unset
``Callback``   Arbitrary ``std::function<…>``     Invoked at execute time
============== ================================== ===================================

Implicit conversions let callers write whichever they prefer:

.. code-block:: cpp

   cg::ViewAxis::range(0, 5)              // Const, Const
   cg::ViewAxis::range(0, "n_occ")        // Const, Param
   cg::ViewAxis::range("first", "last")   // Param, Param
   cg::ViewAxis::range(0, [&]{ return computed_hi; })  // Const, Callback

**Optimization passes treat parameter values as opaque.** A pass may inspect
``BoundExpr::is_const()`` (structural — true at graph build) and use the
constant value (``e.const_value()``) to specialize. It must not branch on the
runtime value of a ``Param`` — the value isn't known when passes run, so any
specialization would be invalidated the moment the user calls ``set_param``.

If a value should drive optimization decisions, make it a build-time
constant (``Const(...)``). If the value should vary across runs without
rebuilding the graph, make it a ``Param``.

Graph editor (Einsums Studio) integration
=========================================

The visual graph editor surfaces all three operations in the right-click
"Add Node" menu:

- **View** (Reshape category, violet) — input pin ``A``, output pin
  ``slice``, single string parameter ``axes``. The editor parses the
  ``axes`` string at codegen time and emits ``cg::view(parent, ...)``
  with the deduced T and Rank — no template arguments to spell out by
  hand. Examples for the parameter:

  - ``"full,full"`` — rank-2 alias of the entire parent (default).
  - ``"full,0..n_occ"`` — column slice; ``n_occ`` resolves from the
    Pipeline parameter table at execute time.
  - ``"0..2,full"`` — first two rows.

- **Write Param** (Memory category, slate) — single scalar input
  ``value``, no output, string parameter ``name``. Emits
  ``cg::write_param("name", value)``.

- **Trace** (Tensor Algebra category, warm orange) — single tensor
  input ``A``, single scalar output ``result``. Emits a hoisted host
  scalar plus ``cg::trace(&result, A)`` so the value is visible to
  Loop convergence pins between iterations.

Saved ``.eingraph`` files round-trip through ``op_kind_from_string`` —
the string identifiers are ``"View"``, ``"WriteParam"``, and ``"Trace"``
respectively.

v1 limitations
==============

These are documented restrictions for now; each will be lifted in a follow-up:

- Only ``Full`` and ``Range`` axes. ``Drop`` (rank-reducing single-index
  pick) throws at capture; rank-preserving slicing covers the HF use case.
- Only ``ResultRank == parent.rank()``.
- Slice strides are inherited from the parent. No stride remapping; no
  transpose-via-view.
- Single-node only: slicing a distributed tensor whose partition straddles
  the slice yields undefined data. A pass-level check will be added when
  distributed View support lands.
- GPU placement: a View follows its parent's residency. If GPU passes
  promote the parent, you'll get host-residency mismatches.

Reference
=========

.. cpp:class:: einsums::compute_graph::BoundExpr

   Scalar integer expression resolved at execute time. Variants ``Const``,
   ``Param``, ``Callback``. Constructible from any integral type, ``char
   const *`` or ``std::string`` (Param), or ``std::function<int64_t()>``
   (Callback).

.. cpp:class:: einsums::compute_graph::ParamTable

   Mutable name → ``int64_t`` table held by ``Pipeline``. ``View`` executors
   read it; ``cg::write_param`` and ``Pipeline::set_param`` write to it.

.. cpp:struct:: einsums::compute_graph::ViewAxis

   Per-axis spec for a ``cg::view`` call. Static factories:

   - ``ViewAxis::full()`` — keep entire axis.
   - ``ViewAxis::range(lo, hi)`` — keep ``[lo, hi)``.
   - ``ViewAxis::drop(i)`` — pick single index (v2; throws today).

.. cpp:function:: template<typename T, size_t Rank, ...> \
                  TensorView<T, Rank>& cg::view(ParentT &parent, Axes &&...axes)

   Record a ``View`` node and return a graph-owned slice ``TensorView`` that
   downstream operations can consume. ``parent`` must outlive the graph;
   exactly ``Rank`` axis specs must be supplied (rank-preserving in v1).

.. cpp:function:: template<typename T> void cg::write_param(std::string name, T &source)

   Record a ``WriteParam`` node that reads ``source`` (an arithmetic
   scalar variable) at execute time and stores its value in
   ``params[name]``. The graph captures ``&source``, so any upstream op
   that registers the same address as a scalar input becomes a true
   dataflow predecessor.

.. cpp:function:: void cg::write_param(std::string name, std::function<int64_t()> source_fn)

   Variant taking a free-standing callback; no graph dependency is created.
