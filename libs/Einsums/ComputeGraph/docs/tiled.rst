.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _computegraph_tiled:

============================
Tiled Tensors in the Graph
============================

:cpp:struct:`einsums::TiledRuntimeTensor` is a runtime-rank, tile-wise sparse
tensor: the index range of each axis is partitioned into tiles, every populated
tile is a dense :cpp:type:`~einsums::RuntimeTensor`, and absent tiles are
**rigorously zero** and are not stored.

It is the runtime-rank analogue of :cpp:class:`~einsums::TiledTensor`, which
carries its rank as a template parameter. Being runtime-rank is what makes it
capturable: it is the type Python sees, and the type the ComputeGraph operation
surface accepts.

Unlike :cpp:class:`~einsums::BlockTensor`, which is the block-diagonal
square-tile special case, a tile may sit anywhere on the grid and may be
rectangular. That is what a non-totally-symmetric operator coupling different
symmetry sectors needs.

Why tile inside a graph
=======================

Structural sparsity is free arithmetic. In a symmetry-blocked calculation a
block is zero unless the irrep product contains the totally symmetric
representation, so most tile combinations contribute nothing and, because the
tile is simply absent, nothing is computed for them. On a small CCSD model that
is roughly a 10x flop saving.

The catch is that saving flops is not the same as saving time. A tiled
contraction captures as **one opaque** ``OpKind::Custom`` node executed tile by
tile, and no pass can read inside it: the einsum-rewriting passes all filter on
``OpKind::Einsum``, and a whole-tiled node could not satisfy them anyway, since
GEMMBatching wants one ``lda`` and MemoryPlanning's arena wants one buffer,
neither of which a tile container has. Left alone, a blocked residual spends its
whole arithmetic saving on dispatch overhead.

:ref:`TiledExpansion <computegraph_tiled_expansion>` is the answer. It runs first
in the default pipeline and lowers the opaque node into ordinary dense nodes, at
which point every existing pass applies unchanged.

Building one
============

A tiled tensor is constructed from its **grid**: one list of tile sizes per axis.
Tiles are then declared, materialized, and filled per tile.

.. code-block:: cpp

   using Grid = std::vector<std::vector<int>>;

   // 6x6 logical, partitioned 2|4 on both axes.
   TiledRuntimeTensor<double> A("A", Grid{{2, 4}, {2, 4}});

   auto const &off = A.tile_offsets();   // global start index per tile, per axis
   auto const &sz  = A.tile_sizes();     // tile extents per axis

   for (int ti = 0; ti < 2; ++ti) {
       for (int tj = 0; tj < 2; ++tj) {
           auto &tile = A.tile({ti, tj});   // creates the tile (deferred) if absent
           tile.materialize();
           for (int lr = 0; lr < sz[0][ti]; ++lr) {
               for (int lc = 0; lc < sz[1][tj]; ++lc) {
                   tile(std::vector<size_t>{size_t(lr), size_t(lc)}) =
                       f(off[0][ti] + lr, off[1][tj] + lc);
               }
           }
       }
   }

Leave a tile out and it is a structural zero. ``add_tile(coord)`` declares one
without touching it, ``has_tile(coord)`` asks, and ``num_filled_tiles()`` and
``grid_size()`` report the sparsity.

From Python the dtype-specific classes are ``TiledRuntimeTensorF``, ``...D``,
``...C`` and ``...Z``, and a tile is filled through a numpy-backed view:

.. code-block:: python

   import numpy as np
   import einsums

   A = einsums.TiledRuntimeTensorD("A", [[2, 4], [2, 4]])
   for ti in range(2):
       for tj in range(2):
           A.add_tile([ti, tj])
   A.materialize()

   a = np.asarray(A.tile_view([0, 0]))   # dense RuntimeTensorView over one tile
   a[:] = ...

.. note::

   A multi-tile tensor has no single contiguous backing buffer, so ``data()``
   returns ``nullptr`` (the one-cell degenerate grid returns that tile's
   pointer) and the type exposes no Python buffer protocol. Read and write it
   through ``tile_view()``.

Capturing tiled operations
==========================

Inside a ``CaptureGuard``, tiled operands work through the same call surface as
dense ones:

.. code-block:: cpp

   cg::Graph graph("tiled");
   {
       cg::CaptureGuard const capture(graph);
       cg::einsum("ij <- ik ; kj", &C, A, B);   // one Custom node at capture
   }
   graph.optimize();                            // TiledExpansion lowers it
   graph.execute();

These operations accept tiled operands: ``einsum``, ``permute``, ``scale``,
``axpy``, ``axpby``, ``direct_division``, ``element_transform``, ``conj``,
``real``, ``imag``, ``abs``, ``dot``, ``dotc``, ``norm``, ``trace``, ``syev``
and ``heev``.

Mixing tiled and dense operands in one call is rejected at compile time: a
tiled ``einsum``, ``permute``, ``axpy`` or ``axpby`` requires **every** operand
to be a ``TiledRuntimeTensor``.

Semantics
---------

The ground truth is ``detail::tiled_runtime_einsum``, and every lowering the
optimizer performs must reproduce it exactly:

**Alignment.**
    An index letter shared between operands must carry an **identical** tile
    partition on each of them. A mismatch throws; general re-tiling is out of
    scope.

**Sparsity.**
    A tile pair contributes only when both operand tiles are present. Output
    tile ``C(c)`` exists if and only if some contributing combination reaches it
    (infer-and-create: the tile is made on demand, zeroed).

**The output prefactor applies exactly once.**
    Pre-existing output tiles are scaled by ``c_pf`` up front, and every
    contribution then accumulates with ``beta = 1``.

**Pre-existing output tiles that receive no contribution are still scaled.**
    They are scaled up front regardless of whether anything reaches them.
    Skipping them would be a silent numerical difference.

**A tiled axpy visits the tiles stored in X.**
    The matching Y tile is created when absent, starting zeroed. A tile absent
    from X contributes nothing, and a Y tile with no X counterpart is left alone.

Scalar-output (full-reduction) einsum over tiled operands is not supported.
Reductions go through ``dot``/``dotc``/``norm``/``trace`` instead, which write
into a one-element dense result tensor.

Views of tiles
==============

``cg::tile_view(parent, coord)`` records a dense, parent-aliasing view of a
single tile. Because a tile is a full-rank dense block, that view feeds the
**entire** dense operation surface inside capture, which is exactly how tiled
workflows iterate:

.. code-block:: python

   with cg.capture(g):
       for i in range(2):
           a = cg.tile_view(A, [i, 0])
           c = cg.tile_view(C, [i, 0])
           einsums.einsum("ij <- ik ; kj", c, a, B, c_pf=0.0, ab_pf=1.0)

Three details matter:

- The tile is created (deferred, dims fixed by the grid) at capture if absent.
  That is value-equivalent, since absent tiles are rigorous zeros and a created
  tile materializes zeroed.
- Each replay re-resolves the tile from the **live** parent and re-emplaces the
  view, so a graph-owned deferred parent that FreeInsertion released and
  Materialize re-allocated stays correct.
- The recorded descriptor carries one constant ``[c, c+1)`` range per axis **in
  tile units**, so the disjointness-aware hazard scan proves views of distinct
  tiles non-conflicting and per-tile writes to one output can be scheduled
  concurrently.

Element-granular slicing of a tiled tensor is deliberately not offered: a slice
crossing tile boundaries has no single buffer. For a per-irrep sub-selection use
:cpp:struct:`einsums::TiledRuntimeTensorView` through
``TiledRuntimeTensor::view(spaces)``, which takes one
:cpp:struct:`~einsums::IndexSpace` per axis.

Graph-owned tiled scratch
=========================

``Graph::declare_zero_tiled_tensor(name, tile_sizes, intermediate)`` declares a
graph-owned tiled shell with a **deferred** lifecycle. The shell starts with no
populated tiles, and ops create them on demand, so Materialize and Initialize are
cheap. Registering the handle as deferred is what puts the tensor under the same
lifecycle machinery as dense scratch:

- ``Materialization`` hoists the Materialize/Initialize pair to the loop's parent
  rather than re-running it per iteration.
- ``FreeInsertion`` can release the tile storage after the last consumer.
  ``release()`` keeps the sparsity pattern, so a later replay re-materializes
  into the same structure.
- ``MemoryPlanning`` leaves tiled scratch alone by construction: the arena
  requires ``materialize_into``, and a tile-wise tensor has no single buffer to
  place.

Reusing the scratch across replays is safe for the ops that carry the
leftover-scale rule. ``einsum``'s ``c_pf``, and ``direct_division``'s and
``permute``'s ``beta``, apply to every stored tile, so stale tiles from a
previous iteration are zeroed rather than kept.

.. _computegraph_tiled_expansion:

TiledExpansion
==============

``TiledExpansion`` runs **first** in the default pipeline. It is a lowering step,
so every pass below it should see the per-tile form rather than the opaque node.

It replaces the tiled node with one ordinary dense einsum per contributing tile
combination, built through ``Graph::make_einsum_node``. Because the result is
ordinary dense nodes, everything downstream applies unchanged: the tile GEMMs
are same-shape so GEMMBatching batches them, CSE deduplicates repeated tile work,
MemoryPlanning packs the per-tile buffers (each tile *is* a dense
``RuntimeTensor``, so it has ``materialize_into``), and Reorder schedules them.

Tiled ``scale`` and ``axpy`` lower the same way, into one dense ``OpKind::Scale``
or ``OpKind::Axpby`` per stored tile.

Emission order
--------------

Contracted indices are enumerated **slowest** and output indices fastest, so
every tile GEMM at the same accumulation step is emitted as one contiguous run.
GEMMBatching only batches a group whose span contains no outside node touching
the same buffers; letting the contracted index vary fastest would drop each
output tile's later accumulations in between the first writes, disqualifying
every group and leaving the expansion paying node-count cost for nothing.

Per output tile the contracted steps stay ascending, so each tile accumulates in
the order the runtime uses and the result is bit-identical. Only independent
tiles move relative to one another.

Predicted tile sets
-------------------

A tiled tensor produced *inside* the graph has none of its tiles yet at pass
time, so its sparsity cannot be read off the object, and reading the empty object
would make a consumer expand into nothing. Planning therefore walks the nodes in
execution order carrying a **predicted** tile set per tensor, seeded from the
stored tiles and extended by each producer it plans. That is what lets a
contraction feed another, or feed a scale, and still expand.

The prediction also decides whether the first write to an output tile carries
``c_pf`` or overwrites, so it must be the set as of *this* point in the program,
not the final one.

Stranding
---------

Expanding a node replaces its whole-tensor reads and writes with per-tile ones,
so the whole-tensor ``TensorId`` loses every reference that node owned. A node
left behind that still names that id would have its dependency edge silently
dropped, and with no writer a reader could be scheduled before the tiles are
filled.

A candidate therefore expands only when **every** node touching its tiled tensors
also expands. Since rejecting one candidate can strand another, this iterates to
a fixpoint, and it is decided before any tile is created: minting a per-tile id
creates the tile, and a spurious tile changes how the runtime applies ``c_pf`` on
the path that ends up not expanding.

Zero-tile screening
-------------------

Structural sparsity is free, but a tile that is *stored* and zero still gets a
full GEMM. The ``zero_tile_tolerance`` constructor parameter screens those: a
screened tile is treated as **absent**, so the prefactor and output-existence
rules apply unchanged instead of being re-derived against already-emitted nodes.

Sparsity then propagates on its own. An output tile whose every contribution
screened out is never created, so it is absent to the next contraction, which
screens further work without being told anything.

The default is ``-1.0``, which disables screening entirely: no tile is inspected
and the emitted node set is unchanged. Zero prunes only exactly-zero tiles.
A positive value is a numerical approximation and an accuracy knob. Both origins
are covered: symmetry blocking is exact and wants tolerance 0, while integral
screening and local-correlation domains are approximate and want a real
threshold.

Only operands that **nothing in the graph writes** are screened. A produced
tensor's tiles hold whatever was in them before execution, so inspecting them
would screen on values the graph has not computed yet. Non-materialized tiles are
likewise left alone, since planning must not allocate.

Densifying small tiles
----------------------

Expanding per tile only pays when a tile contraction is worth a dispatch. Often
it is not. A symmetry-blocked CCSD residual on a small model expands to nearly
4000 einsum nodes carrying 1.3 MFLOP between them, a mean of 330 flops per node,
which is less work than the call costs. Blocking saves real arithmetic there and
then spends the whole saving dispatching it.

So when a tile contraction is not worth its dispatch, the pass lowers it
differently: gather each tiled operand into a dense buffer, run **one** dense
einsum, and scatter the result back into exactly the output tiles the per-tile
path would have written. Absent tiles gather as zeros and forbidden output tiles
are still never created, so the answer is the same up to floating-point summation
order.

Measured on the particle-particle ladder of that model, this replaces 256 nodes
taking 0.595 ms with 3 nodes taking 0.214 ms: faster despite doing strictly more
arithmetic, because the dense form reaches about 61 GFLOP/s where 256 tiny
contractions reach about 2.

The trade is arithmetic for throughput, so which lowering wins is a question
about **time**, not about either quantity alone. ``Densify::Auto``, the default,
asks the shared ``CostModel``: sum the estimated time over the tile contractions
that would be emitted, against one dense contraction of the same indices plus
the gather and scatter traffic at memory bandwidth, and densify only when the
second is smaller.

.. note::

   A flop-based gate was tried first and does not work. An inflation cap of 16
   takes a 26 spin-orbital model from 19.0 to 11.0 ms but a 50 spin-orbital model
   from 71.5 to 103.1 ms, and a cap of 4 never fires at either size. Those
   contractions inflate the arithmetic by 4x to 16x either way, so no threshold
   on the ratio separates the two cases. What differs is achievable throughput,
   and only a time comparison sees that.

.. important::

   Densification ignores tile screening. ``zero_tile_tolerance`` prunes
   near-zero operand tiles from the per-tile enumeration; the densified path
   gathers them anyway and contributes their near-zero value rather than
   dropping it. The densified answer is the more accurate of the two.

Gathering an operand once
-------------------------

A densified contraction copies each tiled operand into a dense buffer, and the
same operand is usually contracted several times running: a CCSD residual gathers
the singles amplitudes 33 times and the doubles eleven. Those copies are
identical while nothing has written the tiles between them, so the pass keys each
gather on the exact list of tile ids it covers and reuses the buffer when that
list is already in hand.

Two things keep it honest. A tensor that has *gained* a tile produces a different
list, so it misses and is gathered afresh rather than silently contracting a copy
that is missing a block. And an entry is dropped as soon as any emitted node
writes a tile it covers.

On that residual the gathers fall from 92 nodes to 23, and from 2.20 ms to 0.71
of a 10 ms replay.

Fusing small elementwise tiles
------------------------------

The elementwise ops have the same node-count problem and none of the tension. A
densified CCSD residual is 82% tiled ``scale`` and ``axpy`` nodes, 1152 of 1404,
each touching a handful of kilobytes. Unlike a contraction there is nothing to
trade: running the same tiles from inside one node does exactly the same
arithmetic and touches exactly the same memory, so fusing is never slower.

What per-tile nodes buy is optimizer exposure, and no cost model can price that.
So ``FuseTiles::Auto``, the default, asks the question it *can* answer: is a
tile's own memory traffic worth the dispatch it costs? It compares the tiles'
total traffic at memory bandwidth against the cost model's per-node overhead, and
collapses the group into a single ``OpKind::TileElementwise`` node when the
dispatches dominate.

On that residual the graph falls from 1404 nodes to 209 and replay from 10.1 to
9.9 ms. The node count is the larger prize: the executor's own per-node cost is
about 0.2 us, so the replay time recovered is a few percent, not the fifth the
node count suggests.

Configuration
-------------

.. code-block:: cpp

   // The default-pipeline configuration, spelled out.
   pm.add<cg::passes::TiledExpansion>(
       /*max_nodes=*/4096,
       /*zero_tile_tolerance=*/-1.0,
       cg::passes::Densify::Auto,
       cg::passes::FuseTiles::Auto,
       cost_model);

``max_nodes``
    Expansion produces up to (tiles of A) x (tiles of B) nodes, and per-node
    bookkeeping is on the order of microseconds, so a large grid can cost more in
    overhead than the contraction saves. The pass **declines** above this budget
    rather than guessing: a gate cannot be subtly wrong the way a cost heuristic
    can.

``zero_tile_tolerance``
    See `Zero-tile screening`_ above.

``Densify``
    ``Never`` always emits one node per contributing tile combination, ``Auto``
    lets the cost model decide per contraction, ``Always`` densifies whenever it
    is structurally possible (for tests).

``FuseTiles``
    ``Never`` always emits one dense node per tile, ``Auto`` fuses when the tiles
    cost more to dispatch than to touch, ``Always`` fuses every group of two or
    more (for tests).

Limits
------

- Sparsity is decided at pass time. For an operand written earlier in the same
  graph that means a prediction, which holds only while every writer is one this
  pass understands.
- The pass **declines** rather than guessing whenever tile sparsity is not
  decidable, or the partitions are misaligned, or the budget is exceeded, or the
  candidate shares a tiled tensor with a node that cannot expand. The surviving
  opaque node is correct, merely unoptimized.
- Creating the predicted output tiles is a side effect of ``apply()`` on user
  data, earlier than the execute-time infer-and-create it replaces.

The pass reports ``num_expanded()``, ``num_tile_nodes()``, ``num_declined()``,
``num_screened()``, ``num_densified()``, ``num_fused()`` and
``num_gathers_reused()``.

See also
========

- :doc:`optimization_passes` for where TiledExpansion sits in the pipeline.
- :doc:`views` for the dense view surface that ``tile_view`` feeds.
- :doc:`workspace` for the deferred-allocation lifecycle tiled scratch joins.
