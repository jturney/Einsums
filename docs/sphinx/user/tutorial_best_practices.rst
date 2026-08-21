..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-best-practices:

************************************
Tutorial: Performance Best Practices
************************************

This page collects the practices that get the best performance out of Einsums.
None of them are folklore: each one was earned by a measurement campaign on the
DLPNO coupled-cluster port, usually by killing a plausible hypothesis first,
and the rationale is stated with each rule so you can judge when it applies to
your code.
For understanding WHERE your time goes (dispatch routes, profiling, PackedGemm),
see :doc:`tutorial_performance`; this page is about how to structure code so
the time goes to arithmetic.

Capture once, replay many
=========================

An iterative algorithm should capture its body into a graph once and replay it,
rather than re-dispatching eager calls every iteration.
Eager execution is the reference semantics and stays the correctness oracle,
but every optimization the library offers - the passes, the batched calls, the
planned thread widths, the memory planning arena - operates on captured graphs.
A replayed graph pays no Python dispatch, no per-call overload resolution, and
no capture cost after the first pass, and the one-time graph build amortizes
with iteration count.

.. code-block:: python

    import einsums.graph as cg

    g = cg.Graph("iteration body")
    with cg.capture(g):
        emit_residual(state)          # ordinary einsums calls, recorded
    g.apply(cg.default_pass_manager())
    for _ in range(n_iterations):
        g.execute()                   # replayed, no re-dispatch

Two habits keep captures healthy.
Build every view BEFORE the capture: a view created inside a capture records a
node of its own, and a per-request view can cost as many nodes as the
arithmetic it feeds.
And use the pointer-writer forms of returning-style operations (``syev``,
``svd``, ``dot``): the returning forms throw during capture by design.

Allocate transient state from a memory pool
===========================================

Constructing a large owning tensor writes every page of it before you ever see
it: the backing ``std::vector`` value-initializes, which at seven megabytes
costs hundreds of microseconds of pure memory writing - measured at 370 us per
tensor against half a microsecond for a pooled carve.
A phase that builds thousands of scratch tensors serially pays seconds for
"allocation" that is really zero-fill, and the zero-fill is silent: the tensor
looks uninitialized and is not.

A :cpp:class:`einsums::MemoryPool` carves tensors out of a pre-reserved
mimalloc arena instead.
A pooled ``empty`` tensor is GENUINELY uninitialized - the carve is a pointer
claim, and first touch happens wherever your compute first writes, which in a
replayed graph is parallel code.
That makes write-before-read a real contract: every element must be written
before it is read, which the library's assign-on-zero-prefactor semantics and
beta-zero kernels are designed to honor.

.. code-block:: python

    import einsums

    pool = einsums.MemoryPool(reserve_bytes, "my phase")
    with pool.epoch():                       # cohort scope: bulk-free on exit
        t = pool.empty([nq, nt, nt], dtype="float64", name="(Q|a b)")
        ...                                  # build, execute, harvest results

Guidance that came from measured failures rather than taste:

* Make pools FEW and LONG-LIVED, reserved to their peak once.
  The arena address space is never returned to the OS, and growing in many
  small steps exhausts what one honest reservation would not.
* Leave pools UNPINNED (the default) unless they are small, hot scratch.
  A pinned pool never shrinks below its high water; a multi-gigabyte pinned
  pool on a laptop turned a working phase into an OS-thrashing one.
* An epoch is a leak detector as much as a scope: closing it throws if
  something still holds a carve, which is exactly how a graph that adopted a
  pooled operand and outlived its chunk announces itself - loudly, on the main
  thread, instead of as a use-after-free during the next replay.

Budget memory explicitly, and count what is actually allocated
==============================================================

``--einsums:max-memory`` (default: 80% of physical RAM) is a planning ceiling:
chunked algorithms size their working sets to it, and a pool reservation that
would push the process past it is refused at the reserve site with both numbers
in the message - on the owning thread, before the operating system is asked
for anything, and never inside a kernel.

Two rules make a chunk budget honest.
Charge :cpp:func:`einsums::pool_reserve_cost` for pooled state, not the raw
byte sum, so the plan and the reservation agree about headroom and rounding.
And when an intermediate stops being allocated, remove it from the accounting
the same day: a budget that describes tensors which no longer exist will
under-fill chunks, and one that misses real tensors will thrash.

Footprint is a cost even when allocation is free.
A working set that approaches physical RAM pushes the OS into reclaim and
compression, and the penalty lands on whatever runs during the high water -
in one measured case a three-second, 24% penalty on a phase whose kernels were
individually running at hardware speed.

Never materialize scratch that dies on first read
=================================================

The pattern to look for: gather or copy a block, feed it to exactly one
consumer, drop it.
Materializing that block costs the write traffic, the read-back, the
footprint, and (see above) possibly an OS-reclaim penalty - and all of it
disappears if the producer and consumer are fused so the data streams through
cache-sized tiles instead.

Einsums ships grouped operations for exactly these shapes:
``grouped_gather_rotate`` streams an index-selected block of a shared source
through L2-sized tiles and writes only the final rotated result, and
``grouped_sandwich`` does the same for dress-and-contract sequences.
In the DLPNO port, replacing a gather-then-rotate emission with the grouped
node deleted 55 GB of intermediate traffic per run, halved the phase's
footprint, and let the chunker run whole passes as single chunks - three wins
from one structural change.
The grouped nodes are also deterministic by construction: their tiling does
not change the bits.

If your pattern is symmetric (same index list and transform on both axes),
the grouped node applies as-is; an asymmetric rotation belongs on the plain
gather-plus-einsum path.

Respect the threading model
===========================

Plan, do not improvise.
Captured graphs get their thread widths from ``plan_widths``; kernels the
library threads itself consume the width they are granted, and vendor BLAS
calls are fenced to a single width because concurrent vendor callers that
disagree about thread count corrupt results with an OpenMP-built OpenBLAS.
Width-one LAPACK nodes run CONCURRENTLY - a graph of 64 independent
eigendecompositions was measured at 9.1x on ten threads - so expressing
independent work as independent nodes is how it parallelizes, not by threading
inside your own emission code.

A batched GEMM is only threaded if something gives it the team: under the
OpenMP executor, a graph whose level holds exactly one batched node runs it
unwrapped with the full team, which is why performance-critical batched calls
are emitted one to a graph.

And never call BLAS-backed einsums operations from multiple PYTHON threads:
an OpenMP OpenBLAS corrupts under callers with differing thread settings.
Parallelism belongs inside the library - graphs, grouped nodes, batched calls -
not above it.

Measure before optimizing, and attribute before believing
=========================================================

The campaign this page distills killed more hypotheses than it confirmed.
The allocator was blamed for cost that was value-initialization; a thread
fence was blamed for serialization that did not exist; a gather kernel was
suspected slow while running at memcpy speed; and hot allocate-free
microbenchmark loops exonerated an allocator that was genuinely costing
seconds in the monotonic, held-live pattern the real code runs.

The practices that survive:

* Time at the granularity of the thing you intend to blame - per graph, per
  chunk, with bytes and FLOP rates alongside - before changing anything.
  A phase timer that mixes a memcpy-bound copy with near-peak GEMMs will
  mislead you about both.
* Benchmark the PATTERN your code runs, not a loop that flatters caches:
  allocators, in particular, behave completely differently under
  allocate-free-allocate than under allocate-and-hold.
* Run comparisons PAIRED, with an in-run control on the same machine in the
  same minutes; absolute numbers from different sessions on a busy machine
  differ by 20% and mean nothing.
* Let the dispatch tell you what it did: ``last_dispatch_route`` and the
  profile annotations name the kernel route a call took, so a silent fallback
  to a generic loop is a fact you check, not a surprise you find in a flame
  graph.

Trust zero extents
==================

Zero-extent tensors are valid operands everywhere: empty BLAS and LAPACK
calls are quick-return no-ops, empty contractions still apply the output
prefactor exactly once, and the grouped nodes skip empty members.
Write chunking and screening code that passes empties through rather than
special-casing them - the special cases are where the off-by-one bugs live -
and never let a workspace-size formula reach zero.
