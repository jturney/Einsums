..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-performance:

*****************************
Tutorial: Performance Tuning
*****************************

This tutorial explains how Einsums chooses a kernel for a contraction, how to use the profiler to
find out where time actually goes, and how to write code that gives the library something to work
with.

For structuring code so the time goes to arithmetic in the first place, see
:doc:`tutorial_best_practices`. For choosing a cheaper *algorithm*, by factoring an integral or
truncating a space, see :doc:`optimizer`.

.. contents:: On this page
    :local:
    :depth: 2

How a Contraction Is Written
============================

An einsum names its operands' axes with a string spec, and the pattern is analysed when the call
runs:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * -
      - String spec
    * - How you write it
      - ``cg::einsum("ik;kj->ij", &C, A, B)``
    * - Pattern analysed
      - when the call runs; inside a graph, once at capture and again at each replay
    * - Used by
      - eager C++ on :cpp:type:`einsums::Tensor` and :cpp:type:`einsums::RuntimeTensor`, the
        ComputeGraph, the Python bindings
    * - Names its route
      - ``last_dispatch_route()``
    * - Route names
      - ``gemm_direct``, ``gemm_direct_runtime``, ``packed_gemm``, ``generic_loop``, ...

A route with a ``_runtime`` suffix is the same kernel reached from runtime-rank operands.

The compile-time-index form, ``einsum(Indices{i,j}, &C, Indices{i,k}, A, Indices{k,j}, B)``,
remains for eager code written against it. It analyses the pattern while your code compiles and
does not share that analysis with the string form, so the two can route one contraction
differently; the string form, which ComputeGraph and the Python bindings use, is the one this page
describes, and the one to tune for.

How Dispatch Chooses
====================

Dispatch walks a ladder, stopping at the first rung that fits. Two cases are settled before it:
a zero-length dimension, which only scales the output, and a letter repeated inside one operand or
summed out of one operand alone, which goes to a repeat-aware loop.

1. **A vendor BLAS call**, when the contraction already is one: ``DOT`` for a scalar result over
   identical index packs, ``GER`` for an outer product, ``GEMV`` for matrix times vector, ``GEMM``
   for matrix times matrix, and an elementwise product. Nothing is copied and nothing is packed.

2. **PackedGemm**, when the contraction has a valid decomposition into output dimensions from A
   (M), output dimensions from B (N), summed dimensions (K), and batch dimensions present in both.
   This covers most higher-rank tensor contractions. See `Understanding PackedGemm`_.

3. **A generic loop nest**, for everything else. Correct, and the slowest option.

Dispatch never permutes an operand to reach a BLAS call. It uses the transposition flags a BLAS
call already offers, so :math:`C_{ik} = \sum_j A_{ji} B_{kj}` still reaches one ``GEMM``; any other
index order goes to PackedGemm, which rearranges the data as it packs it. If you know a
permutation would pay for a contraction that lands on the generic loop, do it yourself with
``cg::permute`` and contract the result.

`Dispatch Reference`_ below has the measured route for every common shape.

Using the Profiler
==================

Einsums is instrumented throughout. The profiler is not a sampling profiler attached from outside:
every BLAS wrapper, every einsum, every graph node and every pass already opens a named region, so
a report tells you which *operations* your time went to rather than which functions the optimizer
happened to leave standing.

It records three things:

**Zones**
    A named, nested region with a start and an end. Zones nest by call depth, so the report is a
    tree, and each one carries its file, line and enclosing function.

**Annotations**
    Key-value pairs attached to the enclosing zone. This is how an einsum records which route it
    took, what dtype it was, and the ranks of its operands, and it is why the profile can answer
    "which kernel ran" without a second tool.

**Counters**
    Hardware counters per zone where the platform provides them, reported alongside the timings.

What It Costs
-------------

Instrumentation is on by default, at build time through ``EINSUMS_WITH_PROFILER`` and at run time
unless you disable it. A zone entry is a few tens of nanoseconds, which disappears next to a
contraction and does not disappear next to a graph of many tiny nodes.

If you are measuring a workload made of many small operations, turn it off for the measurement:

.. code-block:: bash

    ./my_program --einsums:profile:disable

``EINSUMS_WITH_PROFILER_INTERNAL`` is a separate, default-off build option that adds zones inside
Einsums' own hot paths. It is for working on the library, not for profiling a calculation.

Getting a Report
----------------

A report is written at shutdown, with no code change required:

.. code-block:: cpp

    #include <Einsums/Runtime.hpp>

    int main(int argc, char **argv) {
        einsums::initialize(argc, argv);

        // ... your calculation ...

        einsums::finalize();   // the report is written here
        return 0;
    }

By default it lands in ``profile.txt`` in the working directory, appended rather than truncated.
The options that shape it:

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Option
      - Effect
    * - :option:`--einsums:profile:disable`
      - Record nothing. The large speedup for many-small-operations workloads.
    * - :option:`--einsums:profile:report`
      - Write the report at exit. On by default; turn it off with ``--einsums:profile:no-report``.
    * - :option:`--einsums:profile:filename`
      - Where the report goes. Defaults to ``profile.txt``.
    * - :option:`--einsums:profile:append`
      - Append rather than truncate. On by default.
    * - :option:`--einsums:profile:detailed`
      - Report every zone with its counters, rather than the summary.
    * - :option:`--einsums:profile:save`
      - Also write the session as JSON, for the viewer to load later. Requires
        ``--einsums:profile:server``, and says so if it is missing.
    * - :option:`--einsums:profile:server`
      - Serve live data over TCP so a viewer can attach.
    * - :option:`--einsums:profile:port`
      - The port to serve on. Defaults to 19216.
    * - :option:`--einsums:profile:wait-for-viewer`
      - Block at startup until a viewer connects, so nothing is missed.

Reading the Report
------------------

The report is one tree per thread. Indentation is nesting, and a zone's own annotations follow its
name:

.. code-block:: text

    Thread: main (22306818)  (total exclusive:   2.354 ms)
      total(ms)    count       mean(ms)     name                     file:line              function
          0.458           1    0.458±0.000  Runtime constructor      Runtime.cpp:274        Runtime
          0.263           1    0.263±0.000  Calling startup routines Runtime.cpp:410        call_startup_functions
          0.789           9    0.088±0.050  dgemm                    gemm.cpp:123           dgemm
          0.531           1    0.531±0.000  demo run                 probe_profiler.cpp:26  main
          0.002           1    0.002±0.000    iteration 0            probe_profiler.cpp:34  main  iter=0
          0.027           1    0.027±0.000      build_intermediate   probe_profiler.cpp:18  build_intermediate  stage=intermediate
          0.239           1    0.239±0.000        cg::einsum: i,j <- i,k ; k,j   StringDispatch.hpp:492  string_einsum  a_rank=2 b_rank=2 c_rank=2 dispatch=gemm_direct_runtime

Four things are worth pointing at in that output.

``total(ms)`` is inclusive of nested zones, so a parent's number contains its children. The
thread header reports total **exclusive** time, which is the figure to compare against wall clock.

``count`` and ``mean±stddev`` are what distinguish "one slow call" from "a million fast ones". A
large standard deviation on a repeated zone usually means load imbalance or a cold first call.

``dgemm`` appears at the top level rather than nested under the einsum that called it, because
the BLAS wrappers are instrumented independently. Nine calls for five iterations is the give-away
that something ran more often than the loop count suggests.

The trailing ``dispatch=gemm_direct_runtime`` is the einsum telling you which kernel it chose.
That annotation is the most direct answer to "is this contraction accelerated", and it is present
in every report without a special flag.

Instrumenting Your Own Code
---------------------------

Use the ``LabeledSection`` macros. They open a zone that closes when the enclosing scope exits, so
an early return or a thrown exception cannot leave the tree unbalanced:

.. code-block:: cpp

    #include <Einsums/Profile/Profile.hpp>

    void build_fock(Tensor<double, 2> &F) {
        LabeledSection0();                       // zone named after the function

        for (int iter = 0; iter < n; ++iter) {
            LabeledSection("iteration {}", iter); // name built per call
            ProfileAnnotate("stage", "fock");
            ProfileAnnotate("iter", static_cast<std::int64_t>(iter));
            // ... work ...
        }
    }

.. list-table::
    :header-rows: 1
    :widths: 34 66

    * - Macro
      - Use
    * - ``LabeledSection0()``
      - A zone named after the enclosing function. The common case.
    * - ``LabeledSection("literal")``
      - A zone with a fixed name, interned once per call site.
    * - ``LabeledSection("fmt {}", args...)``
      - A name built per call. The format string must be a literal.
    * - ``LabeledSectionRuntime(expr)``
      - A name only known at run time. Interned on every entry, which takes a lock, so prefer one of the above where the label can be a literal.
    * - ``ProfileAnnotate(key, value)``
      - Attach a string, integer or double to the enclosing zone.
    * - ``ProfileAnnotateDims(key, dims)``
      - Attach a sequence of extents as ``key.0``, ``key.1``, ...

These are the library's own instrumentation, used at well over seven hundred call sites, and they
compile to nothing when ``EINSUMS_WITH_PROFILER`` is off, so an instrumented function needs no
preprocessor guard of its own.

There is also a lower-level ``Profiler::instance().push(name)`` and ``pop()`` pair. Prefer the
macros: a manual pair leaks a zone if anything between them throws.

From Python
-----------

``einsums.profile`` exposes the same two facilities as a context manager and a helper:

.. code-block:: python

    import einsums.profile as profile

    with profile.section("ao_to_mo"):
        transform(...)
        profile.annotate_dims("shape", [nocc, nvirt, naux])

The Live Viewer
---------------

``devtools/profiling/profile_viewer.py`` is a terminal interface that attaches to a running
program and shows the tree as it fills. Start the program with the server enabled, then attach:

.. code-block:: bash

    # Terminal 1: run with the profile server on (defaults to port 19216)
    ./my_program --einsums:profile:server

    # Terminal 2: attach
    python devtools/profiling/profile_viewer.py

The viewer finds local sessions over mDNS; ``--host`` and ``--port`` reach one explicitly, and
``--no-mdns`` turns discovery off. If a program is short enough that you would miss the start, run
it with :option:`--einsums:profile:wait-for-viewer` and it will block until you attach.

The keys worth knowing:

.. list-table::
    :header-rows: 1
    :widths: 14 86

    * - Key
      - Action
    * - ``/``
      - Filter zones by name, regular expressions accepted.
    * - ``u``
      - Bottom-up view: who called the expensive leaves.
    * - ``H``
      - Top-N hotspots panel.
    * - ``G``
      - Thread Gantt chart, for seeing imbalance across workers.
    * - ``C``
      - Compare two sessions side by side, which is how you show a change helped.
    * - ``b`` / ``B``
      - Bookmark a zone, and jump to the next bookmark.
    * - ``V``
      - Source panel for the selected zone.
    * - ``S``
      - Save the session to JSON.

Sessions can be recorded and replayed without a live program, which is the usual way to compare a
run before and after a change:

.. code-block:: bash

    python devtools/profiling/profile_viewer.py --record before.jsonl
    # ... make the change, run again ...
    python devtools/profiling/profile_viewer.py --replay before.jsonl

A program run with :option:`--einsums:profile:save` writes a session file directly, which
``--load`` opens; ``--load`` accepts several files at once for comparison.

One constraint on that option: it writes nothing unless :option:`--einsums:profile:server` is
also given, because the export goes through the server object. Asking for a session without one
is reported rather than ignored, so the run says so instead of leaving you to notice a missing
file.

The session is exported during teardown, whichever way a program ends, so it holds the same run
the text report does.

Finding Out Which Kernel Ran
----------------------------

Three ways, in the order they are usually easiest.

**The profile annotation.** Every einsum zone carries a ``dispatch`` annotation naming its route,
and PackedGemm records why it declined under ``packed_gemm_skip`` and which path it took under
``packed_gemm_path``. No flags, no code change.

**The log.** PackedGemm explains its declines at INFO level, which
:option:`--einsums:log:level` controls. A contraction that lands on the generic loop is not
logged; the annotation above and the route below are where it shows.

**Programmatically.** A thread-local naming the last route:

.. code-block:: cpp

    #include <Einsums/ComputeGraph/StringDispatch.hpp>

    cg::einsum("ik;kj->ij", &C, A, B);
    std::printf("%s\n", cg::dispatch::last_dispatch_route());   // gemm_direct_runtime

It exists for test introspection. Assert on it in a test that means to pin a fast path; do not
branch on it inside a calculation.

Performance Tips
================

1. Prefer einsum over manual loops
-----------------------------------

Einsums dispatches to BLAS when it can. Hand-written loops almost never will:

.. code-block:: cpp

    // SLOW: manual loop
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C(i, j) += A(i, k) * B(k, j);

    // FAST: dispatches to BLAS GEMM
    cg::einsum("ij <- ik ; kj", &C, A, B);

2. Use views instead of copying
--------------------------------

Views avoid an allocation and a copy:

.. code-block:: cpp

    // SLOW: copies a submatrix
    auto block = Tensor<double, 2>("block", 3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            block(i, j) = A(i + 2, j + 5);

    // FAST: zero-copy view
    auto block = A(Range{2, 5}, Range{5, 8});

3. Mind index ordering for cache efficiency
--------------------------------------------

Einsums tensors are column-major, so the first index varies fastest and a run along it is
contiguous. A contraction is cheapest when the indices it walks innermost are the contiguous
ones, because PackedGemm can then fill its blocks from stride-1 runs rather than gathering
element by element:

.. code-block:: cpp

    // C_il = A_ijk * B_jkl
    // i varies fastest in A, so runs over i are contiguous
    // j varies fastest in B, so runs over j are contiguous

4. Use ComputeGraph for iterative code
---------------------------------------

Capture once, replay many times. The graph pays the pattern analysis once, and every optimization
the library performs happens on the captured form:

.. code-block:: cpp

    cg::Graph graph("iteration");
    { /* capture the body once */ }

    graph.optimize();

    for (int iter = 0; iter < 1000; iter++) {
        graph.execute();       // no re-dispatch, no re-analysis
    }

One rule to get right before measuring anything: a graph-owned tensor you read after
``execute()`` must be created as a result rather than as scratch, or the optimizer removes the
node that writes it. See :ref:`howto-graphs-intermediate`.

5. Use TaskPool for data-parallel workloads
--------------------------------------------

For workloads with thousands of independent tasks, such as integral generation, use the TaskPool
rather than raw OpenMP:

.. code-block:: cpp

    #include <Einsums/TaskPool/TaskPool.hpp>

    auto &pool = einsums::task_pool::TaskPool::get_singleton();

    // parallel_for: chunked, work-stealing, load-balanced
    pool.parallel_for("integral_batches", 0, num_shell_pairs, [&](size_t pair) {
        compute_integrals(pair, &F_local);
    });

    // parallel_reduce: thread-local accumulators, no false sharing
    double energy = pool.parallel_reduce<double>("energy", 0, N,
        []() { return 0.0; },
        [&](size_t i, double &acc) { acc += compute_contribution(i); },
        [](double &g, double const &l) { g += l; }
    );

What it buys over an OpenMP loop: work-stealing absorbs load imbalance, continuations
(``.then()``) and dataflow remove hand-written synchronization, and every task is a profiler zone,
so per-task and per-worker timings show up in the report without extra instrumentation.

6. State a memory cap instead of writing the loop
--------------------------------------------------

A captured program declares the tensors its equations mention, and for a density-fitted method
those are exactly the objects density fitting exists to avoid holding. The usual answer is to
write the pair loop by hand. The other answer is to state the footprint you have to live inside
and let ``AxisTiling`` derive the loop:

.. code-block:: python

    tiling = cg.AxisTiling()
    tiling.set_memory_cap(4096)          # bytes
    pm = cg.PassManager()
    pm.add(tiling)
    g.apply(pm)
    g.apply(cg.default_pass_manager())

The pass slices free axes of the program and streams every intermediate that carries them, so the
body is the same algebra at one slice, and a chunk of slices deeper than one takes the grouped
kernels. The cap, :option:`--einsums:graph:tiling-memory-cap`, is read three times over: whether
to tile at all, since a program whose largest intermediate already fits is left alone; which axes
to slice; and how deep a chunk may be.

It is **not** in the default pipeline, and the default cap is deliberately far above anything a
captured program declares. Slicing a program that already fits costs kernel efficiency for memory
nobody was short of, so this fires when a caller asks for it. Zero is the spelling for the
captured schedule.

On the full-axis DF-MP2 capture at water in cc-pVDZ, a cap of 4096 bytes takes the largest
intermediate inside the loop from 72200 bytes to 2888 and turns six captured nodes into a
``Scale`` and a ``Loop``. ``optimizer_tour.py`` in the ComputeGraph examples prints the decision
and the buffers the tiled graph allocates.

7. Give a search pass room, or take its allowance away
-------------------------------------------------------

Most passes walk the graph once. ``MultiTermFactorization`` searches, and its runtime is a
function of how many candidates the program offers rather than of how large it is, which is why
the search half is off unless a program asks for it with
:option:`--einsums:graph:structural-search`.

:option:`--einsums:graph:optimizer-budget` is the wall-clock allowance each search pass gets, in
milliseconds, with ``PassManager.set_optimizer_budget`` as the per-pipeline override and zero
meaning unlimited. Running out of it costs optimization rather than correctness: the pass keeps
the best candidate it had reached and reports that it was cut off. That candidate is a
**different graph** from the one an unbounded search would emit, so a program that compares two
emitted graphs, or a measurement that has to be reproducible across machines, should take the
allowance away with ``set_optimizer_budget(0)`` and check ``was_cut_off``.

Within one process, the plan a search chose is cached and replayed for a structurally identical
graph, which is what makes a pipeline whose stages present the same program search once. See
:doc:`optimizer`.

Understanding PackedGemm
========================

PackedGemm is the BLIS-inspired backend that handles contractions beyond a plain GEMM. It

1. classifies the contraction into M dimensions (output, from A), N dimensions (output, from B),
   K dimensions (summed) and batch dimensions (present in both operands and the output);
2. takes a vendor GEMM directly when the strides already describe one, either per batch slice,
   over flattened multi-K buffers, or as a single ``gemm_batch`` call;
3. otherwise packs A and B into cache-friendly contiguous buffers and contracts each block with
   either its own register-blocked micro-kernel or one vendor GEMM per block, whichever is faster
   on this CPU.

It beats the generic loop nest because the data is laid out for L1/L2/L3 reuse, the inner kernel
is vectorized (the vendor's, or a per-instruction-set micro-kernel chosen at run time), and the
packing cost is amortized over the whole contraction. Batch contractions go through ``gemm_batch``,
which runs the slices in parallel.

The requirement is a valid M/N/K decomposition: every output index must classify as M-only from A,
N-only from B, or batch from both. Rank is otherwise unrestricted.

See :doc:`architecture` for the full route list and how the micro-kernel rung is chosen.

.. _dispatch-reference:

Dispatch Reference
==================

The routes below are measured, not predicted: each one is the value
``last_dispatch_route()`` returned for that spec on the build these docs were generated from, and
``docs/doctests`` pins them so the table cannot quietly go stale.

Through the string form
-----------------------

.. list-table::
    :header-rows: 1
    :widths: 34 34 32

    * - Spec
      - Route
      - What runs
    * - ``i;i->``
      - ``dot_runtime``
      - one ``DOT``
    * - ``i;j->ij``
      - ``ger_runtime``
      - rank-1 update
    * - ``ij;j->i``
      - ``gemv_mat_vec_runtime``
      - one ``GEMV``
    * - ``ik;kj->ij``
      - ``gemm_direct_runtime``
      - one ``GEMM``
    * - ``ki;jk->ij``
      - ``gemm_direct_runtime``
      - one ``GEMM``, using its transpose flags
    * - ``ij;ij->ij``
      - ``direct_product_runtime``
      - elementwise
    * - ``ijk;jkl->il``
      - ``packed_gemm``
      - multi-K
    * - ``ijk;klm->ijlm``
      - ``packed_gemm``
      - multi-M and multi-N
    * - ``bik;bkj->bij``
      - ``packed_gemm``
      - batch
    * - ``ijkl;klmn->ijmn``
      - ``packed_gemm``
      - rank-4 both sides
    * - ``ijk;k->ij``
      - ``packed_gemm``
      - all output indices from A
    * - ``ij;ji->i``
      - ``packed_gemm``
      - shared index plus a summed one
    * - ``ii;i->i``
      - ``generic_loop_repeated_indices``
      - diagonal
    * - ``ikk;k->i``
      - ``generic_loop_repeated_indices``
      - diagonal with a summed index

A contraction with a zero-length dimension reports ``empty_input_scale_only`` or
``empty_output_noop`` and still applies the output prefactor exactly once.

A scalar over *permuted* index packs, such as :math:`s = \sum_{ij} A_{ij} B_{ji}`, falls to the
generic loop: ``DOT`` needs the two operands to name their indices in the same order, and
PackedGemm does not produce a scalar.

What is still not accelerated
-----------------------------

**A letter repeated inside one operand.** ``A(i,k,k)`` is a diagonal, and a diagonal is not a
matrix multiplication, so it runs on a repeat-aware loop. If a diagonal sits inside an iteration,
extract it once outside the loop.

**A scalar result over permuted index packs.** ``DOT`` requires the two operands to name their
indices in the same order, so :math:`s = \sum_{ij} A_{ij} B_{ji}` takes the generic loop. Permute
one operand first if this is hot.

**Mixed value types.** A contraction whose operands and output are not all the same scalar type
runs on ``generic_loop_mixed_precision``, never BLAS or PackedGemm. Convert first if you care
about the speed.

**Tiled tensors.** A contraction over ``TiledRuntimeTensor`` operands runs tile by tile, each tile
a dense einsum that is accelerated. The per-tile work around it is not free, so a structure with
very many tiny tiles spends its time in bookkeeping.

What's Not Covered
==================

This page is about the CPU schedule: which kernel runs, and how to see it. Two neighbours cover
the rest.

Choosing a cheaper *algorithm*, by factoring an integral, decoupling an energy denominator or
truncating a space, is :doc:`optimizer`.

Structuring code so the time reaches arithmetic at all, including pooling transient state and not
materializing scratch that dies on first read, is :doc:`tutorial_best_practices`.

GPU execution is placed by the graph's GPU passes; :doc:`tutorial_compute_graph` covers offload
and the transfers around it.
