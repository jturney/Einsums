..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-write-a-pass:

**************************
Write an Optimization Pass
**************************

A pass inspects a captured graph and may rewrite it.
:doc:`/user/optimizer` documents the passes that ship and how to drive them; this page is about writing one.

.. contents:: On this page
    :local:
    :depth: 1

The minimum
===========

A pass derives from :cpp:class:`~einsums::compute_graph::OptimizerPass` and implements two functions:

.. code-block:: cpp

    class MyPass : public OptimizerPass {
      public:
        std::string name() const override { return "MyPass"; }

        bool run(Graph &graph) override {
            // inspect and possibly modify graph.nodes()
            return false;   // true if the graph was modified
        }
    };

The return value from ``run`` is how the manager knows whether anything changed, which drives re-running analysis passes and the report.
Returning ``true`` when nothing changed causes needless work; returning ``false`` after a rewrite is a correctness problem.

If your pass produces a valid topological ordering, call ``graph.mark_sorted()`` so the graph is not sorted again.

Headers go in ``libs/Einsums/ComputeGraph/include/Einsums/ComputeGraph/Passes/`` and implementations in ``libs/Einsums/ComputeGraph/src/Passes/``.

Declare the phase
=================

The phase answers one question: may a saved graph keep this pass's output, or must it be re-derived on the machine that loads it?

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Phase
      - Meaning
    * - ``Analysis``
      - Writes annotations only, never rewrites the node set. ``run()`` returns false.
    * - ``StructuralAlgebraic``
      - Machine-independent rewrites of the mathematics. **The only phase a saved graph persists.**
    * - ``StructuralResource``
      - Changes the node set for machine-dependent reasons: tiling, placement, distribution.
    * - ``Tuning``
      - Schedule, memory, batching, and thread decisions over a final node set.
    * - ``Diagnostic``
      - Read-only reporting. Changes nothing, not even annotations.

.. code-block:: cpp

    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }

The default is ``Tuning``, deliberately.
An unclassified pass must not have its output written into a graph file, so the safe default costs a re-run rather than a wrong answer on the next machine.

The rule that decides this for you: if your pass's answer could differ on another machine, with another cost model, or with another thread count, it is not ``StructuralAlgebraic``.
Tiling, placement, distribution, and batching all rewrite the node set for machine reasons, which is why ``StructuralResource`` exists rather than folding them into the structural phase.

Declare the tier
================

The tier is orthogonal to the phase and answers what the pass may do to the numbers.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Tier
      - Meaning
    * - ``BitwiseExact``
      - Identical to the last bit, on every toolchain.
    * - ``ReAssociating``
      - Mathematically equivalent but not bit-identical, because floating-point addition is not associative. Declares a bound and is validated against it.
    * - ``Tuning``
      - Same operations in the same order, a different schedule.

.. code-block:: cpp

    [[nodiscard]] PassTier tier() const override { return PassTier::BitwiseExact; }

Two points that are easy to get wrong.

A pass whose *algebra* is exact but that hands the vendor a *different kernel* is ``ReAssociating``, not ``BitwiseExact`` with a caveat.
``PermuteFusion`` folds a transpose into a ``transa`` flag, which is exact reasoning and a different BLAS path, and it is bit-identical on five CI legs and not on Accelerate.
A tier whose members each carry their own exception has stopped drawing the line it exists to draw.

Tier membership is measured rather than declared, across the CI legs that carry the tier report.
Do not classify from one machine.

Report what you did
===================

Override ``explain`` to contribute lines to ``Graph::explain()``:

.. code-block:: cpp

    [[nodiscard]] std::vector<std::string> explain() const override;

Keep per-run counters and zero them in ``reset_stats()``.
Expose the counters with ``APIARY_GETTER`` so Python tests can assert on them:

.. code-block:: cpp

    APIARY_EXPOSE APIARY_GETTER("num_eliminated")
    [[nodiscard]] std::size_t num_eliminated() const { return _num_eliminated; }

    void reset_stats() override;

Report what you declined, too.
The skip tally is the negative half of ``explain``, and it is what turns "the pass did nothing" into "the pass declined because this operand carried no space annotation".
From verbosity two up it appears in the report, and it saves the next person reading your source to find out why.

Decline early and cheaply
=========================

Your pass will run on every graph anyone optimizes, and almost none of them will be graphs it can help.
Give it a cheap gate at the top of ``run()``, before it does any real work:

.. code-block:: cpp

    bool MyPass::run(Graph &graph) {
        if (!has_anything_i_can_use(graph)) {
            return false;               // one cheap scan, then out
        }
        // ... the expensive part
    }

``DeltaElimination`` is the model to copy.
It is in the default pipeline, so it runs on every graph anyone optimizes, and almost none of them declare a Kronecker delta.
One scan of the tensor map is what keeps it from costing a region formation and a raise per region for a guaranteed no-op.

If you build on the region rewrite framework by deriving from ``RegionRewrite`` rather than from ``OptimizerPass`` directly, that gate has a dedicated hook:

.. code-block:: cpp

    [[nodiscard]] bool applicable(Graph const &graph) const override;

Returning false from it records a skip and stops before region formation, so the decline still appears in the report rather than being silent.
Note that this hook belongs to ``RegionRewrite`` and not to the ``OptimizerPass`` base, so a pass deriving straight from the base does the gate in ``run()`` as above.

Test it against the eager oracle
================================

Eager execution is the reference semantics, so the test for a pass is that the optimized graph agrees with it.

#. Build a graph that exercises the pattern your pass rewrites.
#. Execute it unoptimized and keep the result.
#. Apply your pass, execute again, and compare.
#. Assert your counter fired, so a pass that silently stopped matching is a red test rather than a quiet regression.

.. code-block:: cpp

    auto [modified, pass] = graph.apply<cg::passes::MyPass>();
    REQUIRE(modified);
    REQUIRE(pass.num_eliminated() == 1);

For a ``BitwiseExact`` pass, compare exactly.
For ``ReAssociating``, compare against the bound the pass declares, never against bit equality.

Extend the Hypothesis differential fuzzers as well.
``test_hyp_*_python.py`` compares graph results against NumPy oracles, and the right change is usually to widen the generator draws rather than to add fixed cases.

Do not write a test that depends on the clock or on a measured cost decision.
A search pass cut off by its budget returns a different valid graph, so a test pinning a tree should call ``set_optimizer_budget(0)`` and assert nothing was cut off.

Add it to the pipeline
======================

The canonical list is ``PassManager::build_default_passes()`` in ``libs/Einsums/ComputeGraph/src/Optimizer.cpp``.
``populate_default`` and the phase-filtered factories are all built from it, so they cannot drift apart.

Order matters, and the existing entries carry comments explaining why each sits where it does.
Read the neighbours before inserting: the reasoning is generally that a pass must run before another can see what it produced, or after another has cleaned up what it strands.
``DeltaElimination`` runs ahead of the cleanup cluster because dissolving an intermediate strands the allocation that named it, and ``DeadNodeElimination`` further down removes it.

Users can switch your pass off with :option:`--einsums:pass:disable`, and ``PassManager.disable`` is the programmatic half.

Checklist
=========

#. Header in ``Passes/``, implementation in ``src/Passes/``.
#. ``name()`` and ``run()``; ``run()`` returns whether it changed anything.
#. ``phase()`` and ``tier()`` declared, with the tier measured rather than assumed.
#. ``applicable()`` gate so the common case costs almost nothing.
#. ``explain()``, counters, ``reset_stats()``, and skip reasons for what it declined.
#. A test comparing against the unoptimized replay, asserting the counter fired.
#. An entry in ``build_default_passes()``, placed with a comment saying why it goes there.

Next
====

- :doc:`/user/optimizer` for the passes that ship and how they are driven.
- :ref:`howto-graphs` for capturing the graphs your pass will run on.
