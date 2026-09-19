..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-graphs:

**************
Compute Graphs
**************

Recipes for capturing a calculation, optimizing it, and replaying it.
:doc:`/user/tutorial_compute_graph` introduces the machinery and :doc:`/user/optimizer` documents the passes.
This page is the short path from working eager code to a working graph, and the mistakes worth knowing about first.

.. contents:: On this page
    :local:
    :depth: 1

Use RuntimeTensor
=================

Prefer :cpp:type:`einsums::RuntimeTensor` for graph work.
It is the tensor type the graph is built around and the one the Python bindings expose, so a C++ capture and its Python equivalent look the same and behave the same.
The statically ranked :cpp:type:`einsums::Tensor` works in a capture too, and the rest of this page applies to both.

What actually decides whether a graph-owned tensor is scratch or a result is which creator you call, covered next.
Getting that wrong is the one mistake here that can pass silently.

Capture, optimize, replay
=========================

Capture records operations instead of running them.
Outside a capture the same calls run eagerly, so the code you write is the same either way.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Graph.hpp>
            #include <Einsums/ComputeGraph/Operations.hpp>
            #include <Einsums/Tensor/RuntimeTensor.hpp>

            namespace cg = einsums::compute_graph;

            cg::Graph g("demo");
            auto &C = g.create_runtime_tensor<double>("C", {m, n}, /*intermediate=*/false);  // a result, not scratch

            {
                cg::CaptureGuard guard(g);
                cg::einsum("ik;kj->ij", &C, A, B);
            }

            g.optimize();
            g.execute();      // replay as often as you like

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums, einsums.graph as cg

            g = cg.Graph("demo")
            C = g.create_tensor("C", [m, n], intermediate=False)

            with cg.capture(g):
                einsums.einsum("ik;kj->ij", C, A, B)

            g.optimize()
            g.execute()       # replay as often as you like

The whole benefit is in the replay.
Capturing once and executing many times is what an iterative method should do, because every optimization the library performs happens on the captured form and nothing is re-dispatched per iteration.

.. _howto-graphs-intermediate:

Say which tensors are results
=============================

A graph-owned tensor is scratch unless you say otherwise.
``DeadNodeElimination`` removes any node whose outputs are all unread graph-owned intermediates, and it cannot see that your code reads one after ``execute()`` returns.
Say which tensors are results and the pass leaves them alone.

The distinction is ``create_`` versus ``declare_``, and it applies to both tensor types:

.. list-table::
    :header-rows: 1
    :widths: 40 25 35

    * - Creator
      - Means
      - Allocation
    * - ``create_tensor``, ``create_zero_tensor``
      - scratch
      - immediate
    * - ``declare_tensor``, ``declare_zero_tensor``
      - result
      - deferred, placed by the ``Materialization`` pass

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto &T = g.create_zero_tensor<double, 2>("T", m, n);    // scratch
            auto &C = g.declare_zero_tensor<double, 2>("C", m, n);   // result

        The runtime-rank creators take the flag explicitly instead, which is useful when the
        answer is decided at run time:

        .. code-block:: cpp

            auto &T = g.create_runtime_tensor<double>("T", {m, n});                         // scratch
            auto &C = g.create_runtime_tensor<double>("C", {m, n}, /*intermediate=*/false); // result

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            T = g.create_tensor("T", [m, n])                      # scratch
            C = g.create_tensor("C", [m, n], intermediate=False)  # result

A third option is to own the result yourself: create the tensor outside the graph and pass it in by pointer.
A tensor the graph does not own is always live, because the graph cannot know what else touches it.

The two failure modes are worth telling apart.
A tensor declared with ``declare_`` but never materialized throws on ``execute()``, naming the tensor and telling you to run the Materialization pass, which ``optimize()`` does.
A result created with ``create_`` is pruned instead, and the graph runs to completion leaving the buffer as it was.
Since a pruned result and a pruned intermediate are the same thing from inside the graph, the pass names what it dropped rather than guessing:

.. code-block:: text

    DeadNodeElimination: removed 1 dead node(s)
    DeadNodeElimination: dropped the only writer of graph-owned 'C', which nothing in the graph reads
    DeadNodeElimination: if you read one of those after execute(), it is a result rather than
    scratch; create it with declare_tensor/declare_runtime_tensor, or with intermediate=false

If a graph is correct without ``optimize()`` and full of zeros with it, look for that line.

Use the pointer-writing forms
=============================

Operations that return a value throw during capture, because a recorded operation has no value to return until the graph runs:

.. code-block:: text

    RuntimeError: reject_if_capturing: ... is not capturable

Use the form that writes through a pointer instead.
``dot(A, B)`` returns; ``dot(&result, A, B)`` records.
The same holds for the decompositions: prefer ``syev_eig`` and ``svd`` in their output-argument spellings inside a capture.

Find out what the optimizer did
===============================

:cpp:func:`~einsums::compute_graph::Graph::explain` is a report of the last ``optimize()``.
In Python it is a property rather than a method.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            g.optimize();
            std::cout << g.explain() << "\n";

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            g.optimize()
            print(g.explain)      # property, not a call

A typical report names the phase each line came from:

.. code-block:: text

    optimize(O2) on 'demo': 2 -> 2 node(s)
      - [diagnostic] ScalingAnalysis: 1 contraction(s), total flops 2*?i*?j*?k
      - [diagnostic] ScalingAnalysis: rate-limiting node(s) 'einsum: C[i,j] = A[i,k] * B[k,j]' (#1)
      - [tuning] MemoryPlanning: peak 0.00 MB of 0.00 MB total

The ``?`` in ``2*?i*?j*?k`` marks a letter no index-space annotation reaches.
Costs stay symbolic until you say what the axes are, which is what :doc:`/user/optimizer` means by annotating index spaces.
Annotating them is also what lets several passes fire at all.

Find out why a pass did nothing
===============================

Raise the verbosity on the pass manager.
From level two up, each pass narrates whether it changed anything and why it declined:

.. code-block:: python

    pm = cg.default_pass_manager()
    pm.set_verbosity(2)
    g.apply(pm)

.. code-block:: text

    [PassManager] Reorder: no change (2 -> 2 nodes, 0.00 ms)
    [ScalingAnalysis] costed 1 contraction(s), total flops 2*?i*?j*?k
    [PassManager] StreamContractionFusion: no change (2 -> 2 nodes, 0.00 ms)
    [MemoryPlanning] peak host memory 1280 bytes (of 1280 total allocated)

Most passes decline for a stated reason rather than silently.
A pass that wants an annotation you have not supplied says so, which is usually faster than reading the pass source.

Note that ``PassManager.explain`` is a method while ``Graph.explain`` is a property.

When an optimized graph gives the wrong answer
==============================================

Check the pruning report first, per :ref:`howto-graphs-intermediate`, because that is the common cause and it produces exact zeros rather than a wrong number.

If the answer is wrong rather than zero, bisect the pipeline.
``cg.BisectDriver`` automates switching passes off one at a time and comparing against an unoptimized replay, and :option:`--einsums:pass:disable` takes a comma-separated list for doing it by hand.
:doc:`/user/optimizer` documents both.

Be aware that a search pass cut off by its time budget returns a different but still valid graph, so two machines running one script can emit two different programs.
A test that pins a particular tree should remove the allowance with ``set_optimizer_budget(0)`` and assert that nothing was cut off, rather than depending on the clock.

Next
====

- :doc:`/user/optimizer` for the passes, the phases, and the budget.
- :doc:`/user/tutorial_best_practices` for structuring a capture so the passes have something to work with.
- :ref:`howto-ccsd` for a whole method captured and replayed.
