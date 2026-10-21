..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-ccsd:

********************************************
Tutorial: From NumPy to an Optimized CCSD
********************************************

This tutorial takes one program, closed-shell CCSD with density-fitted integrals, from NumPy to a captured graph that the optimizer streams through a memory budget, keeps in files, and runs as a triangle.
Each stage is a complete program in the ComputeGraph examples directory.
The test suite runs all of them, and each one reproduces the first stage's correlation energy to 1e-10.
The code on this page is included from those files, so what you read is what runs.

.. list-table::
   :header-rows: 1
   :widths: 8 30 62

   * - Stage
     - Program
     - What changes
   * - 1
     - ``ccsd_stage1_numpy.py``
     - The reference, in NumPy.
   * - 2
     - ``ccsd_stage2_eager.py``
     - The same equations with Einsums, run as they are reached.
   * - 3
     - ``ccsd_stage3_graph.py``
     - One iteration captured once and replayed.
   * - 4
     - ``ccsd_stage4_optimizer.py``
     - The intermediates declared as scratch and the axes given their spaces.
   * - 5
     - ``ccsd_stage5_memory.py``
     - A memory budget smaller than the program.
   * - 6
     - ``ccsd_stage6_disk.py``
     - The integrals and the amplitudes in files.
   * - 7
     - ``ccsd_stage7_symmetry.py``
     - The amplitudes' pair symmetry declared.
   * - 8
     - ``ccsd_stage8_disk_symmetry.py``
     - Files and symmetry together.

Stages 2 through 8 share ``ccsd_equations.py``, which holds the equations, and ``ccsd_problem.py``, which loads the molecule.
**The equations do not change after stage 2.**
Everything that makes the program faster or smaller afterwards is a statement about the program, made around the equations, never a rewrite of them.

Run any stage from the examples directory:

.. code-block:: bash

    PYTHONPATH=<einsums-build>/lib python ccsd_stage4_optimizer.py

.. contents:: On this page
    :local:
    :depth: 1

The problem
===========

The molecule is water in cc-pVTZ: 5 occupied orbitals, 53 virtual, and 141 auxiliary functions for the density fitting.
It comes from the fixture the DLPNO example carries, so nothing here needs an integral engine or psi4.
``ccsd_problem.py`` diagonalizes the converged Fock matrix for canonical orbitals and transforms the fitted three-index tensor to them, once, in NumPy.
Every two-electron integral is then a product of two of its blocks,

.. math::

    \langle pq | rs \rangle = (pr|qs) = \sum_Q B^Q_{pr} B^Q_{qs}.

Water is small enough that every stage runs in a few seconds, which is what lets the test suite run them all, and large enough that the arithmetic shows in the timings.
The :math:`v^4` block of integrals is 63 MB here, which is what stages 5 and 6 are about: they are written for molecules where it is hundreds of gigabytes, and shown at a size where you can check every number.

Stage 1: the reference in NumPy
===============================

The equations are the spin-adapted closed-shell ones of the psi4numpy RHF-CCSD, Crawford's ``ccenergy`` formulation.
The integral blocks are formed once:

.. literalinclude:: ../examples/ccsd/ccsd_stage1_numpy.py
   :language: python
   :start-after: # [doc:integrals]
   :end-before: # [doc:integrals-end]

The iteration is a transcription of the equations.
The one-particle intermediate :math:`F_{ae}` reads like this:

.. literalinclude:: ../examples/ccsd/ccsd_stage1_numpy.py
   :language: python
   :start-after: # [doc:fae]
   :end-before: # [doc:fae-end]
   :dedent: 4

and the doubles residual like this:

.. literalinclude:: ../examples/ccsd/ccsd_stage1_numpy.py
   :language: python
   :start-after: # [doc:r2]
   :end-before: # [doc:r2-end]
   :dedent: 4

The particle ladder is the one term that would read the :math:`v^4` block of integrals, and it is the term density fitting exists for.
Here the block is formed from its factors at every iteration rather than stored, which is how a DF-CCSD code runs once :math:`v^4` does not fit.
NumPy will not split the three-operand contraction by itself: its path search caps an intermediate at the size of the largest input or output, refuses both ways of splitting this one, and falls back to a single seven-index loop.
Spelling the two steps out is the NumPy programmer's job.

The program checks itself before anything else depends on it.
The MP2 energy of the first amplitudes agrees with psi4's DF-MP2 for the same molecule to every printed digit, and the converged CCSD energy agrees with psi4's DF-CCSD to 1.3e-7.
That last difference is understood: the fixture's SCF was density fitted in a different auxiliary basis than its correlation treatment, and with both fitted in cc-pVTZ-RI the same equations, on psi4's own orbitals and integrals, agree with psi4 to 1e-11.

.. code-block:: text

    MP2  correlation energy = -0.2752048956   psi4 DF-MP2 = -0.2752048956
    CCSD correlation energy = -0.2811389391   psi4 DF-CCSD = -0.2811388115   (27 iterations)

Stage 2: the same equations with Einsums
========================================

``ccsd_equations.py`` is stage 1 transcribed into Einsums calls.
Every call in it runs immediately outside a capture and records a node inside one, so it is written once and used by every stage that follows.

Three things change in the transcription, and each is what a NumPy program has to give up to be captured.

**An operation writes into a tensor you supply.**
``einsums.einsum`` takes its output, with a prefactor on the output and one on the product.
Two helpers name the only two cases the equations need:

.. literalinclude:: ../examples/ccsd/ccsd_equations.py
   :language: python
   :start-after: # [doc:helpers]
   :end-before: # [doc:helpers-end]

The first term of an intermediate assigns it and the rest add to it.
That is a habit worth keeping even eagerly: a graph replays the same calls on the same tensors, so a sum that began from a tensor zeroed once, when it was made, would start the second replay from the first replay's value.

**Every tensor comes from a function the caller supplies.**
``iterate()`` asks ``new(name, shape)`` for each intermediate rather than allocating one itself, which is what lets the later stages decide what an intermediate is without touching the equations.
Here is :math:`F_{ae}` again:

.. literalinclude:: ../examples/ccsd/ccsd_equations.py
   :language: python
   :start-after: # [doc:fae]
   :end-before: # [doc:fae-end]
   :dedent: 4

**The amplitudes are updated in place.**
Rebinding a name, as stage 1 does with ``t1, t2 = iterate(t1, t2)``, changes nothing that a replay reads.
The update divides the residuals by the denominators and adds the step into ``t1`` and ``t2`` themselves:

.. literalinclude:: ../examples/ccsd/ccsd_equations.py
   :language: python
   :start-after: # [doc:update]
   :end-before: # [doc:update-end]
   :dedent: 4

Two things got simpler.
The closed-shell residual symmetrizes most of its terms under the exchange :math:`(i,a) \leftrightarrow (j,b)`, and symmetrizing is linear, so the terms are collected in one tensor ``X`` and symmetrized once, with one permute.
And the ladder is written the way the equation reads, as one contraction over three operands:

.. literalinclude:: ../examples/ccsd/ccsd_equations.py
   :language: python
   :start-after: # [doc:r2]
   :end-before: # [doc:r2-end]
   :dedent: 4

Einsums plans an einsum over three or more operands itself, choosing the order of the pairwise contractions, and limits a partial product by the memory budget, :option:`--einsums:max-memory`, rather than by the size of the inputs.
Eagerly the plan is chosen by the extents in front of it; captured, by the index spaces the operands are annotated with, which is stage 4's subject.

The driver is the loop, with ``new`` making ordinary tensors:

.. literalinclude:: ../examples/ccsd/ccsd_stage2_eager.py
   :language: python
   :start-after: # [doc:run]
   :end-before: # [doc:run-end]

``converge`` is a few lines in ``ccsd_equations.py`` that call the step until the energy settles and time it.
The energy is the same, in the same 27 iterations.
This stage is the reference semantics for everything after it: eager execution is what a captured graph has to reproduce.

Stage 3: capture once, replay
=============================

A graph records the calls instead of running them.
Wrap the same ``iterate()`` and ``energy()`` in a capture, optimize what was recorded, and replay it once per iteration:

.. literalinclude:: ../examples/ccsd/ccsd_stage3_graph.py
   :language: python
   :start-after: # [doc:capture]
   :end-before: # [doc:capture-end]

That is the whole change.
``t1``, ``t2`` and ``e`` are the caller's tensors, made before the capture, so the graph reads and writes them where they are and the loop reads the energy from ``e`` after each replay.
The loop itself stays in Python; ``graph.execute`` is just the step it calls.

``optimize()`` is not optional here, and it is worth knowing why.
The planner's partial products for the ladder are tensors the graph declares and allocates later, and the default pipeline is what allocates them.
A graph is meant to be optimized before it runs.

Inside a capture, ``einsums.zeros`` makes a tensor the graph owns as a **result**: the caller holds it and may read it after ``execute()``, so the graph has to keep all of it.
Every tensor ``iterate()`` asks ``new`` for in this stage is such a result.
That is correct, and it ties the optimizer's hands, which is the next stage's subject.

Stage 4: tell the optimizer what the program is
===============================================

The optimizer can only rewrite what the program has told it it may.
Stage 4 adds two declarations and nothing else.

**Intermediates are scratch.**
``graph.scratch`` declares a tensor that exists only during a replay:

.. literalinclude:: ../examples/ccsd/ccsd_stage4_optimizer.py
   :language: python
   :start-after: # [doc:capture]
   :end-before: # [doc:capture-end]

A scratch tensor may be freed after its last use, share its memory with another, or be held a block at a time.
In exchange its values mean nothing outside a replay: reading one from Python after ``execute()``, or passing it to an eager operation, is refused with an error naming the tensor and the graph.
``e``, the one value the loop reads, is the caller's tensor and so is unaffected.

**Axes have spaces.**
``annotate`` says what each input's axes range over, from three registered index spaces:

.. literalinclude:: ../examples/ccsd/ccsd_equations.py
   :language: python
   :start-after: # [doc:spaces]
   :end-before: # [doc:spaces-end]

A space's extent is not this molecule's size.
It describes the family of molecules the program is for, here a few hundred basis functions, so that a pass weighing two forms of a contraction weighs them where the program will run rather than at water.
Only the inputs are annotated; the passes carry the spaces through every intermediate.

``graph.explain`` reports what the optimizer did:

.. code-block:: text

    optimize(O2) on 'ccsd': 74 -> 132 node(s)
      - [structural-algebraic] LinearCombinationContractionFolding: folded 9 group(s), replacing 18 contraction(s) with 9 fused contraction(s) plus 9 operand-combination node(s)
      - [structural-algebraic] LayoutAssignment: re-laid out 1 intermediate(s) and folded away 0 permute(s), removing 2 operand copy(ies) modelled at 31.811 us per replay
      - [tuning] Materialization: allocated 20 deferred tensor(s), initialized 0 (0 filled whole from a backing)
      - [analysis] SymmetryPropagation: inferred symmetry on 1 tensor(s)
      - [diagnostic] ScalingAnalysis: 47 contraction(s), total flops 4*o*v^2 + 2*o*v^3 + 4*o^2*v + 22*o^2*v^2 + 12*o^2*v^3 + 2*o^2*v^4 + 4*o^3*v + 16*o^3*v^2 + 18*o^3*v^3 + 4*o^4*v + 4*o^4*v^2 + 2*v^4*Q
      - [diagnostic] ScalingAnalysis: rate-limiting node(s) 'einsum: C[i,j,a,b] = A[i,j,e,f] * B[a,e,b,f]' (#63) at 2*o^2*v^4
      - [diagnostic] ScalingAnalysis: intermediate footprint at most 2*o^2 + o^4 + 2*o*v + 9*o^2*v^2 + o^3*v + 2*v^2 + v^4 elements over 18 intermediate(s), a sum of sizes rather than a liveness-aware high-water mark
      - [tuning] InplaceOptimization: merged 0 buffer(s) into dying inputs (13 candidate(s))
      - [tuning] FreeInsertion: 20 intermediate(s) freed after their last consumer
      - [tuning] MemoryPlanning: peak 66.44 MB of 92.87 MB total
      - [tuning]     arena: 62.88 MB hosting 20 intermediate(s) (75.52 MB of buffers)

Read it from the top.

* The spin adaptation writes many terms as ``2 A - B``, where ``A`` and ``B`` are the same integral with two indices exchanged. ``LinearCombinationContractionFolding`` found nine such pairs and runs each as one contraction against the combined integral.
* The cost of an iteration is a polynomial in the spaces' symbols, and the rate-limiting node is the ladder at :math:`2 o^2 v^4`. The term :math:`2 v^4 Q` is the planner's choice for the ladder: with the spaces annotated it forms :math:`\langle ab|ef\rangle` from its factors and then contracts it, the order with the fewest operations for this family.
* The footprint has a :math:`v^4` term, the price of that choice: 63 MB of the 66 MB peak. It is the subject of stage 5.
* The memory passes free every intermediate after its last use and pack 20 of them into one arena.

The same report is available as records a program can read, ``graph.decisions``, each with a stable code; :ref:`tutorial-optimizer` describes them.

Stage 5: run in less memory than the program names
==================================================

The ladder's :math:`v^4` intermediate is 63 MB at water and hundreds of gigabytes for a molecule CCSD is run on.
Tell the program how much memory it has:

.. literalinclude:: ../examples/ccsd/ccsd_stage5_memory.py
   :language: python
   :start-after: # [doc:budget]
   :end-before: # [doc:budget-end]

The runtime reads the budget when it starts, so it is set before ``einsums`` is imported; :option:`--einsums:max-memory` on the command line or ``EINSUMS_MAX_MEMORY`` in the environment do the same.
The capture is stage 4's, unchanged, and the report says what changed:

.. code-block:: text

    slicing Bvv*Bvv[0] x Bvv*Bvv[2] into 2809 slice(s), blocks of 8 x 18 in 4 loop(s) of 21 iteration(s) in all, seed 63123848 bytes -> largest intermediate 3235968 bytes

``AxisTiling`` turned the ladder into a loop over blocks of its two virtual indices :math:`a` and :math:`b`, so :math:`\langle ab|ef\rangle` is built and used 3.2 MB at a time instead of 63 MB.
Nothing in the equations moved, and the energy is the same to every digit.
A program that fits its budget is left alone, which is why stage 4, with the machine's whole memory as its budget, was not tiled.

**A budget has to leave room for what cannot be sliced.**
The program's other tensors, the integral blocks and the four-index intermediates, take about 36 MB here.
Under a 32 MB budget they alone do not fit, and the pass slices nothing at all; the report says so for every candidate:

.. code-block:: text

    the program's other tensors alone exceed the memory budget, so no slice of the region fits beside them

A budget smaller than the inputs is the next stage's problem.

Stage 6: integrals and amplitudes in files
==========================================

Tiling bounds what the program builds.
The inputs are a different problem: at a large enough molecule the integral blocks themselves do not fit, and amplitudes that took hours to converge are worth keeping where a crash cannot take them.
A **backing** says where a tensor lives instead of memory:

.. literalinclude:: ../examples/ccsd/ccsd_stage6_disk.py
   :language: python
   :start-after: # [doc:files]
   :end-before: # [doc:files-end]
   :dedent: 4

The integral blocks are written once and declared on the graph with read backings: each replay reads them from the file, and the caller never holds them.
``t1`` and ``t2`` have **update** backings: each replay starts from what the file holds and leaves the file holding the new amplitudes.
The new values are written beside the old ones and published in one commit at the end of the replay, so a crash leaves the previous iteration's amplitudes, and a later run can start from them.
``iterate()`` and ``energy()`` are handed these tensors and do not know the difference.

The budget is 16 MB, less than the program's inputs, which is now possible because the inputs no longer have to be in memory.
The report says how each one is moved:

.. code-block:: text

    slicing Bvv*Bvv[1] x Bvv*Bvv[3] into 2809 slice(s), blocks of 2 x 4 in 4 loop(s) of 378 iteration(s) in all, seed 63123848 bytes -> largest intermediate 561800 bytes
    SlabStreaming: 'einsum: C[a,e] = A[m,f] * B[m,a,f,e]' reads 'ovvv' in blocks of 23 of 53 along its last axis (a letter the output keeps, so each block writes its own slice); 5955080 bytes read per execution, about 0.2 flop per byte read
    ...
    SlabStreaming: 'einsum: C[a,e,b,f] = A[Q,a,e] * B[Q,b,f]' reads 'Bvv' a box at a time, the box its view names: 119568 bytes in 1 run(s) each time the view is made, in place of the whole tensor
    ...

The ladder is sliced again, in smaller blocks; the 6 MB ``ovvv`` block is read a slab at a time by every contraction that uses it; and inside the ladder's loop, ``Bvv`` is read from the file one box at a time, the box each block needs.
``graph.last_run()`` says what a replay actually moved:

.. literalinclude:: ../examples/ccsd/ccsd_stage6_disk.py
   :language: python
   :start-after: # [doc:measured]
   :end-before: # [doc:measured-end]
   :dedent: 4

.. code-block:: text

    the last replay held at most 11594104 bytes
      Bvv: 129910632 bytes read in 756 read(s), 0 written
      ...
      ovvv: 35730480 bytes read in 18 read(s), 0 written
      t1: 2120 bytes read in 1 read(s), 2120 written
      t2: 561800 bytes read in 1 read(s), 561800 written

The replay held 11.6 MB at most, against 66 MB in stage 4.
That was paid for in reading: ``Bvv`` is 3.2 MB and was read 40 times over, because every block of the ladder reads the box it needs.
The budget steers what is streamed, and is not a hard limit on what a replay holds.
And the commit is not free either: each replay waits for the drive three times, once for each amplitude and once for the commit that publishes both.

Stage 7: declare the symmetry the amplitudes have
=================================================

The closed-shell amplitude is unchanged when the two electrons trade places, :math:`t_{ij}^{ab} = t_{ji}^{ba}`.
Stage 7 is stage 4 with that said:

.. literalinclude:: ../examples/ccsd/ccsd_stage7_symmetry.py
   :language: python
   :start-after: # [doc:symmetry]
   :end-before: # [doc:symmetry-end]

From the declaration the optimizer proves that the ladder's contribution has the same symmetry, and computes only the blocks of it on and above the diagonal, writing each one's mirror from it:

.. code-block:: text

    runs 36 of 64 tile(s) of 7 x 7, writing each off-diagonal tile's mirror with sign +1
    slicing Bvv*Bvv[0] x Bvv*Bvv[2] into 2809 slice(s), a triangle of blocks of 7 x 7 in 4 loop(s) of 36 iteration(s) in all, seed 63123848 bytes -> largest intermediate 1101128 bytes

A declaration is a promise, and the tiled loop checks it at every replay before relying on it.
The energy is unchanged.

Stage 8: files and symmetry together
====================================

The declarations compose.
Stage 8 is stage 6 with stage 7's line, made on the ``t2`` the graph declared with its file backing:

.. literalinclude:: ../examples/ccsd/ccsd_stage8_disk_symmetry.py
   :language: python
   :start-after: # [doc:symmetry]
   :end-before: # [doc:symmetry-end]
   :dedent: 4

A symmetry declared on a tensor the graph declared is read when the graph is optimized, so it may be set after the declaration, as here, or after the capture.
The ladder runs as a triangle again, now with its factor ``Bvv`` read from the file a box at a time inside the loop:

.. code-block:: text

    runs 66 of 121 tile(s) of 5 x 5, writing each off-diagonal tile's mirror with sign +1
    slicing Bvv*Bvv[0] x Bvv*Bvv[2] into 2809 slice(s), a triangle of blocks of 5 x 5 in 4 loop(s) of 66 iteration(s) in all, seed 63123848 bytes -> largest intermediate 561800 bytes

and what the replay read from the file falls with it:

.. code-block:: text

    the last replay held at most 11857352 bytes
      Bvv: 38022624 bytes read in 132 read(s), 0 written
      ...

In stage 6, ``Bvv`` was read in 756 pieces, 130 MB a replay; here it is 132 pieces and 38 MB, because the triangle's loop has fewer, larger blocks and skips half of them.
The blocks are smaller than stage 7's because the 16 MB budget has to hold the mirror's scratch tile beside everything else.

What each stage costs
=====================

Each stage's time per iteration, measured on 2026-10-21 on an Apple M4 laptop on mains power, with the default thread count and no hardware profile, so the optimizer priced its choices from the built-in table.
The programs ran interleaved, five times each, and the table gives the median; the iterations are timed, not the setup or the optimizer.

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - Stage
     - ms per iteration
   * - 1. NumPy
     - 36.8
   * - 2. Einsums, eager
     - 34.8
   * - 3. Captured and replayed
     - 23.8
   * - 4. Scratch and spaces declared
     - 17.3
   * - 5. A 64 MB memory budget
     - 16.4
   * - 6. Integrals and amplitudes in files, a 16 MB budget
     - 74.1
   * - 7. Pair symmetry declared
     - 13.4
   * - 8. Files and symmetry, a 16 MB budget
     - 50.2

Read it as six observations rather than a ranking.

**Eager Einsums is not faster than good NumPy.**
The two run the same algorithm on the same BLAS, and come out even.
Stage 2 exists to fix the semantics, not the speed.

**The speed comes from capturing, and from what you declare.**
Replaying the captured iteration is 1.5 times faster than NumPy, and declaring scratch and spaces takes that to 2.1.
None of it touched the equations.

**Holding less need not cost time.**
Stage 5 holds a 3.2 MB block of :math:`\langle ab|ef\rangle` where stage 4 holds all 63 MB, and runs no slower.

**Files are bought with time.**
Stage 6 holds a sixth of what stage 4 holds and takes more than four times as long.
About half of the difference is the ladder's loop, which reads each block's box of ``Bvv`` from the file, 40 times its size in all; most of the rest is the three waits for the drive that make each commit safe.
At a molecule whose integrals do not fit in memory, there is no faster version to compare it with.

**The triangle pays at water too.**
It halves the ladder's arithmetic, and stage 7 is 23% faster than stage 4.
How large the blocks should be is the optimizer's choice, and it is not the arithmetic that decides it.
A block of the ladder's output is computed by a contraction of its own, and a small block pays for more contractions, more packing of their operands, and more of the mirror's copies; a large block computes more of the diagonal twice.
The optimizer prices each candidate block as the contraction engine would run its tiles, from the hardware profile when ``einsums tune`` has measured one and from a built-in table otherwise, and here it picks blocks of 7 x 7.
On benzene in cc-pVDZ, where the ladder is most of an iteration, the same declaration took an iteration to 0.69 to 0.72 of the time without it.

**With files, the triangle pays more.**
Stage 8 is a third faster than stage 6, because what the ladder's loop costs with files is reading ``Bvv``, and a triangle of fewer, larger blocks reads less of it.

Where to go next
================

* **Approximations.** Every stage here computes the same energy. Density fitting the integrals further, a Laplace transform of the denominators, tensor hypercontraction and truncated virtual spaces change the answer by a bounded amount, and each is something a program asks for by name; :ref:`tutorial-optimizer` describes them under *Trading accuracy for scaling*.
* **The loop inside the graph.** ``graph.add_loop`` moves the convergence loop into the graph, and ``cg.diis`` wraps its predicate to accelerate it; ``ccsd_rhf_graph_numpy_style.py`` in the psi4 bridge examples does both.
* :ref:`tutorial-optimizer` for every declaration on this page in full: spaces, backings, budgets and the report.
* :ref:`tutorial-compute-graph` for capture, replay and control flow.
* :ref:`howto-ccsd` for the spin-orbital CCSD, written eagerly with its antisymmetrizers in the spec.
