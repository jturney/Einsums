..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-optimizer:

*************************************
Tutorial: The Algebraic Optimizer
*************************************

Most optimizers rewrite the schedule of a computation whose mathematics is
fixed. This one can also rewrite the mathematics: it can substitute a
factorization for an integral, replace an energy denominator by a quadrature,
truncate a space, re-bracket a product, and turn a program that declares
four-index tensors into a loop that never holds one.

None of that happens by guesswork. Every rewrite is either exact, or opt-in and
recorded, and each one is taken because the caller said something the graph
could not otherwise know: what an axis ranges over, what a tensor is, or what
factored form is available for it. This page is about those statements and what
each of them buys.

Everything below is written for the Python surface, ``import einsums.graph as
cg``. The C++ spelling is the same vocabulary with the same names;
``TagsAndProviders.cpp`` in the ComputeGraph examples is the C++ tour of the
same ground.

The two runnable tours are ``optimizer_tour.py``, which writes one
density-fitted MP2 energy and drives it through every arm below, and
``annotate_and_rebind.py``, which is the family story on its own. Both live in
the ComputeGraph examples directory and both run offline from the fixture the
DLPNO example carries, so nothing here needs an integral engine.

Capture once, optimize, replay
==============================

The optimizer only ever sees a captured graph. Eager execution gets none of it,
and that is deliberate: eager is the reference semantics this whole layer is
validated against, so a numpy-style script stays eager and gets exactly what it
wrote.

.. code-block:: python

    import einsums
    import einsums.graph as cg

    g = cg.Graph("mp2")
    with cg.capture(g):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        ...

    g.apply(cg.default_pass_manager())
    g.execute()

``default_pass_manager()`` is what a caller gets by asking for nothing. It holds
the analysis passes, the schedule and memory passes, and the structural rewrites
that have been proven exact. **No lossy pass is ever in it.** A factorization, a
quadrature or a basis truncation is something a program asks for by name, and
until it does, the answer the graph computes is the answer the program wrote.

Saying what the axes are
========================

A tensor's dimensions say how big *this* problem is. An index space says what an
axis ranges over and how that set grows, which is a statement about the whole
family of problems the captured program stands for.

Spaces are registered once, with the letter each contributes to a cost
polynomial, an advisory extent for the family, a growth class, and the dim
symbol a rebind solves for:

.. code-block:: python

    registry = cg.global_space_registry()
    for name, symbol, extent, dim in (("occ", "o", 300.0, "nocc"),
                                      ("vir", "v", 2700.0, "nvir"),
                                      ("aux", "x", 9000.0, "naux")):
        registry.register_space(cg.index_space(name, symbol, extent,
                                               cg.GrowthClass.linear(), dim))

Only the ratios of those extents reach the cost model. What they say is which
regime the captured geometry stands for, which matters because the geometry in
front of the optimizer is usually a small one.

Tensors are annotated one entry per axis, by space name:

.. code-block:: python

    cg.annotate(B, ("aux", "occ", "vir"), graph=g)
    cg.annotate(K, ("occ", "vir", "occ", "vir"), graph=g)

Only a program's inputs need annotating. ``SpacePropagation`` carries the
annotation through the intermediates, ``CrossSpaceValidation`` reports a
contraction letter that binds two different spaces, and ``ScalingAnalysis``
reports what the program costs as a polynomial in the scale symbols. Those three
are described in the :ref:`compute graph tutorial <tutorial-compute-graph>`.

Why the family annotation decides rewrites
------------------------------------------

Some rewrites are cheaper at every size and some are cheaper only past a
crossover. The second kind cannot be judged from the extents a capture happens
to hold, and this is not a corner case: it is where the interesting rewrites
live.

The opposite-spin MP2 energy is the plain example. Written as it stands it
contracts the orbital indices into a four-index integral; with the denominator
decoupled it contracts them away first and holds an auxiliary-by-auxiliary
matrix per quadrature point instead. That trades ``o^2 v^2 Q`` for
``o v Q^2 t``, and neither dominates the other, so the answer is a fact about
extents. At water in cc-pVDZ, 84 auxiliary functions against 95
occupied-virtual pairs, the captured form wins and the optimizer says so.
Annotated for a system of a few thousand basis functions, which is where
spin-opposite-scaled MP2 is used, the pairless form wins and the optimizer
reaches it, with no four-index tensor left in the emitted graph.

Both arms are in ``annotate_and_rebind.py``, and the pass is right both times.
What the annotation changed is which problem it was right about.

The same holds for the two rewrites that are family wins and constant-factor
losses at a small molecule:

* The **fourth-order opposite-spin form** above. Annotate the three-index
  tensor with its spaces and the four-index intermediates with theirs; the
  quadrature's own index is a registered space the transform annotates for you,
  but only where the denominator itself is annotated, all axes or none.
* The **tensor hypercontraction ladder**, where fitting the integral and the
  amplitude on one grid takes the doubles ladder from degree six to degree five
  while costing more arithmetic at water in cc-pVDZ, because a grid of a few
  hundred points against a basis of a couple of dozen is what a small molecule's
  grid looks like. What a caller must supply there is the grid space, through
  ``ThcFactorization.register_grid_space``, and the annotation on the tensors it
  fits.

Where a rewrite pays only as a family, the optimizer declines it on an
unannotated program, and the skip tally says the decomposed form is not
symbolically cheaper. That is the correct answer to the question it was asked.

Dim symbols, and a graph that outlives its geometry
---------------------------------------------------

A space carries a dim symbol, and an axis annotated with one is an axis a later
bind may resize. Two things follow.

The first is that the numeric veto on a factorization abstains over such an
axis. A rewrite that is asymptotically better and slower at the size in hand is
normally refused, because a graph rewritten into something slower than what it
replaced is a real regression traded for a promise. Over an axis the graph
itself declares resizable, the capture-time number is a placeholder and the veto
says so rather than acting on it.

The second is reuse. Declare the intermediates over the spaces rather than over
numbers and the whole program moves:

.. code-block:: python

    g.annotate_dims(B, ["naux", "nocc", "nvir"])
    occ, vir = registry.find("occ"), registry.find("vir")
    axes = [cg.SpaceDim(occ), cg.SpaceDim(vir), cg.SpaceDim(occ), cg.SpaceDim(vir)]
    K = g.declare_zero_tensor_over("K", axes, True)

``declare_zero_tensor_over`` sizes the axes, annotates their spaces and writes
their dim symbols in one statement, which is the spelling that cannot drift.
``cg.fixed(8)`` is the literal axis for a dimension that means nothing
chemically. ``annotate_dims`` is the one-axis-at-a-time form for a tensor the
caller already allocated.

``annotate_and_rebind.py`` captures the energy at water in cc-pVDZ, saves it,
binds it to cc-pVTZ, and replays it there against a pair-driven oracle. The
occupied count is unchanged, the virtual count goes from 19 to 53 and the
auxiliary count from 84 to 141, and every intermediate follows the symbols.

.. note::

   The exact algebra is what crosses a size boundary today. A graph carrying a
   quadrature still binds at the geometry whose orbital energies it was fitted
   from, and the interface is no longer the reason: annotate the energy vectors
   and their slots carry the dim symbols, whichever of the handles behind a slot
   the annotation was written on. What is still literal is the quadrature's own
   working set, the exponential matrices and the scaled integrals the transform
   declares, so a bind at another size meets them at the extents they were
   fitted on and is refused there.

Saying what a tensor is
=======================

Spaces give cost. They cannot say that a tensor *is* an electron repulsion
integral, or a Kronecker delta, or an energy denominator, and a pass that wants
to recognize something has to ask that question instead. A provenance tag is the
answer: an open vocabulary name plus free-form key/value attributes.

.. code-block:: python

    cg.annotate(eri, tag="eri", graph=g)
    cg.annotate(delta, tag="identity", graph=g)
    cg.annotate(eri, spaces=("occ", "vir", "occ", "vir"),
                tag={"name": "eri", "basis": "cc-pvdz"}, graph=g)

A tag is a **declaration** and is never inferred from a tensor's contents. A
pass could notice that a tensor happens to hold an identity today, and that
would be wrong in a way easy to miss: a structural rewrite is what a saved graph
keeps, and a later bind may put a different tensor behind the same name.

A tag crosses an operation only where the output holds the same elements as the
input, which today is the axis reorderings and nothing else. A view does **not**
inherit one, and the reason is the tag this rule was written for: a slice of a
Kronecker delta off the diagonal is an ordinary rectangular matrix of ones and
zeros, and eliminating a contraction against it would produce a wrong number
rather than a slower one.

Two tags in the vocabulary have a shape the pass that reads them requires.

``eri``
    Claimed by the factorization providers below. Nothing in the mechanism knows
    what an integral is: density fitting is a provider registered on this name,
    and the chemistry lives in the registration.

``laplace_denominator``
    Read by :ref:`the Laplace transform <optimizer-lossy>`. Its attributes are
    the substance rather than decoration: one entry per axis of the tagged
    tensor, naming the orbital-energy vector that supplies that axis, and one
    giving the sign with which that energy enters. The tagged tensor is the
    **reciprocal**, since that is the object a program forms and multiplies.
    ``cg.LaplaceTransform.denominator_tag(names, signs)`` builds it.

Registering a provider
======================

A factorization provider registers on a tag and states a factored form, the
setup computation that produces the factors, and its accuracy statement. One
generic pass then does the work for all of them: it queries the providers for
each tagged tensor, substitutes the factors, hands the whole leaf set to the
same subset search that brackets ordinary products, and accepts the result only
if it is cheaper.

.. code-block:: python

    registry = cg.FactorizationRegistry()
    registry.add(cg.MetricFitFactorization("eri", three_index, metric, 1e-6))

    factorization = cg.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(cg.ProvenancePropagation())   # carries a tag to the operand handles
    pm.add(factorization)
    g.apply(pm)

The substitution alone would make the arithmetic **worse**. Replacing one tensor
by two factors and contracting them in the captured order is more work, not
less; the whole value is in the re-association afterwards, which is why the pass
costs the result twice before emitting anything: symbolically, which is the
claim about the family, and at the extents the graph holds, which is the veto
described above.

What the library ships
----------------------

``MetricFitFactorization``
    Density fitting: a rank-four tensor as ``B[Q,m,n] B[Q,p,q]`` with
    ``B = J^{-1/2} R`` from a caller-supplied three-index tensor and a symmetric
    positive-definite metric. The bound it records is **asserted**, and it has to
    be: the error of a fit is its difference from the exact four-index tensor,
    and a caller holding that tensor had no reason to fit it. What is measured
    instead, per bind, is how many auxiliary directions the guarded inverse
    square root threw away, which is a specific and common way for accuracy to
    degrade quietly.

``ThcFactorization``
    Tensor hypercontraction: a five-factor chain over a grid, fitted by least
    squares from the three-index tensor. The grid is a registered space with a
    symbolic extent, declared by the caller through ``register_grid_space``,
    because a grid is chosen per problem. It takes one collocation matrix per
    axis, so it can fit an occupied-virtual block rather than the whole basis,
    and on water in cc-pVDZ that block fit is an order of magnitude more accurate
    than the whole-basis fit of the same integrals. Its tolerance is
    :option:`--einsums:graph:thc-epsilon`, asserted in the record, with the
    least-squares residual against the three-index tensor measured per bind.
    A residual far above the asserted tolerance is the signal that the grid was
    too coarse.

``NaturalAuxiliaryFactorization``
    The first provider whose factors **shorten** an index rather than splitting a
    tensor. The three-index tensor, seen as a matrix against the pair index, has
    a singular value decomposition, and the directions below
    :option:`--einsums:graph:naf-threshold` are dropped. Both occurrences of the
    tensor in a contraction are substituted, and the search then contracts the
    two truncated transformations together first, which is where the saving is.
    Nobody had to teach it that the small matrix is nearly an identity, only that
    it is the cheapest thing to build out of two of them. The record is
    **measured**: the norm of the dropped singular values. At water in cc-pVDZ a
    threshold of ``1e-1`` keeps 39 of 84 auxiliary functions at a recorded bound
    of ``1.06e-1``, against an MP2 energy error of ``5.8e-4``. That is a bound on
    the tensor, so it is loose against an energy, and loose in the safe
    direction; ``test_naf_python.py`` sweeps four thresholds and asserts it.

``BasisTruncation``
    Frozen natural orbitals, and the one member of this list that is **not** a
    provider: nothing is factored, a space is replaced by a contained subspace.
    The MP2 virtual-virtual density is diagonalized, the orbitals above
    :option:`--einsums:graph:fno-occupation` are kept, every interface tensor
    over the virtual space is projected into the smaller one, and the truncated
    Fock block is semicanonicalized so the denominators are diagonal again. The
    pass takes the amplitudes, the virtual-virtual Fock block and the occupied
    orbital energies rather than building them. At water in cc-pVDZ a cutoff of
    ``1e-3`` keeps 9 of 19 virtual orbitals.

    Its record is the third kind of thing a record can be: not a claimed
    tolerance and not a dropped norm, but the MP2 correlation energy the
    truncation removed, measured, which composes on the absolute side. A program
    that wants the corrected number reads it off the record and adds it back,
    which is what makes the record worth composing rather than an extra output.
    ``optimizer_tour.py`` prints the corrected energy landing back on the
    untruncated one.

Two providers claiming one tag are alternatives ranked by profitability rather
than a composition. Two passes with a claim on one tensor do not compose at all,
and the tour shows what that looks like: after the truncation projects the
three-index tensor, nothing but the setup reads it, so an auxiliary truncation
looking for a contraction to re-associate around finds none and says so.

.. _optimizer-lossy:

Trading accuracy for scaling
============================

``LaplaceTransform`` is the pass with no provider behind it. It recognizes a
tagged energy denominator, replaces it by a quadrature, and pushes the resulting
per-axis exponentials onto the factors of whatever the denominator multiplies,
which is what decouples the four indices.

.. code-block:: python

    g.annotate_tag(D, cg.LaplaceTransform.denominator_tag(
        ["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-"))

    transform = cg.LaplaceTransform()
    transform.set_epsilon(1e-5)
    transform.add_energy("eps_occ", eps_occ)
    transform.add_energy("eps_vir", eps_vir)
    pm = cg.PassManager()
    pm.add(transform)
    g.apply(pm)

The knob is a target accuracy, :option:`--einsums:graph:laplace-epsilon`, and
not a point count. A tolerance is what composes with the accuracy budget and
what a record can state; a count is a means, and the same count over a wider
spectral range is a different approximation. The count is derived at optimize
time from the tolerance and the range the bound orbital energies span. At water
in cc-pVDZ that is 16 points at ``1e-3`` and 60 at ``1e-8``, and the emitted
node count is the same at both, because the quadrature index becomes an ordinary
contracted letter rather than a loop over terms.

Two things about writing the denominator are worth knowing before the pass
declines your program.

The denominator may be built inside the capture, but only as a **recipe the pass
can read**: an ``outer_sum`` over the tagged energy vectors followed by the
registered ``recip`` element operation. Everything the tag claims is checked
against those nodes, and the chain is then dissolved along with the tensor, so
the four-index denominator is never allocated. An anonymous Python callable in
place of ``recip`` says only that something is applied to every element, and the
pass declines it with that reason.

.. code-block:: python

    with cg.capture(g):
        einsums.linalg.outer_sum(D, [eps_o, eps_v, eps_o, eps_v], [1.0, -1.0, 1.0, -1.0])
        einsums.linalg.element_transform(D, "recip")     # the registered op, not a lambda

The pass takes its own copy of the numerator it rewrites and leaves the
definition standing for whatever else reads it, so **one integral is the correct
program**. A caller who writes the contraction twice gets the same rewrite, the
same point count and the same energy, from a capture one node larger.

Records, and what they are for
------------------------------

Every lossy rewrite leaves a record on the graph, saved with the structure:

.. code-block:: python

    for record in g.approximations():
        print(record.pass_name, record.bound, record.effect, record.origin)

    tolerance = g.approximation_tolerance("E_corr")
    budget = tolerance.absolute + tolerance.relative * abs(reference)

A record carries the tolerance the pass was **asked** for and, separately, the
bound it **states**, which are usually equal and are not the same thing. It
carries the units that bound is in, because a number without units is not a
bound and the three kinds do not convert into one another: an element-wise
bound says nothing about a norm. And it carries whether the number was measured
or asserted, because a bound and a guess are the same double and a reader of a
saved graph months later cannot tell them apart otherwise.

Composition is per unit and is not addition in general. Two absolute bounds add,
by the triangle inequality. Two relative ones compose as ``e1 + e2 + e1*e2``,
because the second rewrite's error is relative to the already perturbed result,
and dropping the product term makes a composed budget quietly optimistic, which
is the one direction an accuracy contract must never fail in. Bounds in
different units do not combine at all and are reported as two numbers.

Two passes that compose on one water program are the basis truncation and the
quadrature, in that order. They act on different quantities, and the one link
between them is the energy vector the denominator recipe names: the truncation
points that recipe at the semicanonical energies it produced, and the quadrature
then verifies the chain it was always going to verify. The other order is
declined, because a quadrature carries exponentials fitted for the space the
denominator ran over and nothing re-derives them from a truncated one.

A graph carries an **accuracy budget** too, which is where a pass refuses rather
than spends:

.. code-block:: python

    g.set_accuracy_budget(cg.ApproximationEffect.NormRelative, 1e-8)

The budget caps one kind of error, so a record in other units is refused rather
than admitted under a cap a caller would reasonably read as covering everything.
It is a property of the run rather than of the graph, so it is never saved: a
loaded graph's caller states their own. The refusal happens before anything is
rewritten, so a graph is never left half rewritten with an unrecorded
approximation in it.

Testing against a lossy graph
-----------------------------

``einsums.testing`` reads the records, so a comparison does not need a tolerance
picked by hand:

.. code-block:: python

    from einsums.testing import assert_close
    assert_close(got, want, graph=g, output="E_corr")

The helper asks the graph for ``approximation_tolerance(output)`` and adds both
sides to the dtype baseline. Added rather than maxed, because rounding and a
deliberate approximation are both present in the result and the bound that holds
is their sum.

The rewrites you will see in the report
=======================================

Three passes in the default pipeline rewrite what a program computes *with*
rather than how it is scheduled, and none of them moves the answer beyond the
bound its tier declares. They are described in full, with what each one reports,
in the :ref:`pass catalog <computegraph_optimization_passes>`; what follows is
what a user meets.

``MultiTermFactorization`` is the search. It takes two decisions as one: how each
product of three or more tensors is bracketed, and which partial products
several of them should share, because fixing either first assumes the other. It
is **off by default**, which is the point rather than caution: every other
structural pass walks the graph once, and a search's runtime is a function of
how many candidates the program offers, which nobody can predict from outside.
Switch it on with :option:`--einsums:graph:structural-search` or, for one
pipeline, ``MultiTermFactorization.set_search_enabled(True)``.

On the coupled-cluster doubles residual it finds the shape a hand-optimized code
reaches by hand: the tau term routed through two different intermediates is one
three-factor product under the flattening, the search shares one ``o^4``
intermediate between both routes, and three contractions replace four with the
``v^4`` tensor never formed.

**You write the integral once.** An author-named intermediate that several
statements read used to pin the algebra around it, so a program wanting one half
of an expression re-bracketed had to write the value twice. The flattener now
inlines a definition into each consumer whose bracketing it improves and keeps
the definition for the rest, with the cost deciding rather than a rule: a
consumer that gains nothing reads the definition instead, and the whole rewrite
is still measured against the captured bracketing before anything is emitted.
:option:`--einsums:graph:factorization-max-readers` bounds how many consumers a
definition may be copied into, four by default, and a definition past it is left
whole with that reason in the tally.

``DeltaElimination`` is the exact one, and it is in the default pipeline.
Contraction against a tagged Kronecker delta is a rename, so it does the rename
and drops the delta; a letter summed over two spaces the registry says share no
element has no term to sum, so the contraction becomes a scaling of the
destination by its own prefactor. Both halves are justified by a **declaration**
rather than by the data, which is why the delta needs its tag and the disjoint
spaces need their relation.

``LayoutAssignment`` treats per-tensor storage order as a global decision. What
it minimizes is not the cost of ``Permute`` nodes, of which a real captured
program has almost none, but the copy a contraction kernel makes inside itself
for an operand it cannot read as a flat matrix, which never appears as a node at
all. It also folds away a permuted copy whose chosen storage order equals its
source's, which is a case a peephole cannot reach when several contractions read
the copy.

Fitting the program into memory
===============================

A full-axis capture declares the tensors the equations mention, and for a
density-fitted method those are exactly the objects density fitting exists to
avoid. ``AxisTiling`` slices free axes of the program and streams every
intermediate that carries them, so the body is the same algebra at one slice.

.. code-block:: python

    tiling = cg.AxisTiling()
    tiling.set_memory_cap(4096)          # bytes
    pm = cg.PassManager()
    pm.add(tiling)
    g.apply(pm)
    g.apply(cg.default_pass_manager())   # the schedule still needs its lifecycles

The cap is :option:`--einsums:graph:tiling-memory-cap`, in bytes, and it is the
first knob that describes the **machine** rather than the program or a
tolerance. It is read three times over: whether to tile at all, since a program
whose largest intermediate already fits is left alone; which axes, since the
pass keeps the candidate set that streams the fewest bytes among those that
bring the footprint under the cap; and how deep, since a chunk of slices is the
grouped form of the same body. Zero is the spelling for a caller who wants the
captured schedule, and the default is deliberately far above anything a captured
program declares, because slicing a program that already fits costs kernel
efficiency for memory nobody was short of. The pass is **not** in the default
pipeline for that reason.

On the full-axis DF-MP2 capture at water in cc-pVDZ, under a cap of 4096 bytes,
it chooses the two occupied axes, 25 slices at depth one, and the largest
intermediate the loop holds falls from 72200 bytes to 2888. Six captured nodes
become two, a ``Scale`` and a ``Loop``, and the energy agrees with the untiled
replay, with a numpy pair loop that never forms a four-index tensor, and with
the fixture's own reference. ``test_axis_tiling_python.py`` asserts every one of
those, and ``optimizer_tour.py`` prints them.

The occupied axes are not chosen because they are the smaller ones. At this
molecule they are the larger. The exchange permutation of the MP2 energy
exchanges the two virtual indices, so a slice of the permuted tensor would need
a slice of its source at a pair the body is not at, and the virtual pair is
rejected structurally, at every extent, on every molecule. The size argument
agrees with that answer at water only by accident.

Two things about the schedule are worth stating plainly.

A tensor written by a node the pass cannot re-emit stays whole and outside the
loop. A denominator built inside the capture is that case, since an
``outer_sum`` records a node the schedule does not slice; the pass streams the
tensors of the region it rewrote, and the tour prints which buffers the tiled
graph actually allocates so the difference is visible. Dissolving a denominator
is what the Laplace transform is for; slicing the rest is what this pass is for.

A tiled graph does not save, and it is not meant to. The file carries the
algebra, a load rebinds it to fresh buffers, and the loop is emitted again on
the other side against whatever cap is in force there. That is the phase rule
read back rather than a limitation met halfway.

Driving the pass manager
========================

Every pass declares a phase, and the phase answers one question: may a saved
graph keep this pass's output, or must the output be re-derived on the machine
that loads it?

.. code-block:: python

    cg.analysis_pass_manager()     # read-only; safe on a graph in any state
    cg.structural_pass_manager()   # machine-independent rewrites; what a file keeps
    cg.resource_pass_manager()     # tiling, placement, distribution; never saved
    cg.tuning_pass_manager()       # batching, memory planning, threads, streams
    cg.default_pass_manager()      # the ones safe by default, in the default order

They are views of the default pipeline rather than a re-planned one, built in
place by ``populate_analysis``, ``populate_structural``, ``populate_resource``,
``populate_tuning`` and ``populate_default``. The two a caller runs over a
loaded graph are the resource and tuning ones, in that order, which is what the
:ref:`round trip <optimizer-round-trip>` below does.

Switching a pass off
--------------------

:option:`--einsums:pass:disable` takes a comma-separated list of pass names, and
``PassManager.disable`` / ``PassManager.enable`` are the programmatic half. An
explicit ``enable`` wins over the option, because the more specific statement
about this pipeline is the one that should win, and a name matching no pass in
the pipeline is reported rather than ignored. Switching passes off one at a time
against an unoptimized replay is what ``cg.BisectDriver`` automates when an
optimized graph produces a wrong number.

The optimizer's own budget
--------------------------

:option:`--einsums:graph:optimizer-budget` is a wall-clock allowance in
milliseconds for each search pass, and ``PassManager.set_optimizer_budget``
overrides it for one pipeline. Zero means unlimited.

**A cut-off search returns a different graph.** It is a valid one, the arithmetic
is equivalent and a lossy pass's error stays inside the bound it recorded, but
the tree is the best candidate the search had reached rather than the best one
there is, so two machines running one script can emit two different programs.
``MultiTermFactorization.was_cut_off`` is what says which happened, and the skip
tally says so from verbosity 2 up. A program that compares two emitted graphs, or
a test that pins one, should take the allowance away with
``set_optimizer_budget(0)`` and assert that nothing was cut off; both example
tours do exactly that and say why in a comment.

Two more bounds on the search are knobs rather than constants because they are
properties of the program a caller brings.
:option:`--einsums:graph:factorization-max-factors` is the largest number of
factors a term may have before it is declined instead of searched, fourteen by
default. The per-term program is ``3^N`` in that count, so a value in the
twenties is a request to run under the budget and keep the best tree found when
it expires rather than a way to hang a pipeline; the option help says so.
:option:`--einsums:graph:factorization-max-readers` is the linear one described
above.

The result cache
----------------

The search is a pure function of a region's structure, so its answer is kept and
replayed when a structurally identical graph comes back. What an entry holds is
a **plan** rather than a graph: the shared pairs that were committed, in order,
the contraction tree chosen for each term, and which consumers a definition was
kept for, written positionally so a plan found on one graph applies to another
that hashes the same. The case it pays for is a pipeline whose stages present
the same program under the same name.

It does not pay for re-running a script, which is a new process with an empty
cache; that case is what saving the optimized graph is for. A search cut off by
its allowance is never stored, so a plan that comes back out is one an unbounded
search would have found. :option:`--einsums:graph:factorization-cache` switches
it off, which is the first thing to do when bisecting a wrong number, since two
supposedly identical graphs getting one plan is the cheapest explanation to rule
out.

.. _optimizer-round-trip:

Saving, loading and binding
===========================

.. code-block:: python

    cg.save_graph(g, "mp2.eig")

    loaded = cg.load_graph("mp2.eig")            # or load_graph_into(path, registry)
    cg.bind(loaded, {"B_ov": B, "D": D, "E_corr": E})
    loaded.apply(cg.resource_pass_manager())
    loaded.apply(cg.tuning_pass_manager())
    loaded.execute()

Three categories of artifact are kept strictly apart, and the split is a
correctness rule rather than a convenience.

**Structure is saved**: nodes, descriptors, edges, the interface manifest, space
annotations, provenance tags and approximation records.

**Tuning is re-derived**: batching and bucket choices, the memory plan, stream
assignment, GPU placement, the distribution plan, per-node thread widths, and
the tiling schedule. Nothing machine-dependent is in the file. Reusing a saved
tuning decision across machines is not a performance risk to weigh against
convenience; a graph planned with mixed thread widths against one BLAS vendor
and replayed against another does not merely run slower.

**Setup is computed at bind**: fitted factors, inverted metrics, quadrature
points and weights. A setup body runs at the first ``execute`` after a bind
rather than during the bind itself, so a bind that is never executed costs
nothing, and ``Graph.run_setup`` is there for a caller who wants the fitting
paid at a moment of their choosing.

Repeated binds of one problem can skip the refit. ``Graph.set_setup_key`` takes a
caller-supplied problem identity, and a body whose key matches what it last
computed for is skipped; the default is an empty key, which refits on every
bind, because refitting is always correct and a graph whose caller has said
nothing about problem identity should get the behavior that cannot be wrong.

Save **before** the resource phase runs. A ``Materialize`` node carries an
allocating closure, and allocation is a resource decision a load re-derives
rather than reads. ``Graph.serializability_report()`` names any node that blocks
a save and the field that blocks it, so "can this graph be saved" has an
actionable answer rather than a boolean; a graph holding a grouped batched GEMM,
a parametric view, or an ``outer_sum`` is refused with that reason.

Space names are what a file carries, and a load resolves them against the
loading process's registry. That is why the default registry is process-global.
A program that does not save has no use for a shared namespace and should take
one of its own, which is one line:

.. code-block:: python

    registry = cg.private_space_registry(g)

A registry refuses a second declaration of one name with different content
rather than overwriting it, so two programs at two problem sizes registering a
space whose extent is derived from the space it sits inside are a conflict, not
a redeclaration. The message names every field the two declarations disagree on
and both values.

Reading the report
==================

When a pass does nothing, the skip tally is the first thing to read, before the
node count and long before the numbers.

.. code-block:: python

    pm = cg.PassManager()
    pm.set_verbosity(2)          # the tally prints from level 2 up
    pm.add(some_pass)
    pm.run(g)
    print(pm.explain())

    for reason, count in some_pass.skip_reasons:
        print(count, reason)

``explain()`` reports what each pass did, which regions formed and which
declined, with the aggregated reason and the number of candidates it applied to.
A pass that ran, did nothing and said nothing is treated as a defect in this
library rather than as normal, so a decline has a reason a caller can act on:
that the structural search is switched off, that a tagged tensor is read by no
two-operand contraction, that the decomposed form is not symbolically cheaper,
that no candidate axis set brings the largest intermediate under the cap.

:option:`--einsums:pass:verbosity` supplies a level only where a program has not
chosen its own with ``set_verbosity``. Level 1 is a summary line per pass, 2 is
each modification and each declined candidate, 3 is the per-candidate detail
behind a decline, including which rung of the cost comparison decided it.

When a rewrite produces a wrong number, :option:`--einsums:graph:dump-regions`
prints the algebra each region rewrite raised, before and after, in the
algebraic form rather than as a node list. A diff of two node-list dumps says
which nodes changed; a diff of the algebra says what the rewrite *claimed*,
which is where the mistake actually is.

What's next
===========

* ``optimizer_tour.py``, ``annotate_and_rebind.py`` and ``TagsAndProviders.cpp``
  in the ComputeGraph examples directory.
* :ref:`tutorial-compute-graph` for capture, replay, control flow and the parts
  of the graph this page assumes.
* :ref:`tutorial-performance` for the schedule and memory side.
* :ref:`The pass catalog <computegraph_optimization_passes>` for what each pass
  of the default pipeline matches and reports.
* :ref:`arguments` for every option named here, with its default, its
  environment variable and its history.
