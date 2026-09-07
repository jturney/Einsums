# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One density-fitted MP2 program, driven through every arm of the optimizer.

The program is the closed-shell MP2 correlation energy over all four orbital
indices, written once, at the top of this file::

    D[i,a,j,b] = 1 / (e_i - e_a + e_j - e_b)      an outer sum and a reciprocal
    K[i,a,j,b] = sum_Q B[Q,i,a] B[Q,j,b]          the density-fitted integral
    T          = K * D                            the amplitude
    E          = sum_iajb (2 K[i,a,j,b] - K[i,b,j,a]) T[i,a,j,b]

Every arm below captures that same program and then asks the optimizer a
different question about it. Nothing in the program changes between arms: what
changes is which passes are in the pipeline and what the caller declared about
the tensors. That is the point of the tour. A user does not rewrite their
equations to get a factorization, a quadrature or a loop; they annotate, tag,
register a provider, and state a memory cap.

The molecule is water in cc-pVDZ, read out of the DLPNO example's fixture, so
this runs offline with no integral engine and no psi4. Five occupied orbitals,
19 virtuals, 84 auxiliary functions.

Run with::

    PYTHONPATH=<einsums-build>/lib python optimizer_tour.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

import einsums
import einsums.graph as cg
from einsums import linalg as la

_FIXTURE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvdz.npz"))

#: The family the caller declares. The numbers are a system of a few thousand
#: basis functions, which is the regime these rewrites are aimed at; only their
#: RATIOS reach the cost model, and declaring them is how a caller says which
#: regime the captured geometry stands for. Without them a comparison between
#: two forms of one equation is settled by the toy extents the capture happened
#: to have.
_FAMILY = (("occ", "o", 300.0, "nocc"),
           ("vir", "v", 2700.0, "nvir"),
           ("aux", "x", 8100.0, "naux"))

#: A cap between one occupied pair of the four-index tensor (19 x 19 doubles)
#: and one occupied row of it (5 x 19 x 19), so it forces per-pair streaming and
#: nothing coarser.
_PAIR_CAP = 4096


def _rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


# ──────────────────────────────────────────────────────────────────────────
# The problem, out of the fixture
# ──────────────────────────────────────────────────────────────────────────


def load_water():
    """Canonical orbitals, the density-fitted three-index tensor, the energies."""
    z = np.load(_FIXTURE, allow_pickle=False)

    overlap, fock = z["S"], z["F"]
    nbf = overlap.shape[0]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])

    metric = z["metric"]
    mvals, mvecs = np.linalg.eigh(metric)
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    three_ao = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])
    three = np.einsum("Pmn,mp,nq->Ppq", three_ao, coefficients, coefficients)

    o, v = slice(0, nocc), slice(nocc, nbf)
    B_ov = np.ascontiguousarray(three[:, o, v])
    ovov = np.einsum("Qia,Qjb->iajb", B_ov, B_ov)
    gaps = (energies[o, None, None, None] - energies[None, v, None, None]
            + energies[None, None, o, None] - energies[None, None, None, v])
    return {
        "nocc": nocc, "nvir": nbf - nocc, "naux": three.shape[0],
        "B_ov": B_ov,
        "amplitudes": np.ascontiguousarray(ovov / gaps),
        "fock_vv": np.ascontiguousarray(np.diag(energies[v])),
        "eps_occ": np.ascontiguousarray(energies[o]),
        "eps_vir": np.ascontiguousarray(energies[v]),
        "reference": float(z["energy_psi4_df_mp2"]),
    }


# ──────────────────────────────────────────────────────────────────────────
# The program, written once
# ──────────────────────────────────────────────────────────────────────────


def capture(problem, name):
    """Capture the DF-MP2 energy and say what its tensors ARE.

    Three kinds of statement about the program go in here, and each one buys a
    different class of rewrite:

    * the SPACES, one per axis, which is what lets a pass compare two forms of
      the equation for the family rather than for this geometry;
    * the TAG on the three-index tensor, which is what a factorization provider
      claims;
    * the TAG on the denominator, which names the orbital-energy vector behind
      each axis and the sign it enters with, and is what the Laplace transform
      recognizes. The denominator is built inside the capture, out of an outer
      sum and the registered ``recip`` element operation, because that is a
      recipe the transform can read and verify. An anonymous Python callable in
      its place says only that something is applied to every element, and the
      transform declines it.

    The integral is written ONCE. Four statements read it, and no pass needs a
    second copy: the transform takes its own copy of the numerator it rewrites
    and leaves the definition standing for the other readers.
    """
    nocc, nvir = problem["nocc"], problem["nvir"]
    shape = [nocc, nvir, nocc, nvir]

    three = _tensor("B_ov", problem["B_ov"])
    occupied = _tensor("eps_occ", problem["eps_occ"])
    virtual = _tensor("eps_vir", problem["eps_vir"])
    energy = einsums.create_zero_tensor("E_corr", [1])

    graph = cg.Graph(name)
    integrals = graph.scratch("K", shape, "float64")
    amplitudes = graph.scratch("T", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    denominator = graph.scratch("D", shape, "float64")

    with cg.capture(graph):
        la.outer_sum(denominator, [occupied, virtual, occupied, virtual], [1.0, -1.0, 1.0, -1.0])
        la.element_transform(denominator, "recip")
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", integrals, three, three)
        la.direct_product(1.0, integrals, denominator, 0.0, amplitudes)
        einsums.permute("iajb <- ibja", exchange, integrals)
        la.axpby(2.0, integrals, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, amplitudes)

    cg.annotate(three, ("aux", "occ", "vir"), graph=graph)
    cg.annotate(occupied, ("occ",), graph=graph)
    cg.annotate(virtual, ("vir",), graph=graph)
    for scratch in (integrals, amplitudes, exchange, combination, denominator):
        cg.annotate(scratch, ("occ", "vir", "occ", "vir"), graph=graph)

    graph.annotate_tag(three, cg.ProvenanceTag.make("eri"))
    graph.annotate_tag(denominator, cg.LaplaceTransform.denominator_tag(
        ["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-"))

    return {"graph": graph, "three": three, "occupied": occupied, "virtual": virtual,
            "denominator": denominator, "energy": energy}


# ──────────────────────────────────────────────────────────────────────────
# What a reader wants to see after each arm
# ──────────────────────────────────────────────────────────────────────────


def largest_intermediate(graph):
    """Bytes in the largest tensor some node of the optimized graph writes.

    Read off the graph document rather than off a pass, so the number means the
    same thing in every arm below and is not any one pass's own accounting.
    Loop bodies are included, since a tensor a body writes is one the program
    allocates.
    """
    document = json.loads(graph.to_json())
    return _largest(document)


_ELEMENT_BYTES = {"float32": 4, "float64": 8, "complex64": 8, "complex128": 16}


def _largest(document):
    sizes = {}
    for entry in document.get("tensors", []):
        count = 1
        for extent in entry.get("dims", []):
            count *= extent
        sizes[entry["id"]] = count * _ELEMENT_BYTES.get(entry.get("dtype", "float64"), 8)
    largest = 0
    for node in document.get("nodes", []):
        for tid in node.get("outputs", []):
            largest = max(largest, sizes.get(tid, 0))
        for body in ("body", "then_body", "else_body"):
            if isinstance(node.get(body), dict):
                largest = max(largest, _largest(node[body]))
    return largest


def report(label, graph, energy, exact, captured=None, passes=()):
    """Node count, largest intermediate, the energy, and every record on it."""
    value = float(np.asarray(energy)[0])
    print(f"\n  {label}")
    if captured is not None:
        print(f"    nodes                  {captured} captured -> {graph.num_nodes()} emitted")
    else:
        print(f"    nodes                  {graph.num_nodes()}")
    print(f"    largest intermediate   {largest_intermediate(graph)} bytes")
    print(f"    E_corr                 {value:.12f} Ha")
    print(f"    error vs the exact     {abs(value - exact):.3e}")
    for record in graph.approximations():
        print(f"    record                 {record.pass_name}: bound {record.bound:.3e}, "
              f"{record.effect.name}, {record.origin.name}")
    tolerance = graph.approximation_tolerance()
    if tolerance.absolute or tolerance.relative:
        composed = tolerance.absolute + tolerance.relative * abs(exact)
        print(f"    composed budget        {composed:.3e} "
              f"(absolute {tolerance.absolute:.3e} + relative {tolerance.relative:.3e})")
    for entry in passes:
        for reason, count in entry.skip_reasons:
            print(f"    declined ({count})           {reason}")
    return value


def print_explanation(manager, keep=None):
    """The pass report, which is where a pass says what it did and what it declined."""
    for line in manager.explain().splitlines():
        if not line.strip():
            continue
        if keep is None or any(word in line for word in keep):
            print(f"    | {line}")


# ──────────────────────────────────────────────────────────────────────────
# The arms
# ──────────────────────────────────────────────────────────────────────────


def arm_default(problem, exact):
    """The pipeline a caller gets by asking for nothing.

    It holds the analysis, resource and tuning passes plus the structural
    rewrites proven exact, and no lossy pass is ever in it. So the answer is the
    answer, and what the passes bought is a schedule.
    """
    _rule("1. The default pipeline")
    program = capture(problem, "mp2 default")
    captured = program["graph"].num_nodes()
    manager = cg.default_pass_manager()
    program["graph"].apply(manager)
    program["graph"].execute()
    value = report("default_pass_manager()", program["graph"], program["energy"], exact, captured)
    print(f"\n    The fixture's psi4 DF-MP2 energy is {problem['reference']:.12f} Ha, which is what "
          f"every arm below is measured against.")
    return value


def arm_search(problem, exact):
    """The structural search, which is off by default and asked for here.

    Every other structural pass is a recognizer whose runtime is a function of
    the node count. A search's runtime is a function of how many candidates the
    program offers, which nobody can predict from outside, so it is opt-in.

    The budget is REMOVED for the same reason a test that pins an emitted tree
    removes it: a search cut off by its wall-clock allowance keeps the best
    candidate it had reached, which is a valid graph but a different one, so the
    tree printed below would otherwise be a property of how fast this machine
    is. ``was_cut_off`` is what says which happened.
    """
    _rule("2. The structural search, switched on")

    # Off, which is what a caller gets without asking. The pass runs and says so
    # rather than doing nothing in silence, which is the difference between "this
    # graph was already optimal" and "the pass that would have optimized it was
    # switched off". The skip tally is what carries that, and it prints from
    # verbosity 2 up.
    asleep = cg.PassManager()
    asleep.set_verbosity(2)
    asleep.add(cg.MultiTermFactorization())
    asleep.run(capture(problem, "mp2 no search")["graph"])
    print()
    print_explanation(asleep, keep=("search",))

    program = capture(problem, "mp2 search")
    captured = program["graph"].num_nodes()

    search = cg.MultiTermFactorization()
    search.set_search_enabled(True)
    manager = cg.PassManager()
    manager.set_optimizer_budget(0)
    manager.add(search)
    manager.run(program["graph"])
    print(f"\n    rebracketed {search.num_rebracketed}, inlined {search.num_inlined}, "
          f"copies {search.num_copies}, cut off: {search.was_cut_off}")
    print_explanation(manager, keep=("MultiTermFactorization",))

    program["graph"].apply(cg.default_pass_manager())
    program["graph"].execute()
    report("structural search", program["graph"], program["energy"], exact, captured)
    print("\n    The energy does not move, because a re-bracketing is the same arithmetic in a"
          "\n    different order. What moves is the tree, and on this program the search's answer"
          "\n    is that the captured bracketing is already the cheapest one over these leaves:"
          "\n    the integral is read by a permute and two scalings, none of which is a product"
          "\n    anything can be folded into. The next arm is where it has something to find.")


def arm_laplace(problem, exact):
    """The lossy transform, at two tolerances, with the records it writes.

    The tag on the denominator is what the pass recognizes; the tolerance is
    what it is asked for; the point count is what it derives from the spectral
    range of the bound orbital energies. The record is MEASURED, which this pass
    can afford because the exact quantity it replaces is a reciprocal it can
    sample.

    The quadrature index becomes an ordinary contracted letter rather than a
    loop over terms, so the emitted node count barely moves with the point
    count. The four-index denominator is DISSOLVED, which is the substitution
    showing up where a caller can see it: it is never allocated again.

    Each tolerance is run twice, and the second run is the point. The transform
    alone pushes an exponential onto each factor of the numerator, so what it
    leaves behind is one dressed integral per quadrature point, which at this
    molecule is bigger than the tensor it replaced. It is the SEARCH that then
    contracts the orbital indices away first and brings the footprint back down.
    Neither pass reaches that on its own.
    """
    _rule("3. The Laplace transform, at two tolerances")
    for epsilon in (1e-3, 1e-8):
        program = capture(problem, f"mp2 laplace {epsilon:g}")
        captured = program["graph"].num_nodes()

        transform = cg.LaplaceTransform()
        transform.set_epsilon(epsilon)
        transform.add_energy("eps_occ", program["occupied"])
        transform.add_energy("eps_vir", program["virtual"])
        manager = cg.PassManager()
        manager.add(transform)
        manager.run(program["graph"])
        print(f"\n  laplace-epsilon = {epsilon:g}")
        print(f"    quadrature points      {transform.last_point_count}")
        print(f"    numerator copies       {transform.num_numerator_copies} "
              f"(the pass keeps the definition the other readers want)")
        print(f"    nodes                  {captured} captured -> {program['graph'].num_nodes()} transformed")
        print(f"    largest intermediate   {largest_intermediate(program['graph'])} bytes, "
              f"before the search")

        search = cg.MultiTermFactorization()
        search.set_search_enabled(True)
        searching = cg.PassManager()
        searching.set_optimizer_budget(0)
        searching.add(search)
        searching.run(program["graph"])
        print(f"    search                 rebracketed {search.num_rebracketed}, "
              f"inlined {search.num_inlined}, cut off: {search.was_cut_off}")

        program["graph"].apply(cg.default_pass_manager())
        program["graph"].execute()
        report("after the transform and the search", program["graph"],
               program["energy"], exact, captured)
        allocated = {node["label"] for node in json.loads(program["graph"].to_json())["nodes"]
                     if node["kind"] == "Materialize"}
        print(f"    denominator allocated  {'materialize(D)' in allocated}")


def arm_natural_auxiliary(problem, exact):
    """A provider that SHORTENS an index rather than splitting a tensor.

    The three-index tensor, seen as a matrix against the pair index, has a
    singular value decomposition, and the directions below a threshold are
    dropped. Both occurrences of the tensor in the contraction are substituted,
    and the search then contracts the two truncated transformations together
    first, which is where the saving is: nobody had to teach it that the small
    matrix is nearly an identity, only that it is the cheapest thing to build.

    What the record carries is not the threshold. It is the norm of the dropped
    singular values, measured, which is a bound on the TENSOR and therefore a
    loose one on any energy computed from it.
    """
    _rule("4. Natural auxiliary functions")
    program = capture(problem, "mp2 naf")
    captured = program["graph"].num_nodes()

    provider = cg.NaturalAuxiliaryFactorization("eri", program["three"], 1e-1)
    registry = cg.FactorizationRegistry()
    registry.add(provider)
    factorization = cg.FactorizationPass(registry)
    manager = cg.PassManager()
    # A tag declared on the graph reaches the operand handles in the analysis phase.
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    manager.run(program["graph"])

    program["graph"].apply(cg.default_pass_manager())
    program["graph"].execute()
    report("naf-threshold = 1e-1", program["graph"], program["energy"], exact, captured)
    print(f"    auxiliary functions    {problem['naux']} -> {provider.kept}")
    print(f"    dropped norm           {provider.dropped_norm:.3e}")


def arm_frozen_natural_orbitals(problem, exact):
    """A pass that replaces a SPACE rather than an expression.

    Nothing is factored here, which is why this is not a provider. The MP2
    virtual-virtual density is diagonalized, the orbitals above an occupation
    cutoff are kept, every tensor over the virtual space is projected into the
    smaller one, and the Fock block is semicanonicalized so the denominators are
    diagonal again. The denominator recipe is repointed at the semicanonical
    energies, which is the one link between this pass and the quadrature and is
    what lets the two compose.

    The correlation energy the truncation removed is the record, measured, and
    a program that wants the corrected number adds it back. That is what makes
    the record worth composing rather than an extra output nobody reads.
    """
    _rule("5. Frozen natural orbitals")
    program = capture(problem, "mp2 fno")
    captured = program["graph"].num_nodes()

    truncation = cg.BasisTruncation()
    truncation.set_amplitudes(_tensor("t2", problem["amplitudes"]))
    truncation.set_fock(_tensor("F_vv", problem["fock_vv"]))
    truncation.set_occupied_energies(_tensor("eps_occ_fno", problem["eps_occ"]))
    truncation.set_occupation(1e-3)
    manager = cg.PassManager()
    manager.add(truncation)
    manager.run(program["graph"])

    program["graph"].apply(cg.default_pass_manager())
    program["graph"].execute()
    value = report("fno-occupation = 1e-3", program["graph"], program["energy"], exact, captured)
    print(f"    virtual orbitals       {problem['nvir']} -> {truncation.kept}")
    print(f"    projected tensors      {list(truncation.projected)}")
    print(f"    MP2 correction         {truncation.correction:.6e} Ha")
    print(f"    corrected energy       {value + truncation.correction:.12f} Ha "
          f"(error {abs(value + truncation.correction - exact):.3e})")


def arm_tiling(problem, exact):
    """A schedule, not an algebra: the pair loop derived rather than written.

    The full-axis capture declares four-index tensors, which is exactly the
    object density fitting exists to avoid. Under a memory cap the pass slices
    the free axes and streams every intermediate that carries them, so the body
    is the same algebra at one occupied pair and the four-index tensors are one
    ``v`` by ``v`` block each. It picks the two OCCUPIED axes, and not on their
    size: the exchange permutation exchanges the two virtual indices, so a slice
    of the permuted tensor would need a slice of its source at a pair the body
    is not at, and the virtual pair is rejected structurally at any extents.

    One tensor stays whole and the print below says which. The denominator's
    recipe is an outer sum, which the schedule cannot re-emit, so it sits
    outside the loop and is formed at full size. Dissolving that tensor is what
    the Laplace arm does; slicing the rest is what this one does.
    """
    _rule("6. AxisTiling under a memory cap")
    program = capture(problem, "mp2 tiled")
    captured = program["graph"].num_nodes()

    tiling = cg.AxisTiling()
    tiling.set_memory_cap(_PAIR_CAP)
    manager = cg.PassManager()
    manager.add(tiling)
    manager.run(program["graph"])

    print(f"\n    cap                    {_PAIR_CAP} bytes")
    print(f"    axes                   {list(tiling.axis_letters)} at extents {list(tiling.axis_extents)}")
    print(f"    slices                 {tiling.slice_count} in chunks of {tiling.depth}")
    print(f"    streamed intermediate  {tiling.largest_before} -> {tiling.largest_after} bytes")
    print(f"    streamed / whole       {list(tiling.streamed)} / {list(tiling.whole)}")
    for reason, count in tiling.skip_reasons:
        print(f"    declined ({count})           {reason}")

    program["graph"].apply(cg.default_pass_manager())
    program["graph"].execute()
    report("tiling-memory-cap = 4096", program["graph"], program["energy"], exact, captured)

    document = json.loads(program["graph"].to_json())
    print(f"    top-level node kinds   {[node['kind'] for node in document['nodes']]}")
    dims = {entry["id"]: (entry["name"], entry["dims"]) for entry in document["tensors"]}
    for node in document["nodes"]:
        if node["kind"] == "Materialize":
            for tid in node.get("outputs", []):
                name, extents = dims.get(tid, ("?", []))
                print(f"    allocated              {name} {extents}")


def arm_round_trip(problem, exact):
    """Save the algebra, load it in a graph with no storage, bind, replay.

    What crosses the file is the structure, the interface and the approximation
    records. What does not is anything a machine decided: allocation, batching,
    thread widths, and the schedule. So a load re-runs the resource and tuning
    phases, which is the same code path a fresh capture takes, and the tiling
    decision above is taken again against whatever cap is in force where the
    file is opened.
    """
    _rule("7. Save, load, bind, replay")
    program = capture(problem, "mp2 round trip")
    transform = cg.LaplaceTransform()
    transform.set_epsilon(1e-6)
    transform.add_energy("eps_occ", program["occupied"])
    transform.add_energy("eps_vir", program["virtual"])
    manager = cg.PassManager()
    manager.add(transform)
    manager.run(program["graph"])

    path = os.path.join(tempfile.mkdtemp(), "optimizer_tour.eig")
    try:
        # Saved BEFORE the default manager runs: a Materialize node holds an
        # allocating closure, and allocation is a resource decision a load
        # re-derives rather than reads.
        cg.save_graph(program["graph"], path)
        loaded = cg.load_graph(path)
        names = set(loaded.manifest_names())
        print(f"\n    manifest               {sorted(names)}")
        print(f"    records carried        {[r.pass_name for r in loaded.approximations()]}")
        print("    the denominator is gone from the interface, and the orbital energies are in "
              "it,\n    which is what lets the quadrature be refitted at whatever a bind supplies.")

        replayed = einsums.create_zero_tensor("E_corr", [1])
        supplied = {"B_ov": program["three"], "E_corr": replayed,
                    "eps_occ": program["occupied"], "eps_vir": program["virtual"]}
        cg.bind(loaded, {name: tensor for name, tensor in supplied.items() if name in names})
        loaded.apply(cg.resource_pass_manager())
        loaded.apply(cg.tuning_pass_manager())
        loaded.execute()
        report("replayed from the file", loaded, replayed, exact)
    finally:
        if os.path.exists(path):
            os.remove(path)


def arm_declines(problem):
    """Two passes that decline, and the tally that says why.

    A pass that does nothing is the ordinary case, and the first thing to read
    is the skip tally rather than the node count. Both declines below are
    statements about the program rather than refusals either pass makes on its
    own account.
    """
    _rule("8. What declines, and why")

    program = capture(problem, "mp2 laplace then tile")
    transform = cg.LaplaceTransform()
    transform.set_epsilon(1e-5)
    transform.add_energy("eps_occ", program["occupied"])
    transform.add_energy("eps_vir", program["virtual"])
    manager = cg.PassManager()
    manager.add(transform)
    manager.run(program["graph"])

    tiling = cg.AxisTiling()
    tiling.set_memory_cap(_PAIR_CAP)
    schedule = cg.PassManager()
    schedule.add(tiling)
    schedule.run(program["graph"])
    print("\n    AxisTiling on the transformed program")
    print(f"      tiled {tiling.num_tiled}, largest intermediate {tiling.largest_before} bytes")
    for reason, count in tiling.skip_reasons:
        print(f"      ({count}) {reason}")
    print("      The transform leaves one exponential-dressed integral per quadrature point,"
          "\n      which at this molecule is LARGER than the tensor it replaced. So the decline is"
          "\n      that no candidate axis set brings the footprint under the cap, and not that"
          "\n      there is nothing left to tile.")

    truncated = capture(problem, "mp2 fno then naf")
    truncation = cg.BasisTruncation()
    truncation.set_amplitudes(_tensor("t2", problem["amplitudes"]))
    truncation.set_fock(_tensor("F_vv", problem["fock_vv"]))
    truncation.set_occupied_energies(_tensor("eps_occ_fno", problem["eps_occ"]))
    truncation.set_occupation(1e-3)
    first = cg.PassManager()
    first.add(truncation)
    first.run(truncated["graph"])

    provider = cg.NaturalAuxiliaryFactorization("eri", truncated["three"], 1e-1)
    registry = cg.FactorizationRegistry()
    registry.add(provider)
    factorization = cg.FactorizationPass(registry)
    second = cg.PassManager()
    second.add(cg.ProvenancePropagation())
    second.add(factorization)
    second.run(truncated["graph"])
    print("\n    NaturalAuxiliaryFactorization after the basis truncation")
    print(f"      factorized {factorization.num_factorized}")
    for reason, count in factorization.skip_reasons:
        print(f"      ({count}) {reason}")
    print("      Two passes with a claim on ONE tensor do not compose: after the projection the"
          "\n      caller's three-index tensor is read by nothing but the setup that projects it,"
          "\n      so there is no contraction left to re-associate around its factors.")


# ──────────────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(_FIXTURE):
        raise SystemExit(f"fixture not present: {_FIXTURE}")
    problem = load_water()

    # The family, and the two spaces a truncation derives from it. The registry
    # is process-global, which is what makes a saved graph's space names
    # resolvable when it is loaded; a program that does not save can take a
    # registry of its own with cg.private_space_registry(graph) instead.
    registry = cg.global_space_registry()
    for name, symbol, extent, dim in _FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    cg.NaturalAuxiliaryFactorization.register_naf_space(cg.Graph("family"))
    cg.BasisTruncation.register_fno_space(cg.Graph("family"))

    print(__doc__.split("Run with::")[0].rstrip())
    print(f"\nwater/cc-pVDZ from the fixture: {problem['nocc']} occupied, {problem['nvir']} virtual, "
          f"{problem['naux']} auxiliary")

    exact = arm_default(problem, problem["reference"])
    arm_search(problem, exact)
    arm_laplace(problem, exact)
    arm_natural_auxiliary(problem, exact)
    arm_frozen_natural_orbitals(problem, exact)
    arm_tiling(problem, exact)
    arm_round_trip(problem, exact)
    arm_declines(problem)

    _rule("Done")
    print("Every arm above ran the same captured program. What changed is what the caller\n"
          "declared and which passes were in the pipeline.")


if __name__ == "__main__":
    main()
