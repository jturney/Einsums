# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""What an index space annotation buys: a decision, and a graph that outlives its geometry.

A tensor's dimensions say how big THIS problem is. An index space says what an
axis ranges over and how that set grows, which is a statement about the family
of problems the captured program stands for. Two things follow from it and
nothing else in the library provides either.

The first is that a rewrite can be judged for the family rather than for the
geometry in hand. The opposite-spin MP2 energy below has two forms: the one it
was written in, which contracts the orbital indices into a four-index integral,
and the pairless one, which decouples the energy denominator and contracts the
orbital indices away first. Neither dominates the other asymptotically, so the
choice between them is a question about extents, and at water in cc-pVDZ the
answer is the captured form: 84 auxiliary functions against 95 occupied-virtual
pairs is not where the decoupled form pays. Annotated for a system of a few
thousand basis functions, which is where spin-opposite-scaled MP2 is actually
used, the answer changes and the optimizer reaches the pairless form.

The second is that the program can be replayed at another size. A dim symbol
ties an axis to a name rather than to a number, so a bind solves the names from
the tensors it is handed and every intermediate follows. The same energy,
captured at cc-pVDZ and saved, is bound to cc-pVTZ below and replays there.

Run with::

    PYTHONPATH=<einsums-build>/lib python annotate_and_rebind.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

import einsums
import einsums.graph as cg
from einsums import linalg as la

_FIXTURES = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "examples", "dlpno", "fixtures"))

#: The family: an occupied space, a virtual space and an auxiliary basis, with the
#: extents a system of a few thousand basis functions has. Only the RATIOS reach
#: the cost model. The dim symbol on each space is the fourth field, and it is what
#: a bind solves for.
_FAMILY = (("occ", "o", 300.0, "nocc"),
           ("vir", "v", 2700.0, "nvir"),
           ("aux", "x", 9000.0, "naux"))


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


def load(stem):
    """Canonical orbitals and the density-fitted three-index tensor for one fixture."""
    z = np.load(os.path.join(_FIXTURES, f"{stem}.npz"), allow_pickle=False)
    overlap, fock = z["S"], z["F"]
    nbf = overlap.shape[0]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])

    mvals, mvecs = np.linalg.eigh(z["metric"])
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    three = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])
    three = np.einsum("Pmn,mp,nq->Ppq", three, coefficients, coefficients)
    o, v = slice(0, nocc), slice(nocc, nbf)
    return {"name": stem, "nocc": nocc, "nvir": nbf - nocc,
            "naux": three.shape[0],
            "B": np.ascontiguousarray(three[:, o, v]),
            "eps_occ": np.ascontiguousarray(energies[o]),
            "eps_vir": np.ascontiguousarray(energies[v])}


def pair_driven_energy(problem):
    """The oracle: the same energy pair by pair, never forming a four-index tensor.

    Written outside einsums on purpose, so what the graph is compared against is
    the ALGORITHM rather than another einsums program.
    """
    B, eo, ev = problem["B"], problem["eps_occ"], problem["eps_vir"]
    total = 0.0
    for i in range(problem["nocc"]):
        for j in range(problem["nocc"]):
            block = B[:, i, :].T @ B[:, j, :]
            gaps = 1.0 / (eo[i] + eo[j] - ev[:, None] - ev[None, :])
            total += float(np.sum(block * block * gaps))
    return total


def written_shapes(graph):
    """The shape of every tensor some node of the graph writes."""
    document = json.loads(graph.to_json())
    dims = {entry["id"]: entry["dims"] for entry in document["tensors"]}
    return [dims[tid] for node in document["nodes"] for tid in node.get("outputs", []) if tid in dims]


# ──────────────────────────────────────────────────────────────────────────
# 1. The family decides which form of the equation is taken
# ──────────────────────────────────────────────────────────────────────────


def capture_with_recipe(problem, name, registry, annotate):
    """``E = sum_iajb (ia|jb)^2 / D``, with the denominator built inside the capture.

    The denominator is an outer sum of the orbital energies followed by the
    registered ``recip`` element operation, which is a recipe the Laplace
    transform can read and check against the nodes. The tag on it names the
    energy vector behind each axis and the sign it enters with.

    ``annotate`` is the whole variable of this arm. It declares which set each
    axis runs over, and nothing else about the program changes.
    """
    nocc, nvir = problem["nocc"], problem["nvir"]
    shape = [nocc, nvir, nocc, nvir]
    three = _tensor("B_ov", problem["B"])
    occupied = _tensor("eps_occ", problem["eps_occ"])
    virtual = _tensor("eps_vir", problem["eps_vir"])
    energy = einsums.create_zero_tensor("E_corr", [1])

    graph = cg.Graph(name)
    graph.set_space_registry(registry)
    integrals = graph.scratch("K", shape, "float64")
    amplitudes = graph.scratch("T", shape, "float64")
    denominator = graph.scratch("D", shape, "float64")
    with cg.capture(graph):
        la.outer_sum(denominator, [occupied, virtual, occupied, virtual], [1.0, -1.0, 1.0, -1.0])
        la.element_transform(denominator, "recip")
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", integrals, three, three)
        la.direct_product(1.0, integrals, denominator, 0.0, amplitudes)
        la.dot(energy, integrals, amplitudes)
    graph.annotate_tag(denominator, cg.LaplaceTransform.denominator_tag(
        ["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-"))

    if annotate:
        cg.annotate(three, ("aux", "occ", "vir"), graph=graph)
        # The denominator too, and not for symmetry: the transform gives its
        # exponentials the quadrature space and the axis they were built from,
        # all axes or none, so an unannotated denominator leaves the quadrature
        # letter anonymous, and one anonymous letter blocks the family's extents
        # for the whole polynomial it appears in.
        for tensor in (integrals, amplitudes, denominator):
            cg.annotate(tensor, ("occ", "vir", "occ", "vir"), graph=graph)

    return {"graph": graph, "three": three, "occupied": occupied,
            "virtual": virtual, "energy": energy}


def optimize_with_search(program, epsilon=1e-8):
    """Transform the denominator, then let the search re-bracket what is left.

    The budget is REMOVED. A search cut off by its wall-clock allowance keeps the
    best candidate it had reached, which is a valid graph but a different one, so
    the form reported below would otherwise be a property of how fast this
    machine is rather than of the algebra. ``was_cut_off`` is what says which
    happened, and it is asserted rather than hoped for.
    """
    transform = cg.LaplaceTransform()
    transform.set_epsilon(epsilon)
    transform.add_energy("eps_occ", program["occupied"])
    transform.add_energy("eps_vir", program["virtual"])
    search = cg.MultiTermFactorization()
    search.set_search_enabled(True)

    manager = cg.PassManager()
    manager.set_optimizer_budget(0)
    manager.add(transform)
    manager.add(search)
    manager.run(program["graph"])
    if search.was_cut_off:
        raise SystemExit("the search ran out of its allowance, so the form below is the machine's")
    program.update(transform=transform, search=search)
    return program


def arm_family(problem, registry):
    print("=" * 78)
    print("1. The same program, annotated and not")
    print("=" * 78)
    oracle = pair_driven_energy(problem)
    four_index = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]

    for annotate in (False, True):
        program = optimize_with_search(
            capture_with_recipe(problem, f"sos annotated={annotate}", registry, annotate))
        graph = program["graph"]
        shapes = written_shapes(graph)
        points = program["transform"].last_point_count

        graph.apply(cg.default_pass_manager())
        graph.execute()
        value = float(np.asarray(program["energy"])[0])
        record = graph.approximations()[0]

        print(f"\n  spaces annotated: {annotate}")
        print(f"    quadrature points            {points}")
        print(f"    a four-index tensor survives {four_index in shapes}")
        matrix = [shape for shape in shapes
                  if sorted(shape) == sorted([problem["naux"], problem["naux"], points])]
        print(f"    a Q by points by Q matrix    {bool(matrix)}")
        print(f"    E_os                         {value:.12f} Ha")
        print(f"    against the pair loop        {abs(value - oracle):.3e}, "
              f"inside a recorded bound of {record.bound * abs(oracle):.3e}")

    print("\n  Both answers are correct and inside the bound the transform recorded. What the"
          "\n  annotation changed is which of two forms of one equation the optimizer took: the"
          "\n  decoupled form trades o^2 v^2 Q for o v Q^2 t, and whether that pays is a fact"
          "\n  about the extents rather than about the expression. Unannotated, the comparison"
          "\n  falls back to the extents this capture happens to have, and at water/cc-pVDZ they"
          "\n  say no. That is the optimizer being right about the wrong problem.")


# ──────────────────────────────────────────────────────────────────────────
# 2. The family moves the problem
# ──────────────────────────────────────────────────────────────────────────


def capture_for_reuse(problem, name, registry):
    """The same energy, written so that a bind can move it to another molecule.

    Three things are different from the capture above and each is required.

    The denominator is built EAGERLY, because an outer sum records a node no
    file can rebuild, and a graph holding one refuses to save with that reason.
    Supplied as an input it is an ordinary interface tensor.

    The interface tensors carry DIM SYMBOLS beside their spaces. A symbol is a
    name for an extent, so ``bind`` solves the names from the tensors it is
    handed and checks that every slot naming one agrees.

    The intermediates are declared over the SPACES rather than over numbers,
    which sizes them, annotates them and writes their symbols in one statement.
    That is the spelling that cannot drift: three separate statements about one
    axis are three chances to disagree.
    """
    three = _tensor("B_ov", problem["B"])
    occupied = _tensor("eps_occ", problem["eps_occ"])
    virtual = _tensor("eps_vir", problem["eps_vir"])
    energy = einsums.create_zero_tensor("E_corr", [1])
    denominator = _tensor("D", np.zeros([problem["nocc"], problem["nvir"],
                                         problem["nocc"], problem["nvir"]]))
    la.outer_sum(denominator, [occupied, virtual, occupied, virtual], [1.0, -1.0, 1.0, -1.0])
    la.element_transform(denominator, "recip")

    graph = cg.Graph(name)
    graph.set_space_registry(registry)
    cg.annotate(three, ("aux", "occ", "vir"), graph=graph)
    cg.annotate(denominator, ("occ", "vir", "occ", "vir"), graph=graph)
    graph.annotate_dims(three, ["naux", "nocc", "nvir"])
    graph.annotate_dims(denominator, ["nocc", "nvir", "nocc", "nvir"])

    occ, vir = registry.find("occ"), registry.find("vir")
    axes = [cg.SpaceDim(occ), cg.SpaceDim(vir), cg.SpaceDim(occ), cg.SpaceDim(vir)]
    integrals = graph.declare_zero_tensor_over("K", axes, True)
    amplitudes = graph.declare_zero_tensor_over("T", axes, True)
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", integrals, three, three)
        la.direct_product(1.0, integrals, denominator, 0.0, amplitudes)
        la.dot(energy, integrals, amplitudes)
    return {"graph": graph, "energy": energy, "integrals": integrals}


def arm_rebind(small, big, registry):
    print("\n" + "=" * 78)
    print("2. Captured on one molecule, replayed on a larger one")
    print("=" * 78)

    program = capture_for_reuse(small, "sos reusable", registry)
    graph = program["graph"]
    print(f"\n  captured on {small['name']}: "
          f"{small['nocc']} occupied, {small['nvir']} virtual, {small['naux']} auxiliary")
    print(f"    dim symbols on K             {graph.tensor_dim_symbols(program['integrals'])}")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    print(f"    E_os                         {float(np.asarray(program['energy'])[0]):.12f} Ha")
    print(f"    against the pair loop        "
          f"{abs(float(np.asarray(program['energy'])[0]) - pair_driven_energy(small)):.3e}")

    path = os.path.join(tempfile.mkdtemp(), "sos.eig")
    # Saved from a second capture, because the file holds the structure and the
    # interface and nothing a machine decided: a Materialize node carries an
    # allocating closure, so a graph is saved before the resource phase runs.
    cg.save_graph(capture_for_reuse(small, "sos to save", registry)["graph"], path)

    # ``load_graph_into`` resolves the file's space names against the registry
    # this program declared them in. The default is the process-global registry,
    # which is what a program that shares its files wants; a program that does
    # not can take one of its own with ``cg.private_space_registry(graph)``.
    loaded = cg.load_graph_into(path, registry)
    names = set(loaded.manifest_names())
    print(f"\n  loaded, with no storage behind it: manifest {sorted(names)}")

    three = _tensor("B_ov", big["B"])
    occupied = _tensor("eps_occ", big["eps_occ"])
    virtual = _tensor("eps_vir", big["eps_vir"])
    energy = einsums.create_zero_tensor("E_corr", [1])
    denominator = _tensor("D", np.zeros([big["nocc"], big["nvir"], big["nocc"], big["nvir"]]))
    la.outer_sum(denominator, [occupied, virtual, occupied, virtual], [1.0, -1.0, 1.0, -1.0])
    la.element_transform(denominator, "recip")

    # One transaction, because a dim symbol is a constraint ACROSS slots: nothing
    # may be repointed until every extent the caller supplied has been read and
    # reconciled. Bound one at a time, the second slot is solved against an
    # interface the first has already half-moved.
    cg.bind(loaded, {"B_ov": three, "D": denominator, "E_corr": energy})
    loaded.apply(cg.resource_pass_manager())
    loaded.apply(cg.tuning_pass_manager())
    loaded.execute()

    value = float(np.asarray(energy)[0])
    oracle = pair_driven_energy(big)
    print(f"  bound to {big['name']}: "
          f"{big['nocc']} occupied, {big['nvir']} virtual, {big['naux']} auxiliary")
    print(f"    E_os                         {value:.12f} Ha")
    print(f"    against the pair loop        {abs(value - oracle):.3e}")
    print("\n  The resource and tuning phases are re-run on this side of the file, which is the"
          "\n  same code path a fresh capture takes. Nothing a machine decided was in the file:"
          "\n  allocation, batching and thread widths are re-derived where the graph is opened.")


def main():
    small, big = load("water-ccpvdz"), load("water-ccpvtz")

    # One registry for this program, rather than the process-global one. A caller
    # that does not share saved files with anyone has no use for a shared
    # namespace, and a registry of its own is one line at the top of the program.
    registry = cg.private_space_registry(cg.Graph("family"))
    for name, symbol, extent, dim in _FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))

    print(__doc__.split("Run with::")[0].rstrip())
    print()
    arm_family(small, registry)
    arm_rebind(small, big, registry)

    print("\n" + "=" * 78)
    print("What does not move yet")
    print("=" * 78)
    print("A graph carrying a quadrature binds at the geometry whose orbital energies it was\n"
          "fitted from: the transform registers those vectors as interface entries with literal\n"
          "extents, so a bind at a larger molecule is refused, naming the axis. What crosses a\n"
          "size boundary today is the exact algebra, which is the case above.")


if __name__ == "__main__":
    main()
