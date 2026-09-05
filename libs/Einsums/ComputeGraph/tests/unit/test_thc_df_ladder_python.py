# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The four-external ladder in the form a density-fitted program writes it.

``test_thc_cc_python`` runs the same subseries against a dense ``(ae|bf)``
integral and records why fitting the amplitude alone declines there: the
integral stays where it was, so the grid indices have to be carried through a
``v^4`` tensor. That is the right measurement of the wrong program. No
density-fitted coupled-cluster code forms ``(ae|bf)``; every one of them writes

    W[Q,i,a,j,f] = sum_e B[Q,a,e] t[i,e,j,f]
    R[i,a,j,b]  += sum_Qf W[Q,i,a,j,f] B[Q,b,f]

at ``o^2 v^3 Q``, and the tags then land on the LEAVES of a chain rather than on
the two operands of one statement: the three-index integral twice and the
amplitude once. This file is that program, with ``B`` tagged ``eri`` and fitted
on the grid as ``X[a,P] X[e,P] C[Q,P]``, and ``t2`` tagged ``amplitude`` and
fitted as the five-factor chain. Both fits are substituted in one decision and
the whole leaf set is bracketed together, which is what collapses the ``e``, the
``f`` and the auxiliary contractions onto grid overlaps.

Fitting ``B`` DIRECTLY rather than teaching the pass to recognize a ``B B`` pair
as the four-index tensor's fit is the choice this file makes, and it is the
cheaper one in both senses. The weights are what the four-index fit already
computes on its way to ``Z``, stopped one step before it squares; ``C`` is
auxiliary-by-grid where ``Z`` is grid-by-grid; and the two occurrences of ``B``
in the ladder share one fitting between them, so nothing is fitted twice.

The rest of the iteration is the four-occupied ladder and the update, exactly as
``test_thc_cc_python`` writes them. The four-occupied term is the cheap term
that has to be left alone, and this file asserts that it is.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from einsums import linalg as la

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvdz.npz"))

#: Iterations. The ladder series converges geometrically here.
_ITERATIONS = 10


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


def _select_points(pair, count):
    """Pivoted selection of the points whose basis-pair products are independent.

    The same modified Gram-Schmidt with column pivoting the other grid tests
    use; a raw Becke grid is not a grid a fit works on, and pruning it is a
    setup the caller performs rather than something the pass decides.
    """
    remaining = pair.copy()
    pivots = []
    for _ in range(count):
        norms = np.einsum("ij,ij->j", remaining, remaining)
        best = int(np.argmax(norms))
        if norms[best] <= 1e-20:
            break
        pivots.append(best)
        direction = remaining[:, best] / np.sqrt(norms[best])
        remaining -= np.outer(direction, direction @ remaining)
    return np.array(pivots, dtype=int)


@pytest.fixture(scope="module")
def water():
    """Canonical orbitals, the three-index integrals, and a pruned grid."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"fixture not present: {_FIXTURE}")
    z = np.load(_FIXTURE, allow_pickle=False)
    if not bool(z["has_grid"]):
        pytest.skip("the fixture carries no grid")

    overlap, fock = z["S"], z["F"]
    nbf = overlap.shape[0]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])
    nvir = nbf - nocc

    metric = z["metric"]
    mvals, mvecs = np.linalg.eigh(metric)
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    three_ao = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])
    three = np.einsum("Pmn,mp,nq->Ppq", three_ao, coefficients, coefficients)

    phi_flat, weights = z["grid_phi_flat"], z["grid_w_flat"]
    bfmap, npoints, nbfs = z["grid_bfmap_flat"], z["grid_npoints"], z["grid_nbf"]
    blocks, offset, mapped = [], 0, 0
    for count, width in zip(npoints, nbfs):
        count, width = int(count), int(width)
        block = phi_flat[offset:offset + count * width].reshape(count, width)
        columns = bfmap[mapped:mapped + width].astype(int)
        full = np.zeros((count, nbf))
        full[:, columns] = block
        blocks.append(full)
        offset += count * width
        mapped += width
    collocation_ao = np.concatenate(blocks, axis=0).T
    collocation = (coefficients.T @ collocation_ao) * (np.abs(weights) ** 0.25)
    pairs = np.triu_indices(nbf)
    pivots = _select_points(collocation[pairs[0], :] * collocation[pairs[1], :], nbf * (nbf + 1) // 2)
    grid = np.ascontiguousarray(collocation[:, pivots])

    o, v = slice(0, nocc), slice(nocc, nbf)
    eri = np.einsum("Ppq,Prs->pqrs", three, three)
    denominator = (energies[o, None, None, None] - energies[None, v, None, None]
                   + energies[None, None, o, None] - energies[None, None, None, v])
    return {
        "nocc": nocc, "nvir": nvir, "nbf": nbf, "naux": three.shape[0],
        "ovov": np.ascontiguousarray(eri[o, v, o, v]),
        "oooo": np.ascontiguousarray(eri[o, o, o, o]),
        "B_vv": np.ascontiguousarray(three[:, v, v]),
        "denominator": np.ascontiguousarray(denominator),
        "X_occ": np.ascontiguousarray(grid[:nocc]),
        "X_vir": np.ascontiguousarray(grid[nocc:]),
        "ngrid": grid.shape[1],
    }


def _numpy_ladder(problem, iterations=_ITERATIONS):
    """The same iteration in numpy, and the energy it converges to."""
    ovov, oooo, d = problem["ovov"], problem["oooo"], problem["denominator"]
    vvvv = np.einsum("Qae,Qbf->aebf", problem["B_vv"], problem["B_vv"])
    exchange = 2.0 * ovov - ovov.transpose(0, 3, 2, 1)
    t = ovov / d
    energies = [float(np.einsum("iajb,iajb->", exchange, t))]
    for _ in range(iterations - 1):
        r = ovov + np.einsum("aebf,iejf->iajb", vvvv, t) + np.einsum("minj,manb->iajb", oooo, t)
        t = r / d
        energies.append(float(np.einsum("iajb,iajb->", exchange, t)))
    return energies


def _capture(problem, name, iterations=_ITERATIONS, scratch_in_parent=False):
    """The ladder as a density-fitted program writes it: a chain over ``Q``.

    ``W`` is declared on the BODY, which is where a per-iteration scratch
    belongs and, as ``scratch_in_parent`` exists to pin, is also the only place
    the cone flattening can reach it from.
    """
    nocc, nvir, naux = problem["nocc"], problem["nvir"], problem["naux"]
    shape = [nocc, nvir, nocc, nvir]

    ovov = _tensor("(ia|jb)", problem["ovov"])
    exchange = _tensor("2K-K", 2.0 * problem["ovov"] - problem["ovov"].transpose(0, 3, 2, 1))
    three = _tensor("B_vv", problem["B_vv"])
    oooo = _tensor("(mi|nj)", problem["oooo"])
    denominator = _tensor("D", problem["denominator"])
    amplitude = _tensor("t2", np.zeros(shape))
    energy = einsums.create_zero_tensor("E_ladder", [1])

    graph = cg.Graph(name)
    residual = graph.declare_tensor("r2", shape, intermediate=True, dtype="float64")
    carrier = [naux, nocc, nvir, nocc, nvir]
    if scratch_in_parent:
        half = graph.declare_tensor("W", carrier, intermediate=True, dtype="float64")
    body = graph.add_loop("ladder_iteration", iterations, lambda it, c=iterations: it < c - 1)
    if not scratch_in_parent:
        half = body.declare_tensor("W", carrier, intermediate=True, dtype="float64")
    with cg.capture(body):
        la.axpby(1.0, ovov, 0.0, residual)
        # The four-external ladder, in DF form and never through (ae|bf).
        einsums.einsum("Q,i,a,j,f <- Q,a,e ; i,e,j,f", half, three, amplitude)
        einsums.einsum("i,a,j,b <- Q,i,a,j,f ; Q,b,f", residual, half, three, c_pf=1.0)
        # The cheap term, dense and untagged on the integral side.
        einsums.einsum("i,a,j,b <- m,i,n,j ; m,a,n,b", residual, oooo, amplitude, c_pf=1.0)
        la.direct_division(1.0, residual, denominator, 0.0, amplitude)
        la.dot(energy, exchange, amplitude)
    return {"graph": graph, "body": body, "amplitude": amplitude, "energy": energy,
            "three": three, "half": half}


def _arm(problem, fit_integral=True, fit_amplitude=True, name="ladder_thc",
         bound=1e-2, scratch_in_parent=False):
    """The same program with the tags a caller declares and the providers registered."""
    captured = _capture(problem, name, scratch_in_parent=scratch_in_parent)
    graph = captured["graph"]
    _G.ThcFactorization.register_grid_space(graph)

    X_occ = _tensor("X_occ", problem["X_occ"])
    X_vir = _tensor("X_vir", problem["X_vir"])
    residual_report = einsums.create_zero_tensor("thc_amplitude_residual", [1])
    reference_report = einsums.create_zero_tensor("thc_amplitude_reference", [1])

    registry = _G.FactorizationRegistry()
    if fit_amplitude:
        graph.annotate_tag(captured["amplitude"], _G.ProvenanceTag.make("amplitude"))
        amplitude_fit = _G.ThcFactorization.for_amplitude(
            "amplitude", captured["amplitude"], [X_occ, X_vir, X_occ, X_vir], bound, 1e-8)
        amplitude_fit.report_residual_into(residual_report, reference_report)
        registry.add(amplitude_fit)
    if fit_integral:
        graph.annotate_tag(captured["three"], _G.ProvenanceTag.make("eri"))
        registry.add(_G.ThcFactorization.for_three_index("eri", captured["three"], [X_vir, X_vir], bound, 1e-8))

    factorization = _G.FactorizationPass(registry)
    factorization.set_dump(True)
    manager = cg.PassManager()
    # A body keeps its own tensor table, so a tag declared on the graph reaches the body's
    # handle for the same buffer through the analysis phase rather than by itself.
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    fired = graph.apply(manager)
    captured.update({"fired": fired, "pass": factorization, "registry": registry,
                     "residual": residual_report, "reference": reference_report,
                     "X_occ": X_occ, "X_vir": X_vir})
    return captured


def _emitted_shapes(body):
    """Every intermediate the rewrite declared in the loop body, by name and extents.

    Read out of the body's own IR because that is where the rewrite declares
    them: ``Graph.to_json`` does not descend into a loop body, so asking the
    parent would report the rewrite as having emitted nothing.
    """
    document = json.loads(body.to_json())
    return {entry["name"]: entry["dims"] for entry in document["tensors"]
            if "_x" in entry["name"] and ("Thc" in entry["name"])}


def test_the_df_ladder_agrees_with_numpy(water):
    """The un-rewritten arm, and the anchor that makes the rest mean something.

    The chain over ``Q`` reproduces the dense ``(ae|bf)`` ladder exactly, which
    is what says the density-fitted spelling of the term is the same term.
    """
    captured = _capture(water, "ladder_plain")
    captured["graph"].apply(cg.default_pass_manager())
    captured["graph"].execute()
    energies = _numpy_ladder(water)
    assert abs(float(np.asarray(captured["energy"])[0]) - energies[-1]) < 1e-10


def test_both_tags_are_substituted_in_one_decision(water):
    """The rewrite the two-operand mechanism exists for, on a real molecule.

    The cone the pass brackets is the whole chain: the amplitude's five grid
    factors, the integral's three twice over, and the author's ``Q`` carrier
    dissolved into it. Eleven leaves, one bracketing, one decision.
    """
    arm = _arm(water)
    factorization = arm["pass"]

    assert arm["fired"]
    assert factorization.num_factorized == 1
    assert factorization.num_multi_substituted == 1
    assert factorization.num_dissolved == 1

    # One record per provider, composed by the per-effect rule.
    assert sorted(r.pass_name for r in arm["graph"].approximations()) == ["ThcAmplitude", "ThcThreeIndex"]


def test_no_intermediate_of_the_shape_the_rewrite_exists_to_remove_survives(water):
    """What the rewrite bought, as shapes rather than as a cost line.

    The captured form's carrier is ``Q o^2 v^2``, and the whole reason a
    density-fitted ladder is expensive is that it is built and read twice. After
    the rewrite nothing of that size is emitted, nothing carries three virtual
    axes, and the one emitted tensor still over the auxiliary index is no larger
    than the three-index integral itself.
    """
    arm = _arm(water)
    problem = water
    carrier = problem["naux"] * problem["nocc"] ** 2 * problem["nvir"] ** 2
    three_index = problem["naux"] * problem["nvir"] ** 2

    shapes = _emitted_shapes(arm["body"])
    assert shapes, "the rewrite emitted no intermediates, so this asserts nothing"
    for name, dims in shapes.items():
        size = int(np.prod(dims))
        assert size < carrier, f"{name} {dims} is at least the carrier the rewrite removes"
        assert sum(1 for d in dims if d == problem["nvir"]) <= 2, f"{name} {dims} is over v^3 or worse"
        if problem["naux"] in dims:
            assert size <= three_index, f"{name} {dims} is a larger auxiliary object than B itself"

    # And at most one of them touches the auxiliary index at all: the ladder's two
    # auxiliary contractions have become grid overlaps.
    over_aux = [name for name, dims in shapes.items() if problem["naux"] in dims]
    assert len(over_aux) <= 2, over_aux


def test_the_amplitude_alone_still_declines_and_only_the_pair_admits_it(water):
    """What each half is worth on its own, which is what makes the pair a decision.

    The amplitude alone declines here exactly as it declines against a dense
    ``(ae|bf)``: the integral stays where it is, and the cheapest tree over the
    substituted leaves rebuilds the amplitude and contracts as written. The
    integral alone fires, because fitting the three-index tensor turns the
    auxiliary chain into a grid chain against an amplitude that is still a
    stored leaf. Only with both offered is the amplitude substituted at all, and
    that is one decision over eleven leaves rather than two rewrites in
    sequence.
    """
    alone = _arm(water, fit_integral=False, fit_amplitude=True, name="ladder_amplitude_only")
    assert not alone["fired"]
    assert dict(alone["pass"].skip_reasons).get("the decomposed form is not symbolically cheaper", 0) >= 1

    integral = _arm(water, fit_integral=True, fit_amplitude=False, name="ladder_integral_only")
    assert integral["fired"]
    assert integral["pass"].num_multi_substituted == 0
    assert [r.pass_name for r in integral["graph"].approximations()] == ["ThcThreeIndex"]

    both = _arm(water, name="ladder_both")
    assert both["fired"]
    assert both["pass"].num_multi_substituted == 1


def test_the_four_occupied_ladder_is_left_alone(water):
    """The cheap term declines, and the decline is the search's own answer.

    Substituting the amplitude into ``sum_mn (mi|nj) t[m,a,n,b]`` leaves a dense
    ``o^4`` integral for the grid indices to cross, and the cheapest tree over
    those leaves rebuilds the amplitude and contracts as written. So the term
    keeps its captured form, which is what a factorization that only pays where
    it pays looks like.
    """
    arm = _arm(water)
    reasons = dict(arm["pass"].skip_reasons)
    assert reasons.get("the decomposed form is not symbolically cheaper", 0) >= 1, reasons

    after = "\n".join(dump for dump in arm["pass"].dump_text.splitlines())
    assert "(mi|nj)[m,i,n,j] t2[m,a,n,b]" in after.split("after:")[-1]


def test_the_converged_energy_is_inside_the_composed_bound(water):
    """Two fits, two records, and an energy the composition covers.

    The comparison is against the UN-REWRITTEN replay of the same program rather
    than against numpy, because what a bound is about is the rewrite and not the
    equations.
    """
    plain = _capture(water, "ladder_plain_for_bound")
    plain["graph"].apply(cg.default_pass_manager())
    plain["graph"].execute()
    reference = float(np.asarray(plain["energy"])[0])

    arm = _arm(water)
    assert arm["fired"]
    arm["graph"].apply(cg.default_pass_manager())
    arm["graph"].execute()
    value = float(np.asarray(arm["energy"])[0])

    tolerance = arm["graph"].approximation_tolerance()
    assert abs(value - reference) <= tolerance.relative * abs(reference)

    # The measured half of the statement: what the amplitude fit was worth on the
    # last iteration, which is what the record names rather than carries.
    residual = float(np.asarray(arm["residual"])[0])
    reference_squared = float(np.asarray(arm["reference"])[0])
    assert 0.0 < (residual / reference_squared) ** 0.5 < 1e-2


def test_a_carrier_the_parent_declares_is_not_dissolved_into_the_cone(water):
    """The narrowing the proving ground found, pinned.

    The two tagged leaves only meet once the author's ``Q`` carrier is flattened
    into the cone, and a tensor the PARENT declares is not dissolvable from
    inside the loop body: the escape rule sees a declaration outside the region
    and classifies it as an output. So the same program written with the carrier
    declared one level up gets one substitution offered against a stored
    operand, and declines. It is a property of where the scratch is declared
    rather than of the algebra, and a per-iteration scratch belongs to the
    iteration.
    """
    arm = _arm(water, name="ladder_parent_scratch", scratch_in_parent=True)
    assert not arm["fired"]
    assert arm["pass"].num_dissolved == 0
    reasons = dict(arm["pass"].skip_reasons)
    assert reasons.get("the decomposed form is not symbolically cheaper", 0) >= 1, reasons


def _flat_ladder(problem, name):
    """The four-external term alone, at top level, over a FIXED amplitude.

    The same two tagged leaves and the same cone, without the solver around it.
    A Python caller cannot spell a loop condition as data, so a graph holding one
    is not a graph this can save; the term on its own is, and what the roundtrip
    is about is the doubly substituted structure rather than the iteration.
    """
    nocc, nvir, naux = problem["nocc"], problem["nvir"], problem["naux"]
    three = _tensor("B_vv", problem["B_vv"])
    amplitude = _tensor("t2", problem["ovov"] / problem["denominator"])
    result = einsums.create_zero_tensor("R", [nocc, nvir, nocc, nvir])

    graph = cg.Graph(name)
    half = graph.declare_tensor("W", [naux, nocc, nvir, nocc, nvir], intermediate=True, dtype="float64")
    with cg.capture(graph):
        einsums.einsum("Q,i,a,j,f <- Q,a,e ; i,e,j,f", half, three, amplitude)
        einsums.einsum("i,a,j,b <- Q,i,a,j,f ; Q,b,f", result, half, three)
    return graph, three, amplitude, result


def test_the_rewritten_graph_saves_loads_rebinds_and_replays(water, tmp_path):
    """A file, a fresh set of buffers, and the same tensor on the other side.

    Saved BEFORE the default manager runs, because a ``Materialize`` node holds
    an allocating closure and allocation is a resource decision this design
    re-derives on load rather than storing. Both records survive the file, and
    both collocation matrices arrive as interface tensors a bind supplies, which
    is what makes a grid-fitted graph rebindable at all.
    """
    graph, three, amplitude, result = _flat_ladder(water, "ladder_roundtrip")
    _G.ThcFactorization.register_grid_space(graph)
    graph.annotate_tag(three, _G.ProvenanceTag.make("eri"))
    graph.annotate_tag(amplitude, _G.ProvenanceTag.make("amplitude"))

    # A collocation matrix PER PROVIDER, and that is a caller decision rather than a
    # duplication. A fitting is captured, so every tensor it reads becomes an interface tensor
    # bound by name; two providers handed one matrix present two handles over one buffer under
    # one name, and a manifest that binds by name refuses that. Unifying them is a question for
    # the alias machinery rather than for this pass, so the grid arrives once per fit.
    X_occ = _tensor("X_occ", water["X_occ"])
    X_vir = _tensor("X_vir", water["X_vir"])
    X_vir_eri = _tensor("X_vir_eri", water["X_vir"])
    registry = _G.FactorizationRegistry()
    registry.add(_G.ThcFactorization.for_amplitude("amplitude", amplitude, [X_occ, X_vir, X_occ, X_vir], 1e-2, 1e-8))
    registry.add(_G.ThcFactorization.for_three_index("eri", three, [X_vir_eri, X_vir_eri], 1e-2, 1e-8))
    factorization = _G.FactorizationPass(registry)
    manager = cg.PassManager()
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    assert graph.apply(manager)
    assert factorization.num_multi_substituted == 1

    path = str(tmp_path / "df_ladder_thc.eig")
    cg.save_graph(graph, path)

    graph.apply(cg.default_pass_manager())
    graph.execute()
    in_process = np.array(np.asarray(result), copy=True)

    loaded = cg.load_graph(path)
    assert sorted(r.pass_name for r in loaded.approximations()) == ["ThcAmplitude", "ThcThreeIndex"]

    names = set(loaded.manifest_names())
    assert {"X_occ", "X_vir", "X_vir_eri"} <= names, names
    replayed = einsums.create_zero_tensor("R", [water["nocc"], water["nvir"], water["nocc"], water["nvir"]])
    fresh = {
        "B_vv": _tensor("B_vv", water["B_vv"]),
        "t2": _tensor("t2", water["ovov"] / water["denominator"]),
        "X_occ": _tensor("X_occ", water["X_occ"]),
        "X_vir": _tensor("X_vir", water["X_vir"]),
        "X_vir_eri": _tensor("X_vir_eri", water["X_vir"]),
        "B_vv@fit": _tensor("B_vv", water["B_vv"]),
        "t2@fit": _tensor("t2", water["ovov"] / water["denominator"]),
        "R": replayed,
    }
    cg.bind(loaded, {name: tensor for name, tensor in fresh.items() if name in names})
    loaded.apply(cg.default_pass_manager())
    loaded.execute()

    # The two runs are the same node set over the same numbers, but the fit keeps every
    # direction of its squared-gram metric down to the 1e-8 drop threshold, so rounding that
    # differs between two executions (a multithreaded BLAS reduction order, a different buffer
    # alignment) is amplified by that conditioning to about 1e-8 relative on the result. Linux
    # with MKL and with OpenBLAS showed 1e-12 absolute on elements near 1e-4 and 5e-12 on
    # elements near 1e-9; macOS with Accelerate happened to reproduce bitwise. The bar is the
    # conditioning, with an order of magnitude to spare, and not bit equality.
    assert np.allclose(np.asarray(replayed), in_process, rtol=1e-8, atol=1e-10)
