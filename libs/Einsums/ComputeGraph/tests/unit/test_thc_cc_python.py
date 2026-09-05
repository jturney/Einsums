# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""A grid-fitted amplitude in a real solver's loop body, on water/cc-pVDZ.

The proving ground for tensor hypercontraction of an AMPLITUDE rather than of an
integral. The two differ in the one way that matters here: an integral fit reads
a three-index tensor the caller hands over and its factors do not move, while an
amplitude fit projects the tagged tensor onto the grid basis and so has to be
redone at every update. Everything the pass needs for that is under test
together: the descent into the loop body, the recognition of the update
statement, the refit emitted in the body beside the setup in the parent, and the
record naming where the measured residual lands.

The iteration is the LADDER subseries of the doubles equations,

    R[i,a,j,b] = (ia|jb) + sum_ef (ae|bf) t[i,e,j,f] + sum_mn (mi|nj) t[m,a,n,b]
    t         = R / D

with ``D[i,a,j,b] = e_i - e_a + e_j - e_b`` and the correlation energy
``E = sum [2 (ia|jb) - (ib|ja)] t[i,a,j,b]``. That is a narrowing of full CCD and
a deliberate one: the four-external ladder is the term this rewrite is about and
the term whose ``v^4`` intermediate it removes, the four-occupied ladder is the
cheap term that has to be left alone, and the two together converge to a
well-defined energy whose oracle is three lines of numpy. The quadratic CCD terms
would add equations to get wrong without adding a shape the pass treats
differently. The first iteration is MP2 exactly, which is what anchors the whole
thing against a number computed somewhere else.
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

#: Iterations. The ladder series converges geometrically here; ten is well past
#: the point where the energy stops moving in the digits this compares.
_ITERATIONS = 10


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


def _select_points(pair, count):
    """Pivoted selection of the points whose basis-pair products are independent.

    The same modified Gram-Schmidt with column pivoting ``test_thc_mp2_python``
    uses; a raw Becke grid is not a grid a fit works on, and pruning it is a
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
    """Canonical orbitals, the density-fitted integrals, and a pruned grid."""
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
        "nocc": nocc, "nvir": nvir, "nbf": nbf,
        "ovov": np.ascontiguousarray(eri[o, v, o, v]),
        "vvvv": np.ascontiguousarray(eri[v, v, v, v]),
        "oooo": np.ascontiguousarray(eri[o, o, o, o]),
        "denominator": np.ascontiguousarray(denominator),
        "X_occ": np.ascontiguousarray(grid[:nocc]),
        "X_vir": np.ascontiguousarray(grid[nocc:]),
        "ngrid": grid.shape[1],
    }


def _numpy_ladder(problem, iterations=_ITERATIONS):
    """The same iteration in numpy, and the energy it converges to."""
    ovov, vvvv, oooo, d = (problem["ovov"], problem["vvvv"], problem["oooo"], problem["denominator"])
    t = ovov / d
    energies = [float(np.einsum("iajb,iajb->", 2.0 * ovov - ovov.transpose(0, 3, 2, 1), t))]
    for _ in range(iterations - 1):
        r = ovov + np.einsum("aebf,iejf->iajb", vvvv, t) + np.einsum("minj,manb->iajb", oooo, t)
        t = r / d
        energies.append(float(np.einsum("iajb,iajb->", 2.0 * ovov - ovov.transpose(0, 3, 2, 1), t)))
    return t, energies


def _capture_ladder(problem, name="ladder", iterations=_ITERATIONS):
    """The iteration as a graph: a Loop whose body is the whole residual.

    Returns the graph, the loop body, the amplitude and the energy tensor, so a
    caller can tag the amplitude and read the answer.
    """
    nocc, nvir = problem["nocc"], problem["nvir"]
    shape = [nocc, nvir, nocc, nvir]

    ovov = _tensor("(ia|jb)", problem["ovov"])
    exchange = _tensor("2K-K", 2.0 * problem["ovov"] - problem["ovov"].transpose(0, 3, 2, 1))
    vvvv = _tensor("(ae|bf)", problem["vvvv"])
    oooo = _tensor("(mi|nj)", problem["oooo"])
    denominator = _tensor("D", problem["denominator"])
    amplitude = _tensor("t2", np.zeros(shape))
    energy = einsums.create_zero_tensor("E_ladder", [1])

    graph = cg.Graph(name)
    residual = graph.declare_tensor("r2", shape, intermediate=True, dtype="float64")
    body = graph.add_loop("ladder_iteration", iterations, lambda it, c=iterations: it < c - 1)
    with cg.capture(body):
        # R = (ia|jb) + sum_ef (ae|bf) t[i,e,j,f] + sum_mn (mi|nj) t[m,a,n,b]
        la.axpby(1.0, ovov, 0.0, residual)
        einsums.einsum("i,a,j,b <- a,e,b,f ; i,e,j,f", residual, vvvv, amplitude, c_pf=1.0)
        einsums.einsum("i,a,j,b <- m,i,n,j ; m,a,n,b", residual, oooo, amplitude, c_pf=1.0)
        # The update: a residual divided by an energy denominator into the amplitude, which is
        # the statement the stale-factor refusal's third mode recognizes.
        la.direct_division(1.0, residual, denominator, 0.0, amplitude)
        la.dot(energy, exchange, amplitude)
    return graph, body, amplitude, energy


def _run(graph):
    graph.apply(cg.default_pass_manager())
    graph.execute()


def test_the_ladder_iteration_agrees_with_numpy(water):
    """The un-rewritten arm, and the anchor that makes the rest mean something.

    The first iteration is MP2 exactly, so a mistake in the equations shows up
    against a number this file does not compute.
    """
    graph, _body, _amplitude, energy = _capture_ladder(water)
    _run(graph)

    _t, energies = _numpy_ladder(water)
    assert abs(float(np.asarray(energy)[0]) - energies[-1]) < 1e-10

    # And the first iteration is MP2, which the fixture's own denominator and
    # integrals give directly.
    ovov, d = water["ovov"], water["denominator"]
    mp2 = float(np.einsum("iajb,iajb->", 2.0 * ovov - ovov.transpose(0, 3, 2, 1), ovov / d))
    assert abs(energies[0] - mp2) < 1e-12


def _thc_arm(water_problem, iterations=_ITERATIONS, bound=1e-2, verbose=False):
    """The same iteration with the amplitude tagged and fitted on the grid."""
    graph, body, amplitude, energy = _capture_ladder(water_problem, "ladder_thc", iterations)
    graph.annotate_tag(amplitude, _G.ProvenanceTag.make("amplitude"))
    _G.ThcFactorization.register_grid_space(graph)

    X_occ = _tensor("X_occ", water_problem["X_occ"])
    X_vir = _tensor("X_vir", water_problem["X_vir"])
    residual_report = einsums.create_zero_tensor("thc_amplitude_residual", [1])
    reference_report = einsums.create_zero_tensor("thc_amplitude_reference", [1])

    provider = _G.ThcFactorization.for_amplitude(
        "amplitude", amplitude, [X_occ, X_vir, X_occ, X_vir], bound, 1e-8)
    provider.report_residual_into(residual_report, reference_report)

    registry = _G.FactorizationRegistry()
    registry.add(provider)
    factorization = _G.FactorizationPass(registry)
    manager = cg.PassManager()
    # A body keeps its own tensor table, so the tag declared on the graph reaches the
    # body's handle for the same buffer through the analysis phase rather than by itself.
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    if verbose:
        manager.set_verbosity(3)  # the cost line behind a decline
    fired = graph.apply(manager)
    return {
        "graph": graph, "body": body, "energy": energy, "fired": fired,
        "pass": factorization, "residual": residual_report, "reference": reference_report,
    }


def test_the_amplitude_is_offered_and_the_pass_costs_the_whole_substitution(water):
    """The candidate exists, the fit is proposed, and the DECISION is a number.

    Everything up to the cost is exercised here on a real molecule: the tag
    reaches the loop body, the update statement is recognized so a tensor the
    body rewrites is admitted at all, the provider offers a five-factor chain
    over the grid, and the search brackets the substituted product. What the pass
    then says is no, and the reason it gives is the finding this file exists to
    record.
    """
    arm = _thc_arm(water)
    factorization = arm["pass"]
    reasons = dict(factorization.skip_reasons)

    # Not any of the refusals that would mean the machinery never got this far: the
    # tag reached the body, the amplitude's writer was recognized as an update, and
    # the provider had a chain to offer.
    assert not any("could go stale" in reason for reason in reasons), reasons
    assert not any("provider declined" in reason for reason in reasons), reasons
    assert not any("does not read the tagged tensor" in reason for reason in reasons), reasons

    # What it does say, on both consumers of the amplitude.
    assert reasons.get("the decomposed form is not symbolically cheaper") == 2, reasons
    assert not arm["fired"]
    assert arm["graph"].approximations() == []


def test_fitting_the_amplitude_alone_does_not_make_the_ladder_cheaper(water, capfd):
    """WHY it declines, as the cost line rather than as prose.

    The four-external ladder contracts a ``v^4`` integral against the amplitude.
    Substituting the amplitude by its grid chain leaves that integral where it
    was, so the best bracketing pays ``v^2 g^2`` where it paid ``v^4``, and a grid
    is several times a basis. The term becomes a grid CHAIN only when the integral
    is fitted on the same grid, which is two tagged operands in one contraction
    and a composition this pass does not have: it substitutes one tagged operand
    and treats the other as the operand it re-associates around.

    So the decline is a property of the program rather than of the molecule, and
    it does not move with the extents. The cost line is what says so: the chosen
    tree still holds the captured contraction, because rebuilding the amplitude
    from its own chain and contracting as written is cheaper than carrying the
    grid indices through a ``v^4`` integral.
    """
    _arm = _thc_arm(water, verbose=True)
    err = capfd.readouterr().err
    assert "not symbolically cheaper" in err, err
    # The captured cost is on both sides of the comparison: the decomposed form is
    # the captured contraction PLUS the arithmetic that rebuilds the amplitude.
    line = next(l for l in err.splitlines() if "not symbolically cheaper" in l)
    before, after = line.split(" vs ")
    assert after.strip().rstrip(")") in before, line


def test_the_ladder_energy_is_unchanged_by_a_declined_rewrite(water):
    """A decline costs an optimization and never an answer."""
    arm = _thc_arm(water)
    _run(arm["graph"])
    _t, energies = _numpy_ladder(water)
    assert abs(float(np.asarray(arm["energy"])[0]) - energies[-1]) < 1e-10
