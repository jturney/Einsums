# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""LaplaceTransform on a real molecule's DF-MP2, offline.

``examples/psi4-bridge/df_mp2_laplace.py`` is the same program driven from psi4.
This runs it from the DLPNO fixture's own buffers, so what the example
establishes on a real system is checked wherever the fixture is, with no psi4
and no integral engine.

The energy is the full-axis one, over all four indices rather than pair by pair:

    E = sum_iajb (2 K[i,a,j,b] - K[i,b,j,a]) K[i,a,j,b] / (e_i + e_j - e_a - e_b)

with ``K = sum_Q B[Q,i,a] B[Q,j,b]``. The oracle is the same program with the
pass off, and every comparison is against the bound the pass RECORDED rather
than a tolerance picked to make it pass.

Three things beyond the arithmetic are pinned here because the real case is what
settled them: the denominator has to be built outside the capture, the density
fit does not compose with the transform in canonical MP2 and both passes say so,
and the transformed graph survives a round trip through a file and refits its
quadrature on the rebind.
"""

from __future__ import annotations

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

#: The bound is a BOUND, so the honest factor is one. It is stated rather than
#: assumed: a comparison that quietly allowed ten times the recorded number
#: would pass for a pass whose error model had come loose.
_SAFETY = 1.0

#: Floating-point rounding over an o^2 v^2 sum, which the record does not claim
#: to cover and which a comparison at 1e-9 would otherwise be dominated by.
_ROUNDING = 1e-12


@pytest.fixture(scope="module")
def water():
    """Canonical orbitals, the fitted three-index tensor, and the raw one."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"fixture not present: {_FIXTURE}")
    z = np.load(_FIXTURE, allow_pickle=True)

    overlap = _tensor("S", z["S"])
    fock = _tensor("F", z["F"])
    nbf = overlap.dim(0)
    orthogonalizer = la.pow(overlap, -0.5, 1e-10)
    half = einsums.create_zero_tensor("X'F", [nbf, nbf])
    ortho = einsums.create_zero_tensor("X'FX", [nbf, nbf])
    la.gemm(1.0, orthogonalizer, fock, 0.0, half, trans_a=True)
    la.gemm(1.0, half, orthogonalizer, 0.0, ortho)
    energies = einsums.create_zero_tensor("eps", [nbf])
    la.syev(ortho, energies, compute_eigenvectors=True)
    coefficients = einsums.create_zero_tensor("C", [nbf, nbf])
    la.gemm(1.0, orthogonalizer, ortho, 0.0, coefficients)

    nocc = int(z["C_occ"].shape[1])
    nvir = nbf - nocc
    naux = int(z["metric"].shape[0])
    eps = np.array(np.asarray(energies))
    coeff = np.array(np.asarray(coefficients))

    ao = _tensor("(Q|mn)", np.asarray(z["eri_3index"], dtype=np.float64))
    metric = _tensor("(P|Q)", np.asarray(z["metric"], dtype=np.float64))
    occupied = _tensor("C_occ", coeff[:, :nocc])
    virtual = _tensor("C_vir", coeff[:, nocc:])
    partial = einsums.create_zero_tensor("(Q|in)", [naux, nocc, nbf])
    three = einsums.create_zero_tensor("(Q|ia)", [naux, nocc, nvir])
    einsums.einsum("Q,m,n ; m,i -> Q,i,n", partial, ao, occupied)
    einsums.einsum("Q,i,n ; n,a -> Q,i,a", three, partial, virtual)
    fitted = einsums.create_zero_tensor("B", [naux, nocc, nvir])
    einsums.einsum("P,Q ; Q,i,a -> P,i,a", fitted, la.pow(metric, -0.5, 1e-10), three)

    return {
        "nocc": nocc, "nvir": nvir, "naux": naux,
        "occupied_energies": _tensor("eps_occ", eps[:nocc]),
        "virtual_energies": _tensor("eps_vir", eps[nocc:]),
        "three": three, "metric": metric, "fitted": fitted,
        "reference": float(z["energy_psi4_df_mp2"]),
    }


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


def _shape(water):
    return [water["nocc"], water["nvir"], water["nocc"], water["nvir"]]


def _tag():
    return _G.LaplaceTransform.denominator_tag(["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-")


def _denominator(water, name="D"):
    """1 / (e_i + e_j - e_a - e_b), eagerly, from the two energy vectors."""
    denominator = einsums.create_zero_tensor(name, _shape(water))
    la.outer_sum(denominator,
                 [water["occupied_energies"], water["virtual_energies"],
                  water["occupied_energies"], water["virtual_energies"]],
                 [1.0, -1.0, 1.0, -1.0])
    la.element_transform(denominator, lambda x: 1.0 / x)
    return denominator


def _capture(graph, water, denominator, energy):
    """The full-axis energy. ``K`` is formed twice because the rewrite dissolves one."""
    shape = _shape(water)
    B = water["fitted"]
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    again = graph.scratch("K_again", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", again, B, B)
        einsums.permute("iajb <- ibja", exchange, again)
        la.axpby(2.0, again, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)


def _transform(water, epsilon):
    transform = cg.LaplaceTransform()
    transform.set_epsilon(epsilon)
    transform.add_energy("eps_occ", water["occupied_energies"])
    transform.add_energy("eps_vir", water["virtual_energies"])
    return transform


def _registry(water):
    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization("eri", water["three"], water["metric"], 1e-10))
    return registry


def _exact(water):
    """The same program with no lossy pass on it."""
    energy = einsums.create_zero_tensor("E_exact", [1])
    graph = cg.Graph("mp2_exact")
    _capture(graph, water, _denominator(water), energy)
    graph.apply(cg.default_pass_manager())
    graph.execute()
    return float(np.asarray(energy)[0])


def _laplace_run(water, epsilon):
    """Capture, transform, execute. Returns the energy and the pass."""
    energy = einsums.create_zero_tensor("E_corr", [1])
    graph = cg.Graph(f"mp2_laplace_{epsilon:g}")
    denominator = _denominator(water)
    _capture(graph, water, denominator, energy)
    graph.annotate_tag(denominator, _tag())
    transform = _transform(water, epsilon)
    manager = cg.PassManager()
    manager.add(transform)
    captured = graph.num_nodes()
    assert graph.apply(manager), f"the pass declined: {transform.skip_reasons}"
    return graph, energy, transform, captured


def test_the_energy_reproduces_psi4_before_anything_is_approximated(water):
    """The program is the right program, which everything below is measured against."""
    exact = _exact(water)
    assert exact == pytest.approx(water["reference"], abs=1e-9), (
        "the full-axis form does not reproduce the fixture's DF-MP2 energy, so a later "
        "comparison against it would be measuring the wrong thing")


@pytest.mark.parametrize("epsilon", [1e-3, 1e-5, 1e-8])
def test_the_transform_fires_and_lands_inside_its_own_bound(water, epsilon):
    """The pass rewrites, and the energy it produces is inside the record it wrote."""
    exact = _exact(water)
    graph, energy, transform, captured = _laplace_run(water, epsilon)

    assert transform.num_transformed == 1
    assert transform.last_point_count >= 2
    assert transform.last_measured_error <= epsilon
    assert graph.num_nodes() > captured, "a rewrite that emitted nothing did not happen"

    records = graph.approximations()
    assert [r.pass_name for r in records] == ["LaplaceTransform"]
    record = records[0]
    assert record.origin == _G.ApproximationOrigin.Measured
    assert record.effect == _G.ApproximationEffect.NormRelative
    assert record.tolerance == pytest.approx(epsilon)
    assert record.bound <= epsilon

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = float(np.asarray(energy)[0])
    assert abs(value - exact) <= _SAFETY * record.bound * abs(exact) + _ROUNDING, (
        f"energy error {abs(value - exact):.3e} against a recorded bound of "
        f"{record.bound * abs(exact):.3e}")


def test_tightening_the_tolerance_buys_accuracy_with_points(water):
    """The knob bites on the real spectrum: the count grows and the error follows it down."""
    exact = _exact(water)
    results = []
    for epsilon in (1e-3, 1e-6, 1e-9):
        graph, energy, transform, _ = _laplace_run(water, epsilon)
        graph.apply(cg.default_pass_manager())
        graph.execute()
        results.append((transform.last_point_count, abs(float(np.asarray(energy)[0]) - exact)))

    counts = [count for count, _ in results]
    errors = [error for _, error in results]
    assert counts == sorted(counts) and counts[0] < counts[-1], (
        f"the point count did not move with the tolerance: {counts}")
    assert errors[0] > errors[1] > errors[2], f"the error did not follow the tolerance down: {errors}"


def test_the_transformed_graph_saves_loads_rebinds_and_replays(water, tmp_path):
    """A file, a fresh set of buffers, and the same energy on the other side.

    Saved BEFORE the default manager runs, because a ``Materialize`` node holds
    an allocating closure and allocation is a resource decision this design
    re-derives on load rather than storing.
    """
    exact = _exact(water)
    graph, energy, transform, _ = _laplace_run(water, 1e-6)
    path = str(tmp_path / "mp2_laplace.eig")
    cg.save_graph(graph, path)

    graph.apply(cg.default_pass_manager())
    graph.execute()
    in_process = float(np.asarray(energy)[0])

    loaded = cg.load_graph(path)
    assert [r.pass_name for r in loaded.approximations()] == ["LaplaceTransform"]
    assert loaded.approximations()[0].bound == pytest.approx(transform.last_measured_error)

    # The denominator is gone from the interface, which is the substitution
    # showing up where a caller can see it, and the energies are in it, which is
    # what lets the quadrature be refitted at whatever is bound.
    names = set(loaded.manifest_names())
    assert {"eps_occ", "eps_vir"} <= names
    assert "D" not in names

    replayed = einsums.create_zero_tensor("E_corr", [1])
    mapping = {"B": water["fitted"], "E_corr": replayed,
               "eps_occ": water["occupied_energies"], "eps_vir": water["virtual_energies"]}
    cg.bind(loaded, {name: tensor for name, tensor in mapping.items() if name in names})
    loaded.apply(cg.default_pass_manager())
    loaded.execute()

    value = float(np.asarray(replayed)[0])
    assert value == pytest.approx(in_process, abs=1e-14), (
        "the loaded graph refitted to something else on the same problem")
    record = loaded.approximations()[0]
    assert abs(value - exact) <= _SAFETY * record.bound * abs(exact) + _ROUNDING


def test_a_denominator_the_graph_builds_is_refused(water):
    """Why the denominator is built outside the capture, pinned.

    Forming it inside the graph is the first thing a caller writes, and it is
    refused: the quadrature is fitted once per bind, so a graph that recomputes
    the denominator on every replay is describing something the substitution
    cannot follow.
    """
    energy = einsums.create_zero_tensor("E_corr", [1])
    graph = cg.Graph("mp2_denominator_in_graph")
    denominator = einsums.create_zero_tensor("D", _shape(water))
    with cg.capture(graph):
        la.outer_sum(denominator,
                     [water["occupied_energies"], water["virtual_energies"],
                      water["occupied_energies"], water["virtual_energies"]],
                     [1.0, -1.0, 1.0, -1.0])
        la.element_transform(denominator, lambda x: 1.0 / x)
    _capture(graph, water, denominator, energy)
    graph.annotate_tag(denominator, _tag())

    transform = _transform(water, 1e-6)
    manager = cg.PassManager()
    manager.add(transform)
    assert not graph.apply(manager)
    assert transform.num_transformed == 0
    assert any("written by this graph" in reason for reason, _count in transform.skip_reasons)


def _dense_program(water):
    """Canonical MP2 over a stored ``(ia|jb)``, tagged on both sides."""
    shape = _shape(water)
    dense = einsums.create_zero_tensor("(ia|jb)", shape)
    einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", dense, water["fitted"], water["fitted"])

    energy = einsums.create_zero_tensor("E_corr", [1])
    denominator = _denominator(water)
    graph = cg.Graph("mp2_dense")
    T = graph.scratch("T", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    with cg.capture(graph):
        la.direct_product(1.0, dense, denominator, 0.0, T)
        einsums.permute("iajb <- ibja", exchange, dense)
        la.axpby(2.0, dense, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)
    graph.annotate_tag(dense, _G.ProvenanceTag.make("eri"))
    graph.annotate_tag(denominator, _tag())
    return graph, energy


def test_the_dense_integral_form_is_declined_by_each_pass_alone(water):
    """Canonical MP2 gives each pass nothing on its own, and both say so.

    The form a caller holding only ``(ia|jb)`` writes: the integral is read
    elementwise and through a dot, never as a contraction operand, so the fit
    has no candidate at all, and the amplitude's numerator is then a stored
    tensor rather than a contraction, so the transform has nothing to push its
    exponentials onto. Both report it.
    """
    graph, energy = _dense_program(water)

    factorization = _G.FactorizationPass(_registry(water))
    transform = _transform(water, 1e-6)
    manager = cg.PassManager()
    manager.add(factorization)
    manager.add(transform)

    assert not graph.apply(manager)
    assert factorization.num_factorized == 0
    assert transform.num_transformed == 0
    assert graph.approximations() == []
    assert any("no two-operand contraction" in reason for reason, _count in factorization.skip_reasons), (
        f"the fit passed the tag over in silence: {factorization.skip_reasons}")
    assert any("does not form" in reason for reason, _count in transform.skip_reasons), (
        f"the transform did not say why: {transform.skip_reasons}")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    assert float(np.asarray(energy)[0]) == pytest.approx(_exact(water), abs=1e-12), (
        "nothing was rewritten, so nothing may have moved")


def test_the_pair_is_costed_as_one_decision_and_declines_on_the_joint_number(water):
    """The pair is asked together, and the answer on canonical MP2 is still no.

    Handing the transform to the fit makes a tagged integral multiplied by a
    tagged denominator a candidate the fit did not have: the substituted product
    is emitted into an intermediate, the quadrature is applied to the trial, and
    the cost of the PAIR is what decides. The design expected that to accept.

    It declines, and the reason is structural rather than a threshold. The
    amplitude is a stored ``o^2 v^2`` tensor, so the decoupled form has to
    rebuild it as a sum over the quadrature index and the auxiliary one where
    the captured form built it with one elementwise multiply: the pair is one
    whole scale order worse, by the point count times the auxiliary dimension.
    What would pay is never forming the amplitude at all, which needs the energy
    expression's own contraction re-associated and is a different rewrite.

    Pinned because the decline is now INFORMATIVE where it used to be two
    unrelated silences: one line, naming the pair and both costs.
    """
    graph, energy = _dense_program(water)

    factorization = _G.FactorizationPass(_registry(water))
    transform = _transform(water, 1e-6)
    factorization.set_laplace_transform(transform)
    manager = cg.PassManager()
    manager.add(factorization)

    assert not graph.apply(manager)
    assert factorization.num_factorized == 0
    assert factorization.num_joint == 0
    assert graph.approximations() == [], "a declined pair must record no approximation"

    # The joint verdict, not either half's: the fit now HAS a candidate and the
    # quadrature was fitted on the trial before anything was costed.
    assert any("the fit and the quadrature together" in reason
               for reason, _count in factorization.skip_reasons), (
        f"the pair was not costed as one decision: {factorization.skip_reasons}")
    assert not any("no two-operand contraction" in reason
                   for reason, _count in factorization.skip_reasons), (
        "the tag was reported unclaimed, so the joint candidate was never formed")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    assert float(np.asarray(energy)[0]) == pytest.approx(_exact(water), abs=1e-12), (
        "nothing was rewritten, so nothing may have moved")


def test_a_fit_and_a_transform_compose_on_one_output(water):
    """Two lossy records on one real result, from this molecule's own integrals.

    ``G[i,a] = sum_jb (ia|jb) U[j,b]`` divided by ``e_i - e_a``: the orbital
    response update, which is the MP2-adjacent shape where the integral IS a
    contraction operand. The fit re-associates around its factors and the
    transform then rides on the factor carrying i and a.

    The graph is annotated for its family, which is what makes the fit's numeric
    veto abstain: 84 auxiliary directions against 95 occupied-virtual pairs is
    not cheaper at THIS size and is cheaper at every interesting one, and an
    extent the graph declares resizable is not the one to decide on.
    """
    nocc, nvir = water["nocc"], water["nvir"]
    dense = einsums.create_zero_tensor("(ia|jb)", [nocc, nvir, nocc, nvir])
    einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", dense, water["fitted"], water["fitted"])

    trial = einsums.create_zero_tensor("U", [nocc, nvir])
    np.asarray(trial)[...] = np.random.default_rng(20260904).standard_normal((nocc, nvir))
    gaps = einsums.create_zero_tensor("D_ov", [nocc, nvir])
    la.outer_sum(gaps, [water["occupied_energies"], water["virtual_energies"]], [1.0, -1.0])
    la.element_transform(gaps, lambda x: 1.0 / x)
    oracle = np.einsum("iajb,jb->ia", np.asarray(dense), np.asarray(trial)) * np.asarray(gaps)

    update = einsums.create_zero_tensor("X", [nocc, nvir])
    graph = cg.Graph("orbital_response")
    contracted = graph.scratch("G", [nocc, nvir], "float64")
    with cg.capture(graph):
        einsums.einsum("i,a,j,b ; j,b -> i,a", contracted, dense, trial)
        la.direct_product(1.0, contracted, gaps, 0.0, update)
    graph.annotate_tag(dense, _G.ProvenanceTag.make("eri"))
    graph.annotate_tag(gaps, _G.LaplaceTransform.denominator_tag(["eps_occ", "eps_vir"], "+-"))
    for tensor, symbols in ((dense, ["nocc", "nvir", "nocc", "nvir"]), (trial, ["nocc", "nvir"]),
                            (gaps, ["nocc", "nvir"]), (update, ["nocc", "nvir"])):
        graph.annotate_dims(tensor, symbols)

    factorization = _G.FactorizationPass(_registry(water))
    transform = _transform(water, 1e-6)
    manager = cg.PassManager()
    manager.add(factorization)
    manager.add(transform)
    assert graph.apply(manager)
    assert factorization.num_factorized == 1, f"the fit did not fire: {factorization.skip_reasons}"
    assert transform.num_transformed == 1, f"the transform did not fire: {transform.skip_reasons}"

    names = sorted(r.pass_name for r in graph.approximations())
    assert names == ["LaplaceTransform", "MetricFit"]

    # Composed, not summed: both are relative, so the bound is e1 + e2 + e1*e2.
    bounds = [r.bound for r in graph.approximations()]
    composed = bounds[0] + bounds[1] + bounds[0] * bounds[1]
    tolerance = graph.approximation_tolerance("X")
    assert tolerance.relative == pytest.approx(composed)

    graph.apply(cg.default_pass_manager())
    graph.execute()
    scale = float(np.max(np.abs(oracle)))
    deviation = float(np.max(np.abs(np.asarray(update) - oracle)))
    assert deviation <= _SAFETY * composed * scale + _ROUNDING, (
        f"deviation {deviation:.3e} against a composed bound of {composed * scale:.3e}")


def test_the_veto_stands_where_nothing_is_annotated(water):
    """Without the family annotation the fit declines on the size in hand, and says so."""
    nocc, nvir = water["nocc"], water["nvir"]
    dense = einsums.create_zero_tensor("(ia|jb)", [nocc, nvir, nocc, nvir])
    einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", dense, water["fitted"], water["fitted"])
    trial = einsums.create_zero_tensor("U", [nocc, nvir])
    np.asarray(trial)[...] = 1.0
    out = einsums.create_zero_tensor("G", [nocc, nvir])

    graph = cg.Graph("orbital_response_unannotated")
    with cg.capture(graph):
        einsums.einsum("i,a,j,b ; j,b -> i,a", out, dense, trial)
    graph.annotate_tag(dense, _G.ProvenanceTag.make("eri"))

    factorization = _G.FactorizationPass(_registry(water))
    manager = cg.PassManager()
    manager.add(factorization)
    assert not graph.apply(manager)
    assert any("not cheaper at the extents" in reason for reason, _count in factorization.skip_reasons)
