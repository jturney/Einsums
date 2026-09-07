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
    """The full-axis energy, over ONE integral that four statements read."""
    shape = _shape(water)
    B = water["fitted"]
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.permute("iajb <- ibja", exchange, K)
        la.axpby(2.0, K, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)


def _capture_over_two_integrals(graph, water, denominator, energy):
    """The same energy written with a second copy of the integral, which is also correct.

    Every ground here wrote this until the transform learned to copy the numerator
    it rides on, and a caller who reads that older advice, or who simply has two
    integrals in hand, must get the same answer out. That is what the case over
    this spelling asserts; nothing else uses it.
    """
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


def _searching_manager(*passes, budget_ms=0):
    """A pipeline whose search is allowed to finish.

    What the cases below assert is the TREE the search emits, and a wall-clock
    allowance makes that tree a function of how fast the machine is: a search
    that runs out of time keeps the best candidate it has found so far and
    reports having been cut off, which on a slower runner is a different and
    worse tree over the same program. So a case that pins a shape removes the
    allowance, and ``einsums:graph:optimizer-budget`` is what it removes; the
    per-pipeline setting wins over the option, which is what lets one file do
    this without touching process-global configuration.

    ``budget_ms`` is for the one case that pins the other side of that rule.
    """
    manager = cg.PassManager()
    manager.set_optimizer_budget(budget_ms)
    for entry in passes:
        manager.add(entry)
    return manager


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


def _laplace_run(water, epsilon, capture=None):
    """Capture, transform, execute. Returns the energy and the pass."""
    energy = einsums.create_zero_tensor("E_corr", [1])
    graph = cg.Graph(f"mp2_laplace_{epsilon:g}")
    denominator = _denominator(water)
    (capture or _capture)(graph, water, denominator, energy)
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


def test_a_program_that_writes_the_integral_twice_gets_the_same_rewrite(water):
    """The older spelling is still a correct program and still gets the same answer.

    The transform copies the numerator it rides on, so one integral is what a
    caller should write and what every ground here writes. A caller who writes
    two is not wrong: the second definition is the copy the pass would otherwise
    have taken, so what they must get is the same rewrite at the same point count
    and the same energy, from a capture that is one node larger and an emitted
    graph that is not.
    """
    one_graph, one_energy, one_pass, one_captured = _laplace_run(water, 1e-6)
    two_graph, two_energy, two_pass, two_captured = _laplace_run(water, 1e-6, _capture_over_two_integrals)

    assert one_pass.num_numerator_copies == 1
    assert two_pass.num_numerator_copies == 0, "a private numerator is dissolved, not copied"
    assert two_captured == one_captured + 1, "the second integral is one captured node"
    assert two_graph.num_nodes() == one_graph.num_nodes()
    assert one_pass.last_point_count == two_pass.last_point_count

    for graph in (one_graph, two_graph):
        graph.apply(cg.default_pass_manager())
        graph.execute()
    assert float(np.asarray(two_energy)[0]) == float(np.asarray(one_energy)[0]), (
        "the two spellings are the same arithmetic and must give the same bits")


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


def _build_denominator(graph, water, denominator, coefficients=(1.0, -1.0, 1.0, -1.0), op="recip"):
    """The chain the pass verifies, captured: an outer sum and a reciprocal."""
    with cg.capture(graph):
        la.outer_sum(denominator,
                     [water["occupied_energies"], water["virtual_energies"],
                      water["occupied_energies"], water["virtual_energies"]],
                     list(coefficients))
        la.element_transform(denominator, op)


def test_the_denominator_may_be_a_deferred_intermediate_the_graph_builds(water):
    """The chain the pass can read is accepted, dissolved, and never allocated.

    The first thing a caller writes is the denominator inside the capture, and
    it used to be refused outright. The refusal's argument is that a graph
    recomputing the denominator on every replay disagrees with a quadrature
    fitted once per bind, and it holds for a recipe the pass cannot read. It does
    not hold for the one the tag already describes, so that one is verified
    against the nodes and dissolved with the tensor.
    """
    exact = _exact(water)
    energy = einsums.create_zero_tensor("E_deferred", [1])
    graph = cg.Graph("mp2_denominator_deferred")
    denominator = graph.scratch("D", _shape(water), "float64")
    _build_denominator(graph, water, denominator)
    _capture(graph, water, denominator, energy)
    graph.annotate_tag(denominator, _tag())

    transform = _transform(water, 1e-8)
    manager = cg.PassManager()
    manager.add(transform)
    assert graph.apply(manager), f"the pass declined: {transform.skip_reasons}"
    assert transform.num_transformed == 1

    kinds = [node["kind"] for node in json.loads(graph.to_json())["nodes"]]
    assert "ElementTransform" not in kinds, "the reciprocal survived the dissolution"

    graph.apply(cg.default_pass_manager())
    materialized = {node["label"] for node in json.loads(graph.to_json())["nodes"]
                    if node["kind"] == "Materialize"}
    assert "materialize(D)" not in materialized, materialized

    graph.execute()
    value = float(np.asarray(energy)[0])
    record = graph.approximations()[0]
    assert abs(value - exact) <= _SAFETY * record.bound * abs(exact) + _ROUNDING


@pytest.mark.parametrize("label,coefficients,op,extra", [
    ("a sign the tag does not say", (1.0, 1.0, 1.0, -1.0), "recip", False),
    ("an element operation that is not the reciprocal", (1.0, -1.0, 1.0, -1.0), "negate", False),
    ("a third writer", (1.0, -1.0, 1.0, -1.0), "recip", True),
])
def test_a_denominator_with_a_writer_the_pass_cannot_verify_is_refused(water, label, coefficients, op, extra):
    """Everything the tag describes is CHECKED against the nodes, not taken on its word."""
    energy = einsums.create_zero_tensor("E_corr", [1])
    graph = cg.Graph(f"mp2_denominator_{label}")
    denominator = graph.scratch("D", _shape(water), "float64")
    _build_denominator(graph, water, denominator, coefficients, op)
    if extra:
        with cg.capture(graph):
            la.scale(1.0, denominator)
    _capture(graph, water, denominator, energy)
    graph.annotate_tag(denominator, _tag())

    transform = _transform(water, 1e-6)
    manager = cg.PassManager()
    manager.add(transform)
    assert not graph.apply(manager)
    assert transform.num_transformed == 0
    assert any("cannot verify" in reason for reason, _count in transform.skip_reasons), (
        transform.skip_reasons)


def test_an_anonymous_reciprocal_is_not_a_recipe_the_pass_can_read(water):
    """A Python callable says only that something is applied to every element."""
    energy = einsums.create_zero_tensor("E_corr", [1])
    graph = cg.Graph("mp2_denominator_lambda")
    denominator = graph.scratch("D", _shape(water), "float64")
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
    assert any("cannot verify" in reason for reason, _count in transform.skip_reasons), (
        transform.skip_reasons)


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

    REVISITED 2026-09-05, and it stands. The expectation was that admitting the
    direct product and the dot to the flattener would remove the decline's
    premise, since the amplitude is then no longer a stored leaf. It does not,
    and the reason is where the two decisions are taken: this one is
    ``FactorizationPass`` costing the tagged tensor's own CONE, which is the
    direct product and nothing else, and the amplitude is materialized there
    whatever a later pass would do with it. What the flattener changed is the
    other half of the sentence this decline already carried, that what would pay
    is never forming the amplitude at all: ``MultiTermFactorization`` can now
    reach that, and the opposite-spin proving ground below is it doing so. The
    two are complementary rather than one superseding the other.
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
    """Without the family annotation the fit declines on the size in hand, and says so.

    WHICH check applies those extents moved. The accept comparison sets bound extents now, as
    the rung below scale order and typical extents, so a form that is more expensive at the
    sizes this graph holds is refused by the comparison itself and the separate veto below it
    is never reached. The verdict and the extents behind it are the same; the line naming them
    is the comparison's.
    """
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
    assert any("not symbolically cheaper" in reason for reason, _count in factorization.skip_reasons)


# ──────────────────────────────────────────────────────────────────────────
# The opposite-spin proving ground
#
# SOS-MP2 written over all four indices, and what the optimizer makes of it.
# The target is the pairless form: with the denominator decoupled, the energy
# is one nine-factor product and its cheapest bracketing contracts the orbital
# indices away first, leaving a Q-by-Q matrix per quadrature point and no tensor
# over o and v at all.
#
# The oracle is the PAIR-DRIVEN energy, which is the memory-optimal spelling and
# the one production uses; the full-axis form is a correctness proving ground
# until the tiling pass can derive the pair loop from it.
# ──────────────────────────────────────────────────────────────────────────

#: The family the caller declares, and the whole reason the rewrite is taken.
#:
#: Decoupling trades ``o^2 v^2 Q`` for ``o v Q^2 t``, so it pays exactly where
#: the auxiliary dimension is smaller than the occupied-virtual product by more
#: than the point count. At water/cc-pVDZ it is NOT: 84 auxiliary directions
#: against 95 occupied-virtual pairs, so the rewrite is a loss at this size and
#: the pass says so. These extents are a system of a few thousand basis
#: functions, which is where SOS-MP2 is used, and declaring them is how a caller
#: says which regime they are in. It is the same abstention the density fit's
#: extent veto makes over an annotated axis.
_SOS_FAMILY = (("occ", "o", 300.0, "nocc"), ("vir", "v", 2700.0, "nvir"), ("aux", "x", 9000.0, "naux"))


def _sos_registry():
    registry = cg.SpaceRegistry()
    for name, symbol, extent, dim in _SOS_FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    return registry


def _sos_capture(graph, water, energy):
    """``E = sum_iajb (ia|jb)^2 / D``, over all four indices and ONE integral.

    The direct product and the dot read the same ``K``. The transform takes its
    own copy of the numerator it rewrites and leaves that definition standing for
    the dot, and the search then inlines it, so the rewritten graph forms the
    integral once and then not at all.
    """
    shape = _shape(water)
    B = water["fitted"]
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    denominator = graph.scratch("D", shape, "float64")
    _build_denominator(graph, water, denominator)
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        la.dot(energy, K, T)
    graph.annotate_tag(denominator, _tag())
    return denominator, (B, K, T)


def _annotate_sos(graph, water, tensors, denominator):
    B, K, T = tensors
    cg.annotate(B, ("aux", "occ", "vir"), graph=graph)
    # The DENOMINATOR too, and not for symmetry: the transform gives its
    # exponentials the quadrature space and the axis they were built from, all
    # axes or none, so an unannotated denominator leaves the quadrature letter
    # anonymous and one anonymous letter blocks the family's extents for the
    # whole polynomial it appears in.
    for tensor in (K, T, denominator):
        cg.annotate(tensor, ("occ", "vir", "occ", "vir"), graph=graph)


def _opposite_spin_pair_driven(water):
    """The oracle: the same energy pair by pair, never forming a four-index tensor."""
    B = water["fitted"]
    nocc, nvir = water["nocc"], water["nvir"]
    occupied = np.asarray(water["occupied_energies"])
    gaps = einsums.create_zero_tensor("-ea-eb", [nvir, nvir])
    la.outer_sum(gaps, [water["virtual_energies"], water["virtual_energies"]], [-1.0, -1.0])

    integral = einsums.create_zero_tensor("K_ij", [nvir, nvir])
    squared = einsums.create_zero_tensor("K2_ij", [nvir, nvir])
    weights = einsums.create_zero_tensor("W_ij", [nvir, nvir])
    partial = einsums.create_zero_tensor("e_ij", [1])

    total = 0.0
    for i in range(nocc):
        left = B[:, i, :]
        for j in range(nocc):
            einsums.einsum("Q,a ; Q,b -> a,b", integral, left, B[:, j, :])
            la.axpby(1.0, gaps, 0.0, weights)
            shift = float(occupied[i] + occupied[j])
            la.element_transform(weights, lambda x, s=shift: 1.0 / (x + s))
            la.direct_product(1.0, integral, integral, 0.0, squared)
            la.dot(partial, squared, weights)
            total += float(np.asarray(partial)[0])
    return total


def _sos_run(water, epsilon, annotate):
    energy = einsums.create_zero_tensor(f"E_sos_{epsilon:g}_{annotate}", [1])
    graph = cg.Graph(f"sos_{epsilon:g}_{annotate}")
    registry = _sos_registry()
    graph.set_space_registry(registry)
    denominator, tensors = _sos_capture(graph, water, energy)
    if annotate:
        _annotate_sos(graph, water, tensors, denominator)
    captured = graph.num_nodes()

    transform = _transform(water, epsilon)
    search = cg.MultiTermFactorization()
    search.set_search_enabled(True)
    manager = _searching_manager(transform, search)
    manager.run(graph)
    assert not search.was_cut_off, "the search was cut off, so the tree below is the machine's"
    return graph, energy, transform, search, captured, registry


def _written_dims(graph):
    """The shape of every tensor some node in the optimized graph writes."""
    ir = json.loads(graph.to_json())
    dims = {tensor["id"]: tensor["dims"] for tensor in ir["tensors"]}
    out = []
    for node in ir["nodes"]:
        for tid in node.get("outputs", []):
            if tid in dims:
                out.append(dims[tid])
    return out


def _integrals_formed(graph, water):
    """Every contraction in the optimized graph that BUILDS ``(ia|jb)``.

    Named by what it WRITES, since what is under test is how many of them there
    are: a program that formed the integral once and then rebuilt the same
    product inside a consumer reads as one definition and two contractions.

    Recognized by the OUTPUT it writes and not by its operands alone, which is
    what the shape test below is for. A pairless bracketing contracts the fitted
    factor with itself over the pair index and writes a ``[Q, Q']`` or a
    ``[Q, i, Q']`` intermediate, which is the form the whole rewrite exists to
    reach and which forms no integral at all; a rule reading "both operands are
    the fitted factor" counts that as one and reports the successful rewrite as
    the failure it is the opposite of.

    A node forms the integral, then, when it writes a rank-four tensor over the
    occupied and virtual axes OUT OF the density fit. The shape is what says it
    is the integral rather than a decoupled product over the quadrature, which
    also lands on those four axes and is built from no fitted factor at all; the
    operand is what says the value was BUILT here rather than transformed, which
    keeps the permute of the integral and the elementwise products over it out.
    """
    ir = json.loads(graph.to_json())
    names_by_id = {tensor["id"]: tensor["name"] for tensor in ir["tensors"]}
    dims_by_id = {tensor["id"]: tensor["dims"] for tensor in ir["tensors"]}
    four_index = _shape(water)
    written = []
    for node in ir["nodes"]:
        if node["kind"] != "Einsum":
            continue
        if "B" not in [names_by_id.get(t) for t in node.get("inputs", [])]:
            continue
        written += [names_by_id.get(t) for t in node.get("outputs", [])
                    if dims_by_id.get(t) == four_index]
    return written


def _readers_of(graph, name):
    """The kind of every node in the optimized graph that reads the tensor called @p name."""
    ir = json.loads(graph.to_json())
    wanted = {tensor["id"] for tensor in ir["tensors"] if tensor["name"] == name}
    return sorted(node["kind"] for node in ir["nodes"] if wanted & set(node.get("inputs", [])))


def test_the_transform_copies_a_numerator_the_dot_also_reads(water):
    """One integral, read by the direct product and by the dot, and the pass copies it.

    The rewrite dissolves the numerator it rides on, because the exponentials go
    onto that numerator's factors and the value is never formed again. A program
    that writes one integral and reads it twice still has to keep it, so the pass
    takes a COPY of the definition and leaves the original standing: the dot's
    operand is the tensor the author wrote, and the transform rewrites the
    product over the two factors behind it.

    What the copy costs is nothing in the graph that runs. The transform dissolves
    the copy in the same rewrite that makes it, so the integral is formed exactly
    once after the transform, and the search then inlines that one definition into
    the dot and reaches the pairless chain over the density-fitting factor itself,
    which is zero.
    """
    oracle = _opposite_spin_pair_driven(water)
    energy = einsums.create_zero_tensor("E_one_integral", [1])
    graph = cg.Graph("sos_one_integral")
    graph.set_space_registry(_sos_registry())

    shape = _shape(water)
    B = water["fitted"]
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    denominator = graph.scratch("D", shape, "float64")
    _build_denominator(graph, water, denominator)
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        la.dot(energy, K, T)
    graph.annotate_tag(denominator, _tag())
    cg.annotate(B, ("aux", "occ", "vir"), graph=graph)
    for tensor in (K, T, denominator):
        cg.annotate(tensor, ("occ", "vir", "occ", "vir"), graph=graph)

    transform = _transform(water, 1e-8)
    manager = cg.PassManager()
    manager.add(transform)
    assert graph.apply(manager), f"the pass declined: {transform.skip_reasons}"
    assert transform.num_transformed == 1
    assert transform.num_numerator_copies == 1, transform.skip_reasons

    # The definition stands, once, and the dot still reads it.
    assert _integrals_formed(graph, water) == ["K"], _integrals_formed(graph, water)
    assert _readers_of(graph, "K") == ["Dot"], _readers_of(graph, "K")

    search = cg.MultiTermFactorization()
    search.set_search_enabled(True)
    after = _searching_manager(search)
    assert graph.apply(after), search.skip_reasons
    assert not search.was_cut_off, "the search was cut off, so the tree below is the machine's"
    assert _integrals_formed(graph, water) == [], _integrals_formed(graph, water)

    four_index = [water["nocc"], water["nvir"], water["nocc"], water["nvir"]]
    written = _written_dims(graph)
    assert four_index not in written, f"a tensor over o and v survived the rewrite: {written}"
    naux, points = water["naux"], transform.last_point_count
    assert any(sorted(shape_) == sorted([naux, naux, points]) for shape_ in written), (
        f"the pairless chain was not reached: {written}")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = float(np.asarray(energy)[0])
    record = graph.approximations()[0]
    assert abs(value - oracle) <= _SAFETY * record.bound * abs(oracle) + _ROUNDING, (
        f"the pairless energy {value} is outside the recorded bound against the "
        f"pair-driven {oracle}")


def test_the_opposite_spin_energy_reaches_the_pairless_form(water):
    """No four-index tensor survives, and a Q-by-Q-by-points object appears."""
    oracle = _opposite_spin_pair_driven(water)
    graph, energy, transform, search, captured, _registry = _sos_run(water, 1e-8, annotate=True)

    assert transform.num_transformed == 1
    assert search.num_rebracketed == 1, search.skip_reasons

    points = transform.last_point_count
    four_index = [water["nocc"], water["nvir"], water["nocc"], water["nvir"]]
    written = _written_dims(graph)
    assert four_index not in written, (
        f"a tensor over o and v survived the rewrite: {written}")

    naux = water["naux"]
    assert any(sorted(shape) == sorted([naux, naux, points]) for shape in written), (
        f"no Q by Q by points object was emitted: {written}")

    # The emitted count does not move with the point count, which is the transform's own
    # property surviving the re-association: one contraction sums over the quadrature letter.
    emitted = graph.num_nodes()
    assert emitted == 8, f"{captured} captured, {emitted} emitted"

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = float(np.asarray(energy)[0])
    record = graph.approximations()[0]
    assert abs(value - oracle) <= _SAFETY * record.bound * abs(oracle) + _ROUNDING, (
        f"the pairless energy {value} is outside the recorded bound against the "
        f"pair-driven {oracle}")


def test_the_pairless_form_is_declined_where_it_does_not_pay(water):
    """Unannotated, the comparison falls back to the extents this capture has.

    At water/cc-pVDZ the auxiliary set is 84 directions against 95
    occupied-virtual pairs, so trading ``o^2 v^2 Q`` for ``o v Q^2 t`` is a loss
    and the search says so. The rewrite is a property of the family, not of the
    expression, and the pass is right to decline it here.
    """
    graph, _energy, transform, search, _captured, _registry = _sos_run(water, 1e-8, annotate=False)
    assert transform.num_transformed == 1
    four_index = [water["nocc"], water["nvir"], water["nocc"], water["nvir"]]
    assert four_index in _written_dims(graph), (
        "the search took the decoupled form at extents where it costs more")


# ──────────────────────────────────────────────────────────────────────────
# Full MP2 leaves the same-spin term alone
#
# SOS-MP2 drops the exchange term and scales the other, which is a decision the
# PROGRAM makes by writing SOS-MP2. A full-MP2 capture must therefore keep its
# exchange term, and what the optimizer owes is that it never makes it worse and
# never moves the number.
# ──────────────────────────────────────────────────────────────────────────

def _full_mp2_run(water, epsilon):
    """The example's own full-axis MP2 over ONE integral, transformed and then searched.

    The permute and the two scalings read the same ``K`` the direct product does,
    so the transform copies the numerator it rides on and the definition stays for
    them. It stays for good: none of the three is a product the flattener can
    inline into, so the exchange half keeps the four-index tensor its permute
    makes and the graph forms the integral exactly once.
    """
    energy = einsums.create_zero_tensor(f"E_full_{epsilon:g}", [1])
    graph = cg.Graph(f"full_mp2_{epsilon:g}")
    registry = _sos_registry()
    graph.set_space_registry(registry)
    shape = _shape(water)
    B = water["fitted"]
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    denominator = graph.scratch("D", shape, "float64")
    _build_denominator(graph, water, denominator)
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.permute("iajb <- ibja", exchange, K)
        la.axpby(2.0, K, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)
    graph.annotate_tag(denominator, _tag())
    cg.annotate(B, ("aux", "occ", "vir"), graph=graph)
    for tensor in (K, T, exchange, combination, denominator):
        cg.annotate(tensor, ("occ", "vir", "occ", "vir"), graph=graph)

    transform = _transform(water, epsilon)
    search = cg.MultiTermFactorization()
    search.set_search_enabled(True)
    manager = _searching_manager(transform, search)
    manager.run(graph)
    assert not search.was_cut_off, "the search was cut off, so the tree below is the machine's"
    return graph, energy, transform, search, registry


def test_full_mp2_keeps_its_exchange_term_and_its_number(water):
    """The exchange combination survives, and the energy is inside the bound.

    The example writes both spin cases as ONE reduction over ``2K - K^T``, so
    there is no separable opposite-spin term for the search to take apart: the
    exchange combination is a materialized ``o^2 v^2`` tensor the dot reads, and
    no bracketing of the product it appears in can make it not exist. What the
    search does is re-bracket what is there, and what it owes is that the number
    does not move.
    """
    exact = _exact(water)
    graph, energy, transform, search, _registry = _full_mp2_run(water, 1e-8)

    assert transform.num_transformed == 1
    assert search.num_rebracketed >= 1, search.skip_reasons

    four_index = [water["nocc"], water["nvir"], water["nocc"], water["nvir"]]
    assert four_index in _written_dims(graph), (
        "the exchange combination cannot be bracketed away and must still be formed")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = float(np.asarray(energy)[0])
    record = graph.approximations()[0]
    assert abs(value - exact) <= _SAFETY * record.bound * abs(exact) + _ROUNDING


def _split_over_one_integral(water, epsilon, budget_ms=0):
    """``E = 2 sum K K D - sum K^T K D`` as two reductions over ONE stored integral.

    Each half is given its own numerator, so that the one thing the two halves
    share is the integral they READ, which is what the flattener has to decide
    about per consumer. A half whose numerator the other statements also read
    would have the transform take a copy of it, and the decision under test here
    would then be two decisions.
    """
    shape = _shape(water)
    B = water["fitted"]
    opposite = einsums.create_zero_tensor(f"E_os_{epsilon:g}", [1])
    same = einsums.create_zero_tensor(f"E_ss_{epsilon:g}", [1])

    graph = cg.Graph(f"full_mp2_split_{epsilon:g}")
    graph.set_space_registry(_sos_registry())
    names = ("K_os", "T_os", "D_os", "K_ss", "T_ss", "X_ss", "D_ss", "A")
    held = {name: graph.scratch(name, shape, "float64") for name in names}
    for half in ("os", "ss"):
        with cg.capture(graph):
            la.outer_sum(held[f"D_{half}"],
                         [water["occupied_energies"], water["virtual_energies"],
                          water["occupied_energies"], water["virtual_energies"]],
                         [1.0, -1.0, 1.0, -1.0])
            la.element_transform(held[f"D_{half}"], "recip")
    with cg.capture(graph):
        for half in ("os", "ss"):
            einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", held[f"K_{half}"], B, B)
            la.direct_product(1.0, held[f"K_{half}"], held[f"D_{half}"], 0.0, held[f"T_{half}"])
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", held["A"], B, B)
        einsums.permute("iajb <- ibja", held["X_ss"], held["A"])
        la.dot(opposite, held["A"], held["T_os"])
        la.dot(same, held["X_ss"], held["T_ss"])
    for half in ("os", "ss"):
        graph.annotate_tag(held[f"D_{half}"], _tag())
    cg.annotate(B, ("aux", "occ", "vir"), graph=graph)
    for tensor in held.values():
        cg.annotate(tensor, ("occ", "vir", "occ", "vir"), graph=graph)

    transform = _transform(water, epsilon)
    search = cg.MultiTermFactorization()
    search.set_search_enabled(True)
    manager = _searching_manager(transform, search, budget_ms=budget_ms)
    manager.run(graph)
    assert transform.num_transformed == 2
    return graph, opposite, same, transform, search


def test_the_opposite_spin_half_takes_a_copy_of_the_one_integral_the_exchange_half_keeps(water):
    """Two reductions, one integral, and only the half that pays takes a copy.

    The integral both halves read is one tensor. The exchange half reads it
    through a permute, which is not a product and cannot take it, so the
    definition is kept; the opposite-spin dot takes a copy and re-brackets it
    into the pairless form. The emitted graph holds exactly ONE contraction
    forming the integral, which is the point: giving each half its own copy of
    the integral is what this used to need.

    Which is the design's own point restated by the implementation: the author
    has to write SOS-MP2 as SOS-MP2, and what that costs is one copy in the half
    that profits rather than a second integral in the program.
    """
    exact = _exact(water)
    graph, opposite, same, transform, search = _split_over_one_integral(water, 1e-3)
    assert not search.was_cut_off, "the search was cut off, so the tree below is the machine's"
    assert search.num_rebracketed >= 1, search.skip_reasons

    # Kept, because the permute cannot take it, and copied into the one consumer
    # that profits.
    assert search.num_copies == 1, search.skip_reasons
    assert _integrals_formed(graph, water) == ["A"], _integrals_formed(graph, water)

    # The exchange half's permuted integral is still formed, and must be.
    four_index = [water["nocc"], water["nvir"], water["nocc"], water["nvir"]]
    written = _written_dims(graph)
    assert four_index in written
    # And the opposite-spin half reached the pairless form beside it, over the
    # integral it copied rather than over one the program formed twice.
    naux, points = water["naux"], transform.last_point_count
    assert any(sorted(shape_) == sorted([naux, naux, points]) for shape_ in written), (
        f"the opposite-spin half did not reach the pairless form: {written}")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = 2.0 * float(np.asarray(opposite)[0]) - float(np.asarray(same)[0])
    record = graph.approximations()[0]
    assert abs(value - exact) <= _SAFETY * record.bound * abs(exact) + _ROUNDING, (
        f"the split energy {value} is outside the bound against {exact}")


def test_the_same_program_at_a_tighter_tolerance_reads_the_integral_instead(water):
    """The copy is a cost decision, and the point count is one of its terms.

    Decoupling trades ``o^2 v^2 Q`` for ``o v Q^2 t``, so the quadrature's own
    point count is on the losing side of it. At ``1e-3`` it is sixteen points and
    the copy pays; at ``1e-8`` it is sixty and it does not, so the opposite-spin
    half reads the integral the exchange half keeps rather than rebuilding it
    pairlessly. Both halves then read one tensor, which is the other outcome the
    sharing has to reach, and the integral is still formed exactly once.

    The tolerance SOS-MP2 is actually run at is the loose one, which is where the
    decoupling fires. This is the same program said at the other end of it.
    """
    exact = _exact(water)
    graph, opposite, same, transform, search = _split_over_one_integral(water, 1e-8)

    assert not search.was_cut_off, "the search was cut off, so the tree below is the machine's"
    assert search.num_copies == 0, search.skip_reasons
    assert any("buys that consumer nothing" in reason for reason, _count in search.skip_reasons), (
        search.skip_reasons)
    assert _integrals_formed(graph, water) == ["A"], _integrals_formed(graph, water)

    naux, points = water["naux"], transform.last_point_count
    written = _written_dims(graph)
    assert not any(sorted(shape_) == sorted([naux, naux, points]) for shape_ in written), (
        f"the pairless form was taken where it costs more: {written}")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = 2.0 * float(np.asarray(opposite)[0]) - float(np.asarray(same)[0])
    record = graph.approximations()[0]
    assert abs(value - exact) <= _SAFETY * record.bound * abs(exact) + _ROUNDING, (
        f"the split energy {value} is outside the bound against {exact}")


def test_a_search_cut_off_by_its_allowance_emits_a_different_tree(water):
    """The emitted tree is a function of the allowance, which is why the cases above remove it.

    The same program at ``1e-3``, searched under an allowance too small to
    finish in. The pass keeps the best candidate it had found, says so through
    ``was_cut_off``, and the graph it emits forms the four-index integral a
    second time, because the candidate that reads the one the exchange half
    already computes is never reached. Nothing about that is wrong: a cut-off
    costs optimization rather than correctness, and the energy is still inside
    the recorded bound.

    What it is wrong for is a case that PINS a shape. This is the graph the two
    Linux CI legs emitted against a commit that passed on a faster machine, and
    the difference between the machines was the allowance rather than anything
    numerical: the point count, the letter extents and every comparison the
    search makes are the same on both.
    """
    exact = _exact(water)
    graph, opposite, same, _transform, search = _split_over_one_integral(water, 1e-3, budget_ms=1)
    assert search.was_cut_off, search.skip_reasons
    assert any("wall-clock budget" in reason for reason, _count in search.skip_reasons), search.skip_reasons
    assert len(_integrals_formed(graph, water)) > 1, (
        f"the allowance did not change the tree: {_integrals_formed(graph, water)}")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = 2.0 * float(np.asarray(opposite)[0]) - float(np.asarray(same)[0])
    record = graph.approximations()[0]
    assert abs(value - exact) <= _SAFETY * record.bound * abs(exact) + _ROUNDING, (
        f"the cut-off split energy {value} is outside the bound against {exact}")

    # And the same program with the allowance removed, which is what every case
    # above runs, forms it exactly once.
    unbounded, _o, _s, _t, finished = _split_over_one_integral(water, 1e-3)
    assert not finished.was_cut_off, finished.skip_reasons
    assert _integrals_formed(unbounded, water) == ["A"], _integrals_formed(unbounded, water)


def test_the_fit_contracted_with_itself_over_the_pair_forms_no_integral(water):
    """``sum_ia B[Q,i,a] B[Q',i,a]`` is the pairless form, not the integral.

    The decoupled opposite-spin energy is built out of exactly this product, so
    a reading of "a contraction of the fit with itself forms the integral" calls
    the rewrite's whole point a regression. What separates the two is the shape
    that comes out: the integral is rank four over the occupied and virtual
    axes, and this is rank two over the auxiliary one.
    """
    naux = water["naux"]
    graph = cg.Graph("pairless_shape")
    matrix = graph.scratch("X", [naux, naux], "float64")
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; R,i,a -> Q,R", matrix, water["fitted"], water["fitted"])
    assert _integrals_formed(graph, water) == [], _integrals_formed(graph, water)
