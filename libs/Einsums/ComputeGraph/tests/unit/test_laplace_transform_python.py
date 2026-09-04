# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The Laplace-transform surface, from Python.

The design's sequencing rule says a feature is not done until it is reachable
from Python, so what this covers is a caller who has only Python driving the
whole thing: tag the denominator through ``cg.annotate``, hand the pass its
energy vectors, run it, read the approximation record off the graph, and get
the right numbers out.

The oracle is the exact denominator: the same program with the pass off, and
numpy beside it. Both are held to the bound the pass RECORDED rather than to a
tolerance chosen to make the comparison pass, which is what makes the record
evidence rather than decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
import einsums._core.graph as _G

#: Real only. A complex orbital energy is not a thing, and the exponential
#: integral that represents a reciprocal needs a spectral range on the real line.
REAL_DTYPES = ("float32", "float64")

_NOCC, _NVIR, _NLNK = 4, 5, 3


def _energies(dtype):
    """Occupied and virtual energies with a gap that never approaches zero."""
    eo = np.array([-0.9 + 0.15 * i for i in range(_NOCC)], dtype=dtype)
    ev = np.array([0.05 + 0.5 * a for a in range(_NVIR)], dtype=dtype)
    return eo, ev


def _denominator(eo, ev):
    """The reciprocal the caller tags: 1 / (eps_v[a] - eps_o[i])."""
    return (1.0 / (ev[None, :] - eo[:, None])).astype(eo.dtype)


def _program(dtype, seed):
    """P = (A B) * D, with the numerator a graph-owned intermediate."""
    rng = np.random.default_rng(seed)
    eo, ev = _energies(dtype)
    a = rng.standard_normal((_NOCC, _NLNK)).astype(dtype)
    b = rng.standard_normal((_NLNK, _NVIR)).astype(dtype)
    return eo, ev, a, b, _denominator(eo, ev)


def _capture(graph, A, B, D, P, dtype):
    numerator = graph.scratch(f"numerator", [_NOCC, _NVIR], dtype)
    with cg.capture(graph):
        einsums.einsum("i,k ; k,a -> i,a", numerator, A, B)
        einsums.linalg.direct_product(1.0, numerator, D, 0.0, P)


def _tag():
    """The tag as a MAPPING, which is the spelling a Python caller writes."""
    return {
        "name": "laplace_denominator",
        "axis0": "eps_o",
        "sign0": "-",
        "axis1": "eps_v",
        "sign1": "+",
    }


def _skips(pass_object):
    """What the pass declined, and how many candidates hit each reason."""
    return [reason for reason, _count in pass_object.skip_reasons]


def _mentions(lines, fragment):
    return any(fragment in line for line in lines)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_the_whole_loop_from_python(dtype):
    """Tag, supply the energies, transform, execute, and land inside the record."""
    tolerance = 1e-4 if dtype == "float32" else 1e-8
    eo, ev, a, b, d = _program(dtype, 20260904)

    exact = einsums.zeros((_NOCC, _NVIR), dtype=dtype)
    reference = cg.Graph("laplace_reference")
    _capture(reference, einsums.asarray(a), einsums.asarray(b), einsums.asarray(d), exact, dtype)
    reference.apply(cg.default_pass_manager())
    reference.execute()

    out = einsums.zeros((_NOCC, _NVIR), dtype=dtype)
    graph = cg.Graph("laplace_python")
    D = einsums.asarray(d)
    _capture(graph, einsums.asarray(a), einsums.asarray(b), D, out, dtype)
    cg.annotate(D, tag=_tag(), graph=graph)

    laplace = cg.LaplaceTransform()
    laplace.set_epsilon(tolerance)
    add = laplace.add_energy if dtype == "float64" else laplace.add_energy_f32
    add("eps_o", einsums.asarray(eo))
    add("eps_v", einsums.asarray(ev))

    pm = cg.PassManager()
    pm.add(laplace)
    assert graph.apply(pm), "the pass declined a program it should have transformed"
    assert laplace.num_transformed == 1
    assert laplace.last_point_count >= 2
    assert laplace.last_measured_error <= tolerance

    graph.apply(cg.default_pass_manager())
    graph.execute()

    records = graph.approximations()
    assert [r.pass_name for r in records] == ["LaplaceTransform"]
    record = records[0]
    assert record.origin == _G.ApproximationOrigin.Measured
    assert record.effect == _G.ApproximationEffect.NormRelative
    assert record.tolerance == pytest.approx(tolerance)
    assert record.bound <= tolerance
    assert record.setup == "LaplaceTransform(denom)" or record.setup.startswith("LaplaceTransform(")

    # Held to the RECORDED bound plus the dtype's own rounding, which the record
    # does not claim to cover and which a float comparison is dominated by.
    got = np.asarray(out)
    want = np.asarray(exact)
    rounding = float(np.finfo(np.dtype(dtype)).eps) * 64.0
    scale = float(np.max(np.abs(want)))
    assert float(np.max(np.abs(got - want))) <= (record.bound + rounding) * scale

    # And against numpy, which owes nothing to either arm.
    oracle = (a @ b) * d
    assert float(np.max(np.abs(got - oracle))) <= (record.bound + rounding) * float(np.max(np.abs(oracle)))


def test_a_tighter_tolerance_costs_points_and_buys_accuracy():
    """The knob bites: the count grows and the measured error follows it down."""
    eo, ev, a, b, d = _program("float64", 7)

    def run(epsilon):
        out = einsums.zeros((_NOCC, _NVIR), dtype="float64")
        graph = cg.Graph("laplace_knob")
        D = einsums.asarray(d)
        _capture(graph, einsums.asarray(a), einsums.asarray(b), D, out, "float64")
        cg.annotate(D, tag=_tag(), graph=graph)
        laplace = cg.LaplaceTransform()
        laplace.set_epsilon(epsilon)
        laplace.add_energy("eps_o", einsums.asarray(eo))
        laplace.add_energy("eps_v", einsums.asarray(ev))
        pm = cg.PassManager()
        pm.add(laplace)
        assert graph.apply(pm)
        return laplace.last_point_count, laplace.last_measured_error

    loose_points, loose_error = run(1e-3)
    tight_points, tight_error = run(1e-9)
    assert tight_points > loose_points
    assert tight_error < loose_error
    assert loose_error <= 1e-3
    assert tight_error <= 1e-9


def test_an_untagged_graph_is_left_alone():
    """A pass nobody asked anything of does nothing, and says so."""
    _, _, a, b, d = _program("float64", 3)
    out = einsums.zeros((_NOCC, _NVIR), dtype="float64")
    graph = cg.Graph("untagged")
    _capture(graph, einsums.asarray(a), einsums.asarray(b), einsums.asarray(d), out, "float64")

    laplace = cg.LaplaceTransform()
    pm = cg.PassManager()
    pm.add(laplace)
    assert not graph.apply(pm)
    assert laplace.num_transformed == 0


def test_an_energy_the_pass_was_not_given_is_declined():
    """The tag names the energies; a name nothing was registered under is a decline."""
    eo, _, a, b, d = _program("float64", 4)
    out = einsums.zeros((_NOCC, _NVIR), dtype="float64")
    graph = cg.Graph("missing_energy")
    D = einsums.asarray(d)
    _capture(graph, einsums.asarray(a), einsums.asarray(b), D, out, "float64")
    cg.annotate(D, tag=_tag(), graph=graph)

    laplace = cg.LaplaceTransform()
    laplace.add_energy("eps_o", einsums.asarray(eo))
    pm = cg.PassManager()
    pm.add(laplace)
    assert not graph.apply(pm)
    assert _mentions(_skips(laplace), "energy vector this pass cannot use")


def test_a_folded_axis_denominator_is_declined():
    """The pair-driven form names more energies than the tensor has axes."""
    eo, ev, a, b, d = _program("float64", 5)
    out = einsums.zeros((_NOCC, _NVIR), dtype="float64")
    graph = cg.Graph("folded")
    D = einsums.asarray(d)
    _capture(graph, einsums.asarray(a), einsums.asarray(b), D, out, "float64")
    folded = dict(_tag())
    folded.update({"axis2": "eps_o", "sign2": "-", "axis3": "eps_v", "sign3": "+"})
    cg.annotate(D, tag=folded, graph=graph)

    laplace = cg.LaplaceTransform()
    laplace.add_energy("eps_o", einsums.asarray(eo))
    laplace.add_energy("eps_v", einsums.asarray(ev))
    pm = cg.PassManager()
    pm.add(laplace)
    assert not graph.apply(pm)
    assert _mentions(_skips(laplace), "names more energies than the tagged tensor has axes")


def test_the_tag_helper_and_the_names_are_reachable():
    """Everything a caller needs to spell the tag is bound."""
    assert _G.LaplaceTransform.tag_name() == "laplace_denominator"
    tag = _G.LaplaceTransform.denominator_tag(["eps_o", "eps_v"], "-+")
    assert tag.name == "laplace_denominator"
    assert tag.attribute("axis0") == "eps_o"
    assert tag.attribute("sign1") == "+"
    assert _G.LaplaceTransform.error_tensor_name("denom").endswith("measured_error")

    with pytest.raises(Exception):
        _G.LaplaceTransform.denominator_tag(["eps_o"], "-+")

    laplace = cg.LaplaceTransform()
    assert laplace.name == "LaplaceTransform"
    assert laplace.points == 0
    laplace.set_points(9)
    assert laplace.points == 9
    with pytest.raises(Exception):
        laplace.set_points(1)
    with pytest.raises(Exception):
        laplace.set_epsilon(0.0)
    laplace.clear_energies()


# ── The proving ground ──────────────────────────────────────────────────────

_MP2_NOCC, _MP2_NVIR = 3, 4


def _mp2_problem(seed=20260904):
    """Orbital energies with a realistic gap, and an integral tensor with the right symmetry.

    Synthetic rather than from a fixture, because what the case is about is the
    denominator and not the integrals: the gaps run from about 1.0 to 5.9
    Hartree, which is the range a small closed-shell molecule actually has, and
    the denominator is uniformly negative, which is the sign convention half of
    every MP2 expression written down.
    """
    rng = np.random.default_rng(seed)
    eo = np.array([-0.9 + 0.225 * i for i in range(_MP2_NOCC)])
    ev = np.array([0.05 + 0.5 * a for a in range(_MP2_NVIR)])

    # (ia|jb) with the permutational symmetry the expression assumes, built from
    # a three-index tensor so it is a real integral-shaped object rather than noise.
    naux = 5
    three = rng.standard_normal((naux, _MP2_NOCC, _MP2_NVIR))
    eri = np.einsum("Qia,Qjb->iajb", three, three)

    denominator = 1.0 / (
        eo[:, None, None, None] + eo[None, None, :, None] - ev[None, :, None, None] - ev[None, None, None, :]
    )
    return eo, ev, eri, denominator


def _mp2_tag():
    return {
        "name": "laplace_denominator",
        "axis0": "eps_o", "sign0": "+",
        "axis1": "eps_v", "sign1": "-",
        "axis2": "eps_o", "sign2": "+",
        "axis3": "eps_v", "sign3": "-",
    }


def test_mp2_in_the_full_axis_form_survives_the_substitution():
    """The proving ground: MP2 written over all four indices, tagged and transformed.

    ``E = sum_iajb (2 (ia|jb) - (ib|ja)) (ia|jb) / (e_i + e_j - e_a - e_b)``, with
    the denominator formed as a four-axis reciprocal rather than folded into a
    pair prefactor, which is the form this pass is specified for and the reason
    the psi4-bridge example is declined.

    The numerator's product is spelled as an einsum rather than a direct product
    on purpose: the rewrite pushes the per-axis exponentials onto the OPERANDS of
    a contraction, so a numerator that is itself an elementwise node has nothing
    for them to ride on and is declined. Writing it as the contraction it
    mathematically is costs nothing and is what the pass recognizes.
    """
    eo, ev, eri, denom = _mp2_problem()
    tolerance = 1e-8

    exchange = np.ascontiguousarray(eri.transpose(0, 3, 2, 1))  # (ib|ja)
    shape = (_MP2_NOCC, _MP2_NVIR, _MP2_NOCC, _MP2_NVIR)

    def build(graph, out):
        K = einsums.asarray(eri)
        Kx = einsums.asarray(exchange)
        D = einsums.asarray(denom)
        numerator = graph.scratch("mp2_numerator", list(shape), "float64")
        combination = graph.scratch("mp2_combination", list(shape), "float64")
        with cg.capture(graph):
            einsums.linalg.axpby(2.0, K, 0.0, combination)
            einsums.linalg.axpby(-1.0, Kx, 1.0, combination)
            einsums.einsum("i,a,j,b ; i,a,j,b -> i,a,j,b", numerator, combination, K)
            einsums.linalg.direct_product(1.0, numerator, D, 0.0, out)
        return D

    exact = einsums.zeros(shape, dtype="float64")
    reference = cg.Graph("mp2_reference")
    build(reference, exact)
    reference.apply(cg.default_pass_manager())
    reference.execute()

    out = einsums.zeros(shape, dtype="float64")
    graph = cg.Graph("mp2_laplace")
    D = build(graph, out)
    cg.annotate(D, tag=_mp2_tag(), graph=graph)

    laplace = cg.LaplaceTransform()
    laplace.set_epsilon(tolerance)
    laplace.add_energy("eps_o", einsums.asarray(eo))
    laplace.add_energy("eps_v", einsums.asarray(ev))
    pm = cg.PassManager()
    pm.add(laplace)
    assert graph.apply(pm), f"declined: {laplace.skip_reasons}"
    assert laplace.num_transformed == 1

    graph.apply(cg.default_pass_manager())
    graph.execute()

    record = graph.approximations()[0]
    assert record.origin == _G.ApproximationOrigin.Measured
    assert record.bound <= tolerance

    # numpy, which owes nothing to either arm, and the correlation energy the
    # whole expression exists to produce.
    oracle = (2.0 * eri - exchange) * eri * denom
    got = np.asarray(out)
    scale = float(np.max(np.abs(oracle)))
    assert float(np.max(np.abs(got - np.asarray(exact)))) <= (record.bound + 1e-13) * scale
    assert float(np.max(np.abs(got - oracle))) <= (record.bound + 1e-13) * scale

    energy_exact = float(np.sum(oracle))
    energy_laplace = float(np.sum(got))
    assert abs(energy_laplace - energy_exact) <= record.bound * float(np.sum(np.abs(oracle)))


def test_two_lossy_records_land_on_one_output():
    """A density fit and a Laplace transform, composed on the same result.

    The composition rule of the accuracy contract is per effect, and both
    records here are norm-relative, so the graph reports them composed rather
    than summed. What this pins is that the two passes CAN run on one program:
    the fit rewrites the contraction that forms the numerator, and the transform
    then rides on the operands the fit left behind.
    """
    rng = np.random.default_rng(4)
    naux, nbf = 3, 4
    three = rng.standard_normal((naux, nbf, nbf))
    metric = rng.standard_normal((naux, naux))
    metric = metric @ metric.T + naux * np.eye(naux)
    dense = np.einsum("pmn,pq,qab->mnab", three, np.linalg.inv(metric), three)
    operand = rng.standard_normal((nbf, nbf))

    # A denominator over the two axes the contraction leaves free.
    row = np.array([-0.9 + 0.2 * i for i in range(nbf)])
    col = np.array([0.4 + 0.35 * i for i in range(nbf)])
    denom = 1.0 / (col[None, :] - row[:, None])

    def build(graph, out):
        M = einsums.asarray(dense)
        T = einsums.asarray(operand)
        D = einsums.asarray(denom)
        numerator = graph.scratch("df_numerator", [nbf, nbf], "float64")
        with cg.capture(graph):
            einsums.einsum("m,n,p,q ; p,q -> m,n", numerator, M, T)
            einsums.linalg.direct_product(1.0, numerator, D, 0.0, out)
        return M, D

    exact = einsums.zeros((nbf, nbf), dtype="float64")
    reference = cg.Graph("df_laplace_reference")
    build(reference, exact)
    reference.apply(cg.default_pass_manager())
    reference.execute()

    out = einsums.zeros((nbf, nbf), dtype="float64")
    graph = cg.Graph("df_laplace")
    M, D = build(graph, out)
    graph.annotate_tag(M, _G.ProvenanceTag.make("eri"))
    cg.annotate(D, tag={"name": "laplace_denominator", "axis0": "eps_row", "sign0": "-",
                        "axis1": "eps_col", "sign1": "+"}, graph=graph)

    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization(
        "eri", einsums.asarray(three), einsums.asarray(metric), 1e-12))
    factorization = _G.FactorizationPass(registry)

    laplace = cg.LaplaceTransform()
    laplace.set_epsilon(1e-9)
    laplace.add_energy("eps_row", einsums.asarray(row))
    laplace.add_energy("eps_col", einsums.asarray(col))

    pm = cg.PassManager()
    pm.add(factorization)
    pm.add(laplace)
    assert graph.apply(pm)
    assert factorization.num_factorized == 1, "the fit did not fire, so nothing composed"
    assert laplace.num_transformed == 1, f"the transform declined: {laplace.skip_reasons}"

    names = sorted(r.pass_name for r in graph.approximations())
    assert names == ["LaplaceTransform", "MetricFit"]

    # Composed, not summed: both are relative, so the bound is e1 + e2 + e1*e2.
    # Either record's outputs list is empty, which means every output, so the
    # question can be asked under any name.
    tolerance = graph.approximation_tolerance("result")
    bounds = [r.bound for r in graph.approximations()]
    composed = bounds[0] + bounds[1] + bounds[0] * bounds[1]
    assert tolerance.relative == pytest.approx(composed)

    graph.apply(cg.default_pass_manager())
    graph.execute()

    oracle = np.einsum("mnpq,pq->mn", dense, operand) * denom
    got = np.asarray(out)
    scale = float(np.max(np.abs(oracle)))
    assert float(np.max(np.abs(got - oracle))) <= (composed + 1e-12) * scale
    assert float(np.max(np.abs(got - np.asarray(exact)))) <= (composed + 1e-12) * scale
