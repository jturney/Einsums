# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Two lossy passes on one water program, and the two that decline to be a third.

The program is the density-fitted MP2 energy with its denominator built inside
the capture, which is the shape ``test_laplace_mp2_python`` uses. Three lossy
passes have a claim on it: ``BasisTruncation`` truncates the virtual space,
``LaplaceTransform`` replaces the denominator by a quadrature, and
``NaturalAuxiliaryFactorization`` truncates the auxiliary index of the same
three-index tensor the truncation projects.

What composes is the truncation and the quadrature, in that order, because they
act on different quantities and the one link between them is the energy vector
the denominator recipe names. The truncation replaces that vector and rewrites
the recipe to say so, and the quadrature then verifies the chain it was always
going to verify. Two records land on one output and compose by the per-effect
rule, absolute plus relative, and the observed error is inside the composed
bound.

What declines is everything acting on the tensor the truncation projects, and
the reason is the same one seen from two sides: after the projection the tagged
tensor is read by nobody but the setup, so a factorization has no cone to
re-associate. The other order of the quadrature declines too, and that one is a
refusal the pass makes on purpose: a quadrature carries exponentials of the
orbital energies fitted for the space the denominator ran over, and a truncation
after it would leave those describing a space nothing else uses.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from einsums import linalg as la

from test_fno_projection_python import _FAMILY, _mp2, _tensor, family, water  # noqa: F401

#: What the quadrature is asked for. Loose enough that its own record is visible
#: beside the truncation's and tight enough that the energy is not what dominates.
_EPSILON = 1e-5


def _program(problem, name):
    """``E = sum (2K - K^x) * K / D`` with ``D`` built inside the capture.

    The denominator is a chain the quadrature can verify, an outer sum of the two
    energy vectors and a reciprocal, which is what lets both passes see the same
    orbital energies rather than a number one of them computed.
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
    again = graph.scratch("K_again", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    denominator = graph.scratch("D", shape, "float64")

    with cg.capture(graph):
        la.outer_sum(denominator, [occupied, virtual, occupied, virtual], [1.0, -1.0, 1.0, -1.0])
        la.element_transform(denominator, "recip")
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", integrals, three, three)
        la.direct_product(1.0, integrals, denominator, 0.0, amplitudes)
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", again, three, three)
        einsums.permute("iajb <- ibja", exchange, again)
        la.axpby(2.0, again, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, amplitudes)

    cg.annotate(three, ("aux", "occ", "vir"), graph=graph)
    cg.annotate(occupied, ("occ",), graph=graph)
    cg.annotate(virtual, ("vir",), graph=graph)
    for scratch in (integrals, amplitudes, again, exchange, combination, denominator):
        cg.annotate(scratch, ("occ", "vir", "occ", "vir"), graph=graph)
    graph.annotate_tag(denominator, _G.LaplaceTransform.denominator_tag(
        ["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-"))
    graph.annotate_tag(three, _G.ProvenanceTag.make("eri"))

    return {"graph": graph, "three": three, "occupied": occupied, "virtual": virtual,
            "energy": energy, "denominator": denominator}


def _truncation(problem, occupation=1e-3):
    truncation = _G.BasisTruncation()
    truncation.set_amplitudes(_tensor("t2", problem["amplitudes"]))
    truncation.set_fock(_tensor("F_vv", problem["fock_vv"]))
    truncation.set_occupied_energies(_tensor("eps_occ_for_fno", problem["eps_occ"]))
    truncation.set_occupation(occupation)
    return truncation


def _apply(graph, pass_object, provenance=False):
    manager = cg.PassManager()
    if provenance:
        manager.add(cg.ProvenancePropagation())
    manager.add(pass_object)
    return graph.apply(manager)


def _composed_bound(graph, reference):
    """The per-effect rule applied to whatever records the graph carries."""
    tolerance = graph.approximation_tolerance()
    return tolerance.absolute + tolerance.relative * abs(reference)


# ──────────────────────────────────────────────────────────────────────────
# What composes
# ──────────────────────────────────────────────────────────────────────────


def test_the_truncation_and_the_quadrature_compose_on_one_energy(water):
    """Two records on one output, composed by the per-effect rule.

    The truncation's effect is an ENERGY and composes on the absolute side; the
    quadrature's is a norm and composes on the relative side. They are two
    numbers rather than one because they are in different units, and the
    comparison the composed pair licenses is absolute plus relative times the
    reference.
    """
    full = _mp2(water["ovov"], water["eps_occ"], water["eps_vir"])
    assert full == pytest.approx(water["reference"], abs=1e-9)

    program = _program(water, "compose_fno_laplace")
    graph = program["graph"]

    truncation = _truncation(water)
    assert _apply(graph, truncation), dict(truncation.skip_reasons)
    assert list(truncation.projected) == ["B_ov@fno"]

    transform = cg.LaplaceTransform()
    transform.set_epsilon(_EPSILON)
    transform.add_energy("eps_occ", program["occupied"])
    # The vector the recipe now names, which the truncation has already filled
    # with the semicanonical energies the setup will rewrite on every bind.
    transform.add_energy(_G.BasisTruncation.energies_name(), truncation.energies())
    assert _apply(graph, transform), dict(transform.skip_reasons)
    assert transform.num_transformed == 1

    records = graph.approximations()
    assert sorted(r.pass_name for r in records) == ["BasisTruncation", "LaplaceTransform"]
    effects = {r.pass_name: r.effect for r in records}
    assert effects["BasisTruncation"] == _G.ApproximationEffect.EnergyLike
    assert effects["LaplaceTransform"] == _G.ApproximationEffect.NormRelative
    assert all(r.origin == _G.ApproximationOrigin.Measured for r in records)

    tolerance = graph.approximation_tolerance()
    assert tolerance.absolute == pytest.approx(abs(truncation.correction))
    assert tolerance.relative > 0.0 and tolerance.relative <= _EPSILON

    graph.apply(cg.default_pass_manager())
    graph.execute()
    got = float(np.asarray(program["energy"])[0])
    error = abs(got - full)
    assert error <= _composed_bound(graph, full), (
        f"observed {error:.6e} against a composed bound of {_composed_bound(graph, full):.6e}")
    # And it is a real composition rather than one effect swamping the other: the
    # truncation alone would not account for the whole of it.
    assert error != pytest.approx(abs(truncation.correction), rel=1e-9)


def test_the_denominator_recipe_is_rewritten_to_name_the_semicanonical_energies(water, capfd):
    """The one link between the two passes, and the pass that owns it says so.

    A recipe names its energy vectors by NAME. The truncation points the outer
    sum away from the caller's virtual energies and at the semicanonical ones,
    and a recipe left naming the old vector describes a chain that is no longer
    there, which the quadrature reads as unverifiable and declines over.
    """
    program = _program(water, "compose_recipe")
    graph = program["graph"]

    truncation = _truncation(water)
    manager = cg.PassManager()
    manager.set_verbosity(2)
    manager.add(truncation)
    assert graph.apply(manager)
    assert "the denominator recipe on 'D' now names 'fno_energies'" in capfd.readouterr().err

    # Registered under the OLD name, which is what a caller who did not notice
    # would do: the recipe names a vector the quadrature was never given.
    stale = cg.LaplaceTransform()
    stale.set_epsilon(_EPSILON)
    stale.add_energy("eps_occ", program["occupied"])
    stale.add_energy("eps_vir", program["virtual"])
    assert not _apply(graph, stale)


# ──────────────────────────────────────────────────────────────────────────
# What declines
# ──────────────────────────────────────────────────────────────────────────


def test_the_other_order_of_the_quadrature_is_declined(water):
    """A quadrature fitted to the full space's energies is not refitted by this pass.

    Its exponentials are tables over the orbital energies, sized and fitted for
    the space the denominator ran over, and nothing in the node re-derives them
    from a truncated one. The two compose the other way round, which is what the
    decline says.
    """
    program = _program(water, "compose_wrong_order")
    graph = program["graph"]

    transform = cg.LaplaceTransform()
    transform.set_epsilon(_EPSILON)
    transform.add_energy("eps_occ", program["occupied"])
    transform.add_energy("eps_vir", program["virtual"])
    assert _apply(graph, transform), dict(transform.skip_reasons)

    truncation = _truncation(water)
    assert not _apply(graph, truncation)
    reasons = dict(truncation.skip_reasons)
    assert any("truncate the space before transforming the denominator" in reason for reason in reasons), reasons
    assert [r.pass_name for r in graph.approximations()] == ["LaplaceTransform"]


def test_an_auxiliary_truncation_of_the_tensor_the_projection_replaced_is_declined(water):
    """Two passes on ONE tensor do not compose, and the second one says why.

    After the projection the caller's three-index tensor is read by nothing but
    the setup that projects it, so a factorization looking for a contraction to
    re-associate around its factors finds none. That is a statement about this
    program's shape rather than a refusal either pass makes.
    """
    program = _program(water, "compose_naf_after")
    graph = program["graph"]

    truncation = _truncation(water)
    assert _apply(graph, truncation)

    provider = _G.NaturalAuxiliaryFactorization("eri", program["three"], 1e-1)
    registry = _G.FactorizationRegistry()
    registry.add(provider)
    factorization = _G.FactorizationPass(registry)
    assert not _apply(graph, factorization, provenance=True)
    reasons = dict(factorization.skip_reasons)
    assert any("read by no two-operand contraction" in reason for reason in reasons), reasons
    assert [r.pass_name for r in graph.approximations()] == ["BasisTruncation"]
