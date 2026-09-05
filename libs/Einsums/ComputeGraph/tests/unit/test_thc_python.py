# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The grid-fit provider, from Python.

The design's sequencing rule says a feature is not done until it is reachable
from Python, and a provider is exactly the shape that rule was written about:
``ThcFactorization`` is exposed so a registry can hand one back, and it is
useless unless a caller with only Python can construct it, register the grid
space, run the pass and get the right number out.

What is covered here is that loop, the alternative-provider ranking the design
says the registry does between a metric fit and a grid fit on one tag, and the
declines. Authoring a provider in Python is still not possible, which
``test_factorization_python.py`` records as the standing gap.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from einsums.testing import assert_close

#: Small enough that the fit's normal equations are solvable: the functions
#: ``X[m,P] X[n,P]`` are only independent while the grid is no larger than the
#: number of distinct basis pairs.
_NBF, _NGRID, _NAUX = 4, 6, 3


def _problem(seed):
    """A three-index tensor built FROM a grid, so the fit is exact.

    With ``B[A,m,n] = sum_P C[A,P] X[m,P] X[n,P]`` the least-squares solution is
    ``C`` itself, so the chain reproduces the four-index tensor to rounding and
    the test compares against a dense contraction rather than against a
    tolerance nobody derived.
    """
    rng = np.random.default_rng(seed)
    collocation = rng.standard_normal((_NBF, _NGRID))
    weights = rng.standard_normal((_NAUX, _NGRID))
    three = np.einsum("AP,mP,nP->Amn", weights, collocation, collocation)
    dense = np.einsum("Amn,Apq->mnpq", three, three)
    return collocation, three, dense


def _build(seed):
    collocation, three, dense = _problem(seed)
    operand = np.random.default_rng(seed + 1).standard_normal((_NBF, _NBF))

    M = einsums.asarray(np.ascontiguousarray(dense))
    T = einsums.asarray(np.ascontiguousarray(operand))
    C = einsums.zeros((_NBF, _NBF), dtype="float64")

    graph = cg.Graph(f"thc_python_{seed}")
    with cg.capture(graph):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    graph.annotate_tag(M, _G.ProvenanceTag.make("eri"))
    # Annotated for its family. A grid is chosen per problem, so the number of
    # points a capture happened to have is a placeholder; the same is true of a
    # basis, and this is the caller saying so.
    graph.annotate_dims(M, ["nbf", "nbf", "nbf", "nbf"])
    _G.ThcFactorization.register_grid_space(graph)

    provider = _G.ThcFactorization(
        "eri", einsums.asarray(np.ascontiguousarray(three)),
        einsums.asarray(np.ascontiguousarray(collocation)), 1e-8)
    return graph, provider, M, T, C, dense, operand


def test_the_whole_loop_from_python():
    """Register the grid space, register the provider, factorize, execute."""
    graph, provider, _M, _T, C, dense, operand = _build(20260904)
    assert provider.name == "Thc"
    assert provider.tag == "eri"
    assert provider.epsilon == pytest.approx(1e-8)
    assert _G.ThcFactorization.grid_space_name() == "grid"
    assert _G.ThcFactorization.grid_dim_symbol() == "ngrid"

    registry = _G.FactorizationRegistry()
    registry.add(provider)
    factorization = _G.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(factorization)

    assert graph.apply(pm), f"the pass declined: {factorization.skip_reasons}"
    assert factorization.num_factorized == 1
    assert [record.pass_name for record in graph.approximations()] == ["Thc"]

    graph.apply(cg.default_pass_manager())
    # One lifecycle per tensor, including the ones only the setup body writes.
    assert not cg.duplicate_materializations(graph)
    assert not cg.stranded_materializations(graph)

    graph.execute()
    assert_close(C, np.einsum("mnpq,pq->mn", dense, operand), atol=1e-9, rtol=1e-9)

    # The measured residual is named here and read in C++: the parameter table has
    # no Python surface, which ThcFit.cpp covers and this records rather than
    # leaves to be discovered.
    assert _G.ThcFactorization.residual_param_name("Thc", "M").endswith(".residual_squared")
    assert _G.ThcFactorization.reference_param_name("Thc", "M").endswith(".reference_squared")


def test_the_grid_space_is_registered_once_and_carries_its_symbol():
    """A space is a statement about the graph, so a caller makes it; twice is idempotent."""
    graph = cg.Graph("thc_space")
    first = _G.ThcFactorization.register_grid_space(graph)
    second = _G.ThcFactorization.register_grid_space(graph)
    assert first == second, "a repeated identical declaration must not make a second space"


def test_the_two_providers_are_alternatives_on_one_tag():
    """A metric fit and a grid fit both claim ``eri``, and the pass takes one.

    They are ALTERNATIVES rather than a composition, which is what the registry's
    ranking by cost is for: whichever is cheaper on the cone in hand wins, and
    exactly one approximation record lands on the output either way.
    """
    collocation, three, dense = _problem(7)
    metric = np.eye(_NAUX)
    operand = np.random.default_rng(9).standard_normal((_NBF, _NBF))

    M = einsums.asarray(np.ascontiguousarray(dense))
    T = einsums.asarray(np.ascontiguousarray(operand))
    C = einsums.zeros((_NBF, _NBF), dtype="float64")

    graph = cg.Graph("thc_vs_metric")
    with cg.capture(graph):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    graph.annotate_tag(M, _G.ProvenanceTag.make("eri"))
    graph.annotate_dims(M, ["nbf", "nbf", "nbf", "nbf"])
    _G.ThcFactorization.register_grid_space(graph)

    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization(
        "eri", einsums.asarray(np.ascontiguousarray(three)),
        einsums.asarray(np.ascontiguousarray(metric)), 1e-10))
    registry.add(_G.ThcFactorization(
        "eri", einsums.asarray(np.ascontiguousarray(three)),
        einsums.asarray(np.ascontiguousarray(collocation)), 1e-8))
    assert {p.name for p in registry.for_tag("eri")} == {"MetricFit", "Thc"}

    factorization = _G.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(factorization)
    assert graph.apply(pm), f"neither provider was taken: {factorization.skip_reasons}"
    assert factorization.num_factorized == 1

    records = graph.approximations()
    assert len(records) == 1, "two providers on one tag must not both be applied"
    assert records[0].pass_name in ("MetricFit", "Thc")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    assert_close(C, np.einsum("mnpq,pq->mn", dense, operand), atol=1e-9, rtol=1e-9)


def test_a_threshold_that_could_not_mean_anything_is_refused():
    """Zero is spellable because it is the bare guard; a negative one is not."""
    collocation, three, _dense = _problem(3)
    B = einsums.asarray(np.ascontiguousarray(three))
    X = einsums.asarray(np.ascontiguousarray(collocation))
    _G.ThcFactorization("eri", B, X, 1e-8, 0.0)
    with pytest.raises(ValueError):
        _G.ThcFactorization("eri", B, X, 1e-8, -1e-12)
    with pytest.raises(ValueError):
        _G.ThcFactorization("eri", B, X, -1.0)


def test_a_tensor_the_grid_cannot_span_is_declined_with_its_reason():
    """A four-index tensor over another basis is not one this grid produces."""
    collocation, three, _dense = _problem(5)
    other = np.random.default_rng(11).standard_normal((_NBF + 1,) * 4)
    operand = np.random.default_rng(12).standard_normal((_NBF + 1, _NBF + 1))

    M = einsums.asarray(np.ascontiguousarray(other))
    T = einsums.asarray(np.ascontiguousarray(operand))
    C = einsums.zeros((_NBF + 1, _NBF + 1), dtype="float64")

    graph = cg.Graph("thc_mismatch")
    with cg.capture(graph):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    graph.annotate_tag(M, _G.ProvenanceTag.make("eri"))
    _G.ThcFactorization.register_grid_space(graph)

    registry = _G.FactorizationRegistry()
    registry.add(_G.ThcFactorization(
        "eri", einsums.asarray(np.ascontiguousarray(three)),
        einsums.asarray(np.ascontiguousarray(collocation)), 1e-8))
    factorization = _G.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(factorization)

    assert not graph.apply(pm)
    assert factorization.num_factorized == 0
    assert graph.approximations() == []
    assert any("a provider declined" in reason for reason, _count in factorization.skip_reasons)
