# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The factorization surface, from Python.

The design's sequencing rule says a feature is not done until it is reachable
from Python, and this surface was the one it named as unmet: FactorizationPass
was bound and useless, because a registry could not be built, a provider could
not be constructed, and the tag a provider claims could not be made.

What this covers is registering the SHIPPED provider and running the pass.
Authoring a provider in Python is still not possible, and that is a real
remainder rather than an oversight: FactorizationProvider is an abstract base
and the binding generator has no trampoline, so a Python subclass cannot
implement propose().
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
import einsums._core.graph as _G
from einsums.testing import assert_close

#: Sized so the split is actually cheaper. The pass costs the rewrite at the
#: extents the graph holds and declines one that is slower there, which is
#: correct of it and makes a toy case measure nothing: at naux=5, n=3 the two
#: contractions come to 90 flops against the single contraction's 81.
_NAUX, _N = 3, 8


def _fitted_problem(seed):
    """A rank-4 tensor that IS the metric fit of a three-index one.

    Built so the factorization is exact and the test can compare against the
    dense contraction rather than against a tolerance nobody derived.
    """
    rng = np.random.default_rng(seed)
    three = rng.standard_normal((_NAUX, _N, _N))
    metric = rng.standard_normal((_NAUX, _NAUX))
    metric = metric @ metric.T + _NAUX * np.eye(_NAUX)   # symmetric positive definite
    dense = np.einsum("pmn,pq,qab->mnab", three, np.linalg.inv(metric), three)
    return three, metric, dense


def test_registry_holds_and_reports_providers():
    """A registry is buildable, and answers what claims a tag."""
    three, metric, _ = _fitted_problem(11)
    registry = _G.FactorizationRegistry()
    assert registry.size == 0
    assert not registry.claims("eri")

    provider = _G.MetricFitFactorization(
        "eri", einsums.asarray(three), einsums.asarray(metric), 0.0)
    registry.add(provider)

    assert registry.size == 1
    assert registry.claims("eri")
    assert provider.name == "MetricFit"
    assert provider.tag == "eri"
    assert [p.name for p in registry.for_tag("eri")] == ["MetricFit"]

    assert registry.remove("MetricFit")
    assert registry.size == 0


def test_a_tag_is_constructible():
    """ProvenanceTag is an aggregate, so Python builds one through a factory."""
    tag = _G.ProvenanceTag.make("eri")
    assert tag.name == "eri"
    assert tag.valid

    qualified = _G.ProvenanceTag.make_with_attributes("eri", [("basis", "cc-pvdz")])
    assert qualified.attribute("basis") == "cc-pvdz"
    assert qualified.attribute("missing") is None


def test_the_whole_loop_from_python():
    """Tag, register, factorize, execute, and get the dense answer back.

    This is the sequencing rule's actual bar: not that the types exist, but that
    a caller who only has Python can drive the feature end to end.
    """
    three, metric, dense = _fitted_problem(20260901)
    rng = np.random.default_rng(5)
    operand = rng.standard_normal((_N, _N))

    M = einsums.asarray(dense)
    T = einsums.asarray(operand)
    C = einsums.zeros((_N, _N), dtype="float64")

    g = cg.Graph("autodf_python")
    with cg.capture(g):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    g.annotate_tag(M, _G.ProvenanceTag.make("eri"))

    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization(
        "eri", einsums.asarray(three), einsums.asarray(metric), 0.0))

    factorization = _G.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(factorization)

    assert g.apply(pm), "the pass declined; at this size the split should be cheaper"
    assert factorization.num_factorized == 1

    # The graph says what it now computes, which is half the point of the lossy
    # tier: a factorized graph carries its own approximation record.
    assert [record.pass_name for record in g.approximations()] == ["MetricFit"]

    g.apply(cg.default_pass_manager())
    g.execute()
    assert_close(C, np.einsum("mnpq,pq->mn", dense, operand), atol=1e-12, rtol=1e-10)


def test_declines_when_the_split_is_not_cheaper():
    """The profitability veto is visible from Python too.

    An auxiliary index larger than the product it replaces is an ordinary case
    rather than a pathological one, and a caller who sees nothing happen should
    be able to reproduce why.
    """
    naux, n = 5, 3
    rng = np.random.default_rng(3)
    three = rng.standard_normal((naux, n, n))
    metric = rng.standard_normal((naux, naux))
    metric = metric @ metric.T + naux * np.eye(naux)
    dense = np.einsum("pmn,pq,qab->mnab", three, np.linalg.inv(metric), three)

    M = einsums.asarray(dense)
    T = einsums.asarray(rng.standard_normal((n, n)))
    C = einsums.zeros((n, n), dtype="float64")

    g = cg.Graph("autodf_declined")
    with cg.capture(g):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    g.annotate_tag(M, _G.ProvenanceTag.make("eri"))

    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization(
        "eri", einsums.asarray(three), einsums.asarray(metric), 0.0))

    factorization = _G.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(factorization)

    assert not g.apply(pm)
    assert factorization.num_factorized == 0
    assert g.approximations() == [], "a declined rewrite must record no approximation"


def test_provider_cannot_be_authored_in_python():
    """Recorded as a limitation rather than left to be discovered.

    FactorizationProvider is exposed so the registry can hand one back, and a
    Python subclass of it cannot implement propose(): the generator emits no
    trampoline, so the virtual never dispatches back into Python. When that
    changes, this test is the one that should start failing.
    """
    with pytest.raises(TypeError):
        _G.FactorizationProvider()
