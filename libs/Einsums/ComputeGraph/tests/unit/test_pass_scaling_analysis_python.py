# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python mirror of Pass_ScalingAnalysis.cpp.

The C++ tests own the cost algebra: which polynomial a contraction contributes,
how the rate-limiting node is ranked, and what the memory bound does and does
not mean. This file asserts the bound half. ``SymbolicPoly`` is not a Python
type - a polynomial's only readable form is its rendering, and rendering needs
the registry the pass borrowed - so Python reads the same numbers through the
pass's rendered surface: ``total_flops_str``, ``node_flops``,
``rate_limiting_labels`` and ``report_string``.
"""

from __future__ import annotations

import pytest

import einsums
import einsums.graph as cg


@pytest.fixture()
def reg():
    """occ < virt < aux, which is what lets the rate-limiting verdict rank two
    polynomials asymptotically instead of falling through to a numeric rung."""
    registry = cg.SpaceRegistry()
    occ = registry.register_space(cg.index_space("occ", "o", 4.0))
    virt = registry.register_space(cg.index_space("virt", "v", 8.0))
    aux = registry.register_space(cg.index_space("aux", "x", 16.0))
    registry.declare_less(occ, virt)
    registry.declare_less(virt, aux)
    return registry


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_an_annotated_contraction_reports_its_exact_cost_polynomial(reg):
    A = einsums.create_random_tensor("A", [2, 3])  # occ x virt
    B = einsums.create_random_tensor("B", [3, 4])  # virt x aux

    g = cg.Graph("exact_cost")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    scaling = cg.ScalingAnalysis()
    assert _run(scaling, g) is False  # an analysis pass never claims a modification

    assert scaling.num_analyzed == 1
    assert scaling.num_unannotated_nodes == 0

    # The loop space is i, a, x, one distinct letter each, times two for the
    # multiply and the add.
    assert scaling.total_flops_str() == "2*o*v*x"
    assert scaling.node_flops() == ["2*o*v*x"]
    assert scaling.node_labels() == list(scaling.node_labels())  # parallel arrays
    assert len(scaling.node_labels()) == 1


def test_the_rate_limiting_node_is_the_one_whose_flops_dominate(reg):
    A = einsums.create_random_tensor("A", [2, 3])  # occ x virt
    B = einsums.create_random_tensor("B", [3, 2])  # virt x occ

    g = cg.Graph("rate_limiting")
    g.set_space_registry(reg)
    small = g.create_zero_tensor("Small", [2, 2], dtype="float64")  # occ x occ
    big = g.create_zero_tensor("Big", [3, 3], dtype="float64")  # virt x virt
    with cg.capture(g):
        einsums.einsum("ij <- ia ; aj", small, A, B)  # loop space o, o, v
        einsums.einsum("ab <- ai ; ib", big, B, A)  # loop space v, v, o

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "occ"), graph=g)

    scaling = cg.ScalingAnalysis()
    _run(scaling, g)

    assert scaling.num_analyzed == 2
    # With occ declared below virt, 2 o v^2 dominates 2 o^2 v asymptotically.
    assert scaling.node_flops() == ["2*o^2*v", "2*o*v^2"]
    assert len(scaling.rate_limiting_labels()) == 1
    assert scaling.rate_limiting_labels()[0] == scaling.node_labels()[1]
    assert scaling.total_flops_str() == "2*o*v^2 + 2*o^2*v"


def test_intermediates_are_sized_and_summed_into_a_memory_bound(reg):
    A = einsums.create_random_tensor("A", [2, 3])  # occ x virt
    B = einsums.create_random_tensor("B", [3, 4])  # virt x aux
    E = einsums.create_random_tensor("E", [4, 2])  # aux x occ

    g = cg.Graph("intermediates")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    D = g.create_zero_tensor("D", [2, 2], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)
        einsums.einsum("ij <- ix ; xj", D, C, E)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)
    cg.annotate(E, ("aux", "occ"), graph=g)
    cg.annotate(C, ("occ", "aux"), graph=g)  # what SpacePropagation would infer

    scaling = cg.ScalingAnalysis()
    _run(scaling, g)

    assert scaling.intermediate_names() == ["C", "D"]
    assert scaling.intermediate_sizes_str() == ["o*x", "o^2"]
    # An upper bound, deliberately: the sum of the sizes, not a liveness-aware
    # high-water mark.
    assert scaling.memory_bound_str() == "o^2 + o*x"


def test_a_half_annotated_graph_still_reports_in_anonymous_variables(reg):
    A = einsums.create_random_tensor("A", [2, 3])  # occ x virt
    B = einsums.create_random_tensor("B", [3, 4])  # unannotated

    g = cg.Graph("half_annotated")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)

    scaling = cg.ScalingAnalysis()
    _run(scaling, g)

    assert scaling.num_analyzed == 1
    assert scaling.num_unannotated_nodes == 1
    # i and a resolve through A; x met no annotated slot and becomes the
    # anonymous letter variable.
    assert scaling.total_flops_str() == "2*o*v*?x"


def test_an_unannotated_graph_still_yields_a_report():
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 2])

    g = cg.Graph("unannotated")
    C = g.create_zero_tensor("C", [2, 2], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ia ; aj", C, A, B)

    scaling = cg.ScalingAnalysis()
    _run(scaling, g)

    assert scaling.num_analyzed == 1
    assert scaling.num_unannotated_nodes == 1
    assert scaling.total_flops_str() == "2*?a*?i*?j"


def test_the_report_string_carries_the_headline_numbers(reg):
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 4])

    g = cg.Graph("reporting")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    scaling = cg.ScalingAnalysis()
    _run(scaling, g)

    report = scaling.report_string()
    assert "ScalingAnalysis" in report
    assert "2*o*v*x" in report
    assert "memory bound" in report
    assert "rate-limiting" in report
    assert len(report) > 100


def test_an_empty_report_says_so():
    A = einsums.create_random_tensor("A", [2, 2])

    g = cg.Graph("no_contractions")
    C = g.create_zero_tensor("C", [2, 2], dtype="float64")
    with cg.capture(g):
        einsums.linalg.axpby(1.0, A, 1.0, C)

    scaling = cg.ScalingAnalysis()
    _run(scaling, g)

    assert scaling.num_analyzed == 0
    assert scaling.total_flops_str() == "0"
    assert "no contraction nodes analysed" in scaling.report_string()


def test_the_graph_still_executes_and_its_results_are_unchanged(reg):
    import numpy as np

    from einsums.testing import assert_close

    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 4])

    g = cg.Graph("executes")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    _run(cg.ScalingAnalysis(), g)
    g.execute()

    assert_close(C, np.asarray(A) @ np.asarray(B))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
