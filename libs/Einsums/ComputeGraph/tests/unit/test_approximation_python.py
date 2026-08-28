# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The accuracy contract, from Python.

The C++ cases prove the arithmetic; these prove a person can get at it, which the passing
C++ never does. Two of the gaps this file exists to catch were live while it was written:
the record's natural field name is ``pass``, which is a Python keyword and would have bound
to something no Python source can write down, and the struct had no constructor at all, so
the two methods that take one were unreachable.

The last case is the point of the whole part. A differential comparison against an eager
oracle reads the graph's records and widens by exactly what they say, so a lossy pass is
validated against the bound it declared rather than against bit equality it was never going
to meet.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import assert_close, tolerance_for_graph


def _tensor(name, array):
    t = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(t)[...] = array
    return t


def _graph(name):
    """One contraction, so there is a manifest to name outputs in."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 4))
    b = rng.standard_normal((4, 5))
    A = _tensor("A", a)
    B = _tensor("B", b)
    C = _tensor("C", np.zeros((3, 5)))

    g = cg.Graph(name)
    with cg.capture(g):
        einsums.einsum("i,k ; k,j -> i,j", C, A, B)
    return g, A, B, C, a, b


def test_a_graph_with_no_records_is_in_exact_mode():
    g, *_ = _graph("exact")
    assert g.approximations() == []
    tolerance = g.approximation_tolerance()
    assert tolerance.relative == 0.0
    assert tolerance.absolute == 0.0


def test_a_record_is_constructible_and_reads_back():
    g, *_ = _graph("record")
    record = cg.approximation_record(
        "AutoDF", cg.ApproximationEffect.NormRelative, 1e-5, 4e-6, ["C"], [], "df_fit"
    )
    # The field is `pass_name`, not `pass`: the shorter spelling is a Python keyword and
    # `record.pass` would not parse.
    assert record.pass_name == "AutoDF"
    assert record.effect == cg.ApproximationEffect.NormRelative
    assert record.setup == "df_fit"

    g.note_approximation(record)
    assert len(g.approximations()) == 1
    assert g.approximations()[0].pass_name == "AutoDF"
    assert g.approximations()[0].bound == pytest.approx(4e-6)


def test_a_relative_effect_composes_with_the_product_term():
    g, *_ = _graph("compose")
    g.note_approximation(cg.approximation_record("First", cg.ApproximationEffect.NormRelative, 0.1, 0.1, ["C"]))
    g.note_approximation(cg.approximation_record("Second", cg.ApproximationEffect.NormRelative, 0.2, 0.2, ["C"]))

    spent = g.accuracy_spent(cg.ApproximationEffect.NormRelative, "C")
    assert spent == pytest.approx(0.32)
    assert spent > 0.1 + 0.2


def test_a_budget_refuses_the_record_that_would_exceed_it():
    g, *_ = _graph("budget")
    g.set_accuracy_budget(cg.ApproximationEffect.NormRelative, 1e-4)
    assert g.accuracy_budget_value == pytest.approx(1e-4)

    small = cg.approximation_record("Small", cg.ApproximationEffect.NormRelative, 6e-5, 6e-5, ["C"])
    assert g.can_approximate(small) == ""
    g.note_approximation(small)

    second = cg.approximation_record("Second", cg.ApproximationEffect.NormRelative, 6e-5, 6e-5, ["C"])
    reason = g.can_approximate(second)
    assert reason != ""
    assert "budget" in reason
    with pytest.raises(Exception):
        g.note_approximation(second)

    g.clear_accuracy_budget()
    assert g.accuracy_budget_value < 0
    assert g.can_approximate(second) == ""


def test_records_survive_a_save_and_a_load(tmp_path):
    g, *_ = _graph("saved")
    g.note_approximation(cg.approximation_record("AutoDF", cg.ApproximationEffect.NormRelative, 1e-5, 4e-6, ["C"], [], "df_fit"))
    g.note_approximation(cg.approximation_record("AutoTHC", cg.ApproximationEffect.NormRelative, 2e-6, 2e-6, ["C"]))

    path = str(tmp_path / "approx.eig")
    cg.save_graph(g, path)
    loaded = cg.load_graph(path)

    assert [r.pass_name for r in loaded.approximations()] == ["AutoDF", "AutoTHC"]
    assert loaded.approximations()[0].setup == "df_fit"
    assert loaded.approximation_tolerance("C").relative == pytest.approx(
        g.approximation_tolerance("C").relative
    )


def test_the_oracle_comparison_widens_by_what_the_records_say():
    """Part 5.4's tolerance-aware differential mode, end to end.

    The graph computes the exact contraction, so the only thing standing between it and the
    numpy oracle is floating-point rounding. A record is then added claiming a 1% relative
    approximation, and a perturbed "result" of that size is compared: it fails at the exact
    tolerance and passes at the widened one. That is the whole mechanism, and it is asserted
    in both directions so a records-absent graph cannot quietly be widening anything.
    """
    g, _A, _B, C, a, b = _graph("differential")
    g.execute()
    exact = a @ b

    # Exact mode: records absent, tolerance is the dtype default, and it holds.
    assert_close(C, exact, graph=g)
    assert tolerance_for_graph(g, "C", dtype="float64") == pytest.approx(
        tolerance_for_graph(None, "C", dtype="float64")
    )

    perturbed = exact * 1.005  # half a percent off, which no dtype tolerance admits
    with pytest.raises(AssertionError):
        assert_close(perturbed, exact, graph=g, output="C")

    g.note_approximation(cg.approximation_record("Coarse", cg.ApproximationEffect.NormRelative, 1e-2, 1e-2, ["C"]))
    rtol, _atol = tolerance_for_graph(g, "C", dtype="float64")
    assert rtol >= 1e-2

    # The same comparison, against a graph that now says what it cost.
    assert_close(perturbed, exact, graph=g, output="C")

    # And an output the record does not name is still held to the exact tolerance.
    with pytest.raises(AssertionError):
        assert_close(perturbed, exact, graph=g, output="A")


def test_an_absolute_record_widens_the_absolute_side_only():
    g, *_ = _graph("sides")
    g.note_approximation(cg.approximation_record("Rounding", cg.ApproximationEffect.ElementWise, 1e-6, 1e-6, ["C"]))

    rtol, atol = tolerance_for_graph(g, "C", dtype="float64")
    base_rtol, base_atol = tolerance_for_graph(None, "C", dtype="float64")
    assert rtol == pytest.approx(base_rtol)
    assert atol == pytest.approx(base_atol + 1e-6)


def test_every_effect_is_reachable_by_name():
    for effect in (
        cg.ApproximationEffect.ElementWise,
        cg.ApproximationEffect.NormRelative,
        cg.ApproximationEffect.EnergyLike,
    ):
        record = cg.approximation_record("P", effect, 1e-6, 1e-6)
        assert record.effect == effect
