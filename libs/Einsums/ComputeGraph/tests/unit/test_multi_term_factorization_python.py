# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Joint contraction-order and shared-intermediate search, from Python.

A capability that is not reachable from Python is not shipped, and this module's
history is several misses of exactly that kind. The C++ cases prove the search;
these prove a person can construct the pass, switch it on, bound it, run it,
read its counters, and get the same numbers out the other side.

The shape is the one the C++ file describes:

    T1[i,l] = A[i,k] B[k,l]      R1[i,j] = T1[i,l] C[l,j]
    T2[k,j] = B[k,l] D[l,j]      R2[i,j] = A[i,k] T2[k,j]

Two three-factor products written with opposite bracketing. Nothing here is a
duplicate node and nothing here is a sum into one output, so neither CSE nor
DistributiveFactoring sees anything; the shared ``(A B)`` exists only once both
terms are looked at together.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close

# (A B) C is the cheaper bracketing for both terms with these, which is what makes
# building the shared product once worth more than the re-bracketing costs.
I, K, L, J = 3, 12, 3, 12


def _tensor(name, array, dtype):
    t = einsums.create_zero_tensor(name, list(array.shape), dtype=dtype)
    np.asarray(t)[...] = array.astype(dtype)
    return t


def _operands(seed=5):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((I, K)), rng.standard_normal((K, L)),
            rng.standard_normal((L, J)), rng.standard_normal((L, J)))


def _build(graph, a, b, c, d, dtype):
    A = _tensor("A", a, dtype)
    B = _tensor("B", b, dtype)
    C = _tensor("C", c, dtype)
    D = _tensor("D", d, dtype)
    R1 = _tensor("R1", np.zeros((I, J)), dtype)
    R2 = _tensor("R2", np.zeros((I, J)), dtype)
    T1 = graph.declare_tensor("T1", [I, L], intermediate=True, dtype=dtype)
    T2 = graph.declare_tensor("T2", [K, J], intermediate=True, dtype=dtype)
    with cg.capture(graph):
        einsums.einsum("i,l <- i,k ; k,l", T1, A, B)
        einsums.einsum("i,j <- i,l ; l,j", R1, T1, C)
        einsums.einsum("k,j <- k,l ; l,j", T2, B, D)
        einsums.einsum("i,j <- i,k ; k,j", R2, A, T2)
    return R1, R2, (A, B, C, D, T1, T2)


def _expected(a, b, c, d):
    return np.einsum("ik,kl,lj->ij", a, b, c), np.einsum("ik,kl,lj->ij", a, b, d)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_the_shared_product_is_found_and_the_answer_holds(dtype):
    a, b, c, d = _operands()

    graph = cg.Graph("mtf")
    R1, R2, _pool = _build(graph, a, b, c, d, dtype)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.add(mtf)
    pm.add(cg.Materialization())
    assert pm.run(graph), "the pass reported no change on a graph with a product to share"
    graph.execute()

    assert mtf.num_shared == 1
    assert mtf.num_rebracketed == 2
    assert mtf.num_inlined == 2
    assert not mtf.was_cut_off

    want1, want2 = _expected(a, b, c, d)
    assert_close(np.asarray(R1), want1.astype(dtype), dtype=dtype)
    assert_close(np.asarray(R2), want2.astype(dtype), dtype=dtype)


def test_the_search_is_off_unless_it_is_asked_for():
    a, b, c, d = _operands()

    graph = cg.Graph("mtf-off")
    _R1, _R2, _pool = _build(graph, a, b, c, d, "float64")

    mtf = cg.MultiTermFactorization()
    assert not mtf.search_enabled
    pm = cg.PassManager()
    pm.add(mtf)
    pm.set_verbosity(2)  # the skip tally is what a decline is read through
    assert not pm.run(graph)
    assert mtf.num_shared == 0
    assert "structural search is switched off" in pm.explain()


def test_the_second_identical_graph_replays_the_plan():
    """The case the cache exists for: one search, then a replay per stage."""
    a, b, c, d = _operands()

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    assert mtf.cache_enabled
    pm = cg.PassManager()
    pm.add(mtf)

    first = cg.Graph("mtf-cache")
    _R1, _R2, _pool = _build(first, a, b, c, d, "float64")
    assert pm.run(first)
    assert mtf.num_cache_misses == 1
    assert mtf.num_cache_hits == 0
    assert mtf.cache_size == 1

    # The same program under the same name, with fresh tensors and fresh ids. The plan is written
    # in positions, which is what lets it apply.
    second = cg.Graph("mtf-cache")
    R1, R2, _pool2 = _build(second, a, b, c, d, "float64")
    assert pm.run(second)
    assert mtf.num_cache_hits == 1
    assert mtf.num_cache_misses == 0
    assert mtf.num_shared == 1
    assert mtf.num_rebracketed == 2

    pm2 = cg.PassManager()
    pm2.add(cg.Materialization())
    pm2.run(second)
    second.execute()
    want1, want2 = _expected(a, b, c, d)
    assert_close(np.asarray(R1), want1, dtype="float64")
    assert_close(np.asarray(R2), want2, dtype="float64")


def test_the_cache_can_be_switched_off_and_cleared():
    a, b, c, d = _operands()

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    mtf.set_cache_enabled(False)
    assert not mtf.cache_enabled
    pm = cg.PassManager()
    pm.add(mtf)

    graph = cg.Graph("mtf-cache-off")
    _R1, _R2, _pool = _build(graph, a, b, c, d, "float64")
    assert pm.run(graph)
    assert mtf.cache_size == 0

    mtf.set_cache_enabled(True)
    again = cg.Graph("mtf-cache-off")
    _R3, _R4, _pool2 = _build(again, a, b, c, d, "float64")
    assert pm.run(again)
    assert mtf.cache_size == 1

    mtf.clear_cache()
    assert mtf.cache_size == 0


def test_the_report_names_what_it_shared():
    a, b, c, d = _operands()

    graph = cg.Graph("mtf-report")
    _R1, _R2, _pool = _build(graph, a, b, c, d, "float64")

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.add(mtf)
    assert pm.run(graph)

    report = pm.explain()
    assert "MultiTermFactorization" in report
    assert "shared intermediate" in report
    assert "structural-algebraic" in report


def test_a_budget_can_be_set_from_python():
    a, b, c, d = _operands()

    graph = cg.Graph("mtf-budget")
    _R1, _R2, _pool = _build(graph, a, b, c, d, "float64")

    # Generous enough that this search finishes inside it, which is the point: the budget bounds
    # the wait and exhausting it costs optimization rather than correctness.
    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.add(mtf)
    pm.set_optimizer_budget(30000)
    assert pm.run(graph)
    assert not mtf.was_cut_off
    assert mtf.num_shared == 1


def test_the_factor_cap_declines_rather_than_approximating():
    a, b, c, d = _operands()

    graph = cg.Graph("mtf-cap")
    _R1, _R2, _pool = _build(graph, a, b, c, d, "float64")

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    mtf.set_max_factors(2)  # both terms flatten to three factors
    assert mtf.max_factors == 2
    pm = cg.PassManager()
    pm.add(mtf)
    pm.set_verbosity(2)
    assert not pm.run(graph)
    assert "not a product this pass can model" in pm.explain()
