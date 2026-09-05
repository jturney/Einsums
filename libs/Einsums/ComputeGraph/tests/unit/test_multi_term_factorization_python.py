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

import json

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
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


# The CCSD doubles residual receives one three-factor product twice, once through each of two
# intermediates that bracket it differently. Wmnij contracts tau with the integrals over the
# virtual pair first and holds an o^4 tensor; Wabef contracts over the occupied pair first and
# holds a v^4 one. Hand-optimized codes move the whole term into Wmnij; the pass has to find that
# from the graph. The spin-orbital equations are in examples/toy/ccsd_t_spinorbital_toy.py.
O, V = 4, 10


def _ccsd_operands(seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((O, O, V, V)), rng.standard_normal((O, O, V, V))


def _build_ccsd_tau_terms(graph, tau_np, oovv_np, dtype):
    tau = _tensor("tau", tau_np, dtype)
    oovv = _tensor("oovv", oovv_np, dtype)
    t2n = _tensor("t2n", np.zeros((O, O, V, V)), dtype)
    Wmnij = graph.declare_tensor("Wmnij_tau", [O, O, O, O], intermediate=True, dtype=dtype)
    Wabef = graph.declare_tensor("Wabef_tau", [V, V, V, V], intermediate=True, dtype=dtype)
    with cg.capture(graph):
        einsums.einsum("m,n,i,j <- i,j,e,f ; m,n,e,f", Wmnij, tau, oovv)
        einsums.einsum("i,j,a,b <- m,n,a,b ; m,n,i,j", t2n, tau, Wmnij, ab_pf=0.125)
        einsums.einsum("a,b,e,f <- m,n,a,b ; m,n,e,f", Wabef, tau, oovv)
        einsums.einsum("i,j,a,b <- i,j,e,f ; a,b,e,f", t2n, tau, Wabef, c_pf=1.0, ab_pf=0.125)
    return t2n, (tau, oovv, Wmnij, Wabef)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_the_ccsd_tau_terms_share_the_occupied_intermediate(dtype):
    tau_np, oovv_np = _ccsd_operands()
    graph = cg.Graph("ccsd-tau")
    t2n, _pool = _build_ccsd_tau_terms(graph, tau_np, oovv_np, dtype)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    mat = cg.Materialization()
    pm = cg.PassManager()
    pm.add(mtf)
    pm.add(mat)
    assert pm.run(graph)
    assert mtf.num_inlined == 2
    assert mtf.num_shared == 1
    assert mtf.num_rebracketed == 2
    assert not mtf.was_cut_off

    # Both routes now go through one o^4 intermediate, and the v^4 one is gone.
    ir = json.loads(graph.to_json())
    dims = {t["name"]: t["dims"] for t in ir["tensors"]}
    assert [name for name in dims if name.startswith("mtf_shared")] == ["mtf_shared0"]
    assert dims["mtf_shared0"] == [O, O, O, O]
    assert sum(1 for n in ir["nodes"] if n["kind"] == "Einsum") == 3

    # Gone means not allocated either. The dissolved declarations stay, since the caller holds
    # them, but a buffer for a tensor nothing writes is exactly the v^4 cost the rewrite removed.
    assert {n["label"] for n in ir["nodes"] if n["kind"] == "Materialize"} == {"materialize(mtf_shared0)"}
    assert mat.num_unused == 2

    # The report prices the rewrite it emitted: the v^4 loop space is on the before side only,
    # and the after side is not an empty region.
    cost_line = next(line for line in pm.explain().splitlines() if "MultiTermFactorization" in line and " cost " in line)
    before, after = cost_line.split(" cost ", 1)[1].split(" -> ")
    assert "?a*?b*?e*?f" in before
    assert "?a*?b*?e*?f" not in after
    assert after.strip() != "0"

    graph.execute()

    # Re-associating, so the bar is the tier's norm-relative bound against the same program run
    # without the search in the same dtype, as the C++ cases hold it: the shared form sums the
    # same products in a different order, and an elementwise tolerance on a sum of 1600 terms
    # with cancellation would fail on where the small values landed rather than on the answer.
    # The constant is tier_bound(ReAssociating), which is 1024 epsilon.
    plain = cg.Graph("ccsd-tau-plain")
    t2n_plain, _pool_plain = _build_ccsd_tau_terms(plain, tau_np, oovv_np, dtype)
    pm_plain = cg.PassManager()
    pm_plain.add(cg.Materialization())
    pm_plain.run(plain)
    plain.execute()
    got, reference = np.asarray(t2n), np.asarray(t2n_plain)
    assert np.linalg.norm(got - reference) / np.linalg.norm(reference) <= 1024 * np.finfo(np.dtype(dtype)).eps

    # And one anchor in double against numpy, so the two graphs cannot agree on a wrong prefactor.
    if dtype == "float64":
        want = 0.25 * np.einsum("ijef,mnef,mnab->ijab", tau_np, oovv_np, tau_np)
        assert_close(got, want, dtype=dtype)


def test_the_reported_cost_agrees_with_the_nodes_it_emitted():
    """The cost line as a SECOND derivation rather than a claim.

    The report prints what a region cost before and after, and it is derived
    from the algebra alone: a term the rewrite builds carries whatever cost the
    rewrite gave it. Nothing compared that to the nodes the lowering then
    emitted, and the after side read zero on every rewrite this pass had ever
    made. With ``set_verify_costs`` on, the before side is checked against the
    flops of the region's own nodes and the after side against the flops of the
    nodes it emitted, both through the symbolic cost the analysis pass uses.
    """
    tau_np, oovv_np = _ccsd_operands()
    graph = cg.Graph("mtf-cost-check")
    _t2n, _pool = _build_ccsd_tau_terms(graph, tau_np, oovv_np, "float64")

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    mtf.set_verify_costs(True)
    pm = cg.PassManager()
    pm.add(mtf)
    assert pm.run(graph)
    assert mtf.num_shared == 1
    assert mtf.cost_mismatches == [], mtf.cost_mismatches

    # And the check is off unless it is asked for, so the default pipeline pays
    # nothing for it.
    quiet = cg.MultiTermFactorization()
    quiet.set_search_enabled(True)
    second = cg.Graph("mtf-cost-check-off")
    _r, _p = _build_ccsd_tau_terms(second, tau_np, oovv_np, "float64")
    pm2 = cg.PassManager()
    pm2.add(quiet)
    assert pm2.run(second)
    assert quiet.cost_mismatches == []


# ──────────────────────────────────────────────────────────────────────────
# The other two kinds the flattener reads as a product
#
# A correlation energy is not written as a chain of contractions. It is a
# contraction, an elementwise scaling and a reduction to a scalar, and until
# the flattener read all three the amplitude in the middle survived as a stored
# leaf that every candidate had to rebuild. These pin the widening: the
# elementwise intermediate never exists, the reduction becomes an ordinary
# summed letter, and the answer holds.
# ──────────────────────────────────────────────────────────────────────────

#: Extents where re-bracketing the flattened energy pays. The reduction to a
#: scalar is what makes it pay: with no free index to keep, the two operands of
#: the contraction can be folded in one at a time instead of building the whole
#: matrix first.
EI, EJ, EK, EL = 12, 12, 2, 2


def _energy_shaped(graph, arrays, dtype):
    """``s = sum_ij (A B)[i,j] (C D)[i,j] V[i,j]``, written the way a caller does."""
    A = _tensor("A", arrays[0], dtype)
    B = _tensor("B", arrays[1], dtype)
    C = _tensor("C", arrays[2], dtype)
    D = _tensor("D", arrays[3], dtype)
    V = _tensor("V", arrays[4], dtype)
    energy = _tensor("E", np.zeros((1,)), dtype)
    X = graph.declare_tensor("X", [EI, EJ], intermediate=True, dtype=dtype)
    Y = graph.declare_tensor("Y", [EI, EJ], intermediate=True, dtype=dtype)
    M = graph.declare_tensor("M", [EI, EJ], intermediate=True, dtype=dtype)
    with cg.capture(graph):
        einsums.einsum("i,j <- i,k ; k,j", X, A, B)
        einsums.einsum("i,j <- i,l ; l,j", Y, C, D)
        la.direct_product(1.0, X, Y, 0.0, M)
        la.dot(energy, M, V)
    return energy, (A, B, C, D, V, X, Y, M)


def _energy_operands(seed=11):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((EI, EK)), rng.standard_normal((EK, EJ)),
            rng.standard_normal((EI, EL)), rng.standard_normal((EL, EJ)),
            rng.standard_normal((EI, EJ)))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_a_scaled_product_reduced_by_a_dot_is_re_associated(dtype):
    arrays = _energy_operands()
    graph = cg.Graph("mtf-energy")
    energy, _pool = _energy_shaped(graph, arrays, dtype)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    mtf.set_verify_costs(True)
    pm = cg.PassManager()
    pm.add(mtf)
    assert pm.run(graph), pm.explain()

    # Five factors out of four statements: both contractions and the direct
    # product are dissolved, and nothing was shared, so this is the search
    # re-bracketing on its own.
    assert mtf.num_inlined == 3
    assert mtf.num_rebracketed == 1
    assert mtf.num_shared == 0
    assert mtf.cost_mismatches == [], mtf.cost_mismatches

    ir = json.loads(graph.to_json())
    kinds = {node["kind"] for node in ir["nodes"]}
    assert kinds == {"Einsum"}, f"the direct product and the dot survive: {kinds}"

    # The elementwise intermediate never exists. Its declaration stays, because
    # the caller holds the handle, and Materialization leaves an unused deferred
    # intermediate unallocated.
    pm2 = cg.PassManager()
    pm2.add(cg.Materialization())
    pm2.run(graph)
    materialized = {node["label"] for node in json.loads(graph.to_json())["nodes"]
                    if node["kind"] == "Materialize"}
    for dissolved in ("X", "Y", "M"):
        assert f"materialize({dissolved})" not in materialized, materialized

    graph.execute()
    expected = np.sum((arrays[0] @ arrays[1]) * (arrays[2] @ arrays[3]) * arrays[4])
    assert_close(np.asarray(energy)[0], np.asarray(expected).astype(dtype), dtype=dtype, rtol=1e-3)


def test_the_nine_factor_product_is_within_the_default_cap():
    """The cap admits the opposite-spin energy, which is what it was raised for."""
    assert cg.MultiTermFactorization().max_factors == 10


def test_an_accumulating_direct_product_is_not_folded():
    """``C = A*B + C`` is more than one value, so dissolving it would drop the rest."""
    arrays = _energy_operands(seed=13)
    graph = cg.Graph("mtf-energy-rmw")
    A = _tensor("A", arrays[0], "float64")
    B = _tensor("B", arrays[1], "float64")
    V = _tensor("V", arrays[4], "float64")
    seed = _tensor("M0", np.ones((EI, EJ)), "float64")
    energy = _tensor("E", np.zeros((1,)), "float64")
    X = graph.declare_tensor("X", [EI, EJ], intermediate=True, dtype="float64")
    with cg.capture(graph):
        einsums.einsum("i,j <- i,k ; k,j", X, A, B)
        la.direct_product(1.0, X, V, 1.0, seed)   # accumulates into a caller tensor
        la.dot(energy, seed, V)
    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.add(mtf)
    pm.run(graph)
    kinds = {node["kind"] for node in json.loads(graph.to_json())["nodes"]}
    assert "DirectProduct" in kinds

    graph.apply(cg.default_pass_manager())
    graph.execute()
    expected = np.sum((np.ones((EI, EJ)) + (arrays[0] @ arrays[1]) * arrays[4]) * arrays[4])
    assert_close(np.asarray(energy)[0], np.asarray(expected), dtype="float64")
