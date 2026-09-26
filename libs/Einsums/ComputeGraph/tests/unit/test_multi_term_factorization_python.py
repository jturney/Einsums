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

#: Why every pipeline below sets ``set_optimizer_budget(0)``, marked at each site
#: rather than repeated: a search that runs out of its wall-clock allowance keeps
#: the best candidate it has reached, so it emits a VALID graph and a DIFFERENT
#: one. A case asserting what the pass produced would then be asserting how fast
#: the machine is, and a CI runner is slower than the machine the case was
#: written on, which is where that first shows. The per-pipeline setting wins over
#: ``einsums:graph:optimizer-budget``; the one case that pins an allowance sets its
#: own, and ``was_cut_off`` is what says the removal took.
_NO_ALLOWANCE = "einsums:graph:optimizer-budget"

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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm2.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm_plain.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm_plain.add(cg.Materialization())
    pm_plain.run(plain)
    plain.execute()
    got, reference = np.asarray(t2n), np.asarray(t2n_plain)
    assert np.linalg.norm(got - reference) / np.linalg.norm(reference) <= 1024 * np.finfo(np.dtype(dtype)).eps

    # And one anchor in double against numpy, so the two graphs cannot agree on a wrong prefactor.
    if dtype == "float64":
        want = 0.25 * np.einsum("ijef,mnef,mnab->ijab", tau_np, oovv_np, tau_np)
        assert_close(got, want, dtype=dtype)


def _build_ccsd_tau_iteration(graph, tau_np, oovv_np, dtype, iterations):
    """The same four contractions, captured as the loop body they are in a solver.

    The tiled CCSD example writes its iteration this way: a ``Loop`` whose body
    is the whole residual. Nothing about the algebra changes; what changes is
    that the statements are in a sub-graph, which is what a region rewrite could
    not see until it descended.

    ``tau`` is read and never written here, so the body is a repetition of one
    residual rather than an amplitude update. That is deliberate: what is under
    test is that the rewrite fires inside a body and replays correctly, not the
    solver, and a body whose inputs move would make the comparison against the
    flat capture a comparison of two different programs.
    """
    tau = _tensor("tau", tau_np, dtype)
    oovv = _tensor("oovv", oovv_np, dtype)
    t2n = _tensor("t2n", np.zeros((O, O, V, V)), dtype)
    body = graph.add_loop("ccsd_iteration", iterations, lambda it, c=iterations: it < c - 1)
    Wmnij = body.declare_tensor("Wmnij_tau", [O, O, O, O], intermediate=True, dtype=dtype)
    Wabef = body.declare_tensor("Wabef_tau", [V, V, V, V], intermediate=True, dtype=dtype)
    with cg.capture(body):
        einsums.einsum("m,n,i,j <- i,j,e,f ; m,n,e,f", Wmnij, tau, oovv)
        einsums.einsum("i,j,a,b <- m,n,a,b ; m,n,i,j", t2n, tau, Wmnij, ab_pf=0.125)
        einsums.einsum("a,b,e,f <- m,n,a,b ; m,n,e,f", Wabef, tau, oovv)
        einsums.einsum("i,j,a,b <- i,j,e,f ; a,b,e,f", t2n, tau, Wabef, c_pf=1.0, ab_pf=0.125)
    return t2n, body


def test_the_tau_terms_share_the_occupied_intermediate_inside_a_loop_body():
    """The bite test for descending into loop bodies.

    A coupled-cluster iteration is a ``Loop`` whose body is the whole residual,
    so a framework that stopped at the loop header could never see the
    expression this pass exists to rewrite. The flat capture of these same four
    contractions has been pinned since the pass shipped; this is the identical
    algebra written as the iteration it is, and it has to reach the identical
    rewrite.
    """
    iterations = 3
    tau_np, oovv_np = _ccsd_operands()
    graph = cg.Graph("ccsd-tau-loop")
    t2n, body = _build_ccsd_tau_iteration(graph, tau_np, oovv_np, "float64", iterations)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    pm.add(cg.Materialization())
    assert pm.run(graph), mtf.skip_reasons

    # The same counters the flat case pins, reached from inside a body.
    assert mtf.num_inlined == 2
    assert mtf.num_shared == 1
    assert mtf.num_rebracketed == 2
    assert not mtf.was_cut_off

    # And the same shapes: one o^4 intermediate, three contractions, no v^4 tensor.
    # Read off the BODY, which is where the rewrite landed; the parent's own node
    # list holds one Loop and nothing else.
    outer = [n["kind"] for n in json.loads(graph.to_json())["nodes"]]
    assert outer.count("Loop") == 1 and "Einsum" not in outer, outer
    inner = json.loads(body.to_json())
    dims = {t["name"]: t["dims"] for t in inner["tensors"]}
    assert [name for name in dims if name.startswith("mtf_shared")] == ["mtf_shared0"]
    assert dims["mtf_shared0"] == [O, O, O, O]
    assert sum(1 for n in inner["nodes"] if n["kind"] == "Einsum") == 3

    # Replayed over several iterations against the flat capture run the same number
    # of times, which is what says the rewrite survived being in a body rather than
    # merely happening there.
    graph.execute()

    flat = cg.Graph("ccsd-tau-flat-reference")
    t2n_flat, _pool = _build_ccsd_tau_terms(flat, tau_np, oovv_np, "float64")
    pm_flat = cg.PassManager()
    pm_flat.add(cg.Materialization())
    pm_flat.run(flat)
    for _ in range(iterations):
        flat.execute()

    got, reference = np.asarray(t2n), np.asarray(t2n_flat)
    assert np.linalg.norm(got - reference) / np.linalg.norm(reference) <= 1024 * np.finfo(np.float64).eps


def test_a_setup_body_is_not_a_region_this_framework_rewrites():
    """A fitting is not an expression to re-associate on this pass's own terms.

    A setup body runs once per bound problem, and a rewrite of one would have
    nowhere to hoist a setup of its own; the once-per-bind contract is a property
    of the node rather than of the algebra inside it. So the descent stops there,
    by name, and says so rather than skipping silently.
    """
    tau_np, oovv_np = _ccsd_operands()
    graph = cg.Graph("mtf-setup-body")
    t2n = _tensor("t2n", np.zeros((O, O, V, V)), "float64")
    tau = _tensor("tau", tau_np, "float64")
    oovv = _tensor("oovv", oovv_np, "float64")
    fit = graph.add_setup("a_fitting")
    Wmnij = fit.declare_tensor("Wmnij_fit", [O, O, O, O], intermediate=True, dtype="float64")
    with cg.capture(fit):
        einsums.einsum("m,n,i,j <- i,j,e,f ; m,n,e,f", Wmnij, tau, oovv)
        einsums.einsum("i,j,a,b <- m,n,a,b ; m,n,i,j", t2n, tau, Wmnij, ab_pf=0.125)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    assert not pm.run(graph)
    assert any("setup body" in reason for reason, _count in mtf.skip_reasons), mtf.skip_reasons


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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm2.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
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
    pm2.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm2.add(cg.Materialization())
    pm2.run(graph)
    materialized = {node["label"] for node in json.loads(graph.to_json())["nodes"]
                    if node["kind"] == "Materialize"}
    for dissolved in ("X", "Y", "M"):
        assert f"materialize({dissolved})" not in materialized, materialized

    graph.execute()
    expected = np.sum((arrays[0] @ arrays[1]) * (arrays[2] @ arrays[3]) * arrays[4])
    assert_close(np.asarray(energy)[0], np.asarray(expected).astype(dtype), dtype=dtype, rtol=1e-3)


def test_the_cap_comes_from_the_option_and_admits_the_grid_fitted_energy():
    """The default admits fourteen leaves, which is the grid-fitted opposite-spin energy.

    The cap is a property of the program a caller brings rather than of the pass, so
    it is an option with a per-pipeline override, the same two-level shape
    ``search_enabled`` has. Nine leaves is the density-fitted energy and fourteen is
    the grid-fitted one; both are inside the default.
    """
    assert cg.MultiTermFactorization().max_factors == 14

    explicit = cg.MultiTermFactorization()
    explicit.set_max_factors(20)
    assert explicit.max_factors == 20, "an explicit cap has to win over the option"


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
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    pm.run(graph)
    kinds = {node["kind"] for node in json.loads(graph.to_json())["nodes"]}
    assert "DirectProduct" in kinds

    graph.apply(cg.default_pass_manager())
    graph.execute()
    expected = np.sum((np.ones((EI, EJ)) + (arrays[0] @ arrays[1]) * arrays[4]) * arrays[4])
    assert_close(np.asarray(energy)[0], np.asarray(expected), dtype="float64")


# ──────────────────────────────────────────────────────────────────────────
# Per-consumer inlining
#
# A definition several statements read used to stay a stored leaf that pinned
# the algebra around it. It is now inlined into each consumer whose bracketing
# it improves and KEPT for the rest, so the shapes below are about which
# consumer took what and how many contractions form the value afterwards.
# ──────────────────────────────────────────────────────────────────────────

#: A matrix product with a short inner dimension, so contracting a vector into
#: its right factor first is much cheaper than forming the product and then
#: contracting the vector into it. That is what makes one consumer profit.
CI, CJ, CK = 32, 32, 4


def _copy_operands(seed=17):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((CI, CK)), rng.standard_normal((CK, CJ)),
            rng.standard_normal((CJ,)), rng.standard_normal((CI, CJ)))


def _one_profits(graph, arrays, dtype="float64"):
    """``M`` read by a contraction that gains from inlining and a scaling that does not.

    ``R1[i] = M[i,j] v[j]`` re-brackets into ``A (B v)`` and drops a whole
    factor of the loop space. ``R2 = M * w`` cannot: every bracketing of
    ``A B w`` that does not form ``A B`` first costs more than forming it, so
    that consumer reads the definition instead.
    """
    a, b, v, w = arrays
    A = _tensor("A", a, dtype)
    B = _tensor("B", b, dtype)
    V = _tensor("v", v, dtype)
    W = _tensor("w", w, dtype)
    R1 = _tensor("R1", np.zeros((CI,)), dtype)
    R2 = _tensor("R2", np.zeros((CI, CJ)), dtype)
    M = graph.declare_tensor("M", [CI, CJ], intermediate=True, dtype=dtype)
    with cg.capture(graph):
        einsums.einsum("i,j <- i,k ; k,j", M, A, B)
        einsums.einsum("i <- i,j ; j", R1, M, V)
        la.direct_product(1.0, M, W, 0.0, R2)
    return M, R1, R2


def test_a_definition_one_consumer_profits_from_is_copied_and_kept():
    """The rule, at its smallest: one copy, one definition, and both answers.

    The mechanism rather than the outcome. Exactly one consumer takes a copy of
    ``M``, the other reads it, the definition survives as the single contraction
    that forms it, and the pass says so through its counters. Under the
    resolves-to-one-consumer rule that preceded this, ``M`` had two consumers
    and stayed a stored leaf, so neither consumer was re-bracketed and the pass
    declined the region outright.
    """
    arrays = _copy_operands()
    graph = cg.Graph("mtf-one-consumer-profits")
    _M, R1, R2 = _one_profits(graph, arrays)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    pm.add(cg.Materialization())
    assert pm.run(graph), mtf.skip_reasons

    # Kept, not dissolved, and copied exactly once.
    assert mtf.num_inlined == 0, mtf.skip_reasons
    assert mtf.num_copies == 1
    assert mtf.num_shared == 0

    # The consumer that gained nothing said so, and the reason names both ends.
    assert any("buys that consumer nothing" in reason for reason, _count in mtf.skip_reasons), mtf.skip_reasons

    # The emitted node set: M is formed once, R1 goes through the short
    # intermediate the re-bracketing introduced, and R2 reads M.
    ir = json.loads(graph.to_json())
    dims = {t["id"]: t["dims"] for t in ir["tensors"]}
    names = {t["id"]: t["name"] for t in ir["tensors"]}
    einsum_nodes = [n for n in ir["nodes"] if n["kind"] == "Einsum"]
    assert len(einsum_nodes) == 4, [n["label"] for n in einsum_nodes]
    written = [names[t] for n in einsum_nodes for t in n["outputs"]]
    assert written.count("M") == 1, written
    scratch = [name for name in written if "_mtf_t" in name]
    assert len(scratch) == 1, written
    assert dims[next(t for t, name in names.items() if name == scratch[0])] == [CK], (
        "the re-bracketing did not go through B v")

    graph.execute()
    a, b, v, w = arrays
    assert_close(np.asarray(R1), (a @ b) @ v, dtype="float64")
    assert_close(np.asarray(R2), (a @ b) * w, dtype="float64")


def test_a_definition_no_consumer_profits_from_is_left_whole():
    """Two consumers, neither gaining, and the definition is not copied at all.

    The other half of the rule. A copy that only adds arithmetic is refused
    where it lands, so a value two scalings read is computed once and read
    twice, which is what the captured program already said.
    """
    a, b, _v, w = _copy_operands()
    rng = np.random.default_rng(23)
    second = rng.standard_normal((CI, CJ))
    graph = cg.Graph("mtf-neither-consumer-profits")
    A = _tensor("A", a, "float64")
    B = _tensor("B", b, "float64")
    W1 = _tensor("w1", w, "float64")
    W2 = _tensor("w2", second, "float64")
    R1 = _tensor("R1", np.zeros((CI, CJ)), "float64")
    R2 = _tensor("R2", np.zeros((CI, CJ)), "float64")
    M = graph.declare_tensor("M", [CI, CJ], intermediate=True, dtype="float64")
    with cg.capture(graph):
        einsums.einsum("i,j <- i,k ; k,j", M, A, B)
        la.direct_product(1.0, M, W1, 0.0, R1)
        la.direct_product(1.0, M, W2, 0.0, R2)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    pm.add(cg.Materialization())
    pm.run(graph)
    assert mtf.num_inlined == 0
    assert mtf.num_copies == 0
    assert sum(count for reason, count in mtf.skip_reasons
               if "buys that consumer nothing" in reason) == 2, mtf.skip_reasons

    kinds = [n["kind"] for n in json.loads(graph.to_json())["nodes"]]
    assert kinds.count("DirectProduct") == 2, kinds

    graph.execute()
    assert_close(np.asarray(R1), (a @ b) * w, dtype="float64")
    assert_close(np.asarray(R2), (a @ b) * second, dtype="float64")


def _many_consumers(graph, arrays, count):
    """``M`` read by @p count contractions, every one of which would profit."""
    a, b = arrays[0], arrays[1]
    A = _tensor("A", a, "float64")
    B = _tensor("B", b, "float64")
    M = graph.declare_tensor("M", [CI, CJ], intermediate=True, dtype="float64")
    rng = np.random.default_rng(29)
    vectors = [rng.standard_normal((CJ,)) for _ in range(count)]
    held = [_tensor(f"v{n}", vec, "float64") for n, vec in enumerate(vectors)]
    results = [_tensor(f"R{n}", np.zeros((CI,)), "float64") for n in range(count)]
    with cg.capture(graph):
        einsums.einsum("i,j <- i,k ; k,j", M, A, B)
        for vector, result in zip(held, results):
            einsums.einsum("i <- i,j ; j", result, M, vector)
    return results, vectors


def test_a_definition_more_consumers_than_the_cap_admits_is_left_whole():
    """The growth bound, and the cap that states it.

    Every copy is a term of its own, so a definition many statements read
    multiplies the search's size by how many of them there are. Five consumers
    is past the default of four and the definition is left whole with the
    reason; the same program under a cap of five is inlined into all five, which
    is what says the cap decided it rather than the program.
    """
    arrays = _copy_operands()
    assert cg.MultiTermFactorization().max_readers == 4

    capped = cg.Graph("mtf-past-the-reader-cap")
    results, vectors = _many_consumers(capped, arrays, 5)
    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    pm.add(cg.Materialization())
    pm.run(capped)
    assert mtf.num_inlined == 0 and mtf.num_copies == 0
    assert any("more consumers than the reader cap admits" in reason
               for reason, _count in mtf.skip_reasons), mtf.skip_reasons

    lifted = cg.Graph("mtf-reader-cap-lifted")
    lifted_results, _vectors = _many_consumers(lifted, arrays, 5)
    raised = cg.MultiTermFactorization()
    raised.set_search_enabled(True)
    raised.set_max_readers(5)
    assert raised.max_readers == 5
    pm_lifted = cg.PassManager()
    pm_lifted.add(raised)
    pm_lifted.add(cg.Materialization())
    assert pm_lifted.run(lifted), raised.skip_reasons
    # Dissolved once and computed again in each of the other four consumers.
    assert raised.num_inlined == 1
    assert raised.num_copies == 4

    a, b = arrays[0], arrays[1]
    capped.execute()
    lifted.execute()
    for result, lifted_result, vector in zip(results, lifted_results, vectors):
        want = (a @ b) @ vector
        assert_close(np.asarray(result), want, dtype="float64")
        assert_close(np.asarray(lifted_result), want, dtype="float64")


def test_a_definition_whose_operand_is_rewritten_before_its_consumer_stays_put():
    """Inlining moves a read, and a read may not travel past a write.

    A definition is computed where the author put it and used later, so folding
    it into its consumer moves every read it makes to the consumer's position.
    An operand something rewrites in between would then be read at its new
    value, and the program would quietly compute something else. The definition
    is left whole instead.
    """
    rng = np.random.default_rng(31)
    first = rng.standard_normal((CK, CJ))
    second = rng.standard_normal((CK, CJ))
    a = rng.standard_normal((CI, CK))
    v = rng.standard_normal((CJ,))

    graph = cg.Graph("mtf-operand-rewritten-between")
    P1 = _tensor("P1", first, "float64")
    P2 = _tensor("P2", second, "float64")
    A = _tensor("A", a, "float64")
    V = _tensor("v", v, "float64")
    R = _tensor("R", np.zeros((CI,)), "float64")
    scratch = graph.declare_tensor("t", [CK, CJ], intermediate=True, dtype="float64")
    M = graph.declare_tensor("M", [CI, CJ], intermediate=True, dtype="float64")
    with cg.capture(graph):
        la.axpby(1.0, P1, 0.0, scratch)
        einsums.einsum("i,j <- i,k ; k,j", M, A, scratch)
        la.axpby(1.0, P2, 0.0, scratch)      # the operand M read, rewritten
        einsums.einsum("i <- i,j ; j", R, M, V)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    pm.add(cg.Materialization())
    pm.run(graph)
    assert mtf.num_inlined == 0 and mtf.num_copies == 0
    assert any("cannot travel there" in reason for reason, _count in mtf.skip_reasons), mtf.skip_reasons

    graph.execute()
    assert_close(np.asarray(R), (a @ first) @ v, dtype="float64")


@pytest.mark.parametrize("alias", ["same", "view"])
def test_a_definition_whose_operand_its_consumer_overwrites_stays_put(alias):
    """``A = (A B) C``: the consumer's own write is a write the inlined read may not travel past.

    Left to right, ``T = A B`` reads A before ``A = T C`` overwrites it. Inlined and
    re-bracketed as ``A (B C)``, A is read by the contraction that writes it. Defends
    against the travel check stopping one statement short of the consumer: with A itself
    the rebuilt contraction refused to run (the output overlaps an input), and with
    the result written into one view of a tensor while the operand is an overlapping view
    of it, two ids the check compared, the numbers were off by 70.
    """
    rng = np.random.default_rng(113)
    m, k = 12, 3
    a, b, c = rng.standard_normal((m, k)), rng.standard_normal((k, m)), rng.standard_normal((m, k))
    parent = np.zeros((m + 1, k))
    parent[1:] = a
    if alias == "same":
        expected = a @ b @ c
    else:
        expected = parent.copy()
        expected[:m] = a @ b @ c

    graph = cg.Graph("mtf-consumer-overwrites-an-operand")
    B, C = _tensor("B", b, "float64"), _tensor("C", c, "float64")
    if alias == "same":
        result = _tensor("A", a, "float64")
        out = leaf = result
    else:
        result = _tensor("P", parent, "float64")
        out, leaf = result[0:m, 0:k], result[1:m + 1, 0:k]
    T = graph.declare_tensor("T", [m, m], intermediate=True, dtype="float64")
    with cg.capture(graph):
        einsums.einsum("tv <- tu ; uv", T, leaf, B)
        einsums.einsum("tw <- tv ; vw", out, T, C)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    pm.set_optimizer_budget(0)  # see _NO_ALLOWANCE
    pm.add(mtf)
    pm.add(cg.Materialization())
    pm.run(graph)
    assert mtf.num_inlined == 0 and mtf.num_rebracketed == 0
    assert any("cannot travel there" in reason for reason, _count in mtf.skip_reasons), mtf.skip_reasons

    graph.execute()
    assert_close(np.asarray(result), expected, dtype="float64")


def test_a_region_holding_a_permutation_operator_is_not_rewritten_without_it():
    """The algebra a region is raised into has no term for P(...), so raising an antisymmetrized
    contraction described only its identity term. The CCSD tau pair below, with P(i/j) P(a/b) on
    both writes, came back as the unantisymmetrized sum once the search rewrote it."""
    o, v = 3, 4
    rng = np.random.default_rng(0)
    tau = rng.standard_normal((o, o, v, v))
    g_arr = rng.standard_normal((o, o, v, v))

    def antisym(x):
        x = x - x.transpose(1, 0, 2, 3)
        return x - x.transpose(0, 1, 3, 2)

    w1 = np.einsum("ijef,mnef->mnij", tau, g_arr)
    w2 = np.einsum("mnab,mnef->abef", tau, g_arr)
    want = 0.125 * antisym(np.einsum("mnab,mnij->ijab", tau, w1)) + 0.125 * antisym(np.einsum("ijef,abef->ijab", tau, w2))

    graph = cg.Graph("mtf_operator")
    T = einsums.asarray(np.ascontiguousarray(tau))
    G = einsums.asarray(np.ascontiguousarray(g_arr))
    R = einsums.zeros((o, o, v, v), dtype="float64")
    W1 = graph.declare_tensor("w1", [o, o, o, o], intermediate=True, dtype="float64")
    W2 = graph.declare_tensor("w2", [v, v, v, v], intermediate=True, dtype="float64")
    with cg.capture(graph):
        einsums.einsum("m,n,i,j <- i,j,e,f ; m,n,e,f", W1, T, G)
        einsums.einsum("i,j,a,b <- P(i/j) P(a/b) m,n,a,b ; m,n,i,j", R, T, W1, c_pf=0.0, ab_pf=0.125)
        einsums.einsum("a,b,e,f <- m,n,a,b ; m,n,e,f", W2, T, G)
        einsums.einsum("i,j,a,b <- P(i/j) P(a/b) i,j,e,f ; a,b,e,f", R, T, W2, c_pf=1.0, ab_pf=0.125)
    manager = cg.PassManager()
    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    manager.add(mtf)
    manager.set_optimizer_budget(0)
    manager.add(cg.Materialization())
    manager.run(graph)
    graph.execute()
    assert_close(R, want)
