# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Differential for the SymmetrizedAccumulation pass on the CCSD symmetrization
idiom r2 += s*(tmp + P(tmp)), P = swap (i<->j) and (a<->b).

Runtime tensors (the workload path), captured as einsum->tmp, axpby(tmp->r2),
permute(jiba<-ijab), axpby(tmpP->r2). Checked two ways: the pass applied in
isolation (must fold, num_rewritten == 1) and through the wired default pipeline
(cg.default_pass_manager() now includes it) - both compared to a numpy oracle.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _oracle(A_np, B_np, s):
    # tmp[i,j,a,b] = A[i,a] B[j,b]; P swaps i<->j and a<->b.
    tmp = np.einsum("ia,jb->ijab", A_np, B_np)
    P_tmp = np.transpose(tmp, (1, 0, 3, 2))
    return s * (tmp + P_tmp)


def _capture_symacc(A, B, r2, tmp, tmpP, s):
    g = cg.Graph("symacc")
    with cg.capture(g):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(s, tmp, 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(s, tmpP, 1.0, r2)
    return g


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_pass_folds_and_matches_numpy(dtype):
    o, v = 2, 3
    rng = np.random.default_rng(0)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))
    s = 0.5

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    g = _capture_symacc(A, B, r2, tmp, tmpP, s)

    pm = cg.PassManager()
    pass_obj = cg.SymmetrizedAccumulation()
    pm.add(pass_obj)
    modified = g.apply(pm)

    assert modified
    assert pass_obj.num_matched == 1     # APIARY getters are properties
    assert pass_obj.num_rewritten == 1

    g.execute()
    assert_close(r2, _oracle(A_np, B_np, s))


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_default_pipeline_stays_correct(dtype):
    # The pass is wired into create_default; the full default pipeline must fold
    # the pattern and still produce the right answer.
    o, v = 2, 3
    rng = np.random.default_rng(1)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))
    s = -2.0

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    g = _capture_symacc(A, B, r2, tmp, tmpP, s)
    g.apply(cg.default_pass_manager())
    g.execute()

    assert_close(r2, _oracle(A_np, B_np, s))


# ── Loop-body arm ──────────────────────────────────────────────────────────
# The CCSD residual lives in a captured loop body, so the pass's whole reason
# for recurse_into_subgraphs() is folding a site INSIDE a replayed subgraph.
# This fuzzes that path: the symacc idiom captured in a loop body, folded, then
# replayed N times. tmp/tmpP are parent-declared loop scratch (recomputed each
# iteration); r2 accumulates, so after N iterations r2 = N * s * (tmp + P(tmp)).
@given(o=st.integers(1, 3), v=st.integers(1, 3), s=st.sampled_from([1.0, -0.5, 0.5]),
       niters=st.integers(1, 4), dtype=st.sampled_from(["float64", "complex128"]),
       use_default=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_symacc_inside_loop_body(o, v, s, niters, dtype, use_default, seed):
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))

    # A, B are loop-invariant, so each iteration adds the same contribution.
    oracle = niters * _oracle(A_np, B_np, s)

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    g = cg.Graph("symacc_loop")
    loop = g.add_loop("L", niters, lambda it, N=niters: it < N - 1)
    with cg.capture(loop):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(s, tmp, 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(s, tmpP, 1.0, r2)

    if use_default:
        g.apply(cg.default_pass_manager())
    else:
        pm = cg.PassManager()
        pm.add(cg.SymmetrizedAccumulation())
        modified = g.apply(pm)
        # The only foldable site is in the loop body, so a rewrite here proves
        # the pass recursed into the subgraph.
        assert modified, f"pass did not fold the symacc site inside the loop body (o={o} v={v} niters={niters})"

    g.execute()
    assert_close(r2, oracle)


# ── Loop-CARRIED operand arm ───────────────────────────────────────────────
# The arm above holds A and B fixed, so every iteration contributes the same
# value and a fold that lost track of iteration boundaries would still produce
# N * (one iteration). The real CCSD body updates its amplitudes in place
# (t1 += rd1 at the end of each iteration), so the symacc operand DIFFERS every
# replay. This arm reproduces that shape: A += dA at the end of the body, after
# both accumulations, so the fold is still legal but the per-iteration
# contribution changes.
@given(o=st.integers(1, 3), v=st.integers(1, 3), s=st.sampled_from([1.0, -0.5, 0.5]),
       niters=st.integers(2, 4), dtype=st.sampled_from(["float64", "complex128"]),
       use_default=st.booleans(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_symacc_loop_body_with_loop_carried_operand(o, v, s, niters, dtype, use_default, seed):
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    dA_np = rng.standard_normal((o, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))
        dA_np = dA_np + 1j * rng.standard_normal((o, v))

    # Iteration k contracts A_k = A_0 + k*dA (the update lands AFTER both
    # accumulations, so iteration 0 still sees the original A).
    #
    # The oracle builds A_k by the SAME repeated addition the graph performs,
    # rather than the algebraically equal A_0 + k*dA. Those differ by O(k) ULPs in
    # floating point, and the contraction that follows is a cancelling one, so the
    # discrepancy lands in the result at the magnitude of the intermediates rather
    # than of the answer. Comparing against a closed form here was asking the graph
    # to be more accurate than the arithmetic it was told to do.
    oracle = np.zeros((o, o, v, v), dtype=dtype)
    A_k = A_np.copy()
    for _ in range(niters):
        oracle = oracle + _oracle(A_k, B_np, s)
        A_k = A_k + dA_np

    A = einsums.asarray(A_np.copy())
    B = einsums.asarray(B_np)
    dA = einsums.asarray(dA_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    g = cg.Graph("symacc_loop_carried")
    loop = g.add_loop("L", niters, lambda it, N=niters: it < N - 1)
    with cg.capture(loop):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(s, tmp, 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(s, tmpP, 1.0, r2)
        einsums.linalg.axpby(1.0, dA, 1.0, A)  # loop-carried amplitude update

    if use_default:
        g.apply(cg.default_pass_manager())
    else:
        pm = cg.PassManager()
        pm.add(cg.SymmetrizedAccumulation())
        assert g.apply(pm), f"pass did not fold inside the loop body (o={o} v={v} niters={niters})"

    g.execute()
    assert_close(r2, oracle)


# ── Damping-in-the-window arm (guard regression) ───────────────────────────
# The interference guard admits an intervening accumulation into r2 because
# addition commutes with moving the permuted contribution earlier. That holds
# only for beta == 1. A damping/mixing step (r2 = X + damp*r2), routine in SCF
# and DIIS-driven codes, rescales the running r2, so a contribution moved
# across it would pick up a spurious factor of damp. The pass must decline.
@given(o=st.integers(1, 3), v=st.integers(1, 3), s=st.sampled_from([1.0, -0.5]),
       damp=st.sampled_from([0.5, 0.25, -0.75]), niters=st.integers(1, 3),
       dtype=st.sampled_from(["float64", "complex128"]), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_symacc_damping_in_window_is_not_folded(o, v, s, damp, niters, dtype, seed):
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    X_np = rng.standard_normal((o, o, v, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))
        X_np = X_np + 1j * rng.standard_normal((o, o, v, v))

    # Per iteration, in program order:
    #   r2 += s*tmp  ;  r2 = X + damp*r2  ;  r2 += s*P(tmp)
    tmp_np = np.einsum("ia,jb->ijab", A_np, B_np)
    P_tmp = np.transpose(tmp_np, (1, 0, 3, 2))
    oracle = np.zeros((o, o, v, v), dtype=np.dtype(dtype))
    for _ in range(niters):
        oracle = oracle + s * tmp_np
        oracle = X_np + damp * oracle
        oracle = oracle + s * P_tmp

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    X = einsums.asarray(X_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    g = cg.Graph("symacc_damped")
    loop = g.add_loop("L", niters, lambda it, N=niters: it < N - 1)
    with cg.capture(loop):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(s, tmp, 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(1.0, X, damp, r2)  # damping inside the fold window
        einsums.linalg.axpby(s, tmpP, 1.0, r2)

    p = cg.SymmetrizedAccumulation()
    pm = cg.PassManager()
    pm.add(p)
    g.apply(pm)
    assert p.num_rewritten == 0, f"folded across a beta={damp} damping step"

    g.execute()
    assert_close(r2, oracle)


# ── Conditional-branch arm ─────────────────────────────────────────────────
# Conditionals are the control-flow axis with the least pass coverage. A symacc
# site inside a taken branch, inside a loop, exercises two levels of subgraph
# recursion at once.
@given(o=st.integers(1, 3), v=st.integers(1, 3), s=st.sampled_from([1.0, -0.5]),
       niters=st.integers(1, 3), take=st.booleans(),
       dtype=st.sampled_from(["float64", "complex128"]), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_symacc_inside_conditional_in_loop(o, v, s, niters, take, dtype, seed):
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))

    # The then-branch holds the symacc site; the else-branch is empty, so a
    # not-taken predicate must leave r2 at zero.
    oracle = (niters * _oracle(A_np, B_np, s)) if take else np.zeros((o, o, v, v), dtype=np.dtype(dtype))

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    # Subgraphs are declared outside any capture (nested captures are rejected)
    # and each level is then captured on its own.
    g = cg.Graph("symacc_cond")
    loop = g.add_loop("L", niters, lambda it, N=niters: it < N - 1)
    then_g, _else_g = loop.add_conditional("branch", lambda t=take: t)
    with cg.capture(then_g):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(s, tmp, 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(s, tmpP, 1.0, r2)

    nodes_before = then_g.num_nodes()

    p = cg.SymmetrizedAccumulation()
    pm = cg.PassManager()
    pm.add(p)
    g.apply(pm)

    # Assert on the branch itself, NOT on p.num_rewritten: run() zeroes its
    # counters on entry and the recursive driver calls it once per subgraph, so
    # the getters report only the last subgraph visited -- for a conditional
    # that is always the empty else-branch. Node count is the honest signal.
    assert then_g.num_nodes() == nodes_before - 1, "pass did not recurse into the conditional branch"

    g.execute()
    assert_close(r2, oracle)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("nsites", [2, 4])
def test_symacc_sites_sharing_one_scratch_pair(dtype, nsites):
    """The CCSD body's real shape: ONE tmp/tmpP recycled by every symacc site
    (graph-owned scratch reused in place). Whole-graph writer/reader-uniqueness
    used to reject all of them; generation-scoped matching folds each site and
    the folded graph must still match the oracle."""
    o, v = 2, 3
    rng = np.random.default_rng(7)
    ops = []
    for _ in range(nsites):
        A_np = rng.standard_normal((o, v)).astype(dtype)
        B_np = rng.standard_normal((o, v)).astype(dtype)
        if dtype.startswith("complex"):
            A_np = A_np + 1j * rng.standard_normal((o, v))
            B_np = B_np + 1j * rng.standard_normal((o, v))
        ops.append((A_np, B_np))
    s = 0.5

    tensors = [(einsums.asarray(A_np), einsums.asarray(B_np)) for A_np, B_np in ops]
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)

    g = cg.Graph("symacc-shared")
    tmp = g.declare_zero_tensor("tmp", [o, o, v, v], dtype=dtype, intermediate=True)
    tmpP = g.declare_zero_tensor("tmpP", [o, o, v, v], dtype=dtype, intermediate=True)
    with cg.capture(g):
        for A, B in tensors:
            einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
            einsums.linalg.axpby(s, tmp, 1.0, r2)
            einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
            einsums.linalg.axpby(s, tmpP, 1.0, r2)

    pm = cg.PassManager()
    pass_obj = cg.SymmetrizedAccumulation()
    pm.add(pass_obj)
    g.apply(pm)
    assert pass_obj.num_matched == nsites
    assert pass_obj.num_rewritten == nsites

    g.apply(cg.default_pass_manager())  # materialize the (now partly unused) scratch
    g.execute()
    oracle = sum(_oracle(A_np, B_np, s) for A_np, B_np in ops)
    assert_close(r2, oracle)


def _capture_site_with_interloper(interloper):
    """One symacc site with `interloper(r2_view)` captured BETWEEN axpby1 and
    the permute, i.e. inside the interference-guard window, touching r2 only
    through a constant-index view (the DF-ladder shape)."""
    o, v = 2, 3
    rng = np.random.default_rng(11)
    A_np = rng.standard_normal((o, v))
    B_np = rng.standard_normal((o, v))
    S_np = rng.standard_normal((v, v))

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    Sm = einsums.asarray(S_np)
    r2 = einsums.zeros((o, o, v, v), dtype="float64")
    tmp = einsums.zeros((o, o, v, v), dtype="float64")
    tmpP = einsums.zeros((o, o, v, v), dtype="float64")

    g = cg.Graph("symacc-interloper")
    _FULL = (0, 0, 0)
    with cg.capture(g):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(1.0, tmp, 1.0, r2)
        r2_00 = cg.view_indexed(r2, [(2, 0, 0), (2, 0, 0), _FULL, _FULL])
        interloper(r2_00, Sm)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(1.0, tmpP, 1.0, r2)

    sa = cg.SymmetrizedAccumulation()
    pm = cg.PassManager()
    pm.add(sa)
    g.apply(pm)
    # The graph references these; hand them back so they outlive execute().
    keepalive = (A, B, Sm, tmp, tmpP)
    return sa, g, r2, (A_np, B_np, S_np), keepalive


def test_additive_view_accumulation_in_window_does_not_block():
    """r2[0,0] += S@S between the halves is a pure accumulation through a view;
    it commutes with the fold, so the site must still fold - and exactly."""
    def interloper(r2_00, Sm):
        einsums.einsum("a,b <- a,c ; c,b", r2_00, Sm, Sm, c_pf=1.0, ab_pf=1.0)

    sa, g, r2, (A_np, B_np, S_np), _keep = _capture_site_with_interloper(interloper)
    assert sa.num_candidates == 1
    assert sa.num_rewritten == 1

    g.execute()
    oracle = _oracle(A_np, B_np, 1.0)
    oracle[0, 0] += S_np @ S_np
    assert_close(r2, oracle)


def test_view_overwrite_in_window_blocks_the_fold():
    """r2[0,0] = S@S (c_pf=0 OVERWRITE through the view) discards the first
    half's contribution to that slice; moving the permuted half across it would
    change what the overwrite clobbers, so the guard must reject. This used to
    be invisible to the raw-id guard (the write targets the view id, not r2)."""
    def interloper(r2_00, Sm):
        einsums.einsum("a,b <- a,c ; c,b", r2_00, Sm, Sm, c_pf=0.0, ab_pf=1.0)

    sa, g, r2, (A_np, B_np, S_np), _keep = _capture_site_with_interloper(interloper)
    assert sa.num_rewritten == 0

    g.execute()
    tmp_np = np.einsum("ia,jb->ijab", A_np, B_np)
    oracle = tmp_np.copy()
    oracle[0, 0] = S_np @ S_np                      # overwrite clobbers the first half
    oracle += np.transpose(tmp_np, (1, 0, 3, 2))    # permuted half lands after
    assert_close(r2, oracle)
