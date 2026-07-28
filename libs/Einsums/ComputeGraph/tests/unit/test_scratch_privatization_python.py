#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""ScratchPrivatization + Graph.set_executor + disjoint-view scheduling.

The three features ship together: set_executor gives a loop body a parallel
backend, ScratchPrivatization renames reused scratch so false WAR/WAW chains
stop serializing the body, and the disjointness-aware hazard scan lets writes
to provably disjoint views of one tensor run concurrently. Every test checks
results against numpy, since the whole point is rewriting the schedule without
changing a single value.
"""

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la


def _randn(rng, *shape):
    return np.ascontiguousarray(rng.standard_normal(shape))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_set_executor_on_loop_body(rng):
    """A body-installed executor replays the loop correctly; None resets."""
    A = einsums.asarray(_randn(rng, 20, 20), name="A")
    C1 = einsums.create_zero_tensor("C1", [20, 20], dtype="float64")
    C2 = einsums.create_zero_tensor("C2", [20, 20], dtype="float64")
    acc = einsums.create_zero_tensor("acc", [20, 20], dtype="float64")

    g = cg.Graph("exec")
    body = g.add_loop("it", 3, lambda i: True)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", C1, A, A, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("ij <- ki ; kj", C2, A, A, c_pf=0.0, ab_pf=1.0)
        la.axpby(1.0, C1, 1.0, acc)
        la.axpby(1.0, C2, 1.0, acc)

    body.set_executor(cg.DataflowExecutor())
    g.execute()
    An = np.asarray(A)
    ref = 3 * (An @ An + An.T @ An)
    np.testing.assert_allclose(np.asarray(acc), ref, atol=1e-12)

    body.set_executor(None)
    g.execute()
    np.testing.assert_allclose(np.asarray(acc), 2 * ref, atol=1e-12)


def test_privatization_splits_generations(rng):
    """Three write->read episodes of one scratch split onto clones, exactly."""
    ops = [einsums.asarray(_randn(rng, 24, 24), name=f"A{k}") for k in range(3)]
    acc = einsums.create_zero_tensor("acc", [24, 24], dtype="float64")

    g = cg.Graph("gen")
    tmp = g.declare_zero_tensor("tmp", [24, 24], dtype="float64", intermediate=True)
    with cg.capture(g):
        for A in ops:
            einsums.einsum("ij <- ik ; kj", tmp, A, A, c_pf=0.0, ab_pf=1.0)
            la.axpby(1.0, tmp, 1.0, acc)

    sp = cg.ScratchPrivatization()
    sp.set_require_executor(False)
    pm = cg.PassManager()
    pm.add(sp)
    g.apply(pm)
    assert sp.num_tensors_privatized == 1
    # 3 generations: 2 interior renamed, the last stays on the original.
    assert sp.num_copies_created == 2
    assert sp.num_nodes_rebuilt == 4

    g.apply(cg.default_pass_manager())  # materialize the clones
    g.execute()
    ref = sum(np.asarray(A) @ np.asarray(A) for A in ops)
    np.testing.assert_allclose(np.asarray(acc), ref, atol=1e-12)


def test_privatization_requires_executor_by_default(rng):
    """Without an installed executor the default-gated pass leaves graphs alone."""
    A = einsums.asarray(_randn(rng, 16, 16), name="A")
    acc = einsums.create_zero_tensor("acc", [16, 16], dtype="float64")

    g = cg.Graph("gated")
    tmp = g.declare_zero_tensor("tmp", [16, 16], dtype="float64", intermediate=True)
    with cg.capture(g):
        for _ in range(2):
            einsums.einsum("ij <- ik ; kj", tmp, A, A, c_pf=0.0, ab_pf=1.0)
            la.axpby(1.0, tmp, 1.0, acc)

    sp = cg.ScratchPrivatization()
    pm = cg.PassManager()
    pm.add(sp)
    g.apply(pm)
    assert sp.num_tensors_privatized == 0

    g.set_executor(cg.DataflowExecutor())
    g.apply(pm)
    assert sp.num_tensors_privatized == 1


def test_privatization_skips_loop_carried_scratch(rng):
    """A tensor read before its first overwrite carries state; it must not split."""
    A = einsums.asarray(_randn(rng, 12, 12), name="A")
    acc = einsums.create_zero_tensor("acc", [12, 12], dtype="float64")

    g = cg.Graph("carried")
    carry = g.declare_zero_tensor("carry", [12, 12], dtype="float64", intermediate=True)
    with cg.capture(g):
        la.axpby(1.0, carry, 1.0, acc)  # reads carry BEFORE any write
        einsums.einsum("ij <- ik ; kj", carry, A, A, c_pf=0.0, ab_pf=1.0)
        la.axpby(1.0, carry, 1.0, acc)
        einsums.einsum("ij <- ik ; kj", carry, A, A, c_pf=0.0, ab_pf=2.0)
        la.axpby(1.0, carry, 1.0, acc)

    sp = cg.ScratchPrivatization()
    sp.set_require_executor(False)
    pm = cg.PassManager()
    pm.add(sp)
    g.apply(pm)
    assert sp.num_tensors_privatized == 0


def test_privatized_ccsd_style_loop_matches_numpy(rng):
    """The motivating shape: a loop body reusing tmp/tmpP through einsum,
    permute, and axpby, replayed on the Dataflow executor after the default
    pipeline (privatization included) rewrote it."""
    n = 14
    A1 = einsums.asarray(_randn(rng, n, n), name="A1")
    A2 = einsums.asarray(_randn(rng, n, n), name="A2")
    r2 = einsums.create_zero_tensor("r2", [n, n], dtype="float64")

    g = cg.Graph("ccsdish")
    tmp = g.declare_zero_tensor("tmp", [n, n], dtype="float64", intermediate=True)
    tmpP = g.declare_zero_tensor("tmpP", [n, n], dtype="float64", intermediate=True)
    body = g.add_loop("it", 3, lambda i: True)
    with cg.capture(body):
        for A in (A1, A2):
            einsums.einsum("ij <- ik ; kj", tmp, A, A, c_pf=0.0, ab_pf=1.0)
            la.axpby(1.0, tmp, 1.0, r2)
            einsums.permute("j,i <- i,j", tmpP, tmp)
            la.axpby(1.0, tmpP, 1.0, r2)

    body.set_executor(cg.DataflowExecutor())
    g.apply(cg.default_pass_manager())
    g.execute()

    def sym(m):
        return m + m.T

    ref = 3 * (sym(np.asarray(A1) @ np.asarray(A1)) + sym(np.asarray(A2) @ np.asarray(A2)))
    np.testing.assert_allclose(np.asarray(r2), ref, atol=1e-12)


def test_disjoint_view_writes_parallel_and_exact(rng):
    """Writes through disjoint constant-index views of one tensor must not
    serialize (the ladder idiom) and must land exactly where they belong,
    under both the sequential and the Dataflow executor."""
    n, m = 4, 21
    src = einsums.asarray(_randn(rng, n, m, m), name="src")
    out = einsums.create_zero_tensor("out", [n, m, m], dtype="float64")
    _FULL = (0, 0, 0)

    g = cg.Graph("slices")
    with cg.capture(g):
        for k in range(n):
            s_k = cg.view_indexed(src, [(2, k, 0), _FULL, _FULL])
            o_k = cg.view_indexed(out, [(2, k, 0), _FULL, _FULL])
            einsums.einsum("a,b <- a,c ; c,b", o_k, s_k, s_k, c_pf=1.0, ab_pf=1.0)

    src_np = np.asarray(src)
    ref = np.stack([src_np[k] @ src_np[k] for k in range(n)])

    g.execute()
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-12)

    g.set_executor(cg.DataflowExecutor())
    g.execute()
    np.testing.assert_allclose(np.asarray(out), 2 * ref, atol=1e-12)
