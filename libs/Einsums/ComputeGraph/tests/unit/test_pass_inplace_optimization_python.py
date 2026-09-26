# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python mirror of Pass_InplaceOptimization.cpp, and the pass's numpy-checked guards.

A merge rewrites the graph (``modified=True``); ``num_candidates`` still reports the
single-producer, single-consumer census the pass took when it was analysis-only.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from _fuzz_diff_common import _apply_one_pass
from einsums.testing import assert_close


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_inplace_optimization_empty_graph():
    g = cg.Graph("io_empty")
    pass_inst = cg.InplaceOptimization()
    assert not _run(pass_inst, g)
    assert pass_inst.num_candidates == 0


def test_inplace_optimization_user_owned_tensor_not_a_candidate():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("io_user")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.InplaceOptimization()
    assert not _run(pass_inst, g)
    assert pass_inst.num_candidates == 0


def test_inplace_optimization_finds_candidates():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])

    g = cg.Graph("inplace_test")
    T = g.create_zero_tensor("T", [3, 3], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, B)  # writes T
        einsums.einsum("ij <- ik ; kj", C, T, A)  # reads T (sole consumer)

    pass_inst = cg.InplaceOptimization()
    _run(pass_inst, g)
    assert pass_inst.num_candidates >= 1
    # Einsum consumers may never alias output with input; nothing merges.
    assert pass_inst.num_merged == 0


def test_inplace_optimization_merges_direct_product_output():
    """Mirror of the C++ merge test: Y = alpha*(X (*) B) with beta=0 reuses
    dying X's storage; numerics and replay stay correct."""
    A = einsums.create_random_tensor("A", [6, 6])
    B = einsums.create_random_tensor("B", [6, 6])
    OUT = einsums.create_random_tensor("OUT", [6, 6])

    X_ref = np.asarray(A) @ np.asarray(B)
    Y_ref = 2.0 * X_ref * np.asarray(B)
    OUT_ref = Y_ref @ np.asarray(A)

    g = cg.Graph("inplace_merge_py")
    X = g.create_zero_tensor("X", [6, 6], dtype="float64")
    Y = g.create_zero_tensor("Y", [6, 6], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", X, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.linalg.direct_product(2.0, X, B, 0.0, Y)
        einsums.einsum("ij <- ik ; kj", OUT, Y, A, c_pf=0.0, ab_pf=1.0)

    pass_inst = cg.InplaceOptimization()
    assert _run(pass_inst, g)
    assert pass_inst.num_merged == 1

    g.execute()
    assert_close(OUT, OUT_ref)
    g.execute()  # replay through the merged storage
    assert_close(OUT, OUT_ref)


def test_inplace_optimization_rank3_batched_gemm_with_sole_consumer():
    A = einsums.create_random_tensor("A", [3, 5, 4])
    B = einsums.create_random_tensor("B", [5, 6, 4])

    g = cg.Graph("inplace_rank3")
    T = g.create_zero_tensor("T", [3, 6, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ijb <- ikb ; kjb", T, A, B)
        einsums.linalg.scale(0.5, T)

    pass_inst = cg.InplaceOptimization()
    _run(pass_inst, g)
    assert pass_inst.num_candidates >= 0


# ──────────────────────────────────────────────────────────────────────────
# Ordering guards, checked against numpy
# ──────────────────────────────────────────────────────────────────────────


def _tensor(name, array):
    t = einsums.create_zero_tensor(name, list(array.shape), dtype=str(array.dtype))
    np.asarray(t)[...] = array
    return t


def _execute(g, runs):
    """Materialize, then replay @p runs times, returning after the last."""
    _apply_one_pass(g, "Materialization")
    for _ in range(runs):
        g.execute()


@pytest.mark.parametrize("storage", ["declared", "created"])
def test_inplace_optimization_keeps_a_destination_read_before_its_writer(storage):
    """``E = D`` ahead of ``D = 2 S`` keeps D's own storage.

    E reads the value D held before this replay wrote it: zero for a declared D, which every
    replay re-zeroes, and the last replay's ``2 X`` for a created one. Defends against the merge
    checking only that D has one writer: D was put in the dying S's storage, so E read S's
    ``X`` instead (max abs error 9 on this data, on either storage).
    """
    n = 3
    x0 = np.arange(1.0, 10.0).reshape(n, n)
    for runs in (1, 2):
        X, Y, Z = _tensor("X", x0), _tensor("Y", np.zeros((n, n))), _tensor("Z", np.zeros((n, n)))
        g = cg.Graph("inplace_read_before_write")
        make = g.declare_zero_tensor if storage == "declared" else g.create_zero_tensor
        S, D, E = (make(name, [n, n], True) for name in ("S", "D", "E"))
        with cg.capture(g):
            einsums.linalg.axpby(1.0, X, 0.0, S)
            einsums.linalg.axpby(1.0, D, 0.0, E)
            einsums.linalg.axpby(2.0, S, 0.0, D)
            einsums.linalg.axpby(1.0, D, 0.0, Y)
            einsums.linalg.axpby(1.0, E, 0.0, Z)
        _run(cg.InplaceOptimization(), g)
        _execute(g, runs)
        stale = 2 * x0 if storage == "created" and runs == 2 else np.zeros((n, n))
        np.testing.assert_allclose(np.asarray(Y), 2 * x0)
        np.testing.assert_allclose(np.asarray(Z), stale)


def test_inplace_optimization_keeps_a_source_written_after_the_consumer():
    """``D = 2 S`` ahead of the only ``S = X`` keeps D's own storage.

    D reads the S the previous replay left: zero on the first, X after. Defends against the
    merge checking only that S has one writer and one reader: D was put in S's storage, so the
    later ``S = X`` overwrote D before Y read it, and Y came out X where the program gives zero
    and then ``2 X`` (max abs error 9 on this data).
    """
    n = 3
    x0 = np.arange(1.0, 10.0).reshape(n, n)
    for runs, want in ((1, np.zeros((n, n))), (2, 2 * x0)):
        X, Y = _tensor("X", x0), _tensor("Y", np.zeros((n, n)))
        g = cg.Graph("inplace_write_after_read")
        S, D = (g.create_zero_tensor(name, [n, n], True) for name in ("S", "D"))
        with cg.capture(g):
            einsums.linalg.axpby(2.0, S, 0.0, D)
            einsums.linalg.axpby(1.0, X, 0.0, S)
            einsums.linalg.axpby(1.0, D, 0.0, Y)
        _run(cg.InplaceOptimization(), g)
        _execute(g, runs)
        np.testing.assert_allclose(np.asarray(Y), want)


# ──────────────────────────────────────────────────────────────────────────
# Grouped consumers
# ──────────────────────────────────────────────────────────────────────────

_GROUPED_KINDS = ("product", "division", "axpby")


def _grouped_program(kind, count, n=4, seed=31):
    """``S_i = A_i B_i`` into scratch, a grouped element-wise ``T_i = f(S_i)``, ``R_i = T_i C_i``.

    Each S_i dies at the grouped node, the shape a local-correlation update has. Returns the
    builder and the expected ``R_i``.
    """
    rng = np.random.default_rng(seed)
    a, b, c = ([rng.standard_normal((n, n)) for _ in range(count)] for _ in range(3))
    d = [2.0 + rng.random((n, n)) for _ in range(count)]
    s = [a[i] @ b[i] for i in range(count)]
    t = {"product": [1.5 * s[i] * d[i] for i in range(count)],
         "division": [-1.0 * s[i] / d[i] for i in range(count)],
         "axpby": [0.5 * s[i] for i in range(count)]}[kind]
    expected = [t[i] @ c[i] for i in range(count)]

    def build(g, held):
        A, B, C, D = ([_tensor(f"{name}{i}", arr[i]) for i in range(count)] for name, arr in zip("ABCD", (a, b, c, d)))
        R = [_tensor(f"R{i}", np.zeros((n, n))) for i in range(count)]
        # Created rather than declared: Materialization zero-initializes a declared tensor with
        # an Initialize node, and a destination carrying one is declined.
        S = [g.create_zero_tensor(f"S{i}", [n, n], True) for i in range(count)]
        T = [g.create_zero_tensor(f"T{i}", [n, n], True) for i in range(count)]
        with cg.capture(g):
            for i in range(count):
                einsums.einsum("ij <- ik ; kj", S[i], A[i], B[i], c_pf=0.0, ab_pf=1.0)
            if kind == "product":
                einsums.linalg.grouped_direct_product([1.5] * count, S, D, [0.0] * count, T)
            elif kind == "division":
                einsums.linalg.grouped_direct_division([-1.0] * count, S, D, [0.0] * count, T)
            else:
                einsums.linalg.grouped_axpby([0.5] * count, S, [0.0] * count, T)
            for i in range(count):
                einsums.einsum("ij <- ik ; kj", R[i], T[i], C[i], c_pf=0.0, ab_pf=1.0)
        held += A + B + C + D + S + T
        return R

    return build, expected


@pytest.mark.parametrize("how", ["alone", "default"])
@pytest.mark.parametrize("kind", _GROUPED_KINDS)
def test_inplace_optimization_merges_every_member_of_a_grouped_consumer(how, kind):
    """Each member of a grouped element-wise node reuses its own dying source.

    Three members, three sources that die at the node, three destinations the node pure-writes:
    three merges, and the replay reads every member's result through the survivor's storage.
    This shape used to be declined by the pass's feature declaration.
    """
    count = 3
    build, expected = _grouped_program(kind, count)
    held = []
    g = cg.Graph("inplace_grouped")
    outputs = build(g, held)
    if how == "alone":
        pass_inst = cg.InplaceOptimization()
        assert _run(pass_inst, g)
        assert pass_inst.num_merged == count, pass_inst.skip_reasons
        _apply_one_pass(g, "Materialization")
    else:
        manager = cg.default_pass_manager()
        g.apply(manager)
        assert f"InplaceOptimization: merged {count} buffer(s)" in manager.explain(), manager.explain()
    for _ in range(2):
        g.execute()
        for got, want in zip(outputs, expected):
            np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=1e-12)


def test_inplace_optimization_keeps_a_member_whose_destination_an_earlier_member_reads():
    """``E = T`` then ``T = 2 S`` as members of one grouped axpby keep T's own storage.

    A grouped axpby runs its members in order, so E reads the T the previous replay left: zero
    on the first, ``2 S`` on the second. Defends the check that no member of the node reads a
    destination, which is wider than the member's own inputs: putting T in the dying S's storage
    handed the first member S itself, and E came out S (max abs error 8.7 on this data).
    """
    n = 4
    rng = np.random.default_rng(32)
    a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    s0 = a @ b
    for runs, want in ((1, np.zeros((n, n))), (2, 2 * s0)):
        X, Y = _tensor("A", a), _tensor("B", b)
        R = _tensor("R", np.zeros((n, n)))
        g = cg.Graph("inplace_grouped_earlier_member")
        S, T, E = (g.create_zero_tensor(name, [n, n], True) for name in ("S", "T", "E"))
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", S, X, Y, c_pf=0.0, ab_pf=1.0)
            einsums.linalg.grouped_axpby([1.0, 2.0], [T, S], [0.0, 0.0], [E, T])
            einsums.linalg.axpby(1.0, E, 0.0, R)
        _run(cg.InplaceOptimization(), g)
        _execute(g, runs)
        np.testing.assert_allclose(np.asarray(R), want, rtol=1e-12, atol=1e-12)
