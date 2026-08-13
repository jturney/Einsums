# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for ``grouped_dot`` and ``grouped_axpby``.

Both exist so that a caller holding thousands of scalar-sized operations can pay
one node instead of thousands, WITHOUT the answer moving: the entries run in
sequence inside the node, each through the same kernel the single call uses. So
the property under test is bit equality against the loop of single calls, not
closeness to it, and every assertion below is an exact array comparison.

The bindings take HOMOGENEOUS operand lists per slot - every entry of a slot
owning, or every entry a view - which is the same rule ``grouped_batched_gemm``
carries.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES


def _kinds(graph):
    """The captured node kinds, in program order."""
    return [n["kind"] for n in json.loads(graph.to_json())["nodes"]]


def _shapes():
    """Entry shapes that deliberately disagree, including an empty one."""
    return [[4, 5], [7, 1], [4, 5], [1, 1], [0, 3], [11, 9]]


# ──────────────────────────────────────────────────────────────────────────
# grouped_dot
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_grouped_dot_eager_is_bitwise_the_loop(dtype):
    shapes = _shapes()
    A = [einsums.create_random_tensor(f"A{i}", s, dtype=dtype)
         for i, s in enumerate(shapes)]
    B = [einsums.create_random_tensor(f"B{i}", s, dtype=dtype)
         for i, s in enumerate(shapes)]

    ref = []
    for a, b in zip(A, B):
        r = einsums.create_zero_tensor("r", [1], dtype=dtype)
        einsums.linalg.dot(r, a, b)
        ref.append(np.asarray(r).copy())

    got = [einsums.create_zero_tensor(f"g{i}", [1], dtype=dtype)
           for i in range(len(shapes))]
    einsums.linalg.grouped_dot(got, A, B)

    for i, (g, r) in enumerate(zip(got, ref)):
        np.testing.assert_array_equal(np.asarray(g), r, err_msg=f"entry {i}")


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_grouped_dot_captured_is_bitwise_the_loop(dtype):
    shapes = _shapes()
    A = [einsums.create_random_tensor(f"A{i}", s, dtype=dtype)
         for i, s in enumerate(shapes)]
    B = [einsums.create_random_tensor(f"B{i}", s, dtype=dtype)
         for i, s in enumerate(shapes)]

    looped = [einsums.create_zero_tensor(f"l{i}", [1], dtype=dtype)
              for i in range(len(shapes))]
    g_loop = cg.Graph("loop")
    with cg.capture(g_loop):
        for r, a, b in zip(looped, A, B):
            einsums.linalg.dot(r, a, b)
    assert g_loop.num_nodes() == len(shapes)
    g_loop.execute()

    grouped = [einsums.create_zero_tensor(f"g{i}", [1], dtype=dtype)
               for i in range(len(shapes))]
    g_one = cg.Graph("grouped")
    with cg.capture(g_one):
        einsums.linalg.grouped_dot(grouped, A, B)
    assert g_one.num_nodes() == 1
    assert _kinds(g_one) == ["GroupedDot"]
    g_one.execute()

    for i, (a, b) in enumerate(zip(grouped, looped)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                      err_msg=f"entry {i}")


def test_grouped_dot_takes_views():
    # The DLPNO shape: the operands are slices of a padded store while the
    # destinations are scalars of their own.
    store = einsums.create_random_tensor("store", [8, 8], dtype="float64")
    A = [store[0:3, 0:4], store[2:5, 1:5]]
    B = [store[4:7, 2:6], store[1:4, 3:7]]

    ref = []
    for a, b in zip(A, B):
        r = einsums.create_zero_tensor("r", [1], dtype="float64")
        einsums.linalg.dot(r, a, b)
        ref.append(np.asarray(r).copy())

    got = [einsums.create_zero_tensor(f"g{i}", [1], dtype="float64")
           for i in range(2)]
    einsums.linalg.grouped_dot(got, A, B)
    for i, (g, r) in enumerate(zip(got, ref)):
        np.testing.assert_array_equal(np.asarray(g), r, err_msg=f"entry {i}")


def test_grouped_dot_records_every_read_and_write():
    A = [einsums.create_random_tensor(f"A{i}", [3, 3], dtype="float64")
         for i in range(4)]
    B = [einsums.create_random_tensor(f"B{i}", [3, 3], dtype="float64")
         for i in range(4)]
    R = [einsums.create_zero_tensor(f"R{i}", [1], dtype="float64")
         for i in range(4)]

    g = cg.Graph("io")
    with cg.capture(g):
        einsums.linalg.grouped_dot(R, A, B)

    node = json.loads(g.to_json())["nodes"][0]
    assert len(node["inputs"]) == 8
    assert len(node["outputs"]) == 4


def test_grouped_dot_rejects_malformed_runs():
    A = [einsums.create_random_tensor("A", [3, 3], dtype="float64")]
    B = [einsums.create_random_tensor("B", [3, 3], dtype="float64")]
    R = [einsums.create_zero_tensor("R", [1], dtype="float64")]

    with pytest.raises(Exception):
        einsums.linalg.grouped_dot([], [], [])
    with pytest.raises(Exception):
        einsums.linalg.grouped_dot(R, A, B + B)
    wrong = [einsums.create_random_tensor("W", [2, 4], dtype="float64")]
    with pytest.raises(Exception):
        einsums.linalg.grouped_dot(R, A, wrong)


# ──────────────────────────────────────────────────────────────────────────
# grouped_axpby
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_grouped_axpby_eager_is_bitwise_the_loop(dtype):
    shapes = [[4, 5], [1, 1], [0, 3], [9, 2]]
    alphas = [2.0, -3.0, 0.5, 1.25]
    betas = [1.0, -0.75, 2.5, 0.0]

    X = [einsums.create_random_tensor(f"X{i}", s, dtype=dtype)
         for i, s in enumerate(shapes)]
    Y0 = [einsums.create_random_tensor(f"Y{i}", s, dtype=dtype)
          for i, s in enumerate(shapes)]

    looped = [einsums.create_zero_tensor(f"l{i}", s, dtype=dtype)
              for i, s in enumerate(shapes)]
    grouped = [einsums.create_zero_tensor(f"g{i}", s, dtype=dtype)
               for i, s in enumerate(shapes)]
    for dest in (looped, grouped):
        for d, y in zip(dest, Y0):
            einsums.linalg.axpby(1.0, y, 0.0, d)

    for a, x, b, y in zip(alphas, X, betas, looped):
        einsums.linalg.axpby(a, x, b, y)
    einsums.linalg.grouped_axpby(alphas, X, betas, grouped)

    for i, (g, l) in enumerate(zip(grouped, looped)):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(l),
                                      err_msg=f"entry {i}")


def test_grouped_axpby_chain_into_one_destination_keeps_term_order():
    # The idiom the operation exists for: several scalars accumulated into one
    # element of a shared matrix, whose sum has to arrive in emission order.
    n = 6
    S = [einsums.create_random_tensor(f"s{i}", [1, 1], dtype="float64")
         for i in range(n)]
    alphas = [1.0 + 0.25 * i for i in range(n)]
    betas = [1.0] * n

    looped = einsums.create_zero_tensor("looped", [1, 1], dtype="float64")
    for a, s in zip(alphas, S):
        einsums.linalg.axpby(a, s, 1.0, looped)

    grouped = einsums.create_zero_tensor("grouped", [1, 1], dtype="float64")
    g = cg.Graph("chain")
    with cg.capture(g):
        einsums.linalg.grouped_axpby(alphas, S, betas, [grouped] * n)
    assert g.num_nodes() == 1
    assert _kinds(g) == ["GroupedAxpby"]
    g.execute()

    np.testing.assert_array_equal(np.asarray(grouped), np.asarray(looped))


def test_grouped_axpby_records_the_read_only_where_beta_is_nonzero():
    X = [einsums.create_random_tensor(f"X{i}", [3, 3], dtype="float64")
         for i in range(2)]
    Y = [einsums.create_zero_tensor(f"Y{i}", [3, 3], dtype="float64")
         for i in range(2)]

    g = cg.Graph("io")
    with cg.capture(g):
        einsums.linalg.grouped_axpby([1.0, 1.0], X, [0.0, 1.0], Y)

    node = json.loads(g.to_json())["nodes"][0]
    assert len(node["inputs"]) == 3
    assert len(node["outputs"]) == 2


def test_grouped_axpby_rejects_malformed_runs():
    X = [einsums.create_random_tensor("X", [3, 3], dtype="float64")]
    Y = [einsums.create_zero_tensor("Y", [3, 3], dtype="float64")]

    with pytest.raises(Exception):
        einsums.linalg.grouped_axpby([], [], [], [])
    with pytest.raises(Exception):
        einsums.linalg.grouped_axpby([1.0, 1.0], X, [0.0, 0.0], Y)
    wrong = [einsums.create_random_tensor("W", [2, 4], dtype="float64")]
    with pytest.raises(Exception):
        einsums.linalg.grouped_axpby([1.0], wrong, [0.0], Y)


# ──────────────────────────────────────────────────────────────────────────
# the adopted pair
# ──────────────────────────────────────────────────────────────────────────


def test_grouped_pair_keeps_the_edge_between_reduction_and_accumulate():
    # One node reduces a family into scalars, the next accumulates them into a
    # shared element. Two levels, and the interleaved loop's answer.
    n = 5
    A = [einsums.create_random_tensor(f"a{i}", [4, 4], dtype="float64")
         for i in range(n)]
    B = [einsums.create_random_tensor(f"b{i}", [4, 4], dtype="float64")
         for i in range(n)]
    S = [einsums.create_zero_tensor(f"s{i}", [1, 1], dtype="float64")
         for i in range(n)]
    dest = einsums.create_zero_tensor("dest", [1, 1], dtype="float64")

    looped = einsums.create_zero_tensor("looped", [1, 1], dtype="float64")
    scratch = einsums.create_zero_tensor("scratch", [1, 1], dtype="float64")
    for a, b in zip(A, B):
        einsums.linalg.dot(scratch, a, b)
        einsums.linalg.axpby(-2.0, scratch, 1.0, looped)

    g = cg.Graph("pair")
    with cg.capture(g):
        einsums.linalg.grouped_dot(S, A, B)
        einsums.linalg.grouped_axpby([-2.0] * n, S, [1.0] * n, [dest] * n)
    assert g.num_nodes() == 2
    assert _kinds(g) == ["GroupedDot", "GroupedAxpby"]
    g.execute()

    np.testing.assert_array_equal(np.asarray(dest), np.asarray(looped))
