# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""MultiTermFactorization over a grouped pair body.

The grouped node family carries a local-correlation method's per-pair work, and a
grouped node is one operation over a family of members whose extents differ,
which is a contraction with one more free letter. This is what the search does
with a body written in them: the residual terms of one pair body are several
products over the same per-pair operands, and a partial product both of them want
is a GROUPED tensor, allocated per member from the family's extent table.

Every test here runs with the optimizer budget at zero and asserts the pass did
not report a cut-off, so no decision below is the clock's.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

import einsums
import einsums.graph as cg

# The member extents, deliberately different member by member: a shared
# intermediate allocated at one shape could not give the right answer for all
# four, so the numbers below are themselves the evidence that each member got a
# buffer of its own.
M = [3, 2, 4, 3]      # rows per member
K = [20, 24, 18, 22]  # the large link axis
N = [2, 2, 2, 2]      # the shared inner axis, small
P = [20, 18, 22, 20]  # X's column axis
Q = [20, 22, 18, 20]  # Y's column axis


def _pair_body(graph, rng):
    """Two products per member over the same per-pair operands.

    With the author's own bracketing, per member p::

        T1_p = B_p C_p ;  X_p = A_p T1_p        so X = A (B C)
        T2_p = B_p D_p ;  Y_p = A_p T2_p        so Y = A (B D)

    Both want ``A B`` and neither author intermediate is it, so reaching it means
    re-bracketing both and committing the partial product they share. The link
    axis is large and the inner axis small, which is what makes ``A (B C)`` the
    expensive bracketing and ``(A B) C`` the cheap one.
    """
    def arr(name, shape):
        return einsums.array(rng.standard_normal(shape), name=name)

    a = [arr(f"A{i}", (M[i], K[i])) for i in range(len(M))]
    b = [arr(f"B{i}", (K[i], N[i])) for i in range(len(M))]
    c = [arr(f"C{i}", (N[i], P[i])) for i in range(len(M))]
    d = [arr(f"D{i}", (N[i], Q[i])) for i in range(len(M))]
    x = [einsums.array(np.zeros((M[i], P[i])), name=f"X{i}") for i in range(len(M))]
    y = [einsums.array(np.zeros((M[i], Q[i])), name=f"Y{i}") for i in range(len(M))]

    t1 = [graph.create_zero_tensor(f"T1_{i}", [K[i], P[i]], True) for i in range(len(M))]
    t2 = [graph.create_zero_tensor(f"T2_{i}", [K[i], Q[i]], True) for i in range(len(M))]
    with cg.capture(graph):
        cg.grouped_batched_gemm(1.0, b, c, 0.0, t1)
        cg.grouped_batched_gemm(1.0, a, t1, 0.0, x)
        cg.grouped_batched_gemm(1.0, b, d, 0.0, t2)
        cg.grouped_batched_gemm(1.0, a, t2, 0.0, y)
    return x + y


def _kinds(graph):
    """A census of the node kinds, through the report that names them."""
    return collections.Counter(b.kind_name for b in graph.serializability_report())


def _run(search):
    """Build the body, optionally search it, then materialize and replay."""
    rng = np.random.default_rng(5)
    graph = cg.Graph("grouped pair body")
    outputs = _pair_body(graph, rng)

    mtf = None
    if search:
        # The structural-algebraic phase on its own and FIRST. The search declares
        # its intermediates deferred and Materialization is what gives them
        # storage, so a search bolted onto the end of the default pipeline would
        # leave tensors nothing ever allocates.
        mtf = cg.MultiTermFactorization()
        mtf.set_search_enabled(True)
        mtf.set_dump(True)
        first = cg.PassManager()
        first.add(mtf)
        first.set_optimizer_budget(0)
        graph.apply(first)

    rest = cg.PassManager()
    rest.populate_default()
    rest.set_optimizer_budget(0)
    graph.apply(rest)
    graph.execute()
    return graph, mtf, [np.array(t, copy=True) for t in outputs]


def test_a_grouped_pair_body_shares_one_grouped_intermediate():
    plain_graph, _, expected = _run(search=False)
    graph, mtf, actual = _run(search=True)

    assert not mtf.was_cut_off, "the budget is zero, so nothing here may be the clock's decision"
    assert mtf.regions_formed == 1, f"the four grouped statements are one region, got {mtf.regions_formed}"
    assert mtf.regions_rewritten == 1, (
        f"the region was formed but not rewritten; skips: {dict(mtf.skip_reasons)}")

    # THE MECHANISM. One shared intermediate, committed across the two products,
    # and both terms re-bracketed to reach it.
    assert mtf.num_shared == 1, f"expected one shared intermediate, got {mtf.num_shared}"
    assert mtf.num_rebracketed == 4, f"expected four terms re-bracketed, got {mtf.num_rebracketed}"
    assert mtf.num_copies == 2, (
        f"each author intermediate is copied into the consumer that profits, got {mtf.num_copies}")

    # THE MEMBER LETTER, in the algebra the pass rewrote. Every statement of the
    # region carries it, and it is outermost on every operand.
    dump = mtf.dump_text
    assert "#m0" in dump, f"no member letter in the region dump:\n{dump}"
    after = dump.split("after:")[1]
    for line in (l for l in after.splitlines() if "=" in l):
        head = line.split("=")[0]
        assert "#m" in head, f"a rewritten statement lost its member letter: {line}"

    # WHAT WAS EMITTED: grouped nodes, never a per-member loop of ordinary ones.
    # Three grouped batches where the author wrote four, because the shared
    # partial product replaces both author intermediates and dead-node
    # elimination then takes them away.
    kinds = _kinds(graph)
    assert _kinds(plain_graph)["GroupedBatchedGemm"] == 4
    assert kinds["GroupedBatchedGemm"] == 3, f"emitted node kinds: {dict(kinds)}"
    assert kinds["Einsum"] == 0, f"a grouped term must never lower to per-member nodes: {dict(kinds)}"

    # ALLOCATED PER MEMBER. One shared value, four members, so four tensors more
    # than the untouched graph holds and one Materialize node for each.
    assert graph.num_tensors() == plain_graph.num_tensors() + len(M), (
        f"a grouped shared intermediate is one buffer per member; the graph gained "
        f"{graph.num_tensors() - plain_graph.num_tensors()} tensor(s) for {len(M)} member(s)")
    assert kinds["Materialize"] == len(M), (
        f"one Materialize per member is what a grouped intermediate needs, got {dict(kinds)}")

    # THE NUMBERS, to the re-associating tier's bound. Folding a common factor out
    # of a product changes the summation order, so this is norm-relative rather
    # than bitwise.
    for index, (want, got) in enumerate(zip(expected, actual)):
        scale = max(float(np.max(np.abs(want))), 1e-30)
        deviation = float(np.max(np.abs(want - got))) / scale
        assert deviation < 1e-12, f"output {index} moved by {deviation:.3e}"


def test_the_shared_intermediate_is_priced_by_the_families_typical_extents():
    """Which rung of the comparison decides, asserted rather than assumed.

    A ragged letter has one extent per member and no space says which, so it is an
    ANONYMOUS variable in the cost polynomial and the member letter is the
    family's registered space. One anonymous variable blocks the typical-extent
    rung for the whole polynomial, so what decides is the BOUND-extent rung,
    below the scale order, fed with each ragged letter's typical extent. The
    report's cost line is where that shows: the member letter appears as its
    space's scale symbol and every other letter as an anonymous one.
    """
    _, mtf, _ = _run(search=True)
    lines = [line for line in mtf.dump_text.splitlines() if "#" in line and "*gm" in line]
    assert lines, f"no priced statement in the dump:\n{mtf.dump_text}"
    for line in lines:
        cost = line.split("#", 1)[1] if line.count("#") else ""
        assert "gm4" in line, f"the member letter is not priced by its family's space: {line}"
        assert "?#" in line or "?~" in line, (
            f"a ragged letter should be anonymous, so the bound-extent rung is what decides: {line}")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_rewritten_body_agrees_with_numpy(dtype):
    """The oracle, per member, over the shapes the family actually has."""
    rng = np.random.default_rng(11)
    a = [rng.standard_normal((M[i], K[i])).astype(dtype) for i in range(len(M))]
    b = [rng.standard_normal((K[i], N[i])).astype(dtype) for i in range(len(M))]
    c = [rng.standard_normal((N[i], P[i])).astype(dtype) for i in range(len(M))]
    d = [rng.standard_normal((N[i], Q[i])).astype(dtype) for i in range(len(M))]

    ta = [einsums.array(v, name=f"A{i}") for i, v in enumerate(a)]
    tb = [einsums.array(v, name=f"B{i}") for i, v in enumerate(b)]
    tc = [einsums.array(v, name=f"C{i}") for i, v in enumerate(c)]
    td = [einsums.array(v, name=f"D{i}") for i, v in enumerate(d)]
    x = [einsums.array(np.zeros((M[i], P[i]), dtype=dtype), name=f"X{i}") for i in range(len(M))]
    y = [einsums.array(np.zeros((M[i], Q[i]), dtype=dtype), name=f"Y{i}") for i in range(len(M))]

    graph = cg.Graph("grouped pair body oracle")
    t1 = [graph.create_zero_tensor(f"T1_{i}", [K[i], P[i]], True, dtype) for i in range(len(M))]
    t2 = [graph.create_zero_tensor(f"T2_{i}", [K[i], Q[i]], True, dtype) for i in range(len(M))]
    with cg.capture(graph):
        cg.grouped_batched_gemm(1.0, tb, tc, 0.0, t1)
        cg.grouped_batched_gemm(1.0, ta, t1, 0.0, x)
        cg.grouped_batched_gemm(1.0, tb, td, 0.0, t2)
        cg.grouped_batched_gemm(1.0, ta, t2, 0.0, y)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    first = cg.PassManager()
    first.add(mtf)
    first.set_optimizer_budget(0)
    graph.apply(first)
    assert not mtf.was_cut_off
    rest = cg.PassManager()
    rest.populate_default()
    rest.set_optimizer_budget(0)
    graph.apply(rest)
    graph.execute()

    tolerance = 2e-5 if dtype == "float32" else 1e-12
    for i in range(len(M)):
        want_x = a[i] @ (b[i] @ c[i])
        want_y = a[i] @ (b[i] @ d[i])
        for want, got in ((want_x, np.array(x[i])), (want_y, np.array(y[i]))):
            scale = max(float(np.max(np.abs(want))), 1e-30)
            assert float(np.max(np.abs(want - got))) / scale < tolerance
