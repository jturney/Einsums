# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""A per-quadruplet chain in the shape the psi4 DLPNO-CCSDTQ port writes.

The port's per-entity work is a chain of grouped GEMMs with a permute-to-
contiguous copy between them: a rotation writes a block, the block is copied into
the order the next rotation wants, and the next rotation reads the copy. The
triples phase of ``examples/dlpno`` writes exactly that shape today, through
``grouped_permute`` for the copy and ``grouped_batched_gemm`` for the rotations.

What this pins is that such a chain is one region, that every link of it raises
and lowers back to the grouped kind it came from, that the permute survives as a
permute rather than being folded into a contraction it is not, and what the
search and the layout pass make of it.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la

# One entity per member with its own extents, which is what a truncated domain
# gives a local-correlation method and what makes the family ragged.
NQ = [3, 2, 4, 3]
NR = [4, 3, 2, 4]
NS = [2, 4, 3, 2]
MEMBERS = len(NQ)


def _kinds(graph):
    """A census of the node kinds, through the report that names them."""
    return collections.Counter(b.kind_name for b in graph.serializability_report())


def _chain(graph, rng, dtype="float64"):
    """``O = ((A B)^T C) D`` per member, with the transpose as an explicit copy.

    The copy is a node rather than a transpose flag on the next GEMM, which is
    what the port does: the block it rotates next is read by more than one
    consumer and by a kernel that wants it contiguous.
    """
    def arr(name, shape):
        return einsums.array(np.asarray(rng.standard_normal(shape), dtype=dtype), name=name)

    a = [arr(f"A{i}", (NQ[i], NR[i])) for i in range(MEMBERS)]
    b = [arr(f"B{i}", (NR[i], NS[i])) for i in range(MEMBERS)]
    c = [arr(f"C{i}", (NQ[i], NR[i])) for i in range(MEMBERS)]
    d = [arr(f"D{i}", (NR[i], NS[i])) for i in range(MEMBERS)]
    o = [einsums.array(np.zeros((NS[i], NS[i]), dtype=dtype), name=f"O{i}") for i in range(MEMBERS)]

    w = [graph.create_zero_tensor(f"W{i}", [NQ[i], NS[i]], True, dtype) for i in range(MEMBERS)]
    wt = [graph.create_zero_tensor(f"Wt{i}", [NS[i], NQ[i]], True, dtype) for i in range(MEMBERS)]
    v = [graph.create_zero_tensor(f"V{i}", [NS[i], NR[i]], True, dtype) for i in range(MEMBERS)]
    with cg.capture(graph):
        cg.grouped_batched_gemm(1.0, a, b, 0.0, w)                                  # W = A B
        la.grouped_permute("ba <- ab", wt, w, [0.0] * MEMBERS, [1.0] * MEMBERS)     # Wt = W^T, copied
        cg.grouped_batched_gemm(1.0, wt, c, 0.0, v)                                 # V = Wt C
        cg.grouped_batched_gemm(1.0, v, d, 0.0, o)                                  # O = V D
    return dict(a=a, b=b, c=c, d=d, o=o)


def _oracle(pools):
    """What each member's chain computes, per member, in numpy."""
    out = []
    for i in range(MEMBERS):
        a = np.array(pools["a"][i])
        b = np.array(pools["b"][i])
        c = np.array(pools["c"][i])
        d = np.array(pools["d"][i])
        out.append(((a @ b).T @ c) @ d)
    return out


def test_the_chain_is_one_region_and_every_link_round_trips():
    """The raise, proved by lowering it unchanged and demanding the same bits."""
    rng = np.random.default_rng(17)
    graph = cg.Graph("quadruplet chain")
    pools = _chain(graph, rng)
    captured = _kinds(graph)

    identity = cg.RegionIdentity()
    manager = cg.PassManager()
    manager.add(identity)
    manager.set_optimizer_budget(0)
    graph.apply(manager)

    # ONE region over the whole chain: the permute between two rotations is not a
    # barrier, which is the difference this makes to a body written in grouped
    # nodes.
    assert identity.regions_formed == 1, f"the chain is one region, got {identity.regions_formed}"
    assert identity.regions_rewritten == 1, "the region formed but was not lowered"

    # And every link came back as the grouped kind it went in as: three grouped
    # batches and the copy, never a per-member loop of ordinary nodes.
    lowered = _kinds(graph)
    assert lowered["GroupedBatchedGemm"] == 3, f"{dict(lowered)}"
    assert lowered["GroupedPermute"] == 1, f"{dict(lowered)}"
    assert lowered["Permute"] == 0 and lowered["Einsum"] == 0, f"{dict(lowered)}"
    assert lowered["GroupedBatchedGemm"] == captured["GroupedBatchedGemm"]
    assert lowered["GroupedPermute"] == captured["GroupedPermute"]

    rest = cg.PassManager()
    rest.populate_default()
    rest.set_optimizer_budget(0)
    graph.apply(rest)
    graph.execute()

    for index, (want, got) in enumerate(zip(_oracle(pools), pools["o"])):
        assert np.array_equal(want, np.array(got), equal_nan=True) or (
            float(np.max(np.abs(want - np.array(got)))) / max(float(np.max(np.abs(want))), 1e-30) < 1e-13
        ), f"member {index} moved through the round trip"


def test_the_search_leaves_the_chain_alone_and_says_why():
    """What the search makes of it: the author's bracketing is already the best.

    A chain of three rotations has one bracketing the author chose and the search
    ranks the alternatives against it. It declines here, which is the answer, and
    the pin is that it declines on a NUMBER rather than by failing to model the
    chain: the three rotations are products it reads, and only the copy is not.
    """
    rng = np.random.default_rng(17)
    graph = cg.Graph("quadruplet chain")
    _chain(graph, rng)

    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    manager = cg.PassManager()
    manager.add(mtf)
    manager.set_optimizer_budget(0)
    graph.apply(manager)

    assert not mtf.was_cut_off, "the budget is zero, so nothing here may be the clock's decision"
    assert mtf.regions_formed == 1
    assert mtf.regions_rewritten == 0

    reasons = dict(mtf.skip_reasons)
    assert any("beats the captured form" in reason for reason in reasons), (
        f"the search should decline on a cost, not on a shape; reasons were {reasons}")
    # Exactly one statement of the chain is not a product, and it is the copy.
    assert reasons.get("statement is not a product this pass can model") == 1, (
        f"only the permute is unmodellable; reasons were {reasons}")


def test_the_layout_pass_does_not_reach_a_grouped_copy():
    """LayoutAssignment matches the dense permute only, so the copy survives.

    Worth pinning rather than assuming, because a chain whose copy the layout
    pass folded away would compute something else: the next rotation reads the
    copy's order, not the block's. What a reader should take from it is that the
    grouped family is outside the layout pass's reach entirely, so a port that
    writes its copies grouped gets neither the pass's help nor its harm.
    """
    rng = np.random.default_rng(17)
    graph = cg.Graph("quadruplet chain")
    pools = _chain(graph, rng)
    before = _kinds(graph)

    manager = cg.PassManager()
    manager.populate_default()
    manager.set_optimizer_budget(0)
    graph.apply(manager)
    after = _kinds(graph)

    assert after["GroupedPermute"] == before["GroupedPermute"] == 1
    assert after["GroupedBatchedGemm"] == before["GroupedBatchedGemm"] == 3

    graph.execute()
    for index, (want, got) in enumerate(zip(_oracle(pools), pools["o"])):
        scale = max(float(np.max(np.abs(want))), 1e-30)
        assert float(np.max(np.abs(want - np.array(got)))) / scale < 1e-13, f"member {index}"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_chain_agrees_with_numpy_in_both_real_dtypes(dtype):
    rng = np.random.default_rng(23)
    graph = cg.Graph("quadruplet chain")
    pools = _chain(graph, rng, dtype)

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

    tolerance = 2e-5 if dtype == "float32" else 1e-13
    for index, (want, got) in enumerate(zip(_oracle(pools), pools["o"])):
        scale = max(float(np.max(np.abs(want))), 1e-30)
        assert float(np.max(np.abs(want - np.array(got)))) / scale < tolerance, f"member {index}"
