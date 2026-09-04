# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Delta elimination, and the provenance tags it recognizes, from Python.

A capability that is not reachable from Python is not shipped: the annotation
surface has been missed four separate times in this module's history, each time
by a piece of C++ that worked perfectly and had no spelling anyone writing an
example could use. So the C++ cases prove the arithmetic and these prove a
person can get at it.

The numeric comparisons here are BITWISE, for the same reason the C++ ones are:
contraction against an identity has one nonzero term, so the rewritten form owes
the same float rather than a nearby one.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, tolerance_for


def _tensor(name, array, dtype):
    t = einsums.create_zero_tensor(name, list(array.shape), dtype=dtype)
    np.asarray(t)[...] = array.astype(dtype)
    return t


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_delta_between_two_contractions_is_eliminated(dtype):
    rng = np.random.default_rng(11)
    a = rng.standard_normal((4, 5))
    d = rng.standard_normal((5, 3))

    def build(graph):
        A = _tensor("A", a, dtype)
        D = _tensor("D", d, dtype)
        C = _tensor("C", np.zeros((4, 3)), dtype)
        delta = _tensor("delta", np.eye(5), dtype)
        cg.annotate(delta, tag="identity", graph=graph)
        tmp = graph.create_zero_tensor("tmp", [4, 5], intermediate=True, dtype=dtype)
        with cg.capture(graph):
            einsums.einsum("i,j <- i,k ; k,j", tmp, A, delta)
            einsums.einsum("i,l <- i,j ; j,l", C, tmp, D)
        return C

    plain = cg.Graph("plain")
    expected_out = build(plain)
    plain.execute()
    expected = np.asarray(expected_out).copy()

    rewritten = cg.Graph("rewritten")
    actual_out = build(rewritten)
    before = rewritten.num_nodes()

    pass_ = cg.DeltaElimination()
    pm = cg.PassManager()
    pm.add(pass_)
    assert pm.run(rewritten), "the pass reported no change on a graph holding a tagged delta"
    rewritten.execute()

    assert pass_.num_eliminated == 1
    assert pass_.num_dissolved == 1
    assert rewritten.num_nodes() < before, "the delta contraction is still in the graph"

    # Bitwise: the eliminated products were exact zeros.
    np.testing.assert_array_equal(np.asarray(actual_out), expected)


def test_an_untagged_identity_is_left_alone():
    """Recognition is declared, not sniffed from the data.

    A tensor that happens to hold an identity today is not a delta. A rewrite
    justified by its contents would be written into a saved graph and wrong the
    next time a caller binds something else under the same name.
    """
    rng = np.random.default_rng(3)
    A = _tensor("A", rng.standard_normal((4, 5)), "float64")
    delta = _tensor("delta", np.eye(5), "float64")  # an identity, unannounced
    C = _tensor("C", np.zeros((4, 5)), "float64")

    g = cg.Graph("untagged")
    with cg.capture(g):
        einsums.einsum("i,j <- i,k ; k,j", C, A, delta)
    before = g.num_nodes()

    pass_ = cg.DeltaElimination()
    pm = cg.PassManager()
    pm.add(pass_)
    assert not pm.run(g)
    assert g.num_nodes() == before
    assert pass_.num_eliminated == 0


def test_a_tag_is_written_by_name_or_by_mapping():
    """Both spellings, and the attributes surviving the mapping form.

    The mapping form is what a real annotation looks like: a name plus whatever
    qualifies it. A caller who can only pass a bare name has to keep the
    qualification somewhere else, which is where it goes stale.
    """
    t = _tensor("t", np.zeros((2, 2)), "float64")
    g = cg.Graph("spellings")

    cg.annotate(t, tag="identity", graph=g)
    assert g.tensor_tag(t).name == "identity"
    assert g.tensor_tag(t).valid

    cg.annotate(t, tag={"name": "eri", "basis": "cc-pvdz", "screened": "true"}, graph=g)
    tag = g.tensor_tag(t)
    assert tag.name == "eri"
    # Sorted on the way in, so the order a caller wrote them does not leak into
    # the saved bytes.
    assert tag.attributes == [("basis", "cc-pvdz"), ("screened", "true")]
    assert tag.attribute("basis") == "cc-pvdz"
    assert tag.attribute("absent") is None


def test_annotate_takes_spaces_and_a_tag_together():
    """One call, because they are two statements about the same tensor.

    They answer different questions - how big it is, and what it is - and a
    caller who has both should not need two calls and two chances to annotate
    one and forget the other.
    """
    registry = cg.SpaceRegistry()
    registry.register_space(cg.index_space("occ", "o", 0.0, cg.GrowthClass.linear(), "no"))

    t = _tensor("t", np.zeros((3, 3)), "float64")
    g = cg.Graph("both")
    g.set_space_registry(registry)

    cg.annotate(t, spaces=("occ", "occ"), tag="identity", graph=g)
    assert g.tensor_tag(t).name == "identity"
    assert len(g.tensor_spaces(t)) == 2


def test_a_tag_survives_a_save_and_a_load(tmp_path):
    """A tag is saved structure, because nothing can re-derive it.

    A file that dropped it would come back eliminable in principle and not in
    fact, which is the kind of regression that shows up as a benchmark drifting
    rather than as a test failing.
    """
    A = _tensor("A", np.random.default_rng(5).standard_normal((4, 5)), "float64")
    delta = _tensor("delta", np.eye(5), "float64")
    C = _tensor("C", np.zeros((4, 5)), "float64")

    g = cg.Graph("tagged")
    cg.annotate(delta, tag={"name": "identity", "over": "virt"}, graph=g)
    with cg.capture(g):
        einsums.einsum("i,j <- i,k ; k,j", C, A, delta)

    path = str(tmp_path / "tagged.eig.json")
    cg.save_graph(g, path)
    loaded = cg.load_graph(path)

    # The loaded graph is eliminable, which is the property that actually
    # matters: the tag is not merely present, it is still recognized.
    pass_ = cg.DeltaElimination()
    pm = cg.PassManager()
    pm.add(pass_)
    assert pm.run(loaded)
    assert pass_.num_eliminated == 1


def test_the_dump_shows_the_substitution():
    """The before/after dump is what makes a wrong rewrite diagnosable.

    A node-list diff says which nodes changed. This says what the rewrite
    claimed: the delta gone and its letter renamed, in the algebra a person
    reads.
    """
    A = _tensor("A", np.random.default_rng(9).standard_normal((4, 5)), "float64")
    delta = _tensor("delta", np.eye(5), "float64")
    C = _tensor("C", np.zeros((4, 5)), "float64")

    g = cg.Graph("dump")
    cg.annotate(delta, tag="identity", graph=g)
    with cg.capture(g):
        einsums.einsum("i,j <- i,k ; k,j", C, A, delta)

    pass_ = cg.DeltaElimination()
    pass_.set_dump(True)
    pm = cg.PassManager()
    pm.add(pass_)
    assert pm.run(g)

    text = pass_.dump_text
    assert text, "dumping was on and produced nothing"
    before, after = text.split("before:")[1].split("after:")
    # Before: the contraction names the delta and contracts over k.
    assert "delta" in before
    assert "A[i,k]" in before
    # After: the delta is gone and k has become j.
    assert "delta" not in after
    assert "A[i,j]" in after


def test_provenance_propagation_crosses_a_permute():
    """A transposed delta is still a delta.

    Without propagation a tag would describe only the tensor a caller happened
    to annotate, and a graph that permutes before contracting - which is most of
    them - would see none of its deltas.
    """
    delta = _tensor("delta", np.eye(4), "float64")
    swapped = _tensor("swapped", np.zeros((4, 4)), "float64")

    g = cg.Graph("propagate")
    cg.annotate(delta, tag="identity", graph=g)
    with cg.capture(g):
        einsums.permute("j,i <- i,j", swapped, delta)

    pass_ = cg.ProvenancePropagation()
    pm = cg.PassManager()
    pm.add(pass_)
    pm.run(g)

    assert pass_.num_propagated == 1
    assert g.tensor_tag(swapped).name == "identity"


# ── The zero-block half ───────────────────────────────────────────────────────
#
# A letter summed over two spaces that share no element has no term to sum, so
# the contraction contributes exactly nothing. Like the delta half, the claim
# rests on a declaration, and like the delta half the data here honours it: the
# operand whose block would be summed is zero, so the captured program and the
# rewritten one can be compared bitwise instead of one of them measuring the
# annotation's falsehood.


def _disjoint_registry():
    """occ and virt share no element; pno lives inside virt.

    Both are declarations. The registry infers nothing, so what it can prove is
    exactly what was written down.
    """
    registry = cg.SpaceRegistry()
    occ = registry.register_space(cg.index_space("occ", "o", 4.0))
    virt = registry.register_space(cg.index_space("virt", "v", 4.0))
    registry.register_space(cg.index_space("aux", "x", 4.0))
    pno = registry.register_space(cg.index_space("pno", "p", 4.0))
    registry.declare_disjoint(occ, virt)
    registry.declare_contained(pno, virt)
    return registry


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_a_contraction_over_disjoint_spaces_keeps_only_its_prefactor(dtype):
    rng = np.random.default_rng(23)
    a = rng.standard_normal((4, 4))
    c0 = rng.standard_normal((4, 4))

    def build(graph):
        graph.set_space_registry(_disjoint_registry())
        A = _tensor("A", a, dtype)
        B = _tensor("B", np.zeros((4, 4)), dtype)
        C = _tensor("C", c0, dtype)
        with cg.capture(graph):
            einsums.einsum("i,j <- i,k ; k,j", C, A, B, c_pf=2.0)
        cg.annotate(A, ("aux", "occ"), graph=graph)
        cg.annotate(B, ("virt", "aux"), graph=graph)
        return C

    plain = cg.Graph("plain-zero")
    expected_out = build(plain)
    plain.execute()
    expected = np.asarray(expected_out).copy()

    rewritten = cg.Graph("rewritten-zero")
    actual_out = build(rewritten)

    pass_ = cg.DeltaElimination()
    pm = cg.PassManager()
    pm.add(pass_)
    assert pm.run(rewritten), "the pass reported no change on a contraction over disjoint spaces"
    rewritten.execute()

    assert pass_.num_zero_blocks == 1
    assert pass_.num_eliminated == 0
    np.testing.assert_array_equal(np.asarray(actual_out), expected)
    np.testing.assert_array_equal(np.asarray(actual_out), (2.0 * c0).astype(dtype))


def test_an_unrelated_pair_of_spaces_is_declined():
    """Unknown is treated exactly as No.

    The registry holds only what was declared, and a rewrite on the strength of
    a relation nobody wrote down would be a rewrite on the strength of a guess.
    """
    rng = np.random.default_rng(29)
    A = _tensor("A", rng.standard_normal((4, 4)), "float64")
    B = _tensor("B", rng.standard_normal((4, 4)), "float64")
    C = _tensor("C", np.zeros((4, 4)), "float64")

    g = cg.Graph("unrelated")
    g.set_space_registry(_disjoint_registry())
    with cg.capture(g):
        einsums.einsum("i,j <- i,k ; k,j", C, A, B)
    cg.annotate(A, ("virt", "occ"), graph=g)
    cg.annotate(B, ("aux", "virt"), graph=g)

    pass_ = cg.DeltaElimination()
    pm = cg.PassManager()
    pm.add(pass_)
    pm.set_verbosity(2)  # the skip tally is what a decline is read through
    assert not pm.run(g)
    assert pass_.num_zero_blocks == 0
    assert "nothing declared makes a shared letter's two spaces disjoint" in pm.explain()


def test_a_batched_letter_over_disjoint_spaces_is_declined():
    """A batched letter is walked, not summed, so disjointness does not make it zero."""
    rng = np.random.default_rng(31)
    A = _tensor("A", rng.standard_normal((3, 4, 4)), "float64")
    B = _tensor("B", rng.standard_normal((3, 4, 4)), "float64")
    C = _tensor("C", np.zeros((3, 4, 4)), "float64")

    g = cg.Graph("batched-zero")
    g.set_space_registry(_disjoint_registry())
    with cg.capture(g):
        einsums.einsum("b,i,j <- b,i,k ; b,k,j", C, A, B)
    cg.annotate(A, ("occ", "aux", "aux"), graph=g)
    cg.annotate(B, ("virt", "aux", "aux"), graph=g)

    pass_ = cg.DeltaElimination()
    pm = cg.PassManager()
    pm.add(pass_)
    pm.set_verbosity(2)
    assert not pm.run(g)
    assert pass_.num_zero_blocks == 0
    assert "batched rather than summed" in pm.explain()
