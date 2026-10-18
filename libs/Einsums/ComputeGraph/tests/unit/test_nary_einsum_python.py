# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""einsums.einsum over three or more operands, against numpy.

The einsum is contracted two operands at a time in the order
einsums.graph.plan_nary_einsum chooses. Every case is checked eagerly and
captured, on every dtype, against numpy.einsum, which shares no code with
either path.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close


def _tensor(name, shape, dtype, rng):
    data = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        data = data + 1j * rng.standard_normal(shape)
    out = einsums.create_zero_tensor(name, list(shape), dtype=dtype)
    np.asarray(out)[...] = data.astype(dtype)
    return out


def _run(capture, spec, C, operands, **prefactors):
    if not capture:
        einsums.einsum(spec, C, *operands, **prefactors)
        return
    graph = cg.Graph("nary")
    with cg.capture(graph):
        einsums.einsum(spec, C, *operands, **prefactors)
    graph.apply(cg.default_pass_manager())
    graph.execute()


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "captured"])
def test_ladder_with_prefactors(dtype, capture):
    """The DF particle ladder as the equation reads, one operand read twice, both prefactors."""
    rng = np.random.default_rng(1)
    B = _tensor("B", (6, 4, 4), dtype, rng)
    tau = _tensor("tau", (3, 3, 4, 4), dtype, rng)
    C = _tensor("C", (3, 3, 4, 4), dtype, rng)
    start = np.asarray(C).copy()
    c_pf, ab_pf = (0.5 - 0.25j, 1.5 + 0.5j) if np.dtype(dtype).kind == "c" else (0.5, 1.5)

    _run(capture, "i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", C, (B, B, tau), c_pf=c_pf, ab_pf=ab_pf)

    want = c_pf * start + ab_pf * np.einsum("Qae,Qbf,ijef->ijab", np.asarray(B), np.asarray(B), np.asarray(tau))
    assert_close(np.asarray(C), want)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "captured"])
def test_four_operands_shared_and_lone_letters_and_conj(dtype, capture):
    """Q is in three operands and not the output; m is in one operand alone; one operand is conjugated."""
    rng = np.random.default_rng(2)
    A = _tensor("A", (3, 4), dtype, rng)
    B = _tensor("B", (3, 2, 5), dtype, rng)
    W = _tensor("W", (3,), dtype, rng)
    D = _tensor("D", (4, 2), dtype, rng)
    C = einsums.create_zero_tensor("C", [2, 4], dtype=dtype)

    _run(capture, "b,a <- Q,a ; conj(Q,b,m) ; Q ; a,b", C, (A, B, W, D))

    want = np.einsum("Qa,Qbm,Q,ab->ba", np.asarray(A), np.conj(np.asarray(B)), np.asarray(W), np.asarray(D))
    assert_close(np.asarray(C), want)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "captured"])
def test_views_as_operands_and_output(dtype, capture):
    """Strided slices of larger tensors, read and written, as the graph's loops hand them out."""
    rng = np.random.default_rng(3)
    big_a = _tensor("A", (5, 7), dtype, rng)
    big_b = _tensor("B", (7, 6), dtype, rng)
    big_c = _tensor("C", (6, 4), dtype, rng)
    out = einsums.create_zero_tensor("out", [6, 5], dtype=dtype)
    A, B, Cop, R = big_a[1:4, :], big_b[:, 0:5], big_c[0:5, 1:3], out[1:4, 2:4]

    _run(capture, "i,l <- i,j ; j,k ; k,l", R, (A, B, Cop))

    want = np.zeros((6, 5), dtype=dtype)
    want[1:4, 2:4] = np.asarray(A) @ np.asarray(B) @ np.asarray(Cop)
    assert_close(np.asarray(out), want)


def test_prefactors_may_follow_the_operands_positionally():
    rng = np.random.default_rng(4)
    A, B, D = (_tensor(n, (3, 3), "float64", rng) for n in "ABD")
    C = _tensor("C", (3, 3), "float64", rng)
    start = np.asarray(C).copy()
    einsums.einsum("i,l <- i,j ; j,k ; k,l", C, A, B, D, 2.0, -1.0)
    assert_close(np.asarray(C), 2.0 * start - np.asarray(A) @ np.asarray(B) @ np.asarray(D))


def test_two_operands_still_take_positional_prefactors():
    rng = np.random.default_rng(5)
    A, B = _tensor("A", (3, 4), "float64", rng), _tensor("B", (4, 2), "float64", rng)
    C = _tensor("C", (3, 2), "float64", rng)
    start = np.asarray(C).copy()
    einsums.einsum("ij <- ik ; kj", C, A, B, 0.5, 2.0)
    assert_close(np.asarray(C), 0.5 * start + 2.0 * np.asarray(A) @ np.asarray(B))


def test_plan_picks_the_cheaper_order():
    """(ae|bf) first for benzene in cc-pVDZ; Bvv tau first for a large virtual space."""
    spec = "i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f"
    benzene = cg.plan_nary_einsum(spec, [[420, 93, 93], [420, 93, 93], [21, 21, 93, 93]])
    assert (benzene[0].left, benzene[0].right) == (0, 1)
    assert list(benzene[0].dims) == [93, 93, 93, 93]
    large_v = cg.plan_nary_einsum(spec, [[50, 500, 500], [50, 500, 500], [2, 2, 500, 500]])
    assert large_v[0].right == 2


def _chain_over_spaces():
    """D(i,j) = A(i,a) B(a,k) C(k,j): i and j occupied, a virtual, k over a space declared below the virtuals.

    A with B first costs o v s + o^2 s and B with C first o v s + o^2 v, so the
    first is cheaper term for term; these extents, three occupied, two virtual
    and six over s, make the second cheaper, 54 against 90.
    """
    graph = cg.Graph("spaces")
    registry = graph.space_registry
    occ = registry.register_space(cg.index_space("occ", "o"))
    virt = registry.register_space(cg.index_space("virt", "v"))
    small = registry.register_space(cg.index_space("small", "s"))
    registry.declare_less(small, virt)
    return graph, occ, virt, small


def test_declared_spaces_decide_the_order_where_the_extents_would_not():
    spec, dims = "i,j <- i,a ; a,k ; k,j", [[3, 2], [2, 6], [6, 3]]
    graph, o, v, s = _chain_over_spaces()
    by_extent = cg.plan_nary_einsum(spec, dims)
    assert (by_extent[0].left, by_extent[0].right) == (1, 2)
    by_space = cg.plan_nary_einsum_with_spaces(graph, spec, dims, [[o, v], [v, s], [s, o], [o, o]])
    assert (by_space[0].left, by_space[0].right) == (0, 1)
    with pytest.raises(ValueError):
        cg.plan_nary_einsum_with_spaces(graph, spec, dims, [[o, o], [v, s], [s, o], [o, o]])


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_a_captured_einsum_is_planned_by_its_operands_annotations(dtype, monkeypatch):
    graph, o, v, s = _chain_over_spaces()
    rng = np.random.default_rng(8)
    A, B, C = _tensor("A", (3, 2), dtype, rng), _tensor("B", (2, 6), dtype, rng), _tensor("C", (6, 3), dtype, rng)
    D = einsums.create_zero_tensor("D", [3, 3], dtype=dtype)
    # Annotated before capture, as capture reads the annotations.
    cg.annotate(A, (o, v), graph=graph)
    cg.annotate(B, (v, s), graph=graph)
    cg.annotate(C, (s, o), graph=graph)

    plans = []
    planner = cg.plan_nary_einsum_with_spaces

    def recorded(*args):
        plans.append((args[3], planner(*args)))
        return plans[-1][1]

    monkeypatch.setattr(cg, "plan_nary_einsum_with_spaces", recorded)
    with cg.capture(graph):
        einsums.einsum("i,j <- i,a ; a,k ; k,j", D, A, B, C)
    graph.execute()

    assert len(plans) == 1
    spaces, steps = plans[0]
    assert [list(x) for x in spaces] == [[o, v], [v, s], [s, o], []]
    assert (steps[0].left, steps[0].right) == (0, 1)
    assert_close(np.asarray(D), np.einsum("ia,ak,kj->ij", np.asarray(A), np.asarray(B), np.asarray(C)))


def test_mismatched_extents_are_refused():
    rng = np.random.default_rng(6)
    A, B = _tensor("A", (3, 4), "float64", rng), _tensor("B", (5, 2), "float64", rng)
    D, C = _tensor("D", (2, 2), "float64", rng), einsums.create_zero_tensor("C", [3, 2])
    with pytest.raises(ValueError, match="extent"):
        einsums.einsum("i,l <- i,j ; j,k ; k,l", C, A, B, D)


def test_a_captured_operand_keeps_its_graph_identity(tmp_path):
    """A tensor declared on the graph stays that tensor through the lowering, annotations and all.

    Bvv here exists only as a graph declaration with a file backing. Had the
    lowering handed the binary einsums a stand-in rather than the declared
    tensor, nothing would fill it from the file and the result would not be
    the ladder.
    """
    rng = np.random.default_rng(7)
    source = _tensor("Bsrc", (6, 4, 4), "float64", rng)
    path = str(tmp_path / "ints.etn")
    einsums.io.write(path, "Bvv", source)

    graph = cg.Graph("identity")
    B = graph.declare_tensor("Bvv", [6, 4, 4])
    cg.annotate(B, backing=einsums.io.etn(path, "Bvv"), graph=graph)
    tau = _tensor("tau", (3, 3, 4, 4), "float64", rng)
    r2 = einsums.create_zero_tensor("r2", [3, 3, 4, 4])
    with cg.capture(graph):
        einsums.einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", r2, B, B, tau)
    graph.apply(cg.default_pass_manager())
    graph.execute()

    bvv = np.asarray(source)
    assert_close(np.asarray(r2), np.einsum("Qae,Qbf,ijef->ijab", bvv, bvv, np.asarray(tau)))
