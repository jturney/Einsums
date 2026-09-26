# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""End-to-end reproducers for pass defects a read-only audit claimed.

Each case captures a small program, runs it through the named pass alone and through the
pipelines that reach the same rewrite, executes, and compares against numpy. Every case here
reproduced before its fix and now stays as the guard for the defect its docstring names.

Claims the audit made that did NOT reproduce, and why, so nobody re-derives them:

* AxisTiling reading a redirected slot inside a tiled run. Every pass that calls
  ``Graph::redirect_slot`` (CSE, PermuteFusion, InplaceOptimization, BasisTruncation) also
  rewrites ``Node::inputs``, so no op in a run names a redirected id. The only route left is a
  view registered over an eliminated duplicate, which is already wrong before AxisTiling runs
  (the CSE case below).
* A view over a CSE-eliminated duplicate taken INSIDE the capture: the View node reads the
  duplicate's id, so its input is rewritten with the others and the view follows the survivor.
  A view sliced before the capture has no such node and does reproduce.
* ``einsum_is_uniform`` with an Unknown-dtype operand: every einsum Python binds is same-dtype,
  every tensor Python registers carries its dtype, and the graph loader refuses a tensor whose
  dtype is 'unknown', so the case needs a C++ test.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg

la = einsums.linalg

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvdz.npz"))


def _manager(*passes):
    pm = cg.PassManager()
    for p in passes:
        pm.add(p)
    return pm


def _optimize(graph, how, alone):
    """Run the pipeline named by @p how; ``alone`` builds the pass-alone manager."""
    if how == "alone":
        graph.apply(alone())
    elif how == "default":
        graph.apply(cg.default_pass_manager())
    else:
        graph.optimize(getattr(einsums._core.OptLevel, how))


def _check(build, expected, how, alone, rtol=1e-10, atol=1e-12):
    """Capture with @p build, optimize, execute, compare every output to numpy."""
    held = []
    graph = cg.Graph("audit")
    outputs = build(graph, held)
    _optimize(graph, how, alone)
    graph.execute()
    for got, want in zip(outputs, expected):
        np.testing.assert_allclose(np.asarray(got), want, rtol=rtol, atol=atol)


# ──────────────────────────────────────────────────────────────────────────
# AntisymmetrizerFolding
# ──────────────────────────────────────────────────────────────────────────


def _antisymmetrize3(x):
    out = np.zeros_like(x)
    for perm in itertools.permutations(range(3)):
        sign = round(np.linalg.det(np.eye(3)[list(perm)]))
        out += sign * x.transpose(perm)
    return out


def _folding_program(alpha, rank0):
    rng = np.random.default_rng(101)
    n = 4
    wsrc, vsrc = rng.standard_normal((n, n, n)), rng.standard_normal((n, n, n))

    def build(graph, held):
        W = graph.create_zero_tensor("W", [n, n, n])
        V = graph.create_zero_tensor("V", [n, n, n])
        Ws, Vs = einsums.asarray(wsrc), einsums.asarray(vsrc)
        result = einsums.zeros([] if rank0 else [1])
        with cg.capture(graph):
            einsums.permute("ijk <- P(i/j/k) ijk", W, Ws, 0.0, alpha)
            einsums.permute("ijk <- P(i/j/k) ijk", V, Vs, 0.0, 1.0)
            la.dot(result, W, V)
        held += [Ws, Vs]
        return [result]

    expected = alpha * np.sum(_antisymmetrize3(wsrc) * _antisymmetrize3(vsrc))
    return build, [np.reshape(expected, [] if rank0 else [1])]


def _folding_alone():
    return _manager(_G.AntisymmetryInference(), _G.AntisymmetrizerFolding())


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_folding_into_a_bare_scalar_keeps_the_factor(how):
    """AntisymmetrizerFolding keeps the full factor when it declines a dot into a rank-0 result.

    Defends against the pass repointing the dot's operand at the operator's source and
    rebuilding its executor BEFORE the rank-0 decline, which left the N-term sum reduced to one
    term: 1/6 of the energy for P(i/j/k).
    """
    build, expected = _folding_program(alpha=1.0, rank0=True)
    _check(build, expected, how, _folding_alone)


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_folding_keeps_the_operator_permutes_alpha(how):
    """AntisymmetrizerFolding carries the producer's alpha along with the term count.

    Defends against ``read_producer`` recording the source, operators and term count of
    ``W = 2 P(i/j/k) Wsrc`` but not its alpha, so that the folded dot read Wsrc directly and the
    energy came out halved.
    """
    build, expected = _folding_program(alpha=2.0, rank0=False)
    _check(build, expected, how, _folding_alone)


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_folding_leaves_a_dot_over_a_view_of_the_operator_result(how):
    """AntisymmetrizerFolding does not fold a Dot that reads the antisymmetrized tensor through a view.

    ``W = P(i/j/k) Ws`` over ``ijkl`` and the dot reads only the slice ``W[:, :, :, 0:1]``, sliced
    before the capture. The view resolves to W's buffer, so W's permute is found as the operand's
    producer. Defends the ``understands`` check on the dot, which carries Views, a feature the pass
    does not declare: without it the pass repointed the dot's view operand at the whole of Ws,
    shaped ``(3, 3, 3, 2)`` against V's ``(3, 3, 3, 1)``, and the replay stopped with "The
    dimensions of the tensors passed to dot must be the same!".
    """
    rng = np.random.default_rng(109)
    n, m = 3, 2
    wsrc, vsrc = rng.standard_normal((n, n, n, m)), rng.standard_normal((n, n, n, 1))

    def build(graph, held):
        W = graph.create_zero_tensor("W", [n, n, n, m])
        V = graph.create_zero_tensor("V", [n, n, n, 1])
        Ws, Vs = einsums.asarray(wsrc), einsums.asarray(vsrc)
        result = einsums.zeros([1])
        view = W[:, :, :, 0:1]
        with cg.capture(graph):
            einsums.permute("ijkl <- P(i/j/k) ijkl", W, Ws)
            einsums.permute("ijkl <- P(i/j/k) ijkl", V, Vs)
            la.dot(result, view, V)
        held += [Ws, Vs, view]
        return [result]

    def antisymmetrize(x):
        out = np.zeros_like(x)
        for perm in itertools.permutations(range(3)):
            sign = round(np.linalg.det(np.eye(3)[list(perm)]))
            out += sign * x.transpose(perm + (3,))
        return out

    expected = np.sum(antisymmetrize(wsrc)[..., 0:1] * antisymmetrize(vsrc))
    _check(build, [np.array([expected])], how, _folding_alone)


# ──────────────────────────────────────────────────────────────────────────
# AntisymmetrizerLinearity
# ──────────────────────────────────────────────────────────────────────────


def _linearity_program(dtype, first_map, second_map, second_alpha, c):
    rng = np.random.default_rng(102)
    n = 4

    def draw():
        x = rng.standard_normal((n, n))
        return x + 1j * rng.standard_normal((n, n)) if "complex" in dtype else x

    a, b = draw(), draw()

    def build(graph, held):
        D = graph.create_zero_tensor("D", [n, n], dtype=dtype)
        E = graph.create_zero_tensor("E", [n, n], dtype=dtype)
        A, B = einsums.asarray(a), einsums.asarray(b)
        R = einsums.zeros([n, n], dtype=dtype)
        with cg.capture(graph):
            einsums.permute(f"ij <- P(i/j) {first_map}", D, B)
            einsums.permute(f"ij <- P(i/j) {second_map}", E, A, 0.0, second_alpha)
            la.axpby(c, E, 1.0, D)
            la.axpby(1.0, D, 0.0, R)
        held += [A, B]
        return [R]

    def apply(x, index_map):
        x = x if index_map == "ij" else x.T
        return x - x.T

    return build, [apply(b, first_map) + c * second_alpha * apply(a, second_map)]


def _linearity_alone():
    return _manager(_G.AntisymmetrizerLinearity())


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_linearity_does_not_merge_operators_with_different_index_maps(how):
    """AntisymmetrizerLinearity keeps P(i/j) over ``ij`` and over ``ji`` apart.

    ``D = P(i/j)[ij <- ij] B`` and ``E = P(i/j)[ij <- ji] A`` share their groups. Defends against
    the pass comparing only the groups and folding ``D += c E`` into ``P(i/j)[ij <- ij](B + c A)``:
    the second map is a transpose, so that merged sum had the wrong sign on A's term.
    """
    build, expected = _linearity_program("float64", "ij", "ji", 1.0, 0.5)
    _check(build, expected, how, _linearity_alone)


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
@pytest.mark.parametrize("c, second_alpha", [(0.5j, 1.0), (1.0, 0.5j)])
def test_linearity_keeps_a_complex_alpha(how, c, second_alpha):
    """AntisymmetrizerLinearity carries a complex axpby alpha or operator alpha into the merged weight.

    Defends against the pass multiplying the merged weight as ``as<double>`` of each factor,
    where a complex factor made the conversion throw ("lossy complex to real conversion") and
    optimizing the graph failed with RuntimeError.
    """
    build, expected = _linearity_program("complex128", "ij", "ij", second_alpha, c)
    _check(build, expected, how, _linearity_alone)


# ──────────────────────────────────────────────────────────────────────────
# AxisTiling
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cap", [256, 512], ids=["depth1", "depth2"])
def test_axis_tiling_keeps_the_conjugation_of_dotc(cap):
    """AxisTiling keeps ``conjugated`` when it rebuilds a tiled Dot.

    Defends against the rebuild through ``dot_python`` / ``grouped_dot`` dropping the flag, so
    that the streamed ``E = dotc(K, X)`` came back as ``sum K * X`` rather than
    ``sum conj(K) * X``, both for a one-slice body (cap 256) and a chunked two-slice body
    (cap 512). AxisTiling is not in any default pipeline, so only the pass alone reaches it.
    """
    rng = np.random.default_rng(103)
    nq, no, nv = 3, 4, 4
    dtype = "complex128"
    b = rng.standard_normal((nq, no, nv)) + 1j * rng.standard_normal((nq, no, nv))
    x = rng.standard_normal((no, nv, no, nv)) + 1j * rng.standard_normal((no, nv, no, nv))

    def build(graph, held):
        B, X = einsums.asarray(b), einsums.asarray(x)
        E = einsums.zeros([1], dtype=dtype)
        K = graph.scratch("K", [no, nv, no, nv], dtype)
        with cg.capture(graph):
            einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
            la.dotc(E, K, X)
        held += [B, X]
        return [E]

    def alone():
        tiling = _G.AxisTiling()
        tiling.set_memory_cap(cap)
        return _manager(tiling, _G.Materialization())

    K = np.einsum("qia,qjb->iajb", b, b)
    _check(build, [np.array([np.vdot(K, x)])], "alone", alone)


# ──────────────────────────────────────────────────────────────────────────
# ContractionPlanning
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
@pytest.mark.parametrize("second", ["complex128", "float32"])
def test_contraction_planning_uses_each_chains_own_dtype(how, second):
    """ContractionPlanning prices and rebuilds each chain with that chain's own dtype.

    Defends against the pass reading dtype and element size once per scan, from ``chains[0]``.
    The first chain is a float64 product of square matrices, which re-parenthesizing cannot
    improve (the estimates tie exactly). The second chain, of another dtype, is the one worth
    restructuring, and the defect emitted it as Gemm nodes built for float64: execute stopped
    with "an operand holding complex128 was read as float64".

    Whether the pass restructures is the cost model's decision, and Python cannot pin it: the
    only bound constructor measures the machine, and no knob forces or forbids a rewrite. The
    case therefore asserts the numbers against numpy whichever way the pass decides.
    """
    rng = np.random.default_rng(104)

    def draw(shape, dtype):
        x = rng.standard_normal(shape)
        return (x + 1j * rng.standard_normal(shape) if "complex" in dtype else x).astype(dtype)

    s, m, k = 8, 64, 2
    chains = [("float64", [draw((s, s), "float64") for _ in range(3)]),
              (second, [draw((m, k), second), draw((k, m), second), draw((m, k), second)])]

    def build(graph, held):
        outputs = []
        for index, (dtype, (a, b, c)) in enumerate(chains):
            A, B, C = (einsums.asarray(v) for v in (a, b, c))
            T = graph.create_zero_tensor(f"T{index}", [a.shape[0], b.shape[1]], dtype=dtype)
            D = einsums.zeros([a.shape[0], c.shape[1]], dtype=dtype)
            with cg.capture(graph):
                einsums.einsum("ij <- ik ; kj", T, A, B)
                einsums.einsum("ij <- ik ; kj", D, T, C)
            held += [A, B, C]
            outputs.append(D)
        return outputs

    expected = [a @ b @ c for _, (a, b, c) in chains]
    rtol = 1e-4 if second == "float32" else 1e-10
    _check(build, expected, how, lambda: _manager(_G.ContractionPlanning()), rtol=rtol, atol=1e-4 if second == "float32" else 1e-12)


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_contraction_planning_sees_a_view_reader_of_an_interior(how):
    """ContractionPlanning keeps an interior product a view still reads.

    ``T = A B`` feeds ``D = T C`` and is also read, through a view sliced before the capture,
    by an axpby. Defends against the outside-reader scan looking for T's id among other nodes'
    inputs: the view has its own id, so the chain was re-parenthesized as ``A (B C)``, T was
    never written and the view's reader copied zeros.

    Whether the pass restructures is the cost model's decision, and Python cannot pin it (see
    the dtype case above), so the case asserts the numbers against numpy either way.
    """
    rng = np.random.default_rng(105)
    m, k = 64, 2
    a, b, c = rng.standard_normal((m, k)), rng.standard_normal((k, m)), rng.standard_normal((m, k))

    def build(graph, held):
        A, B, C = (einsums.asarray(v) for v in (a, b, c))
        T = graph.create_zero_tensor("T", [m, m])
        D, R = einsums.zeros([m, k]), einsums.zeros([m, m])
        view = T[0:m, 0:m]
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", T, A, B)
            einsums.einsum("ij <- ik ; kj", D, T, C)
            la.axpby(1.0, view, 0.0, R)
        held += [A, B, C, view]
        return [D, R]

    _check(build, [a @ b @ c, a @ b], how, lambda: _manager(_G.ContractionPlanning()))


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_contraction_planning_leaves_a_chain_with_a_view_leaf(how):
    """ContractionPlanning does not re-parenthesize a chain one of whose leaves is a view.

    ``T = A B`` then ``D = T C`` with A a strided sub-block of a larger tensor, sliced before the
    capture, and with the shapes of the dtype case above, for which ``A (B C)`` is far cheaper.
    Defends the ``understands`` filter handed to ``find_contraction_chains``: the pass does not
    declare Views, and without the filter it replaced both einsums, nodes carrying a view, with
    Gemm nodes. The rebuilt chain happens to compute the right numbers, so what fails without
    the filter is the pass audit (``EINSUMS_PASS_VERIFY``), which refuses the rewrite.

    Whether the pass restructures is the cost model's decision, and Python cannot pin it (see
    the dtype case above), so the case asserts the numbers against numpy either way.
    """
    rng = np.random.default_rng(110)
    m, k = 64, 2
    big, b, c = rng.standard_normal((m + 3, k + 1)), rng.standard_normal((k, m)), rng.standard_normal((m, k))
    a = big[3:, 1:]

    def build(graph, held):
        Big, B, C = (einsums.asarray(v) for v in (big, b, c))
        A = Big[3:m + 3, 1:k + 1]
        T = graph.create_zero_tensor("T", [m, m])
        D = einsums.zeros([m, k])
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", T, A, B)
            einsums.einsum("ij <- ik ; kj", D, T, C)
        held += [Big, A, B, C]
        return [D]

    _check(build, [a @ b @ c], how, lambda: _manager(_G.ContractionPlanning()))


# ──────────────────────────────────────────────────────────────────────────
# CSE
# ──────────────────────────────────────────────────────────────────────────

_CSE_N = 4
_CSE_FULL = [(0, _CSE_N), (0, _CSE_N)]


def _cse_data():
    rng = np.random.default_rng(106)
    return rng.standard_normal((_CSE_N, _CSE_N)), rng.standard_normal((_CSE_N, _CSE_N))


@pytest.mark.parametrize("how", ["alone", "default", "O1", "O2"])
def test_cse_sees_an_intervening_write_through_a_view_of_a_shared_input(how):
    """CSE keeps two products of A and B apart across an in-place scale of a view of A.

    Defends against Guard A collecting the survivor's input BUFFERS and checking each
    intervening node's outputs against them by pointer. The scale writes the view, whose handle
    has its own pointer, so the guard passed and the second product, which should see 2A,
    reused the first.
    """
    a, b = _cse_data()
    n = _CSE_N

    def build(graph, held):
        A, B = einsums.asarray(a), einsums.asarray(b)
        R1, R2 = einsums.zeros([n, n]), einsums.zeros([n, n])
        K1, K2 = graph.create_zero_tensor("K1", [n, n]), graph.create_zero_tensor("K2", [n, n])
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", K1, A, B)
            la.scale(2.0, cg.view(A, _CSE_FULL))
            einsums.einsum("ij <- ik ; kj", K2, A, B)
            la.axpby(1.0, K1, 0.0, R1)
            la.axpby(1.0, K2, 0.0, R2)
        held += [A, B]
        return [R1, R2]

    _check(build, [a @ b, 2 * a @ b], how, lambda: _manager(_G.CSE()))


@pytest.mark.parametrize("how", ["alone", "default", "O1", "O2"])
def test_cse_sees_a_later_write_to_the_survivor_through_a_view(how):
    """CSE does not redirect K2's readers onto K1 when a later scale mutates K1 through a view.

    Defends against Guard B counting writers per buffer pointer: the scale writes the view's
    pointer, so K1 looked single-writer and the reader of K2 saw 2 A B.
    """
    a, b = _cse_data()
    n = _CSE_N

    def build(graph, held):
        A, B = einsums.asarray(a), einsums.asarray(b)
        R1, R2 = einsums.zeros([n, n]), einsums.zeros([n, n])
        K1, K2 = graph.create_zero_tensor("K1", [n, n]), graph.create_zero_tensor("K2", [n, n])
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", K1, A, B)
            einsums.einsum("ij <- ik ; kj", K2, A, B)
            la.scale(2.0, cg.view(K1, _CSE_FULL))
            la.axpby(1.0, K1, 0.0, R1)
            la.axpby(1.0, K2, 0.0, R2)
        held += [A, B]
        return [R1, R2]

    _check(build, [2 * a @ b, a @ b], how, lambda: _manager(_G.CSE()))


@pytest.mark.parametrize("how", ["alone", "default", "O1", "O2"])
def test_cse_follows_a_view_of_the_eliminated_duplicate(how):
    """CSE keeps a view sliced from K2 before the capture live when it removes K2's producer.

    Defends against the redirect rewriting only node inputs naming K2 and K2's slot: the view
    is a separate operand whose slot still pointed into K2's storage, which nothing wrote any
    more, so R2 came out zero.
    """
    a, b = _cse_data()
    n = _CSE_N

    def build(graph, held):
        A, B = einsums.asarray(a), einsums.asarray(b)
        R1, R2 = einsums.zeros([n, n]), einsums.zeros([n, n])
        K1, K2 = graph.create_zero_tensor("K1", [n, n]), graph.create_zero_tensor("K2", [n, n])
        view = K2[0:n, 0:n]
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", K1, A, B)
            einsums.einsum("ij <- ik ; kj", K2, A, B)
            la.axpby(1.0, K1, 0.0, R1)
            la.axpby(1.0, view, 0.0, R2)
        held += [A, B, view]
        return [R1, R2]

    _check(build, [a @ b, a @ b], how, lambda: _manager(_G.CSE()))


@pytest.mark.parametrize("how", ["alone", "default", "O1", "O2"])
def test_cse_leaves_duplicates_that_read_a_view(how):
    """CSE neither keeps nor removes a product that reads a view.

    ``K1 = V B`` and ``K2 = V B`` with V a strided slice of a larger tensor, sliced before the
    capture. Defends the ``understands`` check on each candidate: CSE does not declare Views, and
    without the check it removed K2's producer, a node carrying a view, and redirected K2's reader
    onto K1. The merge happens to be value-preserving, so what fails without the check is the pass
    audit (``EINSUMS_PASS_VERIFY``), which refuses the removal.
    """
    rng = np.random.default_rng(111)
    n = _CSE_N
    big, b = rng.standard_normal((n + 1, n)), rng.standard_normal((n, n))

    def build(graph, held):
        Big, B = einsums.asarray(big), einsums.asarray(b)
        view = Big[1:n + 1, 0:n]
        R1, R2 = einsums.zeros([n, n]), einsums.zeros([n, n])
        K1, K2 = graph.create_zero_tensor("K1", [n, n]), graph.create_zero_tensor("K2", [n, n])
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", K1, view, B)
            einsums.einsum("ij <- ik ; kj", K2, view, B)
            la.axpby(1.0, K1, 0.0, R1)
            la.axpby(1.0, K2, 0.0, R2)
        held += [Big, B, view]
        return [R1, R2]

    _check(build, [big[1:] @ b, big[1:] @ b], how, lambda: _manager(_G.CSE()))


# ──────────────────────────────────────────────────────────────────────────
# DistributiveFactoring
# ──────────────────────────────────────────────────────────────────────────
#
# Run with Factor.Always so the grouping is what is tested, not the cost model. At n = 160 the
# default Factor.Auto takes the same wrong rewrite through default_pass_manager() and O2 for the
# index-key and alias cases.


def _factoring_alone():
    return _manager(_G.DistributiveFactoring(_G.Factor.Always))


def _factoring_data():
    rng = np.random.default_rng(107)
    return tuple(rng.standard_normal((6, 6)) for _ in range(4))


def _factoring_program(specs, before_second=None, eager_view=False):
    a, b1, b2, r0 = _factoring_data()
    n = a.shape[0]

    def build(graph, held):
        A, B1, B2, R = (einsums.asarray(v) for v in (a, b1, b2, r0))
        view = B2[0:n, 0:n] if eager_view else None
        with cg.capture(graph):
            einsums.einsum(specs[0], R, A, B1, 1.0)
            if before_second is not None:
                la.scale(2.0, view if eager_view else cg.view(B2, [(0, n), (0, n)]))
            einsums.einsum(specs[1], R, A, B2, 1.0)
        held += [A, B1, B2, view]
        return [R]

    return build, (a, b1, b2, r0)


def test_factoring_does_not_group_different_shared_operand_indices():
    """``R[ia] += A[ik] B1[ka]`` and ``R[ia] += A[ki] B2[ka]`` are not grouped.

    Defends against FactorKey holding only the output id, the shared operand id and the
    non-shared index list, which both members share, so that the pair was grouped and rebuilt
    with A[ik]. The shared operand's own index list was not in the key.
    """
    build, (a, b1, b2, r0) = _factoring_program(["ia <- ik ; ka", "ia <- ki ; ka"])
    _check(build, [r0 + a @ b1 + a.T @ b2], "alone", _factoring_alone)


def test_factoring_does_not_group_different_output_indices():
    """``R[ia] += A[ik] B1[ka]`` and ``R[ai] += A[ik] B2[ka]`` are not grouped.

    Defends against FactorKey omitting C's index list, which grouped the pair and rebuilt it as
    R[ia].
    """
    build, (a, b1, b2, r0) = _factoring_program(["ia <- ik ; ka", "ai <- ik ; ka"])
    _check(build, [r0 + a @ b1 + (a @ b2).T], "alone", _factoring_alone)


@pytest.mark.parametrize("second", ["ij <- P(i/j) ik ; kj", "ij <- ik ; kj"], ids=["both", "mixed"])
def test_factoring_keeps_permutation_operators(second):
    """DistributiveFactoring keeps P(..) operators when it rebuilds the spec.

    Defends against the pass copying only the c, a and b lists into the factored einsum. With
    P(i/j) on both members the operator was lost from the combined contraction; with it on one
    member only, the two were still grouped (the key did not see operators) and the factored
    result was neither.
    """
    build, (a, b1, b2, r0) = _factoring_program(["ij <- P(i/j) ik ; kj", second])

    def asym(x):
        return x - x.T

    tail = asym(a @ b2) if "P(" in second else a @ b2
    _check(build, [r0 + asym(a @ b1) + tail], "alone", _factoring_alone)


@pytest.mark.parametrize("eager_view", [False, True], ids=["captured_view", "eager_view"])
def test_factoring_sees_a_write_through_a_view_between_members(eager_view):
    """A scale of a view of B2 between the members stops the factoring.

    Defends against ``span_interferes`` comparing raw ids, which made the scale invisible: the
    factored sum ``B1 + B2`` was built at the first member's position, before the scale, so the
    second term entered with B2 instead of 2 B2.
    """
    build, (a, b1, b2, r0) = _factoring_program(["ia <- ik ; ka", "ia <- ik ; ka"], before_second=True,
                                                eager_view=eager_view)
    _check(build, [r0 + a @ b1 + 2 * a @ b2], "alone", _factoring_alone)


# ──────────────────────────────────────────────────────────────────────────
# BasisTruncation
# ──────────────────────────────────────────────────────────────────────────

_FAMILY = (("occ", "o", 300.0, "nocc"), ("vir", "v", 2700.0, "nvir"), ("aux", "x", 8100.0, "naux"))


def _host_tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


@pytest.fixture(scope="module")
def water():
    """The water/cc-pVDZ MP2 inputs ``BasisTruncation`` needs, as ``test_fno_projection_python`` builds them."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"fixture not present: {_FIXTURE}")
    registry = cg.global_space_registry()
    for name, symbol, extent, dim in _FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    _G.BasisTruncation.register_fno_space(cg.Graph("audit_fno_family"))

    z = np.load(_FIXTURE, allow_pickle=False)
    overlap, fock = z["S"], z["F"]
    nbf = overlap.shape[0]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])
    mvals, mvecs = np.linalg.eigh(z["metric"])
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    three = np.einsum("Pmn,mp,nq->Ppq", np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"]), coefficients, coefficients)
    o, v = slice(0, nocc), slice(nocc, nbf)
    ovov = np.einsum("Qia,Qjb->iajb", three[:, o, v], three[:, o, v])
    gaps = (energies[o, None, None, None] - energies[None, v, None, None]
            + energies[None, None, o, None] - energies[None, None, None, v])
    return {
        "nocc": nocc,
        "ovov": np.ascontiguousarray(ovov),
        "amplitudes": np.ascontiguousarray(ovov / gaps),
        "fock_vv": np.ascontiguousarray(np.diag(energies[v])),
        "eps_occ": np.ascontiguousarray(energies[o]),
    }


def _truncated_pairs(water, capture):
    """Capture with @p capture, truncate at 1e-3, replay, and return (fired, got, expected)."""
    nocc = water["nocc"]
    integrals = _host_tensor("K", water["ovov"])
    pairs = einsums.create_zero_tensor("pairs", [nocc, nocc])
    graph = cg.Graph("audit_fno")
    held = capture(graph, integrals, pairs)
    cg.annotate(integrals, ("occ", "vir", "occ", "vir"), graph=graph)

    inputs = [_host_tensor("t2", water["amplitudes"]), _host_tensor("F_vv", water["fock_vv"]),
              _host_tensor("eps_occ", water["eps_occ"])]
    truncation = _G.BasisTruncation()
    truncation.set_amplitudes(inputs[0])
    truncation.set_fock(inputs[1])
    truncation.set_occupied_energies(inputs[2])
    truncation.set_occupation(1e-3)
    fired = graph.apply(_manager(truncation))
    graph.apply(cg.default_pass_manager())
    graph.execute()

    U = np.asarray(truncation.transformation())
    projected = np.einsum("iajb,ay,bz->iyjz", water["ovov"], U, U)
    del held
    return fired, np.array(np.asarray(pairs)), np.einsum("iyjz,iyjz->ij", projected, projected)


def test_basis_truncation_repoints_a_view_of_the_projected_source(water):
    """BasisTruncation either follows a pre-capture view of K or declines.

    Defends against the pass rewriting the reader's K operand but not the view beside it: the
    contraction then bound the virtual index to the truncated extent through K@fno and to the
    full extent through the view, and execute refused it ("binds index 'a' to extent 9 and to
    19").
    """
    def capture(graph, integrals, pairs):
        view = integrals[:, :, :, :]
        with cg.capture(graph):
            einsums.einsum("i,a,j,b ; i,a,j,b -> i,j", pairs, integrals, view)
        return view

    fired, got, expected = _truncated_pairs(water, capture)
    assert not fired or np.allclose(got, expected, rtol=1e-10, atol=1e-14)


def test_basis_truncation_reaches_a_loop_body(water):
    """BasisTruncation reaches a reader of K inside a loop body, or declines.

    Defends against ``repoint`` walking only the parent's node list, where the Loop node's own
    I/O is empty, so the body went on contracting the untruncated K: the pass recorded a
    truncation while the replay computed the full-space pairs.
    """
    def capture(graph, integrals, pairs):
        body = graph.add_loop("once", 1, lambda it: False)
        with cg.capture(body):
            einsums.einsum("i,a,j,b ; i,a,j,b -> i,j", pairs, integrals, integrals)
        return None

    fired, got, expected = _truncated_pairs(water, capture)
    assert not fired or np.allclose(got, expected, rtol=1e-10, atol=1e-14)


# ──────────────────────────────────────────────────────────────────────────
# ElementWiseFusion
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("how", ["alone", "default", "O1", "O2"])
@pytest.mark.parametrize("eager_view", [False, True], ids=["captured_view", "eager_view"])
def test_element_wise_fusion_declines_when_x_aliases_y(how, eager_view):
    """ElementWiseFusion declines to compose ``Y = X + Y`` twice when X is a view of Y.

    X is a full view of Y, so the first axpby changes X. Defends against the pass checking
    ``x == y`` by raw id and composing the pair as if X were constant: the true result is 4Y,
    and the composed scalars (alpha = a2 + b2 a1, beta = b1 b2) gave 3Y.
    """
    rng = np.random.default_rng(108)
    n = 4
    y0 = rng.standard_normal((n, n))

    def build(graph, held):
        Y, R = einsums.asarray(y0), einsums.zeros([n, n])
        X = Y[0:n, 0:n] if eager_view else None
        with cg.capture(graph):
            if not eager_view:
                X = cg.view(Y, [(0, n), (0, n)])
            la.axpby(1.0, X, 1.0, Y)
            la.axpby(1.0, X, 1.0, Y)
            la.axpby(1.0, Y, 0.0, R)
        held += [Y, X]
        return [R]

    _check(build, [4 * y0], how, lambda: _manager(_G.ElementWiseFusion()))
