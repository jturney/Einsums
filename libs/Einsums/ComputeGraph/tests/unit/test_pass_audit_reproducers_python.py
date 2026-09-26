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


def _check_fires(build, expected, how, make_pass, fired, report, rtol=1e-10, atol=1e-12):
    """Like @ref _check, and asserts the pass rewrote the program.

    ``alone`` runs the pass made by @p make_pass and asks @p fired of it; ``default`` runs the
    default pipeline and looks for @p report in its explanation, which is where a pass in a
    pipeline says what it did.
    """
    held = []
    graph = cg.Graph("audit")
    outputs = build(graph, held)
    if how == "alone":
        the_pass = make_pass()
        graph.apply(_manager(the_pass))
        assert fired(the_pass), "the pass did not fire"
    else:
        manager = cg.default_pass_manager()
        graph.apply(manager)
        assert report in manager.explain(), manager.explain()
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


def _check_folds(build, expected, how, rtol=1e-10, atol=1e-12):
    """@ref _check for AntisymmetrizerFolding, asserting it folded one contraction.

    ``alone`` reads the pass's counter; ``default`` looks for its line in the default
    pipeline's explanation.
    """
    held = []
    graph = cg.Graph("audit")
    outputs = build(graph, held)
    if how == "alone":
        fold = _G.AntisymmetrizerFolding()
        graph.apply(_manager(_G.AntisymmetryInference(), fold))
        assert fold.num_folded == 1, fold.skip_reasons
    else:
        manager = cg.default_pass_manager()
        graph.apply(manager)
        assert "AntisymmetrizerFolding: collapsed 1 " in manager.explain(), manager.explain()
    graph.execute()
    for got, want in zip(outputs, expected):
        np.testing.assert_allclose(np.asarray(got), want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_folding_keeps_the_operator_permutes_alpha(how):
    """AntisymmetrizerFolding carries the producer's alpha along with the term count.

    Defends against ``read_producer`` recording the source, operators and term count of
    ``W = 2 P(i/j/k) Wsrc`` but not its alpha, so that the folded dot read Wsrc directly and the
    energy came out halved. The contraction is folded.
    """
    build, expected = _folding_program(alpha=2.0, rank0=False)
    if how == "O2":
        _check(build, expected, how, _folding_alone)
    else:
        _check_folds(build, expected, how)


@pytest.mark.parametrize("how", ["alone", "default"])
@pytest.mark.parametrize("conjugated", [False, True], ids=["dot", "dotc"])
@pytest.mark.parametrize("slot", [0, 1])
def test_folding_carries_a_complex_alpha_into_either_slot(how, conjugated, slot):
    """``<W, V>`` with ``W = alpha P(i/j/k) Ws`` in either operand slot, conjugated or not.

    The dot is bilinear, and sesquilinear in its first operand when conjugated, so the fold
    scales by ``N alpha``, or by ``N conj(alpha)`` when W is the conjugated operand. For the
    second slot the first operand's source is rescaled after its operator, which keeps the fold
    off that slot and puts it on W's.
    """
    rng = np.random.default_rng(117)
    n = 3
    ws = rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))
    vs = rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))
    alpha = 0.5 - 1.25j
    w, v = alpha * _antisymmetrize3(ws), _antisymmetrize3(vs)
    a, b = (w, v) if slot == 0 else (v, w)
    expected = np.sum((np.conj(a) if conjugated else a) * b)

    def build(graph, held):
        W = graph.create_zero_tensor("W", [n, n, n], dtype="complex128")
        V = graph.create_zero_tensor("V", [n, n, n], dtype="complex128")
        Ws, Vs = einsums.asarray(ws), einsums.asarray(vs)
        result = einsums.zeros([1], dtype="complex128")
        with cg.capture(graph):
            einsums.permute("ijk <- P(i/j/k) ijk", V, Vs, 0.0, 1.0)
            einsums.permute("ijk <- P(i/j/k) ijk", W, Ws, 0.0, alpha)
            la.scale(2.0, Vs)
            (la.dotc if conjugated else la.dot)(result, *((W, V) if slot == 0 else (V, W)))
        held += [Ws, Vs]
        return [result]

    _check_folds(build, [np.array([expected])], how)


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
def test_contraction_planning_restructures_a_chain_with_a_view_leaf(how):
    """ContractionPlanning re-parenthesizes a chain one of whose leaves is a view, correctly.

    ``T = A B`` then ``D = T C`` with A a strided sub-block of a larger tensor, sliced before the
    capture, and with the shapes of the dtype case above, for which ``A (B C)`` is far cheaper.
    The emitted GEMM reads A through its impl and strides. This case used to be declined by the
    pass's feature declaration; the C++ tests assert that the chain is restructured.

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


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
@pytest.mark.parametrize("slab", ["middle", "leading"])
def test_contraction_planning_reads_density_fitting_slabs(how, slab):
    """A chain over slabs of a three-index tensor, into a slice of a larger result, computes the product.

    ``T_ab = X_Qa Y_Qb`` then ``D = T C`` with X and Y slabs of one ``(naux, nocc, nvir)`` tensor,
    the DF-MP2 shape, and D a block of a larger tensor. A ``B[:, i, :]`` slab is a column-major
    matrix with a long leading dimension; a ``B[i, :, :]`` slab has no unit stride at all, so the
    emitted GEMM reads it through the strided loop. Re-parenthesized, ``X^T (Y C)`` reads each slab
    once through the GEMM that writes the slice.
    """
    rng = np.random.default_rng(112)
    naux, nocc, nvir, k = 48, 3, 40, 2
    if slab == "middle":
        three = rng.standard_normal((naux, nocc, nvir))
        x, y = three[:, 0, :], three[:, 2, :]
    else:
        three = rng.standard_normal((nocc, naux, nvir))
        x, y = three[0], three[2]
    c = rng.standard_normal((nvir, k))
    big_d = rng.standard_normal((nvir + 2, k + 3))
    expected = big_d.copy()
    expected[2:, 1:k + 1] = x.T @ y @ c

    def build(graph, held):
        Three, C, BigD = einsums.asarray(three), einsums.asarray(c), einsums.asarray(big_d)
        if slab == "middle":
            X, Y = Three[:, 0, :], Three[:, 2, :]
        else:
            X, Y = Three[0, :, :], Three[2, :, :]
        D = BigD[2:nvir + 2, 1:k + 1]
        T = graph.create_zero_tensor("T", [nvir, nvir])
        with cg.capture(graph):
            einsums.einsum("ab <- Qa ; Qb", T, X, Y)
            einsums.einsum("ab <- ac ; cb", D, T, C)
        held += [Three, C, BigD, X, Y, D]
        return [BigD]

    _check(build, [expected], how, lambda: _manager(_G.ContractionPlanning()))


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
@pytest.mark.parametrize("alias", ["same", "view"])
def test_contraction_planning_keeps_a_chain_that_writes_its_own_leaf(how, alias):
    """ContractionPlanning does not re-parenthesize ``A = (A B) L``.

    Left to right, the first member reads A before the last one overwrites it. Defends against
    the pass checking a chain's final output only against its interior: re-parenthesized as
    ``A (B L)``, the GEMM that writes A also reads it, and the result was off by 30 on this data
    (``same``). ``view`` writes the result into one view of a tensor and reads the leaf through
    an overlapping one, which only a comparison by buffer sees.
    """
    rng = np.random.default_rng(113)
    m, k = 64, 2
    a, b, l = rng.standard_normal((m, k)), rng.standard_normal((k, m)), rng.standard_normal((m, k))
    parent = np.zeros((m + 1, k))
    if alias == "same":
        expected = a @ b @ l
    else:
        parent[1:] = a
        expected = parent.copy()
        expected[:m] = a @ b @ l

    def build(graph, held):
        B, L = einsums.asarray(b), einsums.asarray(l)
        T = graph.create_zero_tensor("T", [m, m])
        if alias == "same":
            A = einsums.asarray(a)
            out, leaf, result = A, A, A
        else:
            result = einsums.asarray(parent)
            out, leaf = result[0:m, 0:k], result[1:m + 1, 0:k]
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", T, leaf, B)
            einsums.einsum("ij <- ik ; kj", out, T, L)
        held += [B, L, out, leaf, result]
        return [result]

    _check(build, [expected], how, lambda: _manager(_G.ContractionPlanning()))


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
    more, so R2 came out zero. The merge is now refused because a reader reaches K2 through
    another id.
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


def _cse_eliminated(count):
    return lambda the_pass: the_pass.num_eliminated == count


@pytest.mark.parametrize("how", ["alone", "default"])
@pytest.mark.parametrize("captured", [False, True], ids=["eager_view", "captured_view"])
def test_cse_merges_duplicates_that_read_a_view(how, captured):
    """CSE merges two products that read one view, and the survivor's value reaches both readers.

    ``K1 = V B`` and ``K2 = V B`` with V a strided slice of a larger tensor, sliced before the
    capture or by ``cg.view`` inside it. Reading through a view is something the aliasing guards
    see, since they compare the buffers operands land in. This case used to be declined by the
    pass's feature declaration.
    """
    rng = np.random.default_rng(111)
    n = _CSE_N
    big, b = rng.standard_normal((n + 1, n)), rng.standard_normal((n, n))

    def build(graph, held):
        Big, B = einsums.asarray(big), einsums.asarray(b)
        view = None if captured else Big[1:n + 1, 0:n]
        R1, R2 = einsums.zeros([n, n]), einsums.zeros([n, n])
        K1, K2 = graph.create_zero_tensor("K1", [n, n]), graph.create_zero_tensor("K2", [n, n])
        with cg.capture(graph):
            if captured:
                view = cg.view(Big, [(1, n + 1), (0, n)])
            einsums.einsum("ij <- ik ; kj", K1, view, B)
            einsums.einsum("ij <- ik ; kj", K2, view, B)
            la.axpby(1.0, K1, 0.0, R1)
            la.axpby(1.0, K2, 0.0, R2)
        held += [Big, B, view]
        return [R1, R2]

    _check_fires(build, [big[1:] @ b, big[1:] @ b], how, _G.CSE, _cse_eliminated(1), "CSE: eliminated 1")


@pytest.mark.parametrize("how", ["alone", "default", "O1", "O2"])
def test_cse_keeps_a_duplicate_that_writes_a_view(how):
    """CSE does not remove a duplicate whose output is a view of a tensor other nodes read.

    ``K1 = A B`` and ``V = A B`` with V a block of K2 sliced before the capture; the readers take
    K1 and the whole of K2. Redirecting V's slot onto K1 leaves K2's block unwritten. Defends
    the refusal of a node that writes a view and the check that every reader names the duplicate's
    own output; either one alone keeps this merge out.
    """
    a, b = _cse_data()
    n = _CSE_N
    k2 = np.random.default_rng(114).standard_normal((n + 2, n))
    expected = k2.copy()
    expected[1:n + 1] = a @ b

    def build(graph, held):
        A, B = einsums.asarray(a), einsums.asarray(b)
        R1, R2 = einsums.zeros([n, n]), einsums.zeros([n + 2, n])
        K1 = graph.create_zero_tensor("K1", [n, n])
        K2 = graph.create_zero_tensor("K2", [n + 2, n])
        np.asarray(K2)[...] = k2
        view = K2[1:n + 1, 0:n]
        with cg.capture(graph):
            einsums.einsum("ij <- ik ; kj", K1, A, B)
            einsums.einsum("ij <- ik ; kj", view, A, B)
            la.axpby(1.0, K1, 0.0, R1)
            la.axpby(1.0, K2, 0.0, R2)
        held += [A, B, view]
        return [R1, R2]

    _check(build, [a @ b, expected], how, lambda: _manager(_G.CSE()))


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


@pytest.mark.parametrize("how", ["alone", "default"])
@pytest.mark.parametrize("captured", [False, True], ids=["eager_view", "captured_view"])
def test_factoring_sums_views(how, captured):
    """``R += A V1`` and ``R += A V2`` factor into ``R += A (V1 + V2)`` when A, V1 and V2 are views.

    A is a block of one tensor, V1 and V2 two column blocks of another, the shape of a residual
    contracted against slices of an integral. The axpys that build the sum and the contraction
    read every operand through its own strides. This case used to be declined by the pass's
    feature declaration. The default pipeline prices the group with its cost model, so there
    the case asserts the pass reached the group, factored or priced out, and the numbers.
    """
    rng = np.random.default_rng(118)
    n = 5
    big_a, pair, r0 = rng.standard_normal((n + 1, n + 2)), rng.standard_normal((n, 2 * n)), rng.standard_normal((n, n))
    a, v1, v2 = big_a[1:, 2:], pair[:, :n], pair[:, n:]
    expected = r0 + a @ v1 + a @ v2

    def build(graph, held):
        BigA, Pair, R = einsums.asarray(big_a), einsums.asarray(pair), einsums.asarray(r0)
        views = (BigA[1:n + 1, 2:n + 2], Pair[0:n, 0:n], Pair[0:n, n:2 * n]) if not captured else None
        with cg.capture(graph):
            if captured:
                views = (cg.view(BigA, [(1, n + 1), (2, n + 2)]), cg.view(Pair, [(0, n), (0, n)]),
                         cg.view(Pair, [(0, n), (n, 2 * n)]))
            A, V1, V2 = views
            einsums.einsum("ij <- ik ; kj", R, A, V1, 1.0)
            einsums.einsum("ij <- ik ; kj", R, A, V2, 1.0)
        held += [BigA, Pair, *views]
        return [R]

    if how == "alone":
        _check_fires(build, [expected], how, _factoring_alone_pass, lambda p: p.num_groups == 1, None)
        return
    held = []
    graph = cg.Graph("audit")
    outputs = build(graph, held)
    manager = cg.default_pass_manager()
    graph.apply(manager)
    report = manager.explain()
    reached = ("DistributiveFactoring: factored 1 group" in report
               or "DistributiveFactoring: factored 0 group(s), eliminated 0 contraction(s); 1 declined as unprofitable" in report)
    assert reached, report
    graph.execute()
    np.testing.assert_allclose(np.asarray(outputs[0]), expected, rtol=1e-10, atol=1e-12)


def _factoring_alone_pass():
    return _G.DistributiveFactoring(_G.Factor.Always)


def _reused_sum_program(between, summed_view=False):
    """Two groups summing ``B1 + B2``, the second reusing the first's sum, with @p between in the middle.

    ``R1 += A B1 + A B2``, then @p between, then ``R2 += A B1 + A B2``. With @p summed_view both
    groups read B1 through one whole-extent view of it. Returns the builder and the expected R1
    and R2.
    """
    rng = np.random.default_rng(119)
    n = 3
    a, b1, b2, w, c1, c2, r1, r2 = (rng.standard_normal((n, n)) for _ in range(8))
    e1 = r1 + a @ b1 + a @ b2
    b1_later = b1.copy()
    if between == "view":
        b1_later[0:2, 0:2] += 0.5 * w[0:2, 0:2]
    else:
        b1_later += a @ c1 + a @ c2
    e2 = r2 + a @ b1_later + a @ b2

    def build(graph, held):
        A, B1, B2, W, C1, C2, R1, R2 = (einsums.asarray(x) for x in (a, b1, b2, w[0:2, 0:2].copy(), c1, c2, r1, r2))
        with cg.capture(graph):
            S1 = cg.view(B1, [(0, n), (0, n)]) if summed_view else B1
            einsums.einsum("ij <- ik ; kj", R1, A, S1, 1.0)
            einsums.einsum("ij <- ik ; kj", R1, A, B2, 1.0)
            if between == "view":
                la.axpy(0.5, W, cg.view(B1, [(0, 2), (0, 2)]))
            else:
                einsums.einsum("ij <- ik ; kj", B1, A, C1, 1.0)
                einsums.einsum("ij <- ik ; kj", B1, A, C2, 1.0)
            einsums.einsum("ij <- ik ; kj", R2, A, S1, 1.0)
            einsums.einsum("ij <- ik ; kj", R2, A, B2, 1.0)
        held += [A, B1, B2, W, C1, C2, S1]
        return [R1, R2]

    return build, [e1, e2]


@pytest.mark.parametrize("summed_view", [False, True], ids=["summed_tensor", "summed_view"])
def test_factoring_does_not_reuse_a_sum_across_a_write_through_a_view(summed_view):
    """A write through a view of B1 between two ``B1 + B2`` groups gives the second its own sum.

    Defends against the reuse check comparing the ids a node writes with the summed operands'
    ids: the view's id is not B1's, so the second group reused the sum built before the write,
    and R2 missed the update (max abs error 1.6 on this data). ``summed_view`` sums B1 through a
    view of it, so the check must compare buffers on the summed side as well.
    """
    build, expected = _reused_sum_program("view", summed_view)
    _check(build, expected, "alone", _factoring_alone)


def test_factoring_does_not_reuse_a_sum_across_another_groups_write():
    """A factored group writing B1 between two ``B1 + B2`` groups gives the second its own sum.

    ``B1 += A C1 + A C2`` in the middle is a group of its own, rewritten in the same run. Defends
    against the reuse check skipping every node an earlier group removed: the middle group's
    members were removed, but its factored contraction takes the first member's place and writes
    B1 there, so the second ``B1 + B2`` group reused a sum of the old B1 (max abs error 4.8 on
    this data).
    """
    build, expected = _reused_sum_program("group")
    _check(build, expected, "alone", _factoring_alone)


# ──────────────────────────────────────────────────────────────────────────
# LinearCombinationContractionFolding
# ──────────────────────────────────────────────────────────────────────────


def _lccf_fired(the_pass):
    return the_pass.num_groups == 1 and the_pass.num_eliminated == 1


@pytest.mark.parametrize("how", ["alone", "default"])
@pytest.mark.parametrize("where", ["operand", "shared", "transposed", "output", "all"])
@pytest.mark.parametrize("captured", [False, True], ids=["eager_view", "captured_view"])
def test_lccf_folds_a_transposed_pair_over_views(how, where, captured):
    """``C = 0.5 C + 2 A B - A B^T`` folds into one contraction against ``L = 2 B - B^T`` over views.

    @p where names the view: the folded operand B as a block of a larger tensor, the shared
    operand A as one, B as the transpose of a tensor, the output C as a block of a larger
    tensor the fold writes through, or all of them at once. L is built from B through its own
    strides and the fused contraction reads A and writes C through theirs. This case used to be
    declined by the pass's feature declaration.
    """
    rng = np.random.default_rng(120)
    n = 4
    big_b, big_a, big_c = rng.standard_normal((n + 2, n + 1)), rng.standard_normal((n + 1, n + 3)), rng.standard_normal((n + 3, n + 2))
    views = {"operand": ("b",), "shared": ("a",), "transposed": ("bt",), "output": ("c",), "all": ("a", "b", "c")}[where]
    a = big_a[1:, 3:] if "a" in views else big_a[:n, :n].copy()
    b = big_b[2:, 1:] if "b" in views else big_b[:n, :n].T.copy() if "bt" in views else big_b[:n, :n].copy()
    c0 = big_c[3:, 2:] if "c" in views else big_c[:n, :n].copy()
    expected_c = 0.5 * c0 + 2.0 * a @ b - a @ b.T

    def build(graph, held):
        BigA, BigB, BigC = (einsums.asarray(x) for x in (big_a, big_b, big_c))
        A0, B0, C0 = (einsums.asarray(x) for x in (a, b, c0))
        held += [BigA, BigB, BigC, A0, B0, C0]
        slices = {"a": (BigA, [(1, n + 1), (3, n + 3)]), "b": (BigB, [(2, n + 2), (1, n + 1)]), "c": (BigC, [(3, n + 3), (2, n + 2)])}

        def operand(key, plain):
            if key == "bt":
                square = einsums.asarray(big_b[:n, :n].copy())
                held.append(square)
                return cg.permute_view(square, [1, 0])
            if key not in views:
                return plain
            parent, ranges = slices[key]
            if captured:
                return cg.view(parent, ranges)
            (r0, r1), (q0, q1) = ranges
            return parent[r0:r1, q0:q1]

        with cg.capture(graph):
            A = operand("a", A0)
            B = operand("bt" if "bt" in views else "b", B0)
            C = operand("c", C0)
            einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.5, ab_pf=2.0)
            einsums.einsum("ij <- ik ; jk", C, A, B, c_pf=1.0, ab_pf=-1.0)
        held += [A, B, C]
        return [BigC if "c" in views else C0]

    expected = big_c.copy() if "c" in views else expected_c
    if "c" in views:
        expected[3:, 2:] = expected_c
    _check_fires(build, [expected], how, _G.LinearCombinationContractionFolding, _lccf_fired,
                 "LinearCombinationContractionFolding: folded 1 group(s)")


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
    and the composed scalars (alpha = a2 + b2 a1, beta = b1 b2) gave 3Y. The pass used to be
    kept off this pair by its feature declaration; it is now the buffer comparison.
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


@pytest.mark.parametrize("how", ["alone", "default", "O1", "O2"])
def test_element_wise_fusion_declines_two_views_of_one_block(how):
    """ElementWiseFusion declines ``Y = X/2 + Y`` twice when X and Y are two views of one block.

    Both are rows 1..4 of P, taken by two ``cg.view`` calls, so neither is the other's parent
    and neither id names the storage both land in. Defends the comparison of the two operands'
    buffers: composing the pair read the X the first update had already changed, and scaled the
    block by 2 where the program scales it by 2.25.
    """
    rng = np.random.default_rng(115)
    n = 4
    p0 = rng.standard_normal((n + 1, n))
    expected = p0.copy()
    expected[1:] *= 1.5 * 1.5

    def build(graph, held):
        P = einsums.asarray(p0)
        with cg.capture(graph):
            X = cg.view(P, [(1, n + 1), (0, n)])
            Y = cg.view(P, [(1, n + 1), (0, n)])
            la.axpby(0.5, X, 1.0, Y)
            la.axpby(0.5, X, 1.0, Y)
        held += [P, X, Y]
        return [P]

    _check(build, [expected], how, lambda: _manager(_G.ElementWiseFusion()))


@pytest.mark.parametrize("how", ["alone", "default"])
@pytest.mark.parametrize("eager_view", [False, True], ids=["captured_view", "eager_view"])
def test_element_wise_fusion_composes_axpbys_on_views(how, eager_view):
    """ElementWiseFusion composes two axpbys whose X and Y are views of different tensors.

    ``Y = 2X + Y`` then ``Y = -X + 3Y`` with X a block of Q and Y a block of P, the shape of an
    update on a slice of a residual. The two operands land in different buffers, so the second
    update reads the same X. This case used to be declined by the pass's feature declaration.
    """
    rng = np.random.default_rng(116)
    n = 4
    p0, q0 = rng.standard_normal((n + 2, n)), rng.standard_normal((n, n + 1))
    x = q0[:, 1:]
    expected = p0.copy()
    expected[2:] = -x + 3.0 * (2.0 * x + p0[2:])

    def build(graph, held):
        P, Q = einsums.asarray(p0), einsums.asarray(q0)
        X, Y = (Q[0:n, 1:n + 1], P[2:n + 2, 0:n]) if eager_view else (None, None)
        with cg.capture(graph):
            if not eager_view:
                X = cg.view(Q, [(0, n), (1, n + 1)])
                Y = cg.view(P, [(2, n + 2), (0, n)])
            la.axpby(2.0, X, 1.0, Y)
            la.axpby(-1.0, X, 3.0, Y)
        held += [P, Q, X, Y]
        return [P]

    _check_fires(build, [expected], how, _G.ElementWiseFusion, lambda p: p.num_fused == 1, "ElementWiseFusion: fused 1")
