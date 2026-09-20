# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""The Python examples from the user-guide tutorials, run as written.

The tutorials show each operation in a C++ tab and a Python tab. ``TutorialSnippets.cpp`` compiles
and runs the first; this runs the second, so neither half of a page can drift from the library
while the other stays right.

Several of these exist because the two languages genuinely differ, and a tab written by
transliterating the other one would have been wrong: assignment copies in C++ and binds in Python,
``size`` and ``name`` are calls in C++ and properties here, and ``print`` gives a summary rather
than the elements.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la


# ── user/tutorial_tensors.rst ───────────────────────────────────────────────


def test_creating_tensors_of_each_dtype():
    A = einsums.zeros([10, 10], name="A")
    B = einsums.zeros([5, 6, 7], dtype="float32", name="B")
    v = einsums.zeros([100], dtype="complex128", name="v")

    assert A.rank() == 2
    assert B.rank() == 3
    assert v.rank() == 1
    assert np.asarray(B).dtype == np.float32
    assert np.asarray(v).dtype == np.complex128


def test_convenience_creators():
    Z = einsums.create_zero_tensor("Z", [4, 4])
    R = einsums.create_random_tensor("R", [4, 4])
    I = einsums.create_identity_tensor("I", [4, 4])

    assert np.allclose(np.asarray(Z), 0.0)
    assert np.asarray(R).shape == (4, 4)
    assert np.allclose(np.asarray(I), np.eye(4))

    # The numpy-shaped spellings the page also lists.
    assert np.allclose(np.asarray(einsums.zeros([4, 4])), np.zeros((4, 4)))
    assert np.allclose(np.asarray(einsums.ones([4, 4])), np.ones((4, 4)))
    assert np.allclose(np.asarray(einsums.eye(4)), np.eye(4))


def test_identity_need_not_be_square():
    R = einsums.create_identity_tensor("R", [6, 3])
    expect = np.zeros((6, 3))
    expect[0, 0] = expect[1, 1] = expect[2, 2] = 1.0
    assert np.allclose(np.asarray(R), expect)


def test_element_access_and_negative_indices():
    A = einsums.zeros([3, 3], name="A")
    A[0, 0] = 1.0
    A[1, 2] = 3.14
    assert A[1, 2] == pytest.approx(3.14)

    v = einsums.zeros([5], name="v")
    v[0] = 10.0
    v[4] = 20.0
    assert v[4] == 20.0

    assert A[-1, -1] == A[2, 2]


def test_shape_accessors_are_methods_and_properties():
    """The asymmetry with C++, which the page calls out and which is easy to get wrong."""
    A = einsums.zeros([3, 4, 5], name="A")

    assert A.rank() == 3          # method
    assert A.dim(0) == 3          # method
    assert A.size == 60           # property, not a call
    assert A.name == "A"          # property, not a call
    assert np.asarray(A).shape == (3, 4, 5)

    assert callable(A.rank)
    assert callable(A.dim)
    assert not callable(A.size)
    assert not callable(A.name)

    # Neither of these exists on the Python side; the page says to use numpy instead.
    assert not hasattr(A, "dims")
    assert not hasattr(A, "strides")


def test_fill_and_zero():
    A = einsums.zeros([4, 4])
    A.set_all(3.14)
    assert np.allclose(np.asarray(A), 3.14)
    A.zero()
    assert np.allclose(np.asarray(A), 0.0)


def test_print_gives_a_summary_not_the_elements():
    A = einsums.create_random_tensor("A", [3, 3])
    text = repr(A)
    assert "RuntimeTensor" in text
    assert "shape=(3, 3)" in text
    # The values come through numpy.
    assert np.asarray(A).shape == (3, 3)


def test_assignment_binds_while_array_copies():
    """C++ copies on assignment; Python does not. Transliterating that tab would be wrong."""
    A = einsums.create_random_tensor("A", [4, 4])

    B = A
    B[0, 0] = 999.0
    assert A[0, 0] == 999.0, "assignment should bind, not copy"

    C = einsums.array(A)
    C[1, 1] = 111.0
    assert A[1, 1] != 111.0, "einsums.array should copy"

    import copy

    with pytest.raises(TypeError):
        copy.deepcopy(A)


def test_tensors_captured_into_a_graph():
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])
    reference = np.asarray(A) @ np.asarray(B)

    graph = cg.Graph("example")
    C = graph.create_tensor("C", [4, 4], intermediate=False)

    with cg.capture(graph):
        einsums.einsum("ik;kj->ij", C, A, B)

    graph.optimize()
    graph.execute()
    assert np.allclose(np.asarray(C), reference)

    # Replay is the point of a graph.
    graph.execute()
    assert np.allclose(np.asarray(C), reference)


# ── user/tutorial_views.rst ─────────────────────────────────────────────────


def test_slicing_gives_a_view_that_writes_through():
    A = einsums.create_random_tensor("A", [10, 10])

    block = A[0:3, 0:3]
    block[0, 0] = 999.0
    assert A[0, 0] == 999.0

    row = A[5, :]
    col = A[:, 3]
    assert row.dim(0) == 10
    assert col.dim(0) == 10


def test_orbital_blocks():
    n_occ, n_virt = 5, 15
    n_orbs = n_occ + n_virt

    F = einsums.create_random_tensor("Fock", [n_orbs, n_orbs])
    Foo = F[:n_occ, :n_occ]
    Fov = F[:n_occ, n_occ:]
    Fvv = F[n_occ:, n_occ:]

    assert (Foo.dim(0), Foo.dim(1)) == (n_occ, n_occ)
    assert (Fov.dim(0), Fov.dim(1)) == (n_occ, n_virt)
    assert (Fvv.dim(0), Fvv.dim(1)) == (n_virt, n_virt)
    assert Fov[0, 0] == F[0, n_occ]


def test_views_as_contraction_operands():
    C = einsums.create_random_tensor("C", [20, 20])
    Co = C[:, 0:5]
    Cv = C[:, 5:20]

    AO_ints = einsums.create_random_tensor("ints", [20, 20])
    MO_ints = einsums.zeros([5, 15], name="MO_ints")

    tmp = einsums.zeros([5, 20], name="tmp")
    einsums.einsum("ki;kj->ij", tmp, Co, AO_ints)
    einsums.einsum("ik;kj->ij", MO_ints, tmp, Cv)

    expect = np.asarray(Co).T @ np.asarray(AO_ints) @ np.asarray(Cv)
    assert np.allclose(np.asarray(MO_ints), expect)


def test_views_keep_parent_strides_and_column_major_order():
    A = einsums.create_random_tensor("A", [10, 10])

    # Column-major: the FIRST index is the contiguous one.
    assert A.stride(0) == 1
    assert A.stride(1) == 10

    block = A[2:5, 3:7]
    assert (block.dim(0), block.dim(1)) == (3, 4)
    assert block.stride(0) == 1
    assert block.stride(1) == 10


def test_a_python_view_keeps_its_parent_alive():
    """Where C++ would dangle, Python holds a reference. The page says so per tab."""
    import gc

    def make_view():
        local = einsums.create_random_tensor("local", [5, 5])
        return local[0:3, 0:3]

    v = make_view()
    gc.collect()
    assert isinstance(float(v[0, 0]), float)


def test_views_inside_a_capture():
    n_occ, n_virt = 5, 15
    n_orbs = n_occ + n_virt

    F = einsums.create_random_tensor("F", [n_orbs, n_orbs])
    T = einsums.create_random_tensor("T", [n_occ, n_virt])
    Fov = F[:n_occ, n_occ:]

    graph = cg.Graph("blocks")
    out = graph.create_tensor("out", [n_occ, n_occ], intermediate=False)

    with cg.capture(graph):
        einsums.einsum("ia;ja->ij", out, Fov, T)

    graph.optimize()
    graph.execute()

    expect = np.asarray(Fov) @ np.asarray(T).T
    assert np.allclose(np.asarray(out), expect)


# ── user/tutorial_einsum.rst ────────────────────────────────────────────────


def test_matrix_multiplication_spec():
    A = einsums.create_random_tensor("A", [7, 7])
    B = einsums.create_random_tensor("B", [7, 7])
    C = einsums.zeros([7, 7], name="C")

    einsums.einsum("ik;kj->ij", C, A, B)
    assert np.allclose(np.asarray(C), np.asarray(A) @ np.asarray(B))


def test_dot_outer_and_permute():
    u = einsums.create_random_tensor("u", [100])
    v = einsums.create_random_tensor("v", [100])

    result = la.dot(u, v)
    assert result == pytest.approx(float(np.dot(np.asarray(u), np.asarray(v))))

    C = einsums.zeros([100, 100], name="C")
    einsums.einsum("i;j->ij", C, u, v)
    assert np.allclose(np.asarray(C), np.outer(np.asarray(u), np.asarray(v)))

    A = einsums.create_random_tensor("A", [5, 8])
    At = einsums.zeros([8, 5], name="At")
    einsums.permute("ij->ji", At, A)
    assert np.allclose(np.asarray(At), np.asarray(A).T)


def test_prefactors():
    A = einsums.create_random_tensor("A", [7, 7])
    B = einsums.create_random_tensor("B", [7, 7])
    C = einsums.zeros([7, 7], name="C")
    product = np.asarray(A) @ np.asarray(B)

    einsums.einsum("ik;kj->ij", C, A, B)
    assert np.allclose(np.asarray(C), product)

    einsums.einsum("ik;kj->ij", C, A, B, c_pf=1.0, ab_pf=1.0)
    assert np.allclose(np.asarray(C), 2.0 * product)

    einsums.einsum("ik;kj->ij", C, A, B, c_pf=0.0, ab_pf=0.5)
    assert np.allclose(np.asarray(C), 0.5 * product)


def test_higher_rank_contraction():
    T = einsums.create_random_tensor("T", [4, 4, 4, 4])
    U = einsums.create_random_tensor("U", [4, 4, 4, 4])
    W = einsums.zeros([4, 4, 4, 4], name="W")

    einsums.einsum("ijkl;klmn->ijmn", W, T, U)
    expect = np.einsum("ijkl,klmn->ijmn", np.asarray(T), np.asarray(U))
    assert np.allclose(np.asarray(W), expect)


def test_two_step_capture_matches_eager():
    A = einsums.create_random_tensor("A", [7, 7])
    B = einsums.create_random_tensor("B", [7, 7])

    t = einsums.zeros([7, 7])
    o = einsums.zeros([7, 7])
    einsums.einsum("ik;kj->ij", t, A, B)
    einsums.einsum("ik;kj->ij", o, t, A)

    graph = cg.Graph("two steps")
    tmp = graph.create_tensor("tmp", [7, 7])
    out = graph.create_tensor("out", [7, 7], intermediate=False)

    with cg.capture(graph):
        einsums.einsum("ik;kj->ij", tmp, A, B)
        einsums.einsum("ik;kj->ij", out, tmp, A)

    graph.optimize()
    graph.execute()
    assert np.allclose(np.asarray(out), np.asarray(o))


# ── user/tutorial_linalg.rst ────────────────────────────────────────────────


def test_scale_and_axpy():
    A = einsums.create_random_tensor("A", [4, 4])
    before = np.array(np.asarray(A), copy=True)
    la.scale(2.0, A)
    assert np.allclose(np.asarray(A), 2.0 * before)

    X = einsums.create_random_tensor("X", [100])
    Y = einsums.zeros([100], name="Y")
    la.axpy(1.0, X, Y)
    la.axpy(-0.5, X, Y)
    assert np.allclose(np.asarray(Y), 0.5 * np.asarray(X))


def test_gemm_and_its_transpose_keywords():
    A = einsums.create_random_tensor("A", [10, 5])
    B = einsums.create_random_tensor("B", [5, 8])
    C = einsums.zeros([10, 8], name="C")

    la.gemm(1.0, A, B, 0.0, C)
    assert np.allclose(np.asarray(C), np.asarray(A) @ np.asarray(B))

    # Python uses keywords where C++ uses template parameters.
    At = einsums.create_random_tensor("At", [5, 10])
    la.gemm(1.0, At, B, 0.0, C, trans_a=True)
    assert np.allclose(np.asarray(C), np.asarray(At).T @ np.asarray(B))

    Bt = einsums.create_random_tensor("Bt", [8, 5])
    la.gemm(1.0, A, Bt, 0.0, C, trans_b=True)
    assert np.allclose(np.asarray(C), np.asarray(A) @ np.asarray(Bt).T)


def test_syev_and_invert_work_in_place():
    A = einsums.create_random_definite("A", 5)
    original = np.array(np.asarray(A), copy=True)
    w = einsums.zeros([5], name="w")

    la.syev(A, w)
    # A now holds eigenvectors, w the eigenvalues.
    assert np.all(np.diff(np.asarray(w)) >= -1e-10), "eigenvalues should be ascending"
    assert not np.allclose(np.asarray(A), original)

    B = einsums.create_random_definite("B", 4)
    before = np.array(np.asarray(B), copy=True)
    la.invert(B)
    assert np.allclose(before @ np.asarray(B), np.eye(4), atol=1e-8)


def test_gesv_solves_in_place():
    A = einsums.zeros([3, 3], name="A")
    B = einsums.zeros([3, 2], name="B")

    np.asarray(A)[...] = [[4.0, 1.0, 0.0], [1.0, 5.0, 1.0], [0.0, 1.0, 3.0]]
    np.asarray(B)[...] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    expect = np.linalg.solve(np.asarray(A).copy(), np.asarray(B).copy())
    la.gesv(A, B)
    assert np.allclose(np.asarray(B), expect)


def test_norm_dot_and_ger():
    A = einsums.create_random_tensor("A", [5, 5])
    n = la.norm(la.Norm.FROBENIUS, A)
    assert n == pytest.approx(float(np.linalg.norm(np.asarray(A), "fro")))

    x = einsums.create_random_tensor("x", [100])
    y = einsums.create_random_tensor("y", [100])
    assert la.dot(x, y) == pytest.approx(float(np.dot(np.asarray(x), np.asarray(y))))

    u = einsums.create_random_tensor("u", [4])
    v = einsums.create_random_tensor("v", [5])
    M = einsums.zeros([4, 5], name="M")
    la.ger(1.0, u, v, M)
    assert np.allclose(np.asarray(M), np.outer(np.asarray(u), np.asarray(v)))


def test_the_three_rank_constrained_routines_are_unconstrained_in_python():
    """det, svd and qr need a compile-time rank of two in C++ and not here."""
    A = einsums.create_random_tensor("A", [6, 4])
    U, sigma, Vt = la.svd(A)
    assert np.asarray(sigma).shape == (4,)

    Q, R = la.qr(A)

    S = einsums.create_random_tensor("S", [3, 3])
    d = la.det(S)
    assert np.isfinite(d)


def test_captured_gemm_and_scale():
    A = einsums.create_random_tensor("A", [64, 64])
    B = einsums.create_random_tensor("B", [64, 64])
    expect = 0.5 * (np.asarray(A) @ np.asarray(B))

    graph = cg.Graph("update")
    C = graph.create_tensor("C", [64, 64], intermediate=False)

    with cg.capture(graph):
        la.gemm(1.0, A, B, 0.0, C)
        la.scale(0.5, C)

    graph.optimize()
    graph.execute()
    assert np.allclose(np.asarray(C), expect)


# ── user/absolute_beginners.rst ─────────────────────────────────────────────


def test_creating_tensors_of_each_type():
    A = einsums.zeros([2, 2], name="A")
    B = einsums.zeros([2, 2], dtype="float32", name="B")
    C = einsums.zeros([2, 2], dtype="complex64", name="C")
    D = einsums.zeros([2, 2], dtype="complex128", name="D")

    assert np.asarray(A).dtype == np.float64
    assert np.asarray(B).dtype == np.float32
    assert np.asarray(C).dtype == np.complex64
    assert np.asarray(D).dtype == np.complex128


def test_filling_and_scalar_arithmetic():
    A = einsums.zeros([10, 10], name="A")
    B = einsums.create_random_tensor("B", [10, 10])

    A = einsums.array(B)
    assert np.allclose(np.asarray(A), np.asarray(B))

    A.zero()
    A.set_all(0.5)
    A += 2
    A -= 2
    A *= 2
    A /= 2
    assert np.allclose(np.asarray(A), 0.5)

    einsums.einsum("ij;ij->ij", A, A, B)
    assert np.allclose(np.asarray(A), 0.5 * np.asarray(B))


def test_beginner_indexing_and_slicing():
    A = einsums.create_random_tensor("A", [3, 3])

    assert A[-1, -1] == A[2, 2]
    A[2, 2] = 10.0
    assert A[2, 2] == 10.0

    View1 = A[0:2, :]
    View2 = A[2, :]
    View3 = A[:, 2]
    View4 = A[1:3, 0:2]
    assert View1.dim(0) == 2
    assert View2.dim(0) == 3
    assert View3.dim(0) == 3
    assert (View4.dim(0), View4.dim(1)) == (2, 2)


def test_beginner_shape_queries():
    A = einsums.zeros([3, 4, 5], name="A")
    assert A.size == 3 * 4 * 5
    assert A.dim(0) == 3
    assert A.rank() == 3


def test_beginner_permute_with_prefactors():
    A = einsums.create_random_tensor("A", [3, 4, 5])
    B = einsums.create_random_tensor("B", [5, 4, 3])
    C = einsums.array(B)

    einsums.permute("ijk <- kji", C, A, c_pf=1.0, a_pf=0.5)

    expect = np.asarray(B) + 0.5 * np.asarray(A).transpose(2, 1, 0)
    assert np.allclose(np.asarray(C), expect)


def test_beginner_linalg_and_contraction():
    A = einsums.create_random_tensor("A", [10, 10])
    B = einsums.create_random_tensor("B", [10, 10])
    C = einsums.create_random_tensor("C", [10, 10])
    u = einsums.create_random_tensor("u", [10])
    v = einsums.create_random_tensor("v", [10])

    la.gemm(1.0, A, B, 0.0, C)
    assert np.allclose(np.asarray(C), np.asarray(A) @ np.asarray(B))

    val = la.dot(u, v)
    assert val == pytest.approx(float(np.dot(np.asarray(u), np.asarray(v))))

    # true_dot has no binding; dotc is the conjugating form the page points at.
    assert not hasattr(la, "true_dot")
    assert hasattr(la, "dotc")

    W = einsums.zeros([10, 10, 10], name="W")
    einsums.einsum("ik;kj->ijk", W, A, B)
    expect = np.einsum("ik,kj->ijk", np.asarray(A), np.asarray(B))
    assert np.allclose(np.asarray(W), expect)


def test_beginner_closing_programs_agree():
    A = einsums.create_random_tensor("A", [64, 64])
    B = einsums.create_random_tensor("B", [64, 64])

    C = einsums.zeros([64, 64], name="C")
    einsums.einsum("ik;kj->ij", C, A, B)
    la.scale(0.5, C)
    eager = float(C[0, 0])

    graph = cg.Graph("scaled product")
    Cg = graph.create_tensor("C", [64, 64], intermediate=False)

    with cg.capture(graph):
        einsums.einsum("ik;kj->ij", Cg, A, B)
        la.scale(0.5, Cg)

    graph.optimize()
    for _ in range(3):
        graph.execute()

    assert float(Cg[0, 0]) == pytest.approx(eager)
