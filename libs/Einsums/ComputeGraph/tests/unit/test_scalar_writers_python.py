# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for the pointer-writing graph-aware forms of dot, norm, trace.

The returning forms (``einsums.linalg.dot(A, B) -> scalar``) throw inside
``with cg.capture():``. The pointer-writing forms accept a 1-element tensor
as the destination, so the result becomes a graph-tracked slot and the call
can be recorded. Together they unblock graph-only SCF patterns:

    e        = ½ Σ D · (H+F)      → dot(e, D, sum_HF) then scale(0.5, e)
    rms(ΔD)  = ||D − D_old||_F     → axpby + axpy + norm(rms, FROBENIUS, ΔD)
    tr(A)    = Σ A_ii              → trace(t, A)

The result tensor is rank-1 with one element so it composes with downstream
ops; subsequent scale/axpy operate on it as a normal tensor.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES, COMPLEX_DTYPES, assert_close


# Map each input dtype to the dtype of a Frobenius-norm scalar result.
_NORM_RESULT_DTYPE = {
    "float32": "float32",
    "float64": "float64",
    "complex64": "float32",
    "complex128": "float64",
}


# ──────────────────────────────────────────────────────────────────────────
# dot(result, A, B): writes into result[0]
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_dot_writer_eager_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [6], dtype=dtype)
    B = einsums.create_random_tensor("B", [6], dtype=dtype)
    r = einsums.create_zero_tensor("r", [1], dtype=dtype)

    einsums.linalg.dot(r, A, B)

    expected = np.dot(np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_dot_writer_captured_records_node(dtype):
    A = einsums.create_random_tensor("A", [4], dtype=dtype)
    B = einsums.create_random_tensor("B", [4], dtype=dtype)
    r = einsums.create_zero_tensor("r", [1], dtype=dtype)

    g = cg.Graph("dot-writer")
    with cg.capture(g):
        einsums.linalg.dot(r, A, B)

    # 3 tensor slots (A, B, r) + 1 node.
    assert g.num_nodes() == 1
    assert g.num_tensors() == 3


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_dot_writer_captured_matches_eager(dtype):
    A = einsums.create_random_tensor("A", [5], dtype=dtype)
    B = einsums.create_random_tensor("B", [5], dtype=dtype)

    eager = einsums.create_zero_tensor("eager", [1], dtype=dtype)
    capt = einsums.create_zero_tensor("capt", [1], dtype=dtype)

    einsums.linalg.dot(eager, A, B)

    g = cg.Graph("dot-vs-eager")
    with cg.capture(g):
        einsums.linalg.dot(capt, A, B)
    g.execute()

    np.testing.assert_allclose(np.asarray(capt), np.asarray(eager), rtol=1e-5)


def test_dot_writer_zero_size_result_raises():
    A = einsums.create_random_tensor("A", [3])
    B = einsums.create_random_tensor("B", [3])
    r = einsums.create_zero_tensor("r", [0])  # empty
    with pytest.raises(Exception):
        einsums.linalg.dot(r, A, B)


# ──────────────────────────────────────────────────────────────────────────
# dot with views: all combinations of (Result, A, B) x (owning, view)
# ──────────────────────────────────────────────────────────────────────────


def test_dot_both_inputs_are_views():
    """dot(e, view(A), view(B)): graph aliases both inputs to their parents."""
    A = einsums.create_random_tensor("A", [5, 6])
    B = einsums.create_random_tensor("B", [5, 6])
    e = einsums.create_zero_tensor("e", [1])

    g = cg.Graph("dot-vv")
    with cg.capture(g):
        Av = cg.view(A, [(-1, -1), (0, 3)])
        Bv = cg.view(B, [(-1, -1), (0, 3)])
        einsums.linalg.dot(e, Av, Bv)
    g.execute()

    expected = np.sum(np.asarray(A)[:, :3] * np.asarray(B)[:, :3])
    np.testing.assert_allclose(np.asarray(e)[0], expected, rtol=1e-5)


def test_dot_view_input_owning_input_mixed():
    """dot(e, view(C), D): mixed view/owning input pair."""
    C = einsums.create_random_tensor("C", [4, 6])
    D = einsums.create_random_tensor("D", [4, 3])  # already same shape as C[:, :3]
    e = einsums.create_zero_tensor("e", [1])

    g = cg.Graph("dot-vo")
    with cg.capture(g):
        Cv = cg.view(C, [(-1, -1), (0, 3)])
        einsums.linalg.dot(e, Cv, D)
    g.execute()

    expected = np.sum(np.asarray(C)[:, :3] * np.asarray(D))
    np.testing.assert_allclose(np.asarray(e)[0], expected, rtol=1e-5)


def test_dot_result_is_view():
    """dot(view_into_scalar_holder, A, B): even the result tensor can be a view."""
    # Holder has two slots; we write into slot 1 via a view.
    holder = einsums.create_zero_tensor("holder", [2])
    A = einsums.create_random_tensor("A", [4])
    B = einsums.create_random_tensor("B", [4])

    g = cg.Graph("dot-rv")
    with cg.capture(g):
        e_view = cg.view(holder, [(1, 2)])  # holder[1:2]: a 1-element view
        einsums.linalg.dot(e_view, A, B)
    g.execute()

    expected = np.dot(np.asarray(A), np.asarray(B))
    holder_np = np.asarray(holder)
    np.testing.assert_allclose(holder_np[1], expected, rtol=1e-5)
    assert holder_np[0] == 0.0  # other slot untouched


def test_dot_all_three_are_views():
    """The full 8th-cell case: result, A, and B all views into larger parents."""
    A_big = einsums.create_random_tensor("A_big", [3, 5])
    B_big = einsums.create_random_tensor("B_big", [3, 5])
    holder = einsums.create_zero_tensor("holder", [4])

    g = cg.Graph("dot-vvv")
    with cg.capture(g):
        Av = cg.view(A_big, [(-1, -1), (0, 3)])
        Bv = cg.view(B_big, [(-1, -1), (0, 3)])
        e_view = cg.view(holder, [(2, 3)])
        einsums.linalg.dot(e_view, Av, Bv)
    g.execute()

    expected = np.sum(np.asarray(A_big)[:, :3] * np.asarray(B_big)[:, :3])
    np.testing.assert_allclose(np.asarray(holder)[2], expected, rtol=1e-5)


def test_dot_owning_result_owning_A_view_B():
    """Cell RRV: owning result, owning A, view B."""
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [4, 6])
    e = einsums.create_zero_tensor("e", [1])

    g = cg.Graph("dot-rrv")
    with cg.capture(g):
        Bv = cg.view(B, [(-1, -1), (0, 3)])
        einsums.linalg.dot(e, A, Bv)
    g.execute()

    expected = np.sum(np.asarray(A) * np.asarray(B)[:, :3])
    np.testing.assert_allclose(np.asarray(e)[0], expected, rtol=1e-5)


def test_dot_view_result_owning_A_view_B():
    """Cell VRV: view result, owning A, view B."""
    holder = einsums.create_zero_tensor("holder", [4])
    A = einsums.create_random_tensor("A", [4, 3])
    B = einsums.create_random_tensor("B", [4, 6])

    g = cg.Graph("dot-vrv")
    with cg.capture(g):
        Bv = cg.view(B, [(-1, -1), (0, 3)])
        e_view = cg.view(holder, [(3, 4)])
        einsums.linalg.dot(e_view, A, Bv)
    g.execute()

    expected = np.sum(np.asarray(A) * np.asarray(B)[:, :3])
    np.testing.assert_allclose(np.asarray(holder)[3], expected, rtol=1e-5)


def test_dot_view_result_view_A_owning_B():
    """Cell VVR: view result, view A, owning B."""
    holder = einsums.create_zero_tensor("holder", [4])
    A = einsums.create_random_tensor("A", [4, 6])
    B = einsums.create_random_tensor("B", [4, 3])

    g = cg.Graph("dot-vvr")
    with cg.capture(g):
        Av = cg.view(A, [(-1, -1), (0, 3)])
        e_view = cg.view(holder, [(0, 1)])
        einsums.linalg.dot(e_view, Av, B)
    g.execute()

    expected = np.sum(np.asarray(A)[:, :3] * np.asarray(B))
    np.testing.assert_allclose(np.asarray(holder)[0], expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# norm(result, norm_type, A): writes the (real) norm into result[0]
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_norm_writer_frobenius_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    r_dtype = _NORM_RESULT_DTYPE[dtype]
    r = einsums.create_zero_tensor("r", [1], dtype=r_dtype)

    einsums.linalg.norm(r, einsums.linalg.Norm.FROBENIUS, A)

    expected = np.linalg.norm(np.asarray(A), ord="fro")
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_norm_writer_captured_matches_eager(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)

    eager = einsums.create_zero_tensor("eager", [1], dtype=dtype)
    capt = einsums.create_zero_tensor("capt", [1], dtype=dtype)

    einsums.linalg.norm(eager, einsums.linalg.Norm.FROBENIUS, A)

    g = cg.Graph("norm-vs-eager")
    with cg.capture(g):
        einsums.linalg.norm(capt, einsums.linalg.Norm.FROBENIUS, A)
    g.execute()

    np.testing.assert_allclose(np.asarray(capt), np.asarray(eager), rtol=1e-5)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_norm_writer_complex_returns_real_dtype(dtype):
    """For complex inputs, the norm is real-valued: result tensor must be real-dtype."""
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    r_dtype = _NORM_RESULT_DTYPE[dtype]
    r = einsums.create_zero_tensor("r", [1], dtype=r_dtype)

    einsums.linalg.norm(r, einsums.linalg.Norm.FROBENIUS, A)

    expected = np.linalg.norm(np.asarray(A), ord="fro")
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# norm with views: all 4 combinations of (Result, A) x (owning, view)
# ──────────────────────────────────────────────────────────────────────────


def test_norm_owning_result_view_A():
    """Cell RV: owning result, view A."""
    A = einsums.create_random_tensor("A", [4, 6])
    r = einsums.create_zero_tensor("r", [1])

    g = cg.Graph("norm-rv")
    with cg.capture(g):
        Av = cg.view(A, [(-1, -1), (0, 3)])
        einsums.linalg.norm(r, einsums.linalg.Norm.FROBENIUS, Av)
    g.execute()

    expected = np.linalg.norm(np.asarray(A)[:, :3], ord="fro")
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


def test_norm_view_result_owning_A():
    """Cell VR: view result, owning A."""
    holder = einsums.create_zero_tensor("holder", [3])
    A = einsums.create_random_tensor("A", [4, 5])

    g = cg.Graph("norm-vr")
    with cg.capture(g):
        r_view = cg.view(holder, [(1, 2)])
        einsums.linalg.norm(r_view, einsums.linalg.Norm.FROBENIUS, A)
    g.execute()

    expected = np.linalg.norm(np.asarray(A), ord="fro")
    np.testing.assert_allclose(np.asarray(holder)[1], expected, rtol=1e-5)


def test_norm_view_result_view_A():
    """Cell VV: view result, view A."""
    holder = einsums.create_zero_tensor("holder", [3])
    A = einsums.create_random_tensor("A", [4, 6])

    g = cg.Graph("norm-vv")
    with cg.capture(g):
        Av = cg.view(A, [(-1, -1), (0, 3)])
        r_view = cg.view(holder, [(2, 3)])
        einsums.linalg.norm(r_view, einsums.linalg.Norm.FROBENIUS, Av)
    g.execute()

    expected = np.linalg.norm(np.asarray(A)[:, :3], ord="fro")
    np.testing.assert_allclose(np.asarray(holder)[2], expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_norm_view_complex_input_real_result(dtype):
    """View of complex input -> view of real-dtype result, all cells exercised already
    in the dtype-mapping; this just confirms it works for view A with the
    proper real-dtype view result."""
    r_dtype = _NORM_RESULT_DTYPE[dtype]
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    holder = einsums.create_zero_tensor("holder", [2], dtype=r_dtype)

    g = cg.Graph("norm-vv-cplx")
    with cg.capture(g):
        Av = cg.view(A, [(-1, -1), (0, 2)])
        r_view = cg.view(holder, [(0, 1)])
        einsums.linalg.norm(r_view, einsums.linalg.Norm.FROBENIUS, Av)
    g.execute()

    expected = np.linalg.norm(np.asarray(A)[:, :2], ord="fro")
    np.testing.assert_allclose(np.asarray(holder)[0], expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# trace(result, A): writes Σ A_ii into result[0]
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_trace_writer_eager_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    r = einsums.create_zero_tensor("r", [1], dtype=dtype)

    einsums.linalg.trace(r, A)

    expected = np.trace(np.asarray(A))
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_trace_writer_captured_matches_eager(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)

    eager = einsums.create_zero_tensor("eager", [1], dtype=dtype)
    capt = einsums.create_zero_tensor("capt", [1], dtype=dtype)

    einsums.linalg.trace(eager, A)

    g = cg.Graph("trace-vs-eager")
    with cg.capture(g):
        einsums.linalg.trace(capt, A)
    g.execute()

    np.testing.assert_allclose(np.asarray(capt), np.asarray(eager), rtol=1e-5)


def test_trace_writer_non_square_raises():
    A = einsums.create_random_tensor("A", [3, 4])
    r = einsums.create_zero_tensor("r", [1])
    with pytest.raises(Exception):
        einsums.linalg.trace(r, A)


def test_trace_writer_rank3_raises():
    A = einsums.create_random_tensor("A", [3, 3, 3])
    r = einsums.create_zero_tensor("r", [1])
    with pytest.raises(Exception):
        einsums.linalg.trace(r, A)


# ──────────────────────────────────────────────────────────────────────────
# trace with views: all 4 combinations of (Result, A) x (owning, view)
# ──────────────────────────────────────────────────────────────────────────


def test_trace_owning_result_view_A():
    """Cell RV: owning result, view of a square sub-block."""
    big = einsums.create_random_tensor("big", [6, 6])
    r = einsums.create_zero_tensor("r", [1])

    g = cg.Graph("trace-rv")
    with cg.capture(g):
        sub = cg.view(big, [(1, 4), (1, 4)])  # 3x3 sub
        einsums.linalg.trace(r, sub)
    g.execute()

    expected = np.trace(np.asarray(big)[1:4, 1:4])
    np.testing.assert_allclose(np.asarray(r)[0], expected, rtol=1e-5)


def test_trace_view_result_owning_A():
    """Cell VR: view result, owning A."""
    holder = einsums.create_zero_tensor("holder", [3])
    A = einsums.create_random_tensor("A", [4, 4])

    g = cg.Graph("trace-vr")
    with cg.capture(g):
        r_view = cg.view(holder, [(1, 2)])
        einsums.linalg.trace(r_view, A)
    g.execute()

    expected = np.trace(np.asarray(A))
    np.testing.assert_allclose(np.asarray(holder)[1], expected, rtol=1e-5)


def test_trace_view_result_view_A():
    """Cell VV: view result + view of a square sub-block."""
    big = einsums.create_random_tensor("big", [6, 6])
    holder = einsums.create_zero_tensor("holder", [3])

    g = cg.Graph("trace-vv")
    with cg.capture(g):
        sub = cg.view(big, [(2, 5), (2, 5)])  # 3x3 sub
        r_view = cg.view(holder, [(2, 3)])
        einsums.linalg.trace(r_view, sub)
    g.execute()

    expected = np.trace(np.asarray(big)[2:5, 2:5])
    np.testing.assert_allclose(np.asarray(holder)[2], expected, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Composed SCF energy pattern: e = ½ Σ D·(H+F)
# ──────────────────────────────────────────────────────────────────────────


def test_scf_energy_pattern_captured():
    n = 5
    D = einsums.create_random_tensor("D", [n, n])
    H = einsums.create_random_tensor("H", [n, n])
    F = einsums.create_random_tensor("F", [n, n])
    sum_HF = einsums.create_zero_tensor("HF", [n, n])
    e = einsums.create_zero_tensor("e", [1])

    g = cg.Graph("scf-energy")
    with cg.capture(g):
        einsums.linalg.axpby(1.0, H, 0.0, sum_HF)  # sum_HF = H
        einsums.linalg.axpy(1.0, F, sum_HF)         # sum_HF += F
        einsums.linalg.dot(e, D, sum_HF)            # e = D · (H + F)
        einsums.linalg.scale(0.5, e)                # e = 0.5 * (D · (H + F))
    g.execute()

    expected = 0.5 * np.sum(np.asarray(D) * (np.asarray(H) + np.asarray(F)))
    np.testing.assert_allclose(np.asarray(e)[0], expected, rtol=1e-5)
