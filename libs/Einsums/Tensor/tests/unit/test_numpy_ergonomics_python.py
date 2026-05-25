# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""numpy-parity ergonomics on RuntimeTensor / RuntimeTensorView.

Covers the thin attribute layer installed by ``_patch_numpy_ergonomics``
in ``einsums/__init__.py`` (the May 2026 "Tier 1 + capture-aware matmul"
pass):

  * ``.shape`` / ``.ndim`` / ``.dtype`` / ``len()`` — read-only,
    consistent across ranks and dtypes, and present on views too.
  * ``.T`` — zero-copy reversed-axis view (the C++ ``transpose_view``),
    matching ``numpy``'s transpose.
  * ``__array__`` — ``np.array(t)`` / ``np.asarray(t)`` honor dtype and
    copy semantics.
  * ``__matmul__`` — ``A @ B`` dispatches to ``einsums.linalg.gemm``
    (NOT numpy), both eager and recorded into a ``cg.capture`` graph,
    with shape/dtype/rank guards that keep the op on Einsums.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES, assert_close


# ──────────────────────────────────────────────────────────────────────────
# shape / ndim / dtype / len
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", [[4], [3, 5], [2, 3, 4], [2, 3, 4, 5]])
def test_shape_ndim_len(dtype, shape):
    t = einsums.create_zero_tensor("t", shape, dtype=dtype)
    assert t.shape == tuple(shape)
    assert t.ndim == len(shape)
    assert len(t) == shape[0]
    # mirror numpy exactly
    ref = np.asarray(t)
    assert t.shape == ref.shape
    assert t.ndim == ref.ndim
    assert len(t) == len(ref)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_dtype_property(dtype):
    t = einsums.create_zero_tensor("t", [3, 3], dtype=dtype)
    assert t.dtype == np.dtype(dtype)
    # round-trips through numpy unchanged
    assert np.asarray(t).dtype == t.dtype


def test_repr_includes_name_shape_dtype():
    t = einsums.create_zero_tensor("myname", [2, 7], dtype="float64")
    r = repr(t)
    assert "myname" in r and "(2, 7)" in r and "float64" in r


# ──────────────────────────────────────────────────────────────────────────
# .T  (zero-copy transpose view)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_transpose_property_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [3, 5], dtype=dtype)
    AT = A.T
    assert AT.shape == (5, 3)
    assert AT.ndim == 2
    assert_close(np.asarray(AT), np.asarray(A).T)


def test_transpose_view_aliases_storage():
    """``.T`` is a view: writing through it changes the parent."""
    A = einsums.create_zero_tensor("A", [2, 3], dtype="float64")
    np.asarray(A.T)[0, 1] = 9.0  # AT[0,1] == A[1,0]
    assert np.asarray(A)[1, 0] == 9.0


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_transpose_matmul_in_capture(dtype):
    """Inside cg.capture, ``.T`` routes through cg.permute_view so a
    transposed operand can feed a captured gemm (used to segfault)."""
    P = einsums.create_random_tensor("P", [4, 3], dtype=dtype)
    Q = einsums.create_random_tensor("Q", [4, 2], dtype=dtype)
    p, q = np.asarray(P).copy(), np.asarray(Q).copy()
    g = cg.Graph("tT")
    with cg.capture(g):
        I = P.T @ Q  # (3,4) @ (4,2)
    g.execute()
    assert I.shape == (3, 2)
    assert_close(np.asarray(I), p.T @ q)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transpose_axpy_in_capture(dtype):
    """``2*M - M.T`` inside capture: the transpose view feeds a captured axpy."""
    M = einsums.create_random_tensor("M", [3, 3], dtype=dtype)
    m = np.asarray(M).copy()
    g = cg.Graph("tT2")
    with cg.capture(g):
        K = 2.0 * M - M.T
    g.execute()
    assert_close(np.asarray(K), 2.0 * m - m.T)


def test_permute_view_general_axes_in_capture():
    """cg.permute_view handles an arbitrary rank-3 axis permutation."""
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype="float64")
    r = np.asarray(R).copy()
    g = cg.Graph("perm")
    with cg.capture(g):
        S = cg.permute_view(R, [2, 0, 1]) + cg.permute_view(R, [2, 0, 1])
    g.execute()
    assert_close(np.asarray(S), 2.0 * np.transpose(r, (2, 0, 1)))


# ──────────────────────────────────────────────────────────────────────────
# .transpose / .swapaxes / .copy
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_transpose_forms(dtype):
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype=dtype)
    r = np.asarray(R).copy()
    assert_close(np.asarray(R.transpose()), r.transpose())                 # default reverse
    assert_close(np.asarray(R.transpose(2, 0, 1)), np.transpose(r, (2, 0, 1)))  # explicit
    assert_close(np.asarray(R.transpose((1, 2, 0))), np.transpose(r, (1, 2, 0)))  # tuple
    assert_close(np.asarray(R.transpose(-1, -2, -3)), np.transpose(r, (-1, -2, -3)))  # negative


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_swapaxes(dtype):
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype=dtype)
    r = np.asarray(R).copy()
    assert_close(np.asarray(R.swapaxes(0, 2)), np.swapaxes(r, 0, 2))
    assert_close(np.asarray(R.swapaxes(-1, 1)), np.swapaxes(r, -1, 1))


def test_transpose_is_a_view():
    """transpose returns a zero-copy view: writes propagate to the parent."""
    A = einsums.create_zero_tensor("A", [2, 3], dtype="float64")
    np.asarray(A.transpose(1, 0))[0, 1] = 9.0  # == A[1, 0]
    assert np.asarray(A)[1, 0] == 9.0


def test_transpose_bad_axes_raises():
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype="float64")
    with pytest.raises(ValueError):
        R.transpose(0, 0, 1)          # not a permutation
    with pytest.raises(ValueError):
        R.transpose(0, 1)             # wrong count


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_copy_is_independent(dtype):
    B = einsums.create_random_tensor("B", [3, 3], dtype=dtype)
    b = np.asarray(B).copy()
    Bc = B.copy()
    assert_close(np.asarray(Bc), b)
    np.asarray(Bc)[0, 0] = 123.0      # mutate the copy
    assert np.asarray(B)[0, 0] == b[0, 0]  # original untouched


# ──────────────────────────────────────────────────────────────────────────
# reductions: .sum / .mean / .max
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_sum_and_mean(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    a = np.asarray(A)
    assert_close(A.sum(), a.sum())
    assert_close(A.mean(), a.mean())


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_max(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    assert_close(A.max(), np.asarray(A).max())


def test_sum_max_on_views():
    """Reductions are stride-correct on transpose / slice views."""
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    a = np.asarray(A)
    assert_close(A.T.sum(), a.T.sum())
    assert_close(A[1:3, :].sum(), a[1:3, :].sum())
    assert_close(A[1:3, :].max(), a[1:3, :].max())


def test_max_complex_raises():
    Z = einsums.create_random_tensor("Z", [2, 2], dtype="complex128")
    with pytest.raises(TypeError):
        Z.max()


def test_reductions_in_capture():
    """sum/mean/max record into graph [1] tensors; read after execute."""
    B = einsums.create_random_tensor("B", [4, 5], dtype="float64")
    b = np.asarray(B).copy()
    g = cg.Graph("red")
    with cg.capture(g):
        s = B.sum()
        m = B.mean()
        mx = B.max()
    g.execute()
    assert_close(np.asarray(s)[0], b.sum())
    assert_close(np.asarray(m)[0], b.mean())
    assert_close(np.asarray(mx)[0], b.max())


def test_reduction_feeds_downstream_op_in_capture():
    """The [1] sum result is a real tensor slot, so downstream ops compose."""
    C = einsums.create_random_tensor("C", [3, 3], dtype="float64")
    c = np.asarray(C).copy()
    g = cg.Graph("red2")
    with cg.capture(g):
        tot = C.sum()
        tot += 100.0
    g.execute()
    assert_close(np.asarray(tot)[0], c.sum() + 100.0)


def test_transpose_swapaxes_copy_in_capture():
    """transpose/swapaxes route through cg.permute_view under capture; copy records."""
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype="float64")
    B = einsums.create_random_tensor("B", [3, 3], dtype="float64")
    r, b = np.asarray(R).copy(), np.asarray(B).copy()
    g = cg.Graph("views")
    with cg.capture(g):
        S = R.transpose(2, 0, 1) + R.transpose(2, 0, 1)  # 2 * permuted
        U = B.swapaxes(0, 1).copy()                       # copy of B^T
    g.execute()
    assert_close(np.asarray(S), 2.0 * np.transpose(r, (2, 0, 1)))
    assert_close(np.asarray(U), b.T)


# ──────────────────────────────────────────────────────────────────────────
# __array__
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_array_protocol_roundtrip(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    via_array = np.array(A)        # copy
    via_asarray = np.asarray(A)    # view
    assert_close(via_array, via_asarray)
    assert via_array.shape == (3, 4)


def test_np_array_makes_a_copy():
    A = einsums.create_zero_tensor("A", [2, 2], dtype="float64")
    c = np.array(A)
    c[0, 0] = 5.0
    assert np.asarray(A)[0, 0] == 0.0  # original untouched


# ──────────────────────────────────────────────────────────────────────────
# __matmul__  (dispatches to einsums.linalg.gemm, NOT numpy)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_matmul_eager_matches_numpy(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [4, 2], dtype=dtype)
    C = A @ B
    assert C.shape == (3, 2)
    assert C.dtype == np.dtype(dtype)
    assert_close(np.asarray(C), np.asarray(A) @ np.asarray(B))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_matmul_with_view_operand(dtype):
    """A.T @ A uses a view on the left — gemm handles it eagerly."""
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    C = A.T @ A
    assert C.shape == (4, 4)
    assert_close(np.asarray(C), np.asarray(A).T @ np.asarray(A))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_matmul_is_captured_into_graph(dtype):
    """Inside cg.capture, ``@`` records a gemm node instead of executing."""
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [4, 2], dtype=dtype)
    g = cg.Graph("mm")
    with cg.capture(g):
        C = A @ B
    # Not computed until execute().
    g.execute()
    assert_close(np.asarray(C), np.asarray(A) @ np.asarray(B))


def test_matmul_rejects_numpy_rhs():
    """Right-hand numpy operand is rejected so the op never silently
    leaves Einsums (would otherwise fall through to numpy's matmul)."""
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    with pytest.raises(TypeError):
        _ = A @ np.zeros((4, 2))


def test_matmul_shape_mismatch_raises():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    B = einsums.create_random_tensor("B", [3, 2], dtype="float64")
    with pytest.raises(ValueError):
        _ = A @ B


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_matmul_matrix_vector(dtype):
    """A @ x (gemv) and v @ A (A^T x) match numpy."""
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    x = einsums.create_random_tensor("x", [4], dtype=dtype)
    v = einsums.create_random_tensor("v", [3], dtype=dtype)
    assert_close(np.asarray(A @ x), np.asarray(A) @ np.asarray(x))
    assert_close(np.asarray(v @ A), np.asarray(v) @ np.asarray(A))


def test_matmul_matvec_in_capture():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    x = einsums.create_random_tensor("x", [4], dtype="float64")
    a, xv = np.asarray(A).copy(), np.asarray(x).copy()
    g = cg.Graph("gv")
    with cg.capture(g):
        y = A @ x
    g.execute()
    assert_close(np.asarray(y), a @ xv)


def test_matmul_vector_inner_product_raises():
    """v @ w (inner product) raises -> use einsums.linalg.dot, so @ always returns a tensor."""
    v = einsums.create_random_tensor("v", [4], dtype="float64")
    with pytest.raises(ValueError):
        _ = v @ v


def test_matmul_dtype_mismatch_raises():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    B = einsums.create_random_tensor("B", [4, 2], dtype="float32")
    with pytest.raises(TypeError):
        _ = A @ B


# ──────────────────────────────────────────────────────────────────────────
# __setitem__ : sub-view assignment
# ──────────────────────────────────────────────────────────────────────────


def test_setitem_element():
    A = einsums.create_zero_tensor("A", [3, 4], dtype="float64")
    A[1, 2] = 9.0
    assert np.asarray(A)[1, 2] == 9.0


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_setitem_row_and_block(dtype):
    A = einsums.create_zero_tensor("A", [4, 4], dtype=dtype)
    row = einsums.create_random_tensor("row", [4], dtype=dtype)
    blk = einsums.create_random_tensor("blk", [2, 4], dtype=dtype)
    A[1] = row                       # rank-reducing int key (A[1, :])
    A[2:4, :] = blk                  # slice block
    assert_close(np.asarray(A)[1], np.asarray(row))
    assert_close(np.asarray(A)[2:4, :], np.asarray(blk))


def test_setitem_scalar_fill_block():
    A = einsums.create_zero_tensor("A", [4, 4], dtype="float64")
    A[0:2, 0:2] = 7.0
    assert np.all(np.asarray(A)[0:2, 0:2] == 7.0)
    assert np.all(np.asarray(A)[2:, :] == 0.0)  # rest untouched


def test_setitem_numpy_rhs():
    A = einsums.create_zero_tensor("A", [3, 4], dtype="float64")
    A[2, :] = np.arange(4.0)
    assert_close(np.asarray(A)[2], np.arange(4.0))


def test_setitem_shape_mismatch_raises():
    A = einsums.create_zero_tensor("A", [3, 4], dtype="float64")
    with pytest.raises(ValueError):
        A[0] = einsums.zeros((3,), dtype="float64")  # wrong length


def test_setitem_block_in_capture():
    """Slice-key block assignment records into a graph (sub-view dims resolve
    at execute, so the assignment is correct even though the captured view's
    dims are a placeholder at capture time)."""
    C = einsums.create_zero_tensor("C", [3, 4], dtype="float64")
    src = einsums.create_random_tensor("src", [2, 4], dtype="float64")
    sv = np.asarray(src).copy()
    g = cg.Graph("si")
    with cg.capture(g):
        C[0:2, :] = src
    g.execute()
    assert_close(np.asarray(C)[0:2, :], sv)


def test_setitem_int_key_in_capture():
    """Int-key assignment in capture works now that __getitem__ rank-reduces
    via the Drop axis."""
    C = einsums.create_zero_tensor("C", [3, 4], dtype="float64")
    row = einsums.create_random_tensor("row", [4], dtype="float64")
    rv = np.asarray(row).copy()
    g = cg.Graph("si-int")
    with cg.capture(g):
        C[1] = row
    g.execute()
    assert_close(np.asarray(C)[1], rv)


# ──────────────────────────────────────────────────────────────────────────
# capture-mode rank-reducing reads (Drop axis)
# ──────────────────────────────────────────────────────────────────────────


def test_rank_reducing_read_in_capture():
    """A[i] / A[:,j,:] / A[i,j] inside cg.capture rank-reduce (Drop)."""
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype="float64")
    r = np.asarray(R).copy()
    g = cg.Graph("drop")
    with cg.capture(g):
        a = R[1]          # (3, 4)
        b = R[:, 2, :]    # (2, 4)
        c = R[0, 1]       # (4,)
        s = a + a         # consume a dropped view in a captured op
    g.execute()
    assert_close(np.asarray(a), r[1])
    assert_close(np.asarray(b), r[:, 2, :])
    assert_close(np.asarray(c), r[0, 1])
    assert_close(np.asarray(s), 2.0 * r[1])


def test_negative_int_index_in_capture():
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype="float64")
    r = np.asarray(R).copy()
    g = cg.Graph("drop-neg")
    with cg.capture(g):
        d = R[-1]
    g.execute()
    assert_close(np.asarray(d), r[-1])


def test_matmul_on_dropped_view_in_capture():
    """A dropped (rank-reduced) view feeds gemv inside capture."""
    R = einsums.create_random_tensor("R", [2, 3, 4], dtype="float64")
    v = einsums.create_random_tensor("v", [4], dtype="float64")
    r, vv = np.asarray(R).copy(), np.asarray(v).copy()
    g = cg.Graph("drop-matvec")
    with cg.capture(g):
        y = R[0] @ v      # R[0] is (3,4); @ v(4) -> (3,)
    g.execute()
    assert_close(np.asarray(y), r[0] @ vv)


def test_too_many_indices_in_capture_raises():
    R = einsums.create_random_tensor("R", [2, 3], dtype="float64")
    g = cg.Graph("drop-bad")
    with pytest.raises(IndexError):
        with cg.capture(g):
            _ = R[0, 1, 2]


# ──────────────────────────────────────────────────────────────────────────
# Tier-2 arithmetic operators (dispatch to einsums.linalg, NOT numpy)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_add_sub_match_numpy(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 4], dtype=dtype)
    a, b = np.asarray(A).copy(), np.asarray(B).copy()
    assert_close(np.asarray(A + B), a + b)
    assert_close(np.asarray(A - B), a - b)
    # operands are never mutated
    assert_close(np.asarray(A), a)
    assert_close(np.asarray(B), b)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_elementwise_mul_matches_numpy(dtype):
    """``A * B`` (both tensors) is element-wise, like numpy."""
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 4], dtype=dtype)
    assert_close(np.asarray(A * B), np.asarray(A) * np.asarray(B))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_scalar_mul_and_div(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    a = np.asarray(A).copy()
    assert_close(np.asarray(3.0 * A), 3.0 * a)   # __rmul__
    assert_close(np.asarray(A * 2.0), 2.0 * a)   # __mul__
    assert_close(np.asarray(A / 4.0), a / 4.0)   # __truediv__


def test_complex_scalar_mul():
    A = einsums.create_random_tensor("A", [2, 3], dtype="complex128")
    assert_close(np.asarray((2 + 1j) * A), (2 + 1j) * np.asarray(A))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_neg_and_pos(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    a = np.asarray(A).copy()
    assert_close(np.asarray(-A), -a)
    assert_close(np.asarray(+A), a)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_inplace_add_sub(dtype):
    C = einsums.create_zero_tensor("C", [3, 4], dtype=dtype)
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 4], dtype=dtype)
    a, b = np.asarray(A).copy(), np.asarray(B).copy()
    C += A
    assert_close(np.asarray(C), a)
    C -= B
    assert_close(np.asarray(C), a - b)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_inplace_scalar_mul_div(dtype):
    A = einsums.create_random_tensor("A", [2, 2], dtype=dtype)
    a = np.asarray(A).copy()
    A *= 3.0
    assert_close(np.asarray(A), a * 3.0)
    A /= 6.0
    assert_close(np.asarray(A), a * 3.0 / 6.0)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_scalar_add_sub(dtype):
    """Scalar +/- (binary and reflected) dispatch to einsums.linalg.shift."""
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    a = np.asarray(A).copy()
    assert_close(np.asarray(A + 2.0), a + 2.0)    # __add__ scalar
    assert_close(np.asarray(2.0 + A), 2.0 + a)    # __radd__
    assert_close(np.asarray(A - 1.5), a - 1.5)    # __sub__ scalar
    assert_close(np.asarray(3.0 - A), 3.0 - a)    # __rsub__
    assert_close(np.asarray(A), a)                # operands untouched


def test_complex_scalar_add():
    A = einsums.create_random_tensor("A", [2, 3], dtype="complex128")
    a = np.asarray(A).copy()
    assert_close(np.asarray(A + (1 + 2j)), a + (1 + 2j))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_inplace_scalar_add_sub(dtype):
    """``A += c`` / ``A -= c`` shift in place (no new tensor)."""
    A = einsums.create_random_tensor("A", [2, 2], dtype=dtype)
    a = np.asarray(A).copy()
    A += 5.0
    assert_close(np.asarray(A), a + 5.0)
    A -= 2.0
    assert_close(np.asarray(A), a + 5.0 - 2.0)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_shift_op_direct(dtype):
    """The underlying einsums.linalg.shift adds a scalar to every element."""
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    a = np.asarray(A).copy()
    einsums.linalg.shift(4.0, A)
    assert_close(np.asarray(A), a + 4.0)


def test_scalar_add_captured_into_graph():
    """Scalar add (binary) and in-place shift both record into a graph."""
    A = einsums.create_random_tensor("A", [3, 3], dtype="float64")
    a = np.asarray(A).copy()
    g = cg.Graph("shift")
    with cg.capture(g):
        E = A + 7.0   # copy + shift
        E += 3.0      # in-place shift
    g.execute()
    assert_close(np.asarray(E), a + 10.0)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_arithmetic_captured_into_graph(dtype):
    """Chained operators inside cg.capture record nodes and allocate
    intermediates on the graph (so they outlive execute())."""
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [3, 4], dtype=dtype)
    a, b = np.asarray(A).copy(), np.asarray(B).copy()
    g = cg.Graph("lincomb")
    with cg.capture(g):
        E = (A + B) * 2.0
        E = E - A
    g.execute()
    assert_close(np.asarray(E), (a + b) * 2.0 - a)


def test_add_rejects_numpy_rhs():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    with pytest.raises(TypeError):
        _ = A + np.zeros((3, 4))


def test_tensor_division_unsupported():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    B = einsums.create_random_tensor("B", [3, 4], dtype="float64")
    with pytest.raises(TypeError):
        _ = A / B


def test_add_shape_mismatch_raises():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    B = einsums.create_random_tensor("B", [2, 2], dtype="float64")
    with pytest.raises(ValueError):
        _ = A + B


def test_inplace_tensor_mul_unsupported():
    """``A *= B`` (tensor) is rejected; use ``A = A * B`` for element-wise."""
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    B = einsums.create_random_tensor("B", [3, 4], dtype="float64")
    with pytest.raises(TypeError):
        A *= B


# ──────────────────────────────────────────────────────────────────────────
# Tier-3 numpy-style constructor aliases
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", [5, [3, 4], (2, 3, 4)])
def test_zeros(dtype, shape):
    t = einsums.zeros(shape, dtype=dtype)
    expected = np.zeros(shape, dtype=dtype)
    assert t.shape == expected.shape
    assert t.dtype == np.dtype(dtype)
    assert_close(np.asarray(t), expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_ones(dtype):
    t = einsums.ones([2, 3], dtype=dtype)
    assert_close(np.asarray(t), np.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_full(dtype):
    t = einsums.full((2, 2), 7, dtype=dtype)
    assert_close(np.asarray(t), np.full((2, 2), 7, dtype=dtype))


def test_full_complex_value():
    t = einsums.full((2,), 1 + 2j, dtype="complex128")
    assert_close(np.asarray(t), np.full((2,), 1 + 2j))


def test_empty_shape_and_dtype():
    t = einsums.empty((3, 5), dtype="float32")
    assert t.shape == (3, 5)
    assert t.dtype == np.dtype("float32")


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_eye(dtype):
    assert_close(np.asarray(einsums.eye(3, dtype=dtype)), np.eye(3, dtype=dtype))
    assert_close(np.asarray(einsums.eye(2, 4, dtype=dtype)), np.eye(2, 4, dtype=dtype))


def test_dtype_normalization():
    # float32 stays; every other real type collapses to float64.
    assert einsums.zeros((2,), dtype=np.float32).dtype == np.dtype("float32")
    assert einsums.zeros((2,), dtype=int).dtype == np.dtype("float64")
    assert einsums.zeros((2,), dtype="float16").dtype == np.dtype("float64")
    # complex64 stays; other complex collapses to complex128.
    assert einsums.zeros((2,), dtype=np.complex64).dtype == np.dtype("complex64")
    assert einsums.zeros((2,), dtype=complex).dtype == np.dtype("complex128")


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_asarray_from_numpy(dtype):
    a = (np.arange(12).reshape(3, 4)).astype(dtype)
    t = einsums.asarray(a)
    assert t.shape == (3, 4)
    assert t.dtype == np.dtype(dtype)
    assert_close(np.asarray(t), a)


def test_asarray_passthrough_for_einsums_tensor():
    t = einsums.create_random_tensor("t", [3, 3], dtype="float64")
    assert einsums.asarray(t) is t                  # no copy when dtype matches
    assert einsums.asarray(t, dtype="float64") is t


def test_asarray_from_nested_list_defaults_float64():
    t = einsums.asarray([[1, 2], [3, 4]])
    assert t.dtype == np.dtype("float64")
    assert_close(np.asarray(t), [[1.0, 2.0], [3.0, 4.0]])


def test_array_always_copies():
    a = np.zeros((2, 2))
    t = einsums.array(a)
    np.asarray(t)[0, 0] = 99.0
    assert a[0, 0] == 0.0  # source untouched


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_like_constructors(dtype):
    ref = einsums.create_random_tensor("ref", [2, 5], dtype=dtype)
    assert einsums.zeros_like(ref).shape == ref.shape
    assert einsums.zeros_like(ref).dtype == ref.dtype
    assert_close(np.asarray(einsums.ones_like(ref)), np.ones((2, 5), dtype=dtype))
    assert_close(np.asarray(einsums.full_like(ref, 3)), np.full((2, 5), 3, dtype=dtype))
    assert einsums.empty_like(ref).shape == ref.shape


def test_constructor_result_has_ergonomics():
    """Tensors from the aliases carry the numpy-ergonomics layer."""
    z = einsums.ones((2, 3))
    assert z.ndim == 2
    assert (z + z).shape == (2, 3)
    assert z.T.shape == (3, 2)


def test_constructor_inside_capture_is_graph_owned():
    """A constructor used inside cg.capture yields a graph-owned tensor that
    survives a chained, reassigning workflow through execute()."""
    A = einsums.create_random_tensor("A", [2, 2], dtype="float64")
    a = np.asarray(A).copy()
    g = cg.Graph("ctor")
    with cg.capture(g):
        acc = einsums.zeros((2, 2))
        acc = acc + A
        acc = acc + A
    g.execute()
    assert_close(np.asarray(acc), 2.0 * a)


# ──────────────────────────────────────────────────────────────────────────
# Stub verification: the runtime-patched ergonomics layer is monkey-patched
# onto the bound classes, so it can't be auto-stubbed from the C++ AST — it's
# injected by tools/einsums-pybind/scripts/aggregate_stubs.py. These tests
# guard against that injection silently breaking (verified end-to-end with
# pyright: 0 errors resolving t.shape / t @ t / t.sum() / einsums.zeros, ...).
# ──────────────────────────────────────────────────────────────────────────

import ast as _ast
import pathlib as _pathlib
import re as _re


def _stub_dir() -> _pathlib.Path:
    return _pathlib.Path(einsums.__file__).parent


_HAS_STUBS = (_stub_dir() / "_core.pyi").is_file()
_skip_no_stubs = pytest.mark.skipif(not _HAS_STUBS, reason="stubs not generated (build PyEinsumsStubs)")

# Ergonomics members (monkey-patched at runtime) that must appear in the
# generated tensor-class stubs — keep in sync with _patch_numpy_ergonomics.
_ERGONOMICS_METHODS = (
    "shape", "ndim", "dtype", "T", "__len__", "__repr__", "__array__",
    "__getitem__", "__setitem__", "transpose", "swapaxes", "copy",
    "sum", "mean", "max", "__matmul__", "__add__", "__radd__", "__sub__",
    "__rsub__", "__mul__", "__rmul__", "__truediv__", "__neg__", "__pos__",
    "__iadd__", "__isub__", "__imul__", "__itruediv__",
)
_CONSTRUCTORS = (
    "zeros", "ones", "empty", "full", "eye", "array", "asarray",
    "zeros_like", "ones_like", "empty_like", "full_like",
)


@_skip_no_stubs
@pytest.mark.parametrize("cls", ["RuntimeTensorD", "RuntimeTensorViewD",
                                 "RuntimeTensorC", "RuntimeTensorViewZ"])
def test_stub_has_ergonomics_methods(cls):
    core = (_stub_dir() / "_core.pyi").read_text()
    m = _re.search(rf"\nclass {cls}:\n", core)
    assert m, f"{cls} not found in _core.pyi"
    block = core[m.end():]
    nxt = _re.search(r"\nclass \w", block)
    if nxt:
        block = block[: nxt.start()]
    missing = [name for name in _ERGONOMICS_METHODS if f"def {name}(" not in block]
    assert not missing, f"{cls} stub missing: {missing}"
    # codegen members must survive the injection
    assert 'def rank(' in block and 'def transpose_view(' in block


@_skip_no_stubs
def test_init_stub_has_constructors():
    init = (_stub_dir() / "__init__.pyi").read_text()
    missing = [fn for fn in _CONSTRUCTORS if f"def {fn}(" not in init]
    assert not missing, f"__init__.pyi missing constructors: {missing}"


@_skip_no_stubs
def test_generated_stubs_parse():
    for name in ("_core.pyi", "__init__.pyi"):
        _ast.parse((_stub_dir() / name).read_text())  # raises SyntaxError on bad stub
