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


def test_matmul_rank_guard():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    v = einsums.create_random_tensor("v", [4], dtype="float64")
    with pytest.raises(ValueError):
        _ = A @ v


def test_matmul_dtype_mismatch_raises():
    A = einsums.create_random_tensor("A", [3, 4], dtype="float64")
    B = einsums.create_random_tensor("B", [4, 2], dtype="float32")
    with pytest.raises(TypeError):
        _ = A @ B


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
