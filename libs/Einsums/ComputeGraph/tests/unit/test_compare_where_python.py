# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""compare and where, against numpy.

Masks are ordinary real tensors holding exactly 1 or 0; there is no bool dtype.
compare follows IEEE and numpy on NaN (every comparison is false except ``ne``),
and where is a true select: the branch not taken is never read into the
answer, so an inf or NaN there cannot leak through as ``0 * inf``.
"""

from __future__ import annotations

import itertools
import operator

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import REAL_DTYPES

_ctr = itertools.count()
NAN = np.nan
INF = np.inf

_OPS = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


def _nm(prefix: str) -> str:
    return f"{prefix}{next(_ctr)}"


def _run(fn, captured: bool):
    if not captured:
        fn()
        return
    g = cg.Graph(_nm("cw"))
    with cg.capture(g):
        fn()
    g.execute()
    g.execute()  # an overwrite replays to the same answer


def _transposed_view(x: np.ndarray, dtype: str):
    parent = einsums.asarray(np.ascontiguousarray(x.T), dtype=dtype)
    return parent.permute_view(list(reversed(range(x.ndim))))


def _with_specials(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    flat = x.reshape(-1)
    flat[0] = NAN
    flat[1] = INF
    flat[2] = -INF
    flat[3] = -0.0
    return x


# ── compare ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize("captured", [False, True])
def test_compare_tensors_matches_numpy(dtype, op, captured):
    rng = np.random.default_rng(1)
    a = _with_specials(rng.integers(-2, 3, size=(3, 4, 2)).astype(dtype))
    b = rng.integers(-2, 3, size=(3, 4, 2)).astype(dtype)
    b.reshape(-1)[4] = NAN
    b.reshape(-1)[3] = 0.0  # -0 against +0 compares equal
    out = einsums.create_zero_tensor(_nm("m"), [3, 4, 2], dtype=dtype)
    A, B = einsums.asarray(a, dtype=dtype), _transposed_view(b, dtype)
    _run(lambda: einsums.linalg.compare(out, A, B, op), captured)
    with np.errstate(invalid="ignore"):
        expected = _OPS[op](a, b).astype(dtype)
    np.testing.assert_array_equal(np.asarray(out), expected)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize("scalar", [0.0, 1.0, -INF, NAN])
@pytest.mark.parametrize("captured", [False, True])
def test_compare_scalar_matches_numpy(dtype, op, scalar, captured):
    rng = np.random.default_rng(2)
    a = _with_specials(rng.integers(-2, 3, size=(5, 3)).astype(dtype))
    out = einsums.create_zero_tensor(_nm("m"), [5, 3], dtype=dtype)
    A = _transposed_view(a, dtype)
    _run(lambda: einsums.linalg.compare(out, A, scalar, op), captured)
    with np.errstate(invalid="ignore"):
        expected = _OPS[op](a, np.asarray(scalar, dtype=dtype)).astype(dtype)
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_compare_nan_is_false_except_ne():
    a = np.array([NAN, NAN, 1.0])
    b = np.array([NAN, 1.0, NAN])
    for op in _OPS:
        out = einsums.create_zero_tensor(_nm("m"), [3])
        einsums.linalg.compare(out, einsums.asarray(a), einsums.asarray(b), op)
        expected = np.ones(3) if op == "ne" else np.zeros(3)
        np.testing.assert_array_equal(np.asarray(out), expected, err_msg=op)


def test_compare_in_place():
    a = np.array([1.0, 2.0, 3.0])
    A = einsums.asarray(a)
    einsums.linalg.compare(A, A, 2.0, "ge")
    np.testing.assert_array_equal(np.asarray(A), [0.0, 1.0, 1.0])


def test_compare_rejects_an_unknown_op_naming_the_valid_ones():
    A = einsums.create_zero_tensor("A", [2])
    out = einsums.create_zero_tensor("out", [2])
    with pytest.raises(ValueError, match="gt.*ge.*lt.*le.*eq.*ne"):
        einsums.linalg.compare(out, A, A, "greater")
    with pytest.raises(ValueError, match="greater"):
        einsums.linalg.compare(out, A, 0.0, "greater")


def test_compare_rejects_a_shape_mismatch():
    A = einsums.create_zero_tensor("A", [2, 3])
    B = einsums.create_zero_tensor("B", [3, 2])
    out = einsums.create_zero_tensor("out", [2, 3])
    with pytest.raises(Exception, match="compare"):
        einsums.linalg.compare(out, A, B, "gt")


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_compare_and_where_refuse_complex(dtype):
    A = einsums.create_random_tensor("A", [2, 2], dtype=dtype)
    out = einsums.create_zero_tensor("out", [2, 2], dtype=dtype)
    with pytest.raises(TypeError):
        einsums.linalg.compare(out, A, A, "eq")
    with pytest.raises(TypeError):
        einsums.linalg.compare(out, A, 0.0, "eq")
    with pytest.raises(TypeError):
        einsums.linalg.where(out, A, A, A)


# ── where ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("captured", [False, True])
def test_where_matches_numpy(dtype, captured):
    rng = np.random.default_rng(3)
    mask = rng.integers(0, 2, size=(4, 3, 2)).astype(dtype)
    a = rng.standard_normal((4, 3, 2)).astype(dtype)
    b = rng.standard_normal((4, 3, 2)).astype(dtype)
    out = einsums.create_zero_tensor(_nm("w"), [4, 3, 2], dtype=dtype)
    M, A, B = _transposed_view(mask, dtype), einsums.asarray(a, dtype=dtype), _transposed_view(b, dtype)
    _run(lambda: einsums.linalg.where(out, M, A, B), captured)
    np.testing.assert_array_equal(np.asarray(out), np.where(mask != 0, a, b))


def test_where_is_a_select_not_arithmetic():
    """The branch not taken holds inf and NaN, and the answer is still clean.

    ``mask * A + (1 - mask) * B`` would turn every one of these into NaN, since
    ``0 * inf`` and ``0 * nan`` are NaN.
    """
    mask = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    a = np.array([1.0, 2.0, INF, NAN, -INF, -0.0])
    b = np.array([INF, NAN, 3.0, 4.0, NAN, 5.0])
    out = einsums.create_zero_tensor(_nm("w"), [6])
    g = cg.Graph(_nm("select"))
    with cg.capture(g):
        einsums.linalg.where(out, einsums.asarray(mask), einsums.asarray(a), einsums.asarray(b))
    g.apply(cg.default_pass_manager())
    g.execute()
    got = np.asarray(out)
    np.testing.assert_array_equal(got, [1.0, 2.0, 3.0, 4.0, -INF, 5.0])
    assert np.all(np.isfinite(got[[0, 1, 2, 3, 5]]))


def test_where_treats_any_nonzero_and_nan_as_true():
    """A mask that came through arithmetic still selects; NaN is truthy, as in numpy."""
    mask = np.array([0.5, -2.0, NAN, 0.0, -0.0])
    a = np.full(5, 1.0)
    b = np.full(5, 2.0)
    out = einsums.create_zero_tensor(_nm("w"), [5])
    einsums.linalg.where(out, einsums.asarray(mask), einsums.asarray(a), einsums.asarray(b))
    np.testing.assert_array_equal(np.asarray(out), np.where(mask.astype(bool), a, b))
    np.testing.assert_array_equal(np.asarray(out), [1.0, 1.0, 1.0, 2.0, 2.0])


@pytest.mark.parametrize("alias", ["mask", "a", "b"])
def test_where_in_place(alias):
    rng = np.random.default_rng(4)
    mask = rng.integers(0, 2, size=6).astype(float)
    a = rng.standard_normal(6)
    b = rng.standard_normal(6)
    M, A, B = einsums.asarray(mask), einsums.asarray(a), einsums.asarray(b)
    out = {"mask": M, "a": A, "b": B}[alias]
    einsums.linalg.where(out, M, A, B)
    np.testing.assert_array_equal(np.asarray(out), np.where(mask != 0, a, b))


def test_where_rejects_a_shape_mismatch():
    M = einsums.create_zero_tensor("M", [2, 3])
    A = einsums.create_zero_tensor("A", [2, 3])
    B = einsums.create_zero_tensor("B", [3, 2])
    out = einsums.create_zero_tensor("out", [2, 3])
    with pytest.raises(Exception, match="where"):
        einsums.linalg.where(out, M, A, B)


# ── masks compose ────────────────────────────────────────────────────────────


def test_masks_compose_with_product_maximum_and_negation():
    """AND is direct_product, OR is maximum, NOT is 1 - m, all on 1/0 tensors."""
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    X = einsums.asarray(x)
    lo = einsums.create_zero_tensor("lo", [4])
    hi = einsums.create_zero_tensor("hi", [4])
    both = einsums.create_zero_tensor("both", [4])
    either = einsums.create_zero_tensor("either", [4])
    g = cg.Graph(_nm("masks"))
    with cg.capture(g):
        einsums.linalg.compare(lo, X, -1.0, "gt")
        einsums.linalg.compare(hi, X, 1.0, "lt")
        einsums.linalg.direct_product(1.0, lo, hi, 0.0, both)
        einsums.linalg.maximum(either, lo, hi)
    g.execute()
    np.testing.assert_array_equal(np.asarray(both), ((x > -1) & (x < 1)).astype(float))
    np.testing.assert_array_equal(np.asarray(either), ((x > -1) | (x < 1)).astype(float))


def test_compare_and_where_in_a_loop_body_under_the_passes():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal((4, 3))
    A, B = einsums.asarray(a), einsums.asarray(b)
    M = einsums.create_zero_tensor("M", [4, 3])
    out = einsums.create_zero_tensor("out", [4, 3])
    g = cg.Graph(_nm("cw_loop"))
    body = g.add_loop("iter", 3, lambda i: i + 1 < 3)
    with cg.capture(body):
        einsums.linalg.compare(M, A, B, "gt")
        einsums.linalg.where(out, M, A, B)
    g.apply(cg.default_pass_manager())
    g.execute()
    np.testing.assert_array_equal(np.asarray(out), np.maximum(a, b))
