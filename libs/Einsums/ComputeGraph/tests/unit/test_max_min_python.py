# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""maximum / minimum, max_axes / min_axes and the scalar min, against numpy.

Every op here is real-only and propagates NaN the way numpy.maximum and
numpy.max do, so the oracle is numpy itself rather than a hand-written rule.
Each case runs eagerly and captured; the captured graph is executed twice so an
accumulate-instead-of-assign bug shows as a wrong second answer.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import REAL_DTYPES

_ctr = itertools.count()
NAN = np.nan
INF = np.inf


def _nm(prefix: str) -> str:
    return f"{prefix}{next(_ctr)}"


def _run(fn, captured: bool, replays: int = 2):
    if not captured:
        fn()
        return
    g = cg.Graph(_nm("mm"))
    with cg.capture(g):
        fn()
    for _ in range(replays):
        g.execute()


def _transposed_view(x: np.ndarray, dtype: str):
    """A view of ``x`` whose storage is laid out transposed, so it walks by stride."""
    parent = einsums.asarray(np.ascontiguousarray(x.T), dtype=dtype)
    return parent.permute_view(list(reversed(range(x.ndim))))


# ── elementwise maximum / minimum ────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("name", ["maximum", "minimum"])
@pytest.mark.parametrize("captured", [False, True])
def test_elementwise_extremum_matches_numpy(dtype, name, captured):
    rng = np.random.default_rng(3)
    a = rng.standard_normal((3, 4, 2)).astype(dtype)
    b = rng.standard_normal((3, 4, 2)).astype(dtype)
    A = einsums.asarray(a, dtype=dtype)
    B = _transposed_view(b, dtype)
    C = einsums.create_zero_tensor(_nm("C"), [3, 4, 2], dtype=dtype)
    _run(lambda: getattr(einsums.linalg, name)(C, A, B), captured)
    np.testing.assert_array_equal(np.asarray(C), getattr(np, name)(a, b))


@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_elementwise_extremum_propagates_nan_and_orders_signed_zeros(name):
    a = np.array([0.0, -0.0, NAN, 1.0, NAN, -INF, INF])
    b = np.array([-0.0, 0.0, 1.0, NAN, NAN, 2.0, 2.0])
    C = einsums.create_zero_tensor(_nm("C"), [a.size], dtype="float64")
    getattr(einsums.linalg, name)(C, einsums.asarray(a), einsums.asarray(b))
    expected = getattr(np, name)(a, b)
    got = np.asarray(C)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.signbit(got), np.signbit(expected))


@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_elementwise_extremum_in_place(name):
    """The destination may be one of its operands, which is how a clip is spelled."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal((4, 3))
    A = einsums.asarray(a)
    _run(lambda: getattr(einsums.linalg, name)(A, A, einsums.asarray(b)), captured=True, replays=1)
    np.testing.assert_array_equal(np.asarray(A), getattr(np, name)(a, b))


def test_clip_is_minimum_of_maximum():
    x = np.array([-3.0, -1.0, 0.5, 2.0, NAN])
    X = einsums.asarray(x)
    lo = einsums.asarray(np.full(5, -1.5))
    hi = einsums.asarray(np.full(5, 1.0))
    g = cg.Graph(_nm("clip"))
    with cg.capture(g):
        einsums.linalg.maximum(X, X, lo)
        einsums.linalg.minimum(X, X, hi)
    g.execute()
    np.testing.assert_array_equal(np.asarray(X), np.minimum(np.maximum(x, -1.5), 1.0))


def test_elementwise_extremum_rejects_a_shape_mismatch():
    A = einsums.create_zero_tensor("A", [2, 3])
    B = einsums.create_zero_tensor("B", [3, 2])
    C = einsums.create_zero_tensor("C", [2, 3])
    with pytest.raises(Exception, match="maximum"):
        einsums.linalg.maximum(C, A, B)


@pytest.mark.parametrize("name", ["maximum", "minimum", "max_axes", "min_axes", "min"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_real_only_ops_refuse_complex(name, dtype):
    """Complex has no order; no complex overload is bound, so the call is refused."""
    A = einsums.create_random_tensor("A", [2, 2], dtype=dtype)
    out = einsums.create_zero_tensor("out", [2, 2], dtype=dtype)
    args = {"max_axes": (out, A, []), "min_axes": (out, A, []), "min": (out, A)}.get(name, (out, A, A))
    with pytest.raises(TypeError):
        getattr(einsums.linalg, name)(*args)


# ── axis-wise max_axes / min_axes ────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("name", ["max", "min"])
@pytest.mark.parametrize("axes", [[0], [1], [2], [0, 2], [2, 0], [0, 1, 2], []])
@pytest.mark.parametrize("captured", [False, True])
def test_reduce_axes_matches_numpy(dtype, name, axes, captured):
    rng = np.random.default_rng(11)
    a = rng.standard_normal((3, 4, 5)).astype(dtype)
    expected = getattr(a, name)(axis=tuple(axes)) if axes else a.copy()
    A = _transposed_view(a, dtype)
    out = einsums.create_zero_tensor(_nm("out"), list(np.shape(expected)), dtype=dtype)
    _run(lambda: getattr(einsums.linalg, f"{name}_axes")(out, A, axes), captured)
    np.testing.assert_array_equal(np.asarray(out), expected)


@pytest.mark.parametrize("name", ["max", "min"])
def test_reduce_axes_nan_and_infinities(name):
    a = np.array([[1.0, NAN, -INF], [2.0, 3.0, -INF], [-0.0, 0.0, -INF]]).T  # columns are the lines
    A = einsums.asarray(a)
    out = einsums.create_zero_tensor(_nm("out"), [3])
    getattr(einsums.linalg, f"{name}_axes")(out, A, [0])
    expected = getattr(a, name)(axis=0)
    got = np.asarray(out)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.signbit(got), np.signbit(expected))


@pytest.mark.parametrize("name", ["max_axes", "min_axes"])
@pytest.mark.parametrize("shape,axes", [((0, 3), [0]), ((2, 0), [1]), ((0, 0), [0]), ((0, 3), [0, 1])])
def test_reduce_axes_over_an_empty_axis_is_refused(name, shape, axes):
    """numpy refuses an empty max; there is no identity a caller would want."""
    A = einsums.create_zero_tensor("A", list(shape))
    kept = [d for i, d in enumerate(shape) if i not in axes]
    out = einsums.create_zero_tensor("out", kept)
    with pytest.raises(ValueError):
        getattr(einsums.linalg, name)(out, A, axes)


def test_reduce_axes_keeping_an_empty_axis_is_fine():
    """Only an empty REDUCED axis is refused: (3, 0) reduced over axis 0 is an empty result."""
    A = einsums.create_zero_tensor("A", [3, 0])
    out = einsums.create_zero_tensor("out", [0])
    einsums.linalg.max_axes(out, A, [0])
    assert np.asarray(out).shape == (0,)


@pytest.mark.parametrize("name", ["max_axes", "min_axes"])
def test_reduce_axes_validation(name):
    A = einsums.create_zero_tensor("A", [2, 3])
    with pytest.raises(ValueError, match="twice"):
        getattr(einsums.linalg, name)(einsums.create_zero_tensor("o", [3]), A, [0, 0])
    with pytest.raises(IndexError):
        getattr(einsums.linalg, name)(einsums.create_zero_tensor("o", [3]), A, [2])
    with pytest.raises(Exception, match="rank"):
        getattr(einsums.linalg, name)(einsums.create_zero_tensor("o", [2, 3]), A, [0])
    with pytest.raises(Exception, match="extent"):
        getattr(einsums.linalg, name)(einsums.create_zero_tensor("o", [2]), A, [0])


# ── scalar min ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("captured", [False, True])
def test_scalar_min_matches_numpy(dtype, captured):
    rng = np.random.default_rng(13)
    a = rng.standard_normal((4, 5)).astype(dtype)
    r = einsums.create_zero_tensor(_nm("r"), [], dtype=dtype)
    _run(lambda: einsums.linalg.min(r, _transposed_view(a, dtype)), captured)
    assert float(np.asarray(r)) == float(a.min())


def test_scalar_min_and_max_on_infinities_and_nan():
    def reduce(name, v):
        r = einsums.create_zero_tensor(_nm("r"), [])
        getattr(einsums.linalg, name)(r, einsums.asarray(np.array(v)))
        return float(np.asarray(r))

    # An all-infinite input reduces to the infinity, never to a finite seed.
    assert reduce("max", [-INF, -INF]) == -INF
    assert reduce("min", [INF, INF]) == INF
    assert np.isnan(reduce("min", [1.0, NAN, -2.0]))
    assert reduce("min", [3.0, -INF, 2.0]) == -INF


def test_scalar_min_of_an_empty_tensor_is_refused():
    r = einsums.create_zero_tensor("r", [])
    with pytest.raises(ValueError):
        einsums.linalg.min(r, einsums.create_zero_tensor("A", [0]))


def test_tensor_min_method():
    a = np.array([[2.0, -1.0], [0.5, 4.0]])
    assert einsums.asarray(a).min() == -1.0
    assert einsums.asarray(a).max() == 4.0


# ── under passes, in a loop body, and through a save ─────────────────────────


def test_max_min_ops_under_the_default_passes_and_in_a_loop():
    rng = np.random.default_rng(17)
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal((4, 3))
    A, B = einsums.asarray(a), einsums.asarray(b)
    C = einsums.create_zero_tensor("C", [4, 3])
    R = einsums.create_zero_tensor("R", [3])
    g = cg.Graph(_nm("mm_passes"))
    body = g.add_loop("iter", 3, lambda i: i + 1 < 3)
    with cg.capture(body):
        einsums.linalg.maximum(C, A, B)
        einsums.linalg.min_axes(R, C, [0])
    g.apply(cg.default_pass_manager())
    g.execute()
    np.testing.assert_array_equal(np.asarray(C), np.maximum(a, b))
    np.testing.assert_array_equal(np.asarray(R), np.maximum(a, b).min(axis=0))
