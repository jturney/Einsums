# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""scan, cumsum, cumprod, cummax and cummin, against numpy.

numpy is the oracle: np.cumsum, np.cumprod, np.maximum.accumulate and
np.minimum.accumulate, with exclusive and reverse built from them by shifting
and flipping. Inputs go through einsums.asarray views so the column-major
layout and numpy's indices line up.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES, assert_close, ordered_extremum_accumulate

_ctr = itertools.count()
NAN = np.nan
INF = np.inf

_NUMPY = {
    "sum": (np.cumsum, 0.0),
    "prod": (np.cumprod, 1.0),
    # Not numpy.maximum.accumulate: it leaves a zero tie to the platform, and einsums orders them.
    "max": (lambda x, axis: ordered_extremum_accumulate("max", x, axis), -INF),
    "min": (lambda x, axis: ordered_extremum_accumulate("min", x, axis), INF),
}


def _nm(prefix: str) -> str:
    return f"{prefix}{next(_ctr)}"


def _oracle(x: np.ndarray, axis: int, op: str, exclusive: bool, reverse: bool) -> np.ndarray:
    fn, identity = _NUMPY[op]
    y = np.flip(x, axis) if reverse else x
    out = fn(y, axis=axis)
    if exclusive:
        out = np.concatenate([np.full_like(np.take(out, [0], axis=axis), identity), np.take(out, range(out.shape[axis] - 1), axis=axis)],
                             axis=axis)
    return np.flip(out, axis) if reverse else out


def _transposed_view(x: np.ndarray, dtype: str):
    parent = einsums.asarray(np.ascontiguousarray(x.T), dtype=dtype)
    return parent.permute_view(list(reversed(range(x.ndim))))


def _ops_for(dtype: str):
    return ["sum", "prod"] if dtype.startswith("complex") else ["sum", "prod", "max", "min"]


def _cases():
    for dtype in ALL_DTYPES:
        for op in _ops_for(dtype):
            yield dtype, op


@pytest.mark.parametrize("dtype,op", list(_cases()))
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_scan_matches_numpy(dtype, op, axis, exclusive, reverse, captured):
    rng = np.random.default_rng(7)
    x = (1.0 + 0.25 * rng.standard_normal((3, 4, 5))).astype(dtype)
    if dtype.startswith("complex"):
        x = (x + 0.1j * rng.standard_normal((3, 4, 5))).astype(dtype)
    A = _transposed_view(x, dtype)
    out = einsums.create_zero_tensor(_nm("s"), [3, 4, 5], dtype=dtype)
    if captured:
        g = cg.Graph(_nm("scan"))
        with cg.capture(g):
            einsums.linalg.scan(out, A, axis, op, exclusive=exclusive, reverse=reverse)
        g.execute()
        g.execute()  # an overwrite replays to the same answer
    else:
        einsums.linalg.scan(out, A, axis, op, exclusive=exclusive, reverse=reverse)
    assert_close(out, _oracle(x, axis, op, exclusive, reverse))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_named_wrappers(dtype):
    rng = np.random.default_rng(8)
    x = (1.0 + 0.25 * rng.standard_normal((6, 3))).astype(dtype)
    A = einsums.asarray(x, dtype=dtype)
    out = einsums.create_zero_tensor(_nm("w"), [6, 3], dtype=dtype)
    einsums.linalg.cumsum(out, A, 0)
    assert_close(out, np.cumsum(x, axis=0))
    einsums.linalg.cumprod(out, A, 1)
    assert_close(out, np.cumprod(x, axis=1))
    if dtype in REAL_DTYPES:
        einsums.linalg.cummax(out, A, 0)
        np.testing.assert_array_equal(np.asarray(out), np.maximum.accumulate(x, axis=0))
        einsums.linalg.cummin(out, A, 1)
        np.testing.assert_array_equal(np.asarray(out), np.minimum.accumulate(x, axis=1))


@pytest.mark.parametrize("op", ["max", "min"])
def test_extremum_scans_propagate_nan_and_order_signed_zeros(op):
    x = np.array([-0.0, 0.0, 2.0, NAN, 5.0, -1.0])
    out = einsums.create_zero_tensor(_nm("n"), [6])
    einsums.linalg.scan(out, einsums.asarray(x), 0, op)
    expected = _NUMPY[op][0](x, axis=0)
    got = np.asarray(out)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.signbit(got[:3]), np.signbit(expected[:3]))


def test_all_nan_line():
    x = np.full((4, 2), NAN)
    out = einsums.create_zero_tensor(_nm("n"), [4, 2])
    for op in ("sum", "prod", "max", "min"):
        einsums.linalg.scan(out, einsums.asarray(x), 0, op)
        assert np.all(np.isnan(np.asarray(out))), op


def test_in_place_scan():
    x = np.arange(1.0, 7.0).reshape(2, 3)
    A = einsums.asarray(x)
    g = cg.Graph(_nm("inplace"))
    with cg.capture(g):
        einsums.linalg.cumsum(A, A, 1)
    g.execute()
    np.testing.assert_array_equal(np.asarray(A), np.cumsum(x, axis=1))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
def test_zero_extents_are_a_no_op(shape):
    A = einsums.create_zero_tensor("A", list(shape))
    out = einsums.create_zero_tensor("out", list(shape))
    for axis in (0, 1):
        einsums.linalg.cumsum(out, A, axis)


def test_rejections():
    A = einsums.create_zero_tensor("A", [2, 3])
    out = einsums.create_zero_tensor("out", [2, 3])
    with pytest.raises(IndexError):
        einsums.linalg.scan(out, A, 2, "sum")
    with pytest.raises(ValueError, match="sum, prod, max, min"):
        einsums.linalg.scan(out, A, 0, "mean")
    with pytest.raises(Exception, match="extent"):
        einsums.linalg.cumsum(einsums.create_zero_tensor("w", [3, 2]), A, 0)
    Z = einsums.create_zero_tensor("Z", [2, 3], dtype="complex128")
    Zout = einsums.create_zero_tensor("Zout", [2, 3], dtype="complex128")
    with pytest.raises(ValueError, match="real"):
        einsums.linalg.scan(Zout, Z, 0, "max")
    with pytest.raises(TypeError):
        einsums.linalg.cummin(Zout, Z, 0)


def test_scan_under_the_default_passes_in_a_loop():
    """A cumulative return: log returns summed along time, then the running peak."""
    rng = np.random.default_rng(9)
    r = 0.01 * rng.standard_normal((5, 50))  # 5 paths, 50 steps
    R = einsums.asarray(r)
    cum = einsums.create_zero_tensor("cum", [5, 50])
    peak = einsums.create_zero_tensor("peak", [5, 50])
    g = cg.Graph(_nm("drawdown"))
    body = g.add_loop("iter", 2, lambda i: i + 1 < 2)
    with cg.capture(body):
        einsums.linalg.cumsum(cum, R, 1)
        einsums.linalg.cummax(peak, cum, 1)
    g.apply(cg.default_pass_manager())
    g.execute()
    assert_close(cum, np.cumsum(r, axis=1))
    assert_close(peak, np.maximum.accumulate(np.cumsum(r, axis=1), axis=1))
