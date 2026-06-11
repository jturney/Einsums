#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Shared helpers for einsums Python tests.

Pure-Python module: importing ``einsums.testing`` must not trigger the
compiled ``einsums._core`` extension to load. Anything in here that
needs a runtime tensor must take it as an argument. The module itself
holds only constants and stateless functions.

Typical usage::

    import numpy as np
    import pytest
    import einsums
    from einsums.testing import ALL_DTYPES, REAL_DTYPES, assert_close

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_axpy_eager(dtype):
        X = einsums.create_random_tensor("X", [4, 3], dtype=dtype)
        Y = einsums.create_random_tensor("Y", [4, 3], dtype=dtype)
        expected = np.asarray(Y) + 2.0 * np.asarray(X)
        einsums.linalg.axpy(2.0, X, Y)
        assert_close(Y, expected)   # dtype inferred from Y / expected
"""
from __future__ import annotations

from typing import Any

import numpy as np


REAL_DTYPES: list[str] = ["float32", "float64"]
COMPLEX_DTYPES: list[str] = ["complex64", "complex128"]
ALL_DTYPES: list[str] = REAL_DTYPES + COMPLEX_DTYPES


# Default relative_tol/absolute_tol per dtype. Picked to match the values that test files
# were already passing by hand: f32/c64 paths lose precision through BLAS,
# f64/c128 paths should be near machine precision. Tests with looser
# expected accuracy, such as eigenvalue ordering and matrix inverse reconstruction,
# override via explicit relative_tol=/absolute_tol= kwargs.
_TOLERANCES: dict[str, tuple[float, float]] = {
    "float32":    (1e-5, 1e-6),
    "float64":    (1e-12, 0.0),
    "complex64":  (1e-5, 1e-6),
    "complex128": (1e-12, 0.0),
}


def tolerance_for(dtype: str) -> tuple[float, float]:
    """Return ``(relative_tol, absolute_tol)`` defaults for ``dtype``.

    Raises ``KeyError`` for unknown dtypes — callers should be using one
    of the names in ``ALL_DTYPES``.
    """
    return _TOLERANCES[dtype]


# f32 and c64 share single-precision mantissa width. f64 and c128 share
# double. The tolerance only cares about the mantissa, so two tiers are
# enough. Whether the type is complex is tracked separately to pick the right key.
_PRECISION_TIER: dict[np.dtype, int] = {
    np.dtype(np.float32):    0,
    np.dtype(np.complex64):  0,
    np.dtype(np.float64):    1,
    np.dtype(np.complex128): 1,
}
_COMPLEX_DTYPES_NP: frozenset[np.dtype] = frozenset(
    {np.dtype(np.complex64), np.dtype(np.complex128)}
)


def _infer_dtype(a: np.ndarray, e: np.ndarray) -> str | None:
    """Pick a dtype key from the inputs' numpy dtypes.

    Returns the least precise tier of the two. ``None`` if neither
    input has a known float/complex dtype (caller falls back to numpy
    defaults).
    """
    ta = _PRECISION_TIER.get(a.dtype)
    te = _PRECISION_TIER.get(e.dtype)
    if ta is None and te is None:
        return None
    tier = min(t for t in (ta, te) if t is not None)
    is_complex = a.dtype in _COMPLEX_DTYPES_NP or e.dtype in _COMPLEX_DTYPES_NP
    if tier == 0:
        return "complex64" if is_complex else "float32"
    return "complex128" if is_complex else "float64"


def assert_close(
    actual: Any,
    expected: Any,
    *,
    dtype: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Thin wrapper over ``np.testing.assert_allclose`` with dtype-aware defaults.

    ``actual`` may be an einsums tensor or anything numpy can coerce. It
    is passed through ``np.asarray`` so test bodies don't have to. The
    dtype key driving the tolerance defaults is inferred from the inputs'
    numpy dtypes (the least precise of the two wins, since that's where
    rounding error was introduced). Pass ``dtype=`` to override the
    inference, and ``rtol`` / ``atol`` to override the tolerance defaults
    per-call.
    """
    a = np.asarray(actual)
    e = np.asarray(expected)
    curr_dtype = dtype
    curr_rtol = rtol
    curr_atol = atol
    if curr_dtype is None:
        curr_dtype = _infer_dtype(a, e)
    if curr_dtype is not None:
        default_rtol, default_atol = tolerance_for(dtype)
        if curr_rtol is None:
            curr_rtol = default_rtol
        if curr_atol is None:
            curr_atol = default_atol
    if curr_rtol is None:
        curr_rtol = 1e-7
    if curr_atol is None:
        curr_atol = 0.0
    np.testing.assert_allclose(a, e, rtol=curr_rtol, atol=curr_atol)


__all__ = [
    "REAL_DTYPES",
    "COMPLEX_DTYPES",
    "ALL_DTYPES",
    "tolerance_for",
    "assert_close",
]
