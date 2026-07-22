#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Shared helpers for einsums Python tests.

Pure-Python module: importing ``einsums.testing`` must NOT trigger the
compiled ``einsums._core`` extension to load. Anything in here that
needs a runtime tensor must take it as an argument; the module itself
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

# Dtypes whose mantissa keeps integer-valued contractions exact at the sizes the
# differential fuzzers use (see :func:`integer_data` / :func:`assert_exact`).
# float32/complex64 are excluded: their 24-bit mantissa (2**24) is too small once
# extents grow, so a large-but-valid contraction would round and break exactness.
EXACT_DTYPES: list[str] = ["float64", "complex128"]

# Prefactors that keep an integer-valued contraction integer-valued (no 0.5 or
# fractional/complex scales). Covers no-accumulate (0), plain add (1), and a
# sign-and-scale (-2) so the accumulate/sign structure is still exercised.
EXACT_PREFACTORS: list[float] = [0.0, 1.0, -2.0]


# Default rtol/atol per dtype. Picked to match the values that test files
# were already passing by hand: f32/c64 paths lose precision through BLAS,
# f64/c128 paths should be near machine precision. Tests with looser
# expected accuracy (eigenvalue ordering, matrix inverse reconstruction)
# override via explicit rtol=/atol= kwargs.
_TOLERANCES: dict[str, tuple[float, float]] = {
    "float32":    (1e-5, 1e-6),
    "float64":    (1e-12, 0.0),
    "complex64":  (1e-5, 1e-6),
    "complex128": (1e-12, 0.0),
}


def tolerance_for(dtype: str) -> tuple[float, float]:
    """Return the default tolerance for a dtype.

    :param dtype: A dtype name from :data:`ALL_DTYPES`.
    :returns: The ``(rtol, atol)`` pair used as the default tolerance for that dtype.
    :raises KeyError: If ``dtype`` is not a known name.
    """
    return _TOLERANCES[dtype]


# f32 and c64 share single-precision mantissa width; f64 and c128 share
# double. The tolerance only cares about the mantissa, so two tiers are
# enough — complex-ness is tracked separately to pick the right key.
_PRECISION_TIER: dict[np.dtype, int] = {
    np.dtype(np.float32):    0,
    np.dtype(np.complex64):  0,
    np.dtype(np.float64):    1,
    np.dtype(np.complex128): 1,
}
_COMPLEX_DTYPES_NP: frozenset[np.dtype] = frozenset(
    {np.dtype(np.complex64), np.dtype(np.complex128)}
)


def _infer_dtype(actual: np.ndarray, expected: np.ndarray) -> str | None:
    """Choose a tolerance dtype key from two arrays.

    The less precise of the two tiers wins. That is where rounding was
    introduced, so its looser tolerance is the correct one to apply.

    :param actual: The array produced by the code under test.
    :param expected: The reference array to compare against.
    :returns: A dtype name from :data:`ALL_DTYPES`, or ``None`` when neither
        array has a known float or complex dtype. A ``None`` result tells the
        caller to fall back to numpy's own defaults.
    """
    ta = _PRECISION_TIER.get(actual.dtype)
    te = _PRECISION_TIER.get(expected.dtype)
    if ta is None and te is None:
        return None
    tier = min(t for t in (ta, te) if t is not None)
    is_complex = actual.dtype in _COMPLEX_DTYPES_NP or expected.dtype in _COMPLEX_DTYPES_NP
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
    """Compare two arrays with dtype-aware tolerance defaults.

    This wraps :func:`numpy.testing.assert_allclose`. Both arguments pass
    through :func:`numpy.asarray`, so ``actual`` may be an einsums tensor or
    any array-like object and test bodies do not have to convert it. When
    ``dtype`` is not given it is inferred from the two arrays, and the less
    precise of the two sets the tolerance.

    :param actual: The value produced by the code under test.
    :param expected: The reference value to compare against.
    :param dtype: A dtype name from :data:`ALL_DTYPES` that selects the
        default tolerance. Inferred from the inputs when ``None``.
    :param rtol: Relative tolerance. Overrides the dtype default when given.
    :param atol: Absolute tolerance. Overrides the dtype default when given.
    :raises AssertionError: If the arrays are not equal within tolerance.
    """
    a = np.asarray(actual)
    e = np.asarray(expected)
    if dtype is None:
        dtype = _infer_dtype(a, e)
    if dtype is not None:
        default_rtol, default_atol = tolerance_for(dtype)
        if rtol is None:
            rtol = default_rtol
        if atol is None:
            atol = default_atol
    if rtol is None:
        rtol = 1e-7
    if atol is None:
        atol = 0.0
    np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)


def integer_data(shape: Any, dtype: str, rng: Any, radius: int = 4) -> np.ndarray:
    """Integer-valued sample data for exact (rounding-free) differential tests.

    Draws integers in ``[-radius, radius]`` (complex dtypes get an integer real
    part and an integer imaginary part) as the requested floating dtype. A
    contraction over such data is bit-exact in float64/complex128 as long as no
    intermediate magnitude exceeds ``2**53``, so the result can be compared with
    :func:`assert_exact` instead of a tolerance -- turning a dropped axis or a
    transposed tail block into a guaranteed mismatch rather than a small error a
    relative tolerance can hide. This mirrors the integer-dtype battery in
    numpy's own einsum tests, which einsums cannot use directly (it has no
    integer tensor dtype).

    :param shape: The output shape (any iterable of non-negative ints).
    :param dtype: A floating dtype name from :data:`EXACT_DTYPES`.
    :param rng: A :class:`numpy.random.Generator`.
    :param radius: Values are drawn from ``[-radius, radius]`` inclusive.
    :returns: A :class:`numpy.ndarray` of ``dtype`` holding integer values.
    """
    dt = np.dtype(dtype)
    sz = tuple(shape)

    def ints() -> np.ndarray:
        return rng.integers(-radius, radius + 1, size=sz).astype(np.float64)

    if dt.kind == "c":
        return (ints() + 1j * ints()).astype(dt)
    return ints().astype(dt)


def assert_exact(actual: Any, expected: Any) -> None:
    """Assert two integer-valued arrays are bit-for-bit equal.

    For use with :func:`integer_data`. First verifies ``expected`` is
    integer-valued: if a contraction ever overflowed the ``2**53`` exact-integer
    range the reference would carry a fractional part, so this raises a clear
    design error (reduce the :func:`integer_data` ``radius`` or the extents)
    instead of a misleading value mismatch. Otherwise asserts exact equality,
    which every kernel accumulation order must satisfy for rounding-free integer
    inputs.

    :param actual: The value produced by the code under test (array-like or an
        einsums tensor; passed through :func:`numpy.asarray`).
    :param expected: The reference value to compare against.
    :raises AssertionError: If ``expected`` is not integer-valued, or if the two
        arrays are not exactly equal.
    """
    a = np.asarray(actual)
    e = np.asarray(expected)
    rounded = np.rint(e.real)
    if np.iscomplexobj(e):
        rounded = rounded + 1j * np.rint(e.imag)
    if not np.array_equal(e, rounded):
        raise AssertionError(
            "assert_exact: reference is not integer-valued -- the contraction exceeded the "
            "2**53 exact range; reduce the integer_data radius or the extents"
        )
    np.testing.assert_array_equal(a, e)


__all__ = [
    "REAL_DTYPES",
    "COMPLEX_DTYPES",
    "ALL_DTYPES",
    "EXACT_DTYPES",
    "EXACT_PREFACTORS",
    "tolerance_for",
    "assert_close",
    "integer_data",
    "assert_exact",
]
