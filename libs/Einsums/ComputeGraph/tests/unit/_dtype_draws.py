#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Dtype draws and rounding-aware comparison for the hypothesis differential fuzzers.

A fuzzer draws one of ``einsums.testing.ALL_DTYPES`` per example and builds its
operands with :func:`random_array`, which rounds the sample data to that dtype
and hands it back in double precision. The einsums tensor then holds exactly the
values the oracle computes with, and the oracle runs in double precision, so the
only difference left between the two sides is the rounding of the kernel under
test. Drawing the dtype costs no examples: a fuzzer that sampled float64 and
complex128 samples four dtypes from the same budget.

:func:`assert_rounding_close` judges that difference against the dtype's
default tolerance from :func:`einsums.testing.tolerance_for`, applied to the
size of the terms each element was computed from rather than to the element
alone. An element that cancelled toward zero carries the absolute error of its
terms, and comparing it by its own small value asks the kernel for digits no
summation order can give.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import strategies as st

from einsums.testing import ALL_DTYPES, COMPLEX_DTYPES, REAL_DTYPES, tolerance_for

DTYPES = st.sampled_from(ALL_DTYPES)
REAL = st.sampled_from(REAL_DTYPES)
COMPLEX = st.sampled_from(COMPLEX_DTYPES)


def is_complex(dtype: str) -> bool:
    return dtype in COMPLEX_DTYPES


def wide(dtype: str) -> str:
    """The double-precision dtype of the same kind, which the oracle computes in."""
    return "complex128" if is_complex(dtype) else "float64"


def real_of(dtype: str) -> str:
    """The real dtype of the same precision, the type of eigenvalues and singular values."""
    return {"float32": "float32", "complex64": "float32"}.get(dtype, "float64")


def random_array(shape: Any, dtype: str, rng: np.random.Generator) -> np.ndarray:
    """Standard normal sample data rounded to ``dtype`` and returned in double precision.

    Draws the same generator values for every dtype of a kind, so a failing
    example replays with the same shapes and the same data at another precision.
    """
    x = rng.standard_normal(shape)
    if is_complex(dtype):
        x = x + 1j * rng.standard_normal(shape)
    return np.asarray(x).astype(dtype).astype(wide(dtype))


def rounded(x: Any, dtype: str) -> np.ndarray:
    """``x`` rounded to ``dtype`` and returned in double precision."""
    return np.asarray(x).astype(dtype).astype(wide(dtype))


def assert_rounding_close(actual: Any, expected: Any, dtype: str, scale: Any, err_msg: str = "", factor: float = 1.0) -> None:
    """Assert ``actual`` matches the double-precision ``expected`` up to the rounding of ``dtype``.

    Each element may differ by ``rtol * max(scale, |expected|) + atol``, where
    ``(rtol, atol)`` is :func:`einsums.testing.tolerance_for` of ``dtype``.

    :param actual: The value the code under test produced (anything numpy.asarray takes).
    :param expected: The oracle, computed in double precision from the rounded operands.
    :param dtype: The dtype the code under test computed in.
    :param scale: The magnitude of the terms behind each element, before any
        cancellation: for a contraction, the same contraction over absolute
        values. Broadcast against ``expected``, so a scalar works for a
        normwise bound.
    :param err_msg: Appended to the failure message.
    :param factor: Multiplies ``rtol``, for a result whose error grows with a
        condition number the caller bounds; 1 for plain arithmetic.
    """
    rtol, atol = tolerance_for(dtype)
    rtol *= factor
    a = np.asarray(actual).astype(wide(dtype))
    e = np.asarray(expected)
    bound = rtol * np.maximum(np.abs(np.asarray(scale, dtype=float)), np.abs(e)) + atol
    err = np.abs(a - e)
    bad = ~(err <= bound)
    if np.any(bad):
        worst = np.unravel_index(np.argmax(np.where(bad, err / bound, 0.0)), err.shape) if err.ndim else ()
        raise AssertionError(
            f"{int(np.count_nonzero(bad))} of {err.size} elements differ beyond the rounding of {dtype} "
            f"(rtol={rtol:g} of the term scale, atol={atol:g}); worst at {worst}: actual={a[worst]!r} "
            f"expected={e[worst]!r} error={err[worst]:.3g} bound={bound[worst] if np.ndim(bound) else bound:.3g}. {err_msg}")
