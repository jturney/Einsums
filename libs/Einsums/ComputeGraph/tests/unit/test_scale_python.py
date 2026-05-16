# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for cg::scale bindings.

Verifies ``einsums.linalg.scale(factor, A)`` multiplies A in place by factor across
all four bound dtypes. Real factors apply to all dtypes; complex factors are
exercised on complex tensors.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums

REAL_DTYPES = ["float32", "float64"]
COMPLEX_DTYPES = ["complex64", "complex128"]
ALL_DTYPES = REAL_DTYPES + COMPLEX_DTYPES


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_scale_by_two(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    before = np.asarray(A).copy()
    einsums.linalg.scale(2.0, A)
    np.testing.assert_allclose(np.asarray(A), 2.0 * before, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_scale_by_zero_zeros_tensor(dtype):
    A = einsums.create_random_tensor("A", [5], dtype=dtype)
    einsums.linalg.scale(0.0, A)
    assert np.all(np.asarray(A) == 0)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_scale_by_one_is_noop(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    before = np.asarray(A).copy()
    einsums.linalg.scale(1.0, A)
    np.testing.assert_array_equal(np.asarray(A), before)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_scale_complex_factor_on_complex_tensor(dtype):
    A = einsums.create_random_tensor("A", [3, 3], dtype=dtype)
    before = np.asarray(A).copy()
    einsums.linalg.scale(1.0 + 2.0j, A)
    np.testing.assert_allclose(np.asarray(A), (1.0 + 2.0j) * before, rtol=1e-5, atol=1e-6)


def test_scale_rank_3():
    A = einsums.create_random_tensor("A", [2, 3, 4])
    before = np.asarray(A).copy()
    einsums.linalg.scale(0.5, A)
    np.testing.assert_allclose(np.asarray(A), 0.5 * before)
