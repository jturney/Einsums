# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for cg::permute bindings.

Verifies ``einsums.permute(spec, C, A, c_pf=, a_pf=)`` against numpy's
``np.einsum``/``np.transpose`` on rank-2 and rank-3 tensors with default
prefactors and explicit kwargs across all four bound dtypes.

The bound spec syntax is ``"<output> <- <input>"`` (e.g. ``"ji <- ij"`` for
a transpose).
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
from einsums.testing import ALL_DTYPES, COMPLEX_DTYPES, assert_close


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_transpose_2d(dtype):
    A = einsums.create_random_tensor("A", [3, 5], dtype=dtype)
    C = einsums.create_zero_tensor("C", [5, 3], dtype=dtype)

    einsums.permute("ji <- ij", C, A)

    assert_close(C, np.asarray(A).T)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_identity_permute(dtype):
    A = einsums.create_random_tensor("A", [4, 4], dtype=dtype)
    C = einsums.create_zero_tensor("C", [4, 4], dtype=dtype)

    einsums.permute("ij <- ij", C, A)

    np.testing.assert_array_equal(np.asarray(C), np.asarray(A))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_rank3_permute(dtype):
    A = einsums.create_random_tensor("A", [2, 3, 4], dtype=dtype)
    C = einsums.create_zero_tensor("C", [4, 3, 2], dtype=dtype)

    einsums.permute("kji <- ijk", C, A)

    expected = np.transpose(np.asarray(A), (2, 1, 0))
    assert_close(C, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_default_prefactors_overwrite_C(dtype):
    A = einsums.create_random_tensor("A", [3, 5], dtype=dtype)
    C = einsums.create_random_tensor("C", [5, 3], dtype=dtype)  # garbage values

    einsums.permute("ji <- ij", C, A)  # c_pf=0, a_pf=1 by default

    assert_close(C, np.asarray(A).T)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_kwargs_accumulate_and_scale(dtype):
    A = einsums.create_random_tensor("A", [3, 5], dtype=dtype)
    C = einsums.create_zero_tensor("C", [5, 3], dtype=dtype)
    np.asarray(C)[...] = 10.0

    einsums.permute("ji <- ij", C, A, c_pf=1.0, a_pf=2.0)

    expected = 10.0 + 2.0 * np.asarray(A).T
    assert_close(C, expected)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_complex_prefactors(dtype):
    A = einsums.create_random_tensor("A", [3, 4], dtype=dtype)
    C = einsums.create_zero_tensor("C", [4, 3], dtype=dtype)

    einsums.permute("ji <- ij", C, A, c_pf=0.0 + 0.0j, a_pf=1.0 + 1.0j)

    expected = (1.0 + 1.0j) * np.asarray(A).T
    assert_close(C, expected)


def test_multi_char_indices_2d():
    # Multi-char tokens are comma-separated (single-char would otherwise
    # ambiguously parse "mu" as two indices m and u).
    A = einsums.create_random_tensor("A", [4, 5])
    C = einsums.create_zero_tensor("C", [5, 4])

    einsums.permute("nu,mu <- mu,nu", C, A)

    np.testing.assert_allclose(np.asarray(C), np.asarray(A).T)


def test_multi_char_indices_rank3():
    A = einsums.create_random_tensor("A", [2, 3, 4])
    C = einsums.create_zero_tensor("C", [4, 3, 2])

    einsums.permute("sigma,nu,mu <- mu,nu,sigma", C, A)

    expected = np.transpose(np.asarray(A), (2, 1, 0))
    np.testing.assert_allclose(np.asarray(C), expected)


def test_mixed_index_lengths():
    # Single-char and multi-char tokens in the same spec.
    A = einsums.create_random_tensor("A", [3, 5])
    C = einsums.create_zero_tensor("C", [5, 3])

    einsums.permute("alpha,i <- i,alpha", C, A)

    np.testing.assert_allclose(np.asarray(C), np.asarray(A).T)
