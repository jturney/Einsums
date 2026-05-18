# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for cg::einsum bindings produced by einsums-pybind.

Verifies the headline Phase C deliverable: ``einsums.einsum(spec, C, A, B)``
on RuntimeTensor-backed objects produces results matching numpy.einsum.

The bound spec syntax uses ``<-`` and ``;`` (e.g. ``"ij <- ik ; kj"``) rather
than numpy's ``"ik,kj->ij"``. Each test pairs the einsums spec with the
equivalent numpy spec for cross-checking.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
from einsums.testing import ALL_DTYPES, REAL_DTYPES, assert_close


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_matmul(dtype):
    A = einsums.create_random_tensor("A", [4, 5], dtype=dtype)
    B = einsums.create_random_tensor("B", [5, 3], dtype=dtype)
    C = einsums.create_zero_tensor("C", [4, 3], dtype=dtype)

    einsums.einsum("ij <- ik ; kj", C, A, B)

    expected = np.einsum("ik,kj->ij", np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_outer_product(dtype):
    A = einsums.create_random_tensor("A", [3], dtype=dtype)
    B = einsums.create_random_tensor("B", [4], dtype=dtype)
    C = einsums.create_zero_tensor("C", [3, 4], dtype=dtype)

    einsums.einsum("ij <- i ; j", C, A, B)

    expected = np.einsum("i,j->ij", np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_batched_matmul(dtype):
    A = einsums.create_random_tensor("A", [2, 3, 4], dtype=dtype)
    B = einsums.create_random_tensor("B", [2, 4, 5], dtype=dtype)
    C = einsums.create_zero_tensor("C", [2, 3, 5], dtype=dtype)

    einsums.einsum("bij <- bik ; bkj", C, A, B)

    expected = np.einsum("bik,bkj->bij", np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5, atol=1e-6)


def test_invalid_spec_raises():
    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    with pytest.raises(ValueError):
        einsums.einsum("ik,kj->ij", C, A, B)  # numpy-style ',' instead of ';'


def test_multi_char_indices_matmul():
    # Multi-char tokens (e.g. "mu", "nu", "rho") are comma-separated.
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [4, 5])
    C = einsums.create_zero_tensor("C", [3, 5])

    einsums.einsum("mu,nu <- mu,rho ; rho,nu", C, A, B)

    expected = np.einsum("ik,kj->ij", np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5, atol=1e-6)


def test_multi_char_indices_batched():
    # Mix multi-char and single-char tokens in a batched contraction.
    A = einsums.create_random_tensor("A", [2, 3, 4])
    B = einsums.create_random_tensor("B", [2, 4, 5])
    C = einsums.create_zero_tensor("C", [2, 3, 5])

    einsums.einsum("batch,i,j <- batch,i,k ; batch,k,j", C, A, B)

    expected = np.einsum("bik,bkj->bij", np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(np.asarray(C), expected, rtol=1e-5, atol=1e-6)
