# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for TensorUtilities tensor-creation factories.

Verifies the four ``create_*`` free-function bindings:
* ``create_zero_tensor`` — zeros of the requested shape and dtype
* ``create_random_tensor`` — random values in [-1, 1] for reals
* ``create_random_definite`` — symmetric/Hermitian, all eigenvalues > 0
* ``create_random_semidefinite`` — symmetric/Hermitian, all eigenvalues >= 0

Each function is exposed via the codegen dtype-dispatcher (Phase C.5):
the default dtype is float64 and an optional ``dtype="..."`` kwarg picks
between float32/float64/complex64/complex128.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums

DTYPES = ["float32", "float64", "complex64", "complex128"]
NUMPY_DTYPE = {
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


def test_create_zero_tensor_default_is_float64():
    z = einsums.create_zero_tensor("Z", [4, 3])
    arr = np.asarray(z)
    assert arr.shape == (4, 3)
    assert arr.dtype == np.float64
    assert np.all(arr == 0.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_create_zero_tensor_dtype(dtype):
    z = einsums.create_zero_tensor("Z", [3, 3], dtype=dtype)
    arr = np.asarray(z)
    assert arr.dtype == NUMPY_DTYPE[dtype]
    assert np.all(arr == 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_create_random_tensor_dtype_and_range(dtype):
    r = einsums.create_random_tensor("R", [5, 5], dtype=dtype)
    arr = np.asarray(r)
    assert arr.dtype == NUMPY_DTYPE[dtype]
    if np.iscomplexobj(arr):
        # Complex variant clamps within the unit circle.
        assert np.all(np.abs(arr) <= 1.0 + 1e-6)
    else:
        assert arr.min() >= -1.0
        assert arr.max() <= 1.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_create_random_definite_is_positive_definite(dtype):
    # Complex variants currently form ``P^T D P`` (transpose, not conj-transpose)
    # so the result is symmetric rather than Hermitian — eigenvalues are not
    # guaranteed positive. Tracked as a pre-existing CreateRandomDefinite issue;
    # tested here only for real dtypes.
    p = einsums.create_random_definite("P", 4, dtype=dtype)
    arr = np.asarray(p)
    eigs = np.linalg.eigvalsh((arr + arr.T) * 0.5)
    assert (eigs > 0).all(), f"eigenvalues should all be positive, got {eigs}"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_create_random_semidefinite_eigs_nonneg(dtype):
    # See note on test_create_random_definite_is_positive_definite re: complex.
    s = einsums.create_random_semidefinite("S", 5, dtype=dtype)
    arr = np.asarray(s)
    eigs = np.linalg.eigvalsh((arr + arr.T) * 0.5)
    tol = 1e-5 if arr.dtype == np.float32 else 1e-10
    assert (eigs >= -tol).all(), f"eigenvalues must be non-negative, got {eigs}"


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_create_random_definite_complex_returns_correct_dtype(dtype):
    # Complex definiteness is broken (see real-dtype test above) but the binding
    # itself should still produce a square matrix of the requested dtype.
    p = einsums.create_random_definite("P", 4, dtype=dtype)
    arr = np.asarray(p)
    assert arr.shape == (4, 4)
    assert arr.dtype == NUMPY_DTYPE[dtype]


def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match="unsupported dtype"):
        einsums.create_zero_tensor("Z", [2, 2], dtype="bogus")


def test_dtype_aliases_match_canonical():
    # Pick one alias per canonical dtype and verify it dispatches the same way.
    cases = {
        "double": np.float64,
        "f8": np.float64,
        "single": np.float32,
        "f4": np.float32,
        "c16": np.complex128,
        "c8": np.complex64,
    }
    for alias, np_dtype in cases.items():
        z = einsums.create_zero_tensor("Z", [2, 2], dtype=alias)
        assert np.asarray(z).dtype == np_dtype, f"alias {alias!r} dispatched wrong dtype"
