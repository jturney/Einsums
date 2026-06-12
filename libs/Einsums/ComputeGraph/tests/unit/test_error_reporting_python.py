# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Error-path coverage for the ComputeGraph Python bindings.

Bad inputs should raise clean exceptions with informative messages rather
than crash, hang, or produce silent garbage. Each test feeds a deliberately
invalid input and asserts a Python-side exception was raised. We don't
overspecify the exception type, pybind11 wraps C++ throws as Python
``RuntimeError`` / ``ValueError`` / ``TypeError`` depending on the cast,
and the bindings can evolve. The contract is: *something* raises.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# Rank guards on runtime-rank operations
# ──────────────────────────────────────────────────────────────────────────


def test_trace_on_non_square_raises():
    """``trace`` requires a square matrix."""
    A = einsums.create_random_tensor("A", [3, 4])
    with pytest.raises(Exception):
        einsums.linalg.trace(A)


def test_gemm_on_rank3_raises():
    """``gemm`` requires all three tensors to be rank-2."""
    A = einsums.create_random_tensor("A", [2, 3, 4])
    B = einsums.create_random_tensor("B", [4, 3])
    C = einsums.create_zero_tensor("C", [2, 3])
    with pytest.raises(Exception):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)


def test_gemv_on_rank3_a_raises():
    """``gemv`` requires A rank-2."""
    A = einsums.create_random_tensor("A", [2, 3, 4])
    x = einsums.create_random_tensor("x", [4])
    y = einsums.create_zero_tensor("y", [2])
    with pytest.raises(Exception):
        einsums.linalg.gemv(1.0, A, x, 0.0, y)


def test_gemv_on_rank2_x_raises():
    """``gemv`` requires x rank-1."""
    A = einsums.create_random_tensor("A", [3, 3])
    x = einsums.create_random_tensor("x", [3, 3])
    y = einsums.create_zero_tensor("y", [3])
    with pytest.raises(Exception):
        einsums.linalg.gemv(1.0, A, x, 0.0, y)


def test_ger_on_rank2_x_raises():
    """``ger`` requires X/Y rank-1."""
    X = einsums.create_random_tensor("X", [3, 3])
    Y = einsums.create_random_tensor("Y", [3])
    A = einsums.create_zero_tensor("A", [3, 3])
    with pytest.raises(Exception):
        einsums.linalg.ger(1.0, X, Y, A)


def test_invert_on_rank3_raises():
    """``invert`` requires a rank-2 input."""
    A = einsums.create_random_tensor("A", [2, 3, 4])
    with pytest.raises(Exception):
        einsums.linalg.invert(A)


def test_gesv_on_rank3_a_raises():
    """``gesv`` requires A rank-2."""
    A = einsums.create_random_tensor("A", [2, 3, 3])
    B = einsums.create_random_tensor("B", [3, 1])
    with pytest.raises(Exception):
        einsums.linalg.gesv(A, B)


# ──────────────────────────────────────────────────────────────────────────
# Returning-form ops throw during graph capture
# (already covered op-by-op in test_lapack_python.py; this one checks a
#  representative cluster in one place for documentation purposes.)
# ──────────────────────────────────────────────────────────────────────────


def test_returning_ops_uniformly_raise_during_capture():
    """det / svd / qr / trace / syev_eig all throw inside ``with cg.capture``."""
    A = einsums.create_random_tensor("A", [3, 3])
    arr = np.asarray(A)
    arr[:] = (arr + arr.T) / 2.0

    callables = [
        ("det", lambda: einsums.linalg.det(A)),
        ("svd", lambda: einsums.linalg.svd(A)),
        ("qr", lambda: einsums.linalg.qr(A)),
        ("trace", lambda: einsums.linalg.trace(A)),
        ("syev_eig", lambda: einsums.linalg.syev_eig(A, compute_eigenvectors=True)),
    ]
    for name, op in callables:
        g = cg.Graph(f"err-{name}")
        with cg.capture(g):
            with pytest.raises(Exception):
                op()


# ──────────────────────────────────────────────────────────────────────────
# einsum / permute spec parsing
# ──────────────────────────────────────────────────────────────────────────


def test_einsum_with_invalid_spec_raises():
    """Garbled spec strings should raise rather than silently misbehave."""
    A = einsums.create_random_tensor("A", [3, 4])
    B = einsums.create_random_tensor("B", [4, 3])
    C = einsums.create_zero_tensor("C", [3, 3])
    with pytest.raises(Exception):
        einsums.einsum("not a valid einsum spec", C, A, B)


def test_permute_with_wrong_rank_indices_raises():
    """Index counts in the spec must match the tensor ranks."""
    A = einsums.create_random_tensor("A", [3, 4])  # rank-2
    C = einsums.create_zero_tensor("C", [4, 3])
    with pytest.raises(Exception):
        # Spec says rank-3 permute but tensors are rank-2.
        einsums.permute("ijk <- kji", C, A)


# ──────────────────────────────────────────────────────────────────────────
# Tensor creation
# ──────────────────────────────────────────────────────────────────────────


def test_create_tensor_rejects_unknown_dtype():
    """A bogus dtype string falls through the dispatcher and should raise."""
    with pytest.raises(Exception):
        einsums.create_zero_tensor("bad", [3, 3], dtype="not_a_real_dtype")
