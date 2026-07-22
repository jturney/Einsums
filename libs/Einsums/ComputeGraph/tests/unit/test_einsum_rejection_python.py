# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Malformed / unsupported einsum specs must be REJECTED, not silently
miscomputed. Mirrors the intent of numpy's ``test_einsum_errors``: an output
label that appears in neither input, ellipsis broadcasting (unsupported by
einsums), and a link index whose extent disagrees between the operands all have
to raise rather than produce a wrong answer. Checked on both the eager
string-dispatch path and the graph capture/execute path.
"""
from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


def _mk(nm, arr):
    t = einsums.create_zero_tensor(nm, list(arr.shape), dtype="float64")
    if arr.size:
        np.asarray(t)[...] = arr
    return t


def _run(mode, spec, C, A, B):
    if mode == "eager":
        einsums.einsum(spec, C, A, B)
    else:
        g = cg.Graph("rej")
        with cg.capture(g):
            einsums.einsum(spec, C, A, B)
        g.execute()


# (label, spec, a_shape, b_shape, c_shape). The C shape is only a plausible
# placeholder - the spec itself is what must be rejected.
_INVALID = [
    # Index-role / extent errors
    ("output label absent from inputs", "ijx <- ij ; jk", (2, 3), (3, 4), (2, 4, 5)),
    ("link extent mismatch",            "ij <- ik ; kj", (2, 3), (4, 5), (2, 5)),
    ("batch extent mismatch",           "ij <- ij ; ij", (2, 3), (2, 4), (2, 3)),
    ("operand rank mismatch",           "ij <- i ; jk",  (2, 3), (3, 4), (2, 4)),
    ("output shape != spec output",     "ij <- ik ; kj", (2, 3), (3, 4), (5, 5)),
    # Malformed spec strings
    ("ellipsis unsupported",            "...ik <- ...ij ; jk", (2, 3, 4), (4, 5), (2, 3, 5)),
    ("stray dots",                      "i.. <- ij ; jk", (2, 3), (3, 4), (2, 4)),
    ("non-letter char '@'",             "i@ <- ij ; jk",  (2, 3), (3, 4), (2, 4)),
    ("non-letter char '$'",             "ij <- i$ ; $j",  (2, 3), (3, 4), (2, 4)),
    ("empty spec",                      "",               (2, 3), (3, 4), (2, 4)),
    ("missing ';' between operands",    "ij <- ik",       (2, 3), (3, 4), (2, 4)),
]


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("label,spec,ash,bsh,csh", _INVALID, ids=[x[0] for x in _INVALID])
def test_invalid_einsum_spec_rejected(label, spec, ash, bsh, csh, mode):
    A = _mk("A", np.ones(ash))
    B = _mk("B", np.ones(bsh))
    C = _mk("C", np.zeros(csh))
    # std::invalid_argument -> ValueError, std::out_of_range -> IndexError,
    # raw throws -> RuntimeError; any of these is an acceptable rejection.
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        _run(mode, spec, C, A, B)


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_repeated_output_index_is_accepted(mode):
    # Unlike numpy (which rejects a repeated output label), einsums treats a
    # repeated OUTPUT index as a diagonal WRITE: "ii <- ij ; ji" fills the
    # diagonal of C. This pins that intentional divergence so it is not
    # "fixed" into a rejection by accident.
    A = _mk("A", np.arange(9.0).reshape(3, 3) + 1.0)
    B = _mk("B", np.arange(9.0).reshape(3, 3) + 1.0)
    C = _mk("C", np.zeros((3, 3)))
    _run(mode, "ii <- ij ; ji", C, A, B)   # must NOT raise
    got = np.asarray(C)
    # Diagonal holds sum_j A[i,j]*B[j,i]; off-diagonal stays zero.
    diag = np.einsum("ij,ji->i", np.asarray(A), np.asarray(B))
    expected = np.diag(diag)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_numbered_indices_are_accepted(mode):
    # einsums allows alphanumeric index names (numbered indices "i1"/"i2"),
    # which the char-validation must NOT reject; only non-alphanumeric garbage
    # ('@', '$', '.') is rejected. Comma-delimited so "i1" is one label.
    A = _mk("A", np.arange(6.0).reshape(2, 3))
    B = _mk("B", np.arange(12.0).reshape(3, 4))
    C = _mk("C", np.zeros((2, 4)))
    _run(mode, "i1,i2 <- i1,i3 ; i3,i2", C, A, B)   # must NOT raise
    np.testing.assert_allclose(np.asarray(C), np.asarray(A) @ np.asarray(B), rtol=1e-12, atol=0.0)
