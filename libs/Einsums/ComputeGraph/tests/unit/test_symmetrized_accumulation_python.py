# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Differential for the SymmetrizedAccumulation pass on the CCSD symmetrization
idiom r2 += s*(tmp + P(tmp)), P = swap (i<->j) and (a<->b).

Runtime tensors (the workload path), captured as einsum->tmp, axpby(tmp->r2),
permute(jiba<-ijab), axpby(tmpP->r2). Checked two ways: the pass applied in
isolation (must fold, num_rewritten == 1) and through the wired default pipeline
(cg.default_pass_manager() now includes it) - both compared to a numpy oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _oracle(A_np, B_np, s):
    # tmp[i,j,a,b] = A[i,a] B[j,b]; P swaps i<->j and a<->b.
    tmp = np.einsum("ia,jb->ijab", A_np, B_np)
    P_tmp = np.transpose(tmp, (1, 0, 3, 2))
    return s * (tmp + P_tmp)


def _capture_symacc(A, B, r2, tmp, tmpP, s):
    g = cg.Graph("symacc")
    with cg.capture(g):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(s, tmp, 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(s, tmpP, 1.0, r2)
    return g


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_pass_folds_and_matches_numpy(dtype):
    o, v = 2, 3
    rng = np.random.default_rng(0)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))
    s = 0.5

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    g = _capture_symacc(A, B, r2, tmp, tmpP, s)

    pm = cg.PassManager()
    pass_obj = cg.SymmetrizedAccumulation()
    pm.add(pass_obj)
    modified = g.apply(pm)

    assert modified
    assert pass_obj.num_matched == 1     # APIARY getters are properties
    assert pass_obj.num_rewritten == 1

    g.execute()
    assert_close(r2, _oracle(A_np, B_np, s))


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_default_pipeline_stays_correct(dtype):
    # The pass is wired into create_default; the full default pipeline must fold
    # the pattern and still produce the right answer.
    o, v = 2, 3
    rng = np.random.default_rng(1)
    A_np = rng.standard_normal((o, v)).astype(dtype)
    B_np = rng.standard_normal((o, v)).astype(dtype)
    if dtype.startswith("complex"):
        A_np = A_np + 1j * rng.standard_normal((o, v))
        B_np = B_np + 1j * rng.standard_normal((o, v))
    s = -2.0

    A = einsums.asarray(A_np)
    B = einsums.asarray(B_np)
    r2 = einsums.zeros((o, o, v, v), dtype=dtype)
    tmp = einsums.zeros((o, o, v, v), dtype=dtype)
    tmpP = einsums.zeros((o, o, v, v), dtype=dtype)

    g = _capture_symacc(A, B, r2, tmp, tmpP, s)
    g.apply(cg.default_pass_manager())
    g.execute()

    assert_close(r2, _oracle(A_np, B_np, s))
