#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""cg.diis: Pulay extrapolation wrapped around a captured graph loop.

The fixture is a Jacobi solve of A x = b captured as a loop body
(r = b - A x; rd = r / diag; x += rd) whose plain iteration converges
linearly with the spectral radius of the off-diagonal part. DIIS must reach
the same fixed point in far fewer replays, writing the extrapolated x back
in place so the captured buffer pointers stay valid.
"""

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
from einsums.testing import assert_close


def _jacobi_problem(n, dtype, seed=5, rho=0.85):
    """A = d*I + R with spectral radius of R/d equal to `rho` (slow Jacobi)."""
    rng = np.random.default_rng(seed)
    d = 3.0
    R = rng.standard_normal((n, n))
    if dtype.startswith("complex"):
        R = R + 1j * rng.standard_normal((n, n))
    R = (R + R.conj().T) / 2
    R *= rho * d / np.abs(np.linalg.eigvalsh(R)).max()
    A_np = d * np.eye(n) + R
    b_np = rng.standard_normal(n)
    if dtype.startswith("complex"):
        b_np = b_np + 1j * rng.standard_normal(n)
    return A_np.astype(dtype), b_np.astype(dtype), d


def _capture_jacobi(A_np, b_np, d, dtype, tol, use_diis, k=8):
    n = len(b_np)
    A = einsums.asarray(np.ascontiguousarray(A_np))
    b = einsums.asarray(np.ascontiguousarray(b_np))
    dv = einsums.asarray(np.full(n, d, dtype=dtype))
    x = einsums.zeros((n,), dtype=dtype)
    r = einsums.zeros((n,), dtype=dtype)
    rd = einsums.zeros((n,), dtype=dtype)
    res = einsums.zeros((1,), dtype="float64" if not dtype.startswith("complex") else dtype)

    g = cg.Graph("jacobi")
    iters = [0]

    def converged(it):
        iters[0] = it + 1
        return float(np.abs(np.asarray(res)[0])) > tol**2

    predicate = cg.diis([(x, rd)], k=k).wrap(converged) if use_diis else converged

    body = g.add_loop("jacobi_iter", 400, predicate)
    with cg.capture(body):
        einsums.einsum("i <- i,j ; j", r, A, x, c_pf=0.0, ab_pf=1.0)
        la.axpby(1.0, b, -1.0, r)                 # r = b - A x
        la.direct_division(1.0, r, dv, 0.0, rd)   # rd = r / diag
        la.axpby(1.0, rd, 1.0, x)                 # x += rd
        la.dot(res, r, r)

    g.execute()
    return np.asarray(x).copy(), iters[0]


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_diis_accelerates_and_matches_direct_solve(dtype):
    A_np, b_np, d = _jacobi_problem(48, dtype)
    x_ref = np.linalg.solve(A_np, b_np)

    x_plain, it_plain = _capture_jacobi(A_np, b_np, d, dtype, 1e-10, use_diis=False)
    x_diis, it_diis = _capture_jacobi(A_np, b_np, d, dtype, 1e-10, use_diis=True)

    # The loop stops at residual 1e-10, so x carries ~1e-10 of leftover error;
    # compare at that scale, not at dtype precision.
    assert_close(einsums.asarray(x_plain), x_ref, atol=1e-8, rtol=0.0)
    assert_close(einsums.asarray(x_diis), x_ref, atol=1e-8, rtol=0.0)
    # rho=0.85 Jacobi needs O(150) iterations; DIIS halves it and then some
    # (146 -> 53 for the float64 seed).
    assert it_plain > 100
    assert it_diis * 2 < it_plain


def test_writeback_is_in_place():
    """Extrapolation must mutate the captured buffer, never rebind it."""
    A_np, b_np, d = _jacobi_problem(16, "float64")
    x = einsums.zeros((16,), dtype="float64")
    rd = einsums.zeros((16,), dtype="float64")
    ptr_before = np.asarray(x).__array_interface__["data"][0]

    acc = cg.diis([(x, rd)], k=4)
    # Drive the accelerator directly: fake a few (t, step) states.
    for i in range(4):
        np.asarray(x)[...] = i + 1.0
        np.asarray(rd)[...] = 1.0 / (i + 1.0)
        acc.step()

    assert np.asarray(x).__array_interface__["data"][0] == ptr_before


def test_diis_argument_validation():
    x = einsums.zeros((4,), dtype="float64")
    rd = einsums.zeros((4,), dtype="float64")
    with pytest.raises(ValueError):
        cg.diis([], k=8)
    with pytest.raises(ValueError):
        cg.diis([(x, rd)], k=1)


def test_wrap_without_condition_runs_to_max_iterations():
    A_np, b_np, d = _jacobi_problem(12, "float64")
    A = einsums.asarray(np.ascontiguousarray(A_np))
    b = einsums.asarray(np.ascontiguousarray(b_np))
    dv = einsums.asarray(np.full(12, d))
    x = einsums.zeros((12,), dtype="float64")
    r = einsums.zeros((12,), dtype="float64")
    rd = einsums.zeros((12,), dtype="float64")

    g = cg.Graph("jacobi_fixed")
    body = g.add_loop("it", 30, cg.diis([(x, rd)]).wrap())
    with cg.capture(body):
        einsums.einsum("i <- i,j ; j", r, A, x, c_pf=0.0, ab_pf=1.0)
        la.axpby(1.0, b, -1.0, r)
        la.direct_division(1.0, r, dv, 0.0, rd)
        la.axpby(1.0, rd, 1.0, x)
    g.execute()

    assert_close(einsums.asarray(np.asarray(x).copy()), np.linalg.solve(A_np, b_np))
