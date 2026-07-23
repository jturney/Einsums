# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Fixed differential regressions over a small square pool (RMW / loop / view / einsum / complex-prefactor shapes).

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

from _fuzz_diff_common import *  # shared fuzz/differential harness


# ──────────────────────────────────────────────────────────────────────────
# Fixed regression programs (index a small square pool).
# ──────────────────────────────────────────────────────────────────────────


def test_regression_mutable_reuse_chain():
    prog = [
        ("axpby", 1.0, 0, 0.0, 1),
        ("axpby", 1.0, 0, 0.0, 2),
        ("scale", 0.5, 1),
        ("axpy", 2.0, 1, 2),
        ("gemm", 1.0, 1, 2, 1.0, 3),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(123)), "reuse_chain")


def test_regression_nested_loop():
    prog = [("loop", 3, [("scale", 0.9, 0), ("loop", 2, [("axpy", 0.5, 0, 1), ("gemm", 1.0, 0, 1, 1.0, 2)])])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(456)), "nested_loop")


def test_regression_sequential_loops():
    prog = [
        ("loop", 3, [("scale", 0.8, 0), ("axpy", 1.0, 0, 1)]),
        ("loop", 2, [("scale", 1.1, 1), ("axpby", 0.5, 1, 0.5, 2)]),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(789)), "seq_loops")


def test_regression_loop_with_conditional():
    prog_then = [("loop", 2, [("cond", True, [("axpy", 0.5, 0, 1)], [("scale", 0.0, 1)]), ("gemm", 1.0, 0, 1, 0.0, 2)])]
    prog_else = [("loop", 2, [("cond", False, [("axpy", 0.5, 0, 1)], [("scale", 2.0, 1)]), ("gemm", 1.0, 0, 1, 0.0, 2)])]
    m, v, t = _square_seed_arrays(np.random.default_rng(1011))
    check_program(prog_then, m, v, t, "loop_cond_then")
    check_program(prog_else, m, v, t, "loop_cond_else")


def test_regression_accumulating_gemm_in_loop():
    prog = [("loop", 2, [("gemm", -0.8137, 3, 3, 1.0, 2), ("gemm", -0.1093, 1, 0, 1.0, 3)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(5026)), "accum_gemm_loop")


def test_regression_einsum_accumulate_in_loop():
    prog = [("loop", 3, [("einsum", "ij <- ik ; kj", 1.0, 0, 1, 1.0, 2)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(2024)), "einsum_accum_loop")


def test_regression_einsum_transpose_patterns():
    prog = [
        ("einsum", "ij <- ik ; kj", 1.0, 0, 1, 0.0, 2),
        ("einsum", "ij <- ki ; kj", 1.0, 0, 1, 0.0, 3),
        ("einsum", "ij <- ik ; jk", 0.5, 2, 3, 0.0, 0),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(333)), "einsum_patterns")


def test_regression_permute_accumulate_in_loop():
    prog = [("perm", 1.0, 0.0, 0, 1), ("loop", 3, [("perm", 0.5, 1.0, 0, 1)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(666)), "perm_accum")


def test_regression_symm_gemm():
    prog = [("loop", 2, [("symm", 0, 1, 2), ("scale", 0.5, 3)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(444)), "symm")


def test_regression_gemv_ger_in_loop():
    prog = [("loop", 3, [("gemv", 0.7, 0, 0, 1.0, 1), ("ger", 0.3, 1, 0, 2)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(555)), "gemv_ger")


def test_regression_batched_einsum_in_loop():
    """Accumulating batched einsum (rank-3) inside a loop reads its destination
    each iteration; exercises the BatchedGemm path under LIH/scheduling."""
    prog = [
        ("beinsum", "ijb <- ikb ; kjb", 1.0, 0, 1, 0.0, 2),   # t2 = t0@t1 (batched, overwrite)
        ("loop", 3, [("beinsum", "ijb <- ikb ; kjb", 0.5, 0, 1, 1.0, 2)]),  # t2 += 0.5*(t0@t1), must stay in loop
    ]
    rng = np.random.default_rng(1717)
    m: list[np.ndarray] = []
    v: list[np.ndarray] = []
    t = [rng.standard_normal((3, 3, 2)) for _ in range(3)]  # t0,t1 inputs; t2 output
    check_program(prog, m, v, t, "batched_einsum")


def test_regression_view_alias_ordering():
    """Mix full-matrix writes and view (sub-block) writes on the same matrix,
    then read the whole matrix; stresses the scheduler's alias resolution
    (a write through a view must be seen as a write to the parent)."""
    prog = [
        ("scale", 0.5, 0),                 # whole m0 *= 0.5
        ("vscale", 3.0, 0, 1, 3, 1, 3),    # m0[1:3,1:3] *= 3
        ("vaxpy", 1.0, 1, 0, 0, 2, 0, 2),  # m0[0:2,0:2] += m1   (m1 is 3x3; block 2x2 -> use a 2x2 src)
        ("gemm", 1.0, 0, 2, 0.0, 3),       # m3 = m0 @ m2  (reads the aliased m0)
    ]
    # m1 must be 2x2 to match the vaxpy block; give a custom square+small pool.
    rng = np.random.default_rng(2929)
    m = [rng.standard_normal((3, 3)), rng.standard_normal((2, 2)),
         rng.standard_normal((3, 3)), rng.standard_normal((3, 3))]
    v: list[np.ndarray] = []
    t: list[np.ndarray] = []
    check_program(prog, m, v, t, "view_alias")


def test_regression_overlapping_views():
    """Two views of the same matrix with overlapping regions, written by
    different ops, then the whole matrix read; the partial writes must keep
    their relative order (both resolve to the same owner)."""
    prog = [
        ("vscale", 2.0, 0, 0, 2, 0, 2),     # m0[0:2,0:2] *= 2
        ("vaxpy", 1.0, 1, 0, 1, 3, 1, 3),   # m0[1:3,1:3] += m1 (overlaps [1:2,1:2])
        ("vscale", 0.5, 0, 0, 3, 0, 3),     # m0[0:3,0:3] *= 0.5 (covers both)
        ("gemm", 1.0, 0, 2, 0.0, 3),        # m3 = m0 @ m2
    ]
    rng = np.random.default_rng(8642)
    m = [rng.standard_normal((3, 3)), rng.standard_normal((2, 2)),
         rng.standard_normal((3, 3)), rng.standard_normal((3, 3))]
    check_program(prog, m, [], [], "overlap_views")


def test_regression_element_transform_in_loop():
    """In-place element-wise transform inside a loop, mixed with a full write."""
    prog = [("loop", 3, [("etransform", 2, 0), ("axpy", 0.5, 1, 0)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(9753)), "etransform_loop")


def test_regression_complex_einsum_prefactor():
    """Complex einsum/batched-einsum prefactors (phase factors) must not be
    truncated to their real part; regression for BatchedGemmDescriptor carrying
    only a real alpha/beta. Covers the direct batched path and accumulation."""
    rng = np.random.default_rng(4242)

    def cplx(sh):
        return rng.standard_normal(sh) + 1j * rng.standard_normal(sh)

    t = [cplx((3, 3, 2)) for _ in range(3)]
    prog = [
        ("beinsum", "ijb <- ikb ; kjb", 1.2 + 0.4j, 0, 1, 0.0, 2),          # t2 = (1.2+0.4j)*(t0@t1)_b
        ("loop", 3, [("beinsum", "ijb <- ikb ; kjb", 0.5 - 0.3j, 0, 1, 1.0 + 0j, 2)]),  # accumulate complex
    ]
    check_program(prog, [], [], t, "cplx_beinsum_pf", dtype="complex128")


def test_regression_complex_gemm_and_2d_einsum_prefactor():
    """Complex prefactors on gemm and 2D einsum (non-batched and via the
    GEMMBatching pass under the default manager)."""
    rng = np.random.default_rng(4343)

    def cplx(sh):
        return rng.standard_normal(sh) + 1j * rng.standard_normal(sh)

    m = [cplx((3, 3)) for _ in range(4)]
    prog = [
        ("gemm", 0.7 + 0.2j, 0, 1, 0.3 - 0.9j, 2),         # complex alpha and beta (accumulate)
        ("einsum", "ij <- ik ; kj", -0.6 + 0.5j, 0, 1, 1.0 + 0j, 3),  # complex ab_pf accumulate
    ]
    check_program(prog, m, [], [], "cplx_gemm_pf", dtype="complex128")


def test_regression_view_in_loop():
    """A view write inside a loop, interleaved with a full-tensor write."""
    prog = [
        ("loop", 3, [
            ("vscale", 0.9, 0, 0, 2, 0, 2),   # m0[0:2,0:2] *= 0.9
            ("scale", 1.0, 0),                # whole-tensor touch (identity scale)
        ]),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(3131)), "view_loop")


def test_regression_mixed_shape_einsum_chain():
    rng = np.random.default_rng(2468)
    m = [rng.standard_normal((2, 4)), rng.standard_normal((4, 3)),
         rng.standard_normal((2, 3)), rng.standard_normal((3, 2))]
    v: list[np.ndarray] = []
    t: list[np.ndarray] = []
    prog = [
        ("einsum", "ij <- ik ; kj", 1.0, 0, 1, 0.0, 2),
        ("perm", 1.0, 0.0, 2, 3),
        ("scale", 0.5, 2),
    ]
    check_program(prog, m, v, t, "mixed_einsum")
