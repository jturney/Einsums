# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""cg.Pipeline multi-stage differential fuzz.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

import einsums.graph as cg
from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness


# ══════════════════════════════════════════════════════════════════════════
# cg.Pipeline multi-stage graphs: stages execute in order over a shared
# tensor pool (a later stage reads what an earlier one wrote). A pipeline
# program is a list of stage sub-programs; the oracle is interp_np over their
# concatenation. Each stage may itself contain loops / conditionals. Run both
# raw (no passes) and optimized (default manager applied per stage).
# ══════════════════════════════════════════════════════════════════════════


def _gen_pipeline(rng, n_stages, depth, max_stmts):
    return [_gen_block(rng, depth, max_stmts) for _ in range(n_stages)]


def _run_pipeline(stages, m_arrays, v_arrays, t_arrays, name, optimize):
    mats, vecs, r3 = _make_pool(m_arrays, v_arrays, t_arrays, name)
    p = cg.Pipeline(name)
    for si, stage in enumerate(stages):
        sg = p.add_stage(f"{name}_s{si}")
        build_cg(stage, sg, mats, vecs, r3, f"{name}_s{si}")
    if optimize:
        p.apply(cg.default_pass_manager())
    p.execute()
    return ([np.asarray(x).copy() for x in mats],
            [np.asarray(x).copy() for x in vecs],
            [np.asarray(x).copy() for x in r3])


def check_pipeline(stages, m_arrays, v_arrays, t_arrays, label, dtype="float64"):
    rtol, atol = _DTYPE_TOL[dtype]
    cap = _DTYPE_CAP[dtype]
    dt = np.dtype(dtype)
    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for stage in stages:  # stages run in order, sharing the pool
            interp_np(stage, om, ov, ot, dt)
    if not _usable(om, ov, ot, cap=cap):
        pytest.skip("oracle overflowed — numerically degenerate program")
    oracle = (om, ov, ot)

    raw = _run_pipeline(stages, m_arrays, v_arrays, t_arrays, f"{label}_raw", optimize=False)
    opt = _run_pipeline(stages, m_arrays, v_arrays, t_arrays, f"{label}_opt", optimize=True)

    def _cmp(stage_name, got):
        for arrs, oarr, kind in zip(got, oracle, "mvt"):
            for i in range(len(oarr)):
                if not np.allclose(arrs[i], oarr[i], rtol=rtol, atol=atol):
                    raise AssertionError(
                        f"{stage_name} disagrees on {kind}{i} (dtype={dtype})\n"
                        f"stages={stages!r}\ngot=\n{arrs[i]}\noracle=\n{oarr[i]}"
                    )

    _cmp("PIPELINE-RAW", raw)
    _cmp("PIPELINE-OPTIMIZED", opt)


def test_regression_pipeline_three_stage_chain():
    """Three stages sharing tensors: stage1 produces, stage2 transforms (with a
    loop), stage3 accumulates: the classic produce/consume chain."""
    stages = [
        [("gemm", 1.0, 0, 1, 0.0, 2)],                       # m2 = m0 @ m1
        [("loop", 3, [("scale", 0.9, 2)]), ("perm", 1.0, 0.0, 2, 3)],  # m2 *= 0.9^3; m3 = m2^T
        [("axpy", 1.0, 3, 0), ("gemm", 1.0, 2, 3, 1.0, 1)],   # m0 += m3; m1 += m2@m3
    ]
    check_pipeline(stages, *_square_seed_arrays(np.random.default_rng(24680)), "pipe_chain")


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_pipeline(seed, dtype):
    rng = np.random.default_rng(160_000 + seed)
    n_stages = int(rng.integers(2, 5))
    stages = _gen_pipeline(rng, n_stages, depth=2, max_stmts=5)
    check_pipeline(stages, *_seed_arrays(rng, dtype), f"pipe{seed}", dtype=dtype)
