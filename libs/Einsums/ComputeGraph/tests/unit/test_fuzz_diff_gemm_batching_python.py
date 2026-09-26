# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""GEMMBatching placement differential fuzz (batchable contractions + consumers).

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

from _fuzz_diff_common import *  # shared fuzz/differential harness


# ══════════════════════════════════════════════════════════════════════════
# Deliberate GEMMBatching attack: identical batchable contractions plus a
# consumer. That defect was found here only by ACCIDENT (the cse_redundant
# generator produced the shape as a side effect of the CSE user-visible
# guard); this generator emits it on purpose so batch PLACEMENT stays probed
# as the pass evolves.
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", fuzz_seeds(100))
def test_fuzz_gemm_batching_with_consumers(seed):
    """N identical (batchable) contractions into distinct user-visible
    outputs, then a consumer reading a randomly chosen member's output, and
    optionally a second consumer of another member. GEMMBatching must place
    the fused gemm_batch node BEFORE every consumer (regression: it used to
    append, and position is program order). Full-buffer differential: every
    output is written by the batch, so nothing is dead."""
    rng = np.random.default_rng(130_000 + seed)
    pool = _sq_pool(rng, 10)
    idx = list(range(10))
    rng.shuffle(idx)
    A, B, E, D1, D2 = idx[:5]
    n_dups = int(rng.integers(2, 5))            # 2-4 batchable members
    outs = idx[5:5 + n_dups]

    spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
    prog = [("einsum", spec, 1.0, A, B, 0.0, o) for o in outs]

    def consumer(victim, dest):
        roll = int(rng.integers(0, 3))
        if roll == 0:
            return ("gemm", 1.0, victim, E, 0.0, dest)
        if roll == 1:
            return ("perm", 1.0, 0.0, victim, dest)
        return ("einsum", _SQ, 1.0, victim, E, 0.0, dest)

    prog.append(consumer(outs[int(rng.integers(0, len(outs)))], D1))
    if rng.random() < 0.5:
        prog.append(consumer(outs[int(rng.integers(0, len(outs)))], D2))

    check_program(prog, pool, [], [], f"gemmbatch_consumer{seed}")


def test_gemm_batching_leaves_antisymmetrized_contractions_alone():
    """A batched GEMM computes the one unpermuted product, so batching two P(p/r) contractions
    returned the plain product in both outputs. Expansion is disabled so the operators reach the
    batching pass, which is the order a caller who disables expansion gets."""
    import einsums
    import einsums.graph as cg

    n, k = 4, 3
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((n, k)), rng.standard_normal((k, n))
    A, B = einsums.asarray(a), einsums.asarray(b)
    R0 = einsums.zeros((n, n), dtype="float64")
    R1 = einsums.zeros((n, n), dtype="float64")

    graph = cg.Graph("batching_operator")
    with cg.capture(graph):
        einsums.einsum("p,r <- P(p/r) p,q ; q,r", R0, A, B)
        einsums.einsum("p,r <- P(p/r) p,q ; q,r", R1, A, B)
    manager = cg.PassManager()
    manager.populate_default()
    manager.disable("AntisymmetrizerExpansion")
    manager.run(graph)
    graph.execute()

    want = a @ b - (a @ b).T
    np.testing.assert_allclose(np.asarray(R0), want, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(R1), want, rtol=1e-12, atol=1e-12)
