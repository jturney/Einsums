# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Re-execution and double-optimize idempotence fuzz.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_reexecution(seed):
    """Execute the optimized graph twice without resetting inputs and compare to
    the oracle applied twice. Iterative solvers replay a graph repeatedly, so a
    pass that frees a still-needed buffer or corrupts state on replay must fail
    here even though a single execution passes."""
    rng = np.random.default_rng(20_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t, runs=2)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    g, mats, vecs, r3 = _build(prog, m, v, t, f"replay{seed}")
    g.apply(cg.default_pass_manager())
    g.execute()
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "REEXECUTED")

@pytest.mark.parametrize("seed", range(200))
def test_fuzz_double_optimize(seed):
    """Apply the default pass manager twice before executing. A non-idempotent
    pass (one that doesn't reach a fixpoint, or that mis-handles its own prior
    output) would diverge on the second application."""
    rng = np.random.default_rng(40_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=4)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    g, mats, vecs, r3 = _build(prog, m, v, t, f"dblopt{seed}")
    g.apply(cg.default_pass_manager())
    g.apply(cg.default_pass_manager())
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "DOUBLE-OPTIMIZE")
