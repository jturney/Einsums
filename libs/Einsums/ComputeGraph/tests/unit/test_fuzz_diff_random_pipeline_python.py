# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Random pass-pipeline permutation fuzz (real dtypes), single + replay.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness


@pytest.mark.parametrize("seed", range(400))
def test_fuzz_random_pipeline(seed):
    """Apply a random *permutation* of the individually-sound passes (rather than
    the curated default order) and demand the result still matches the oracle.
    This is the strongest interaction test: any ordering that miscompiles is a
    real soundness bug or an undocumented ordering dependency."""
    rng = np.random.default_rng(30_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    order = list(_SAFE_PASSES)
    rng.shuffle(order)

    g, mats, vecs, r3 = _build(prog, m, v, t, f"rndpipe{seed}")
    pm = cg.PassManager()
    for name in order:
        pm.add(getattr(_G, name)())
    g.apply(pm)
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "RANDOM-PIPELINE", extra=f"  order={order}")

@pytest.mark.parametrize("seed", range(300))
def test_fuzz_random_pipeline_replay(seed):
    """The meanest combination: optimize with a random pass order, then
    execute twice. A randomly-optimized graph must still replay correctly;
    catches interaction bugs that only manifest on re-execution (e.g. a Free
    inserted by one pass ordering that a second run then needs)."""
    rng = np.random.default_rng(60_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t, runs=2)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    order = list(_SAFE_PASSES)
    rng.shuffle(order)

    g, mats, vecs, r3 = _build(prog, m, v, t, f"rndreplay{seed}")
    pm = cg.PassManager()
    for name in order:
        pm.add(getattr(_G, name)())
    g.apply(pm)
    g.execute()
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "RANDOM-PIPELINE+REPLAY", extra=f"  order={order}")
