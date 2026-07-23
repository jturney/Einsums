# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Random pass-pipeline permutation fuzz over complex128.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

import einsums.graph as cg

from _fuzz_diff_common import *  # shared fuzz/differential harness


@pytest.mark.parametrize("seed", range(250))
def test_fuzz_random_pipeline_complex(seed):
    """Random pass-pipeline permutation over complex128 tensors, the strongest
    interaction test on the complex paths through every pass."""
    rng = np.random.default_rng(110_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    m, v, t = _seed_arrays(rng, "complex128")
    oracle = _oracle(prog, m, v, t)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    order = list(_SAFE_PASSES)
    rng.shuffle(order)

    g, mats, vecs, r3 = _build(prog, m, v, t, f"crnd{seed}")
    pm = cg.PassManager()
    for name in order:
        pm.add(getattr(_G, name)())
    g.apply(pm)
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "COMPLEX-RANDOM-PIPELINE", extra=f"  order={order}")
