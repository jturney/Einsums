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


@pytest.mark.parametrize("seed", fuzz_seeds(250))
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


# ──────────────────────────────────────────────────────────────────────────
# Symmetrization sites over complex128, in the all-passes shuffle.
#
# ``_motif_symacc`` draws the site SymmetrizedAccumulation folds, with the
# transpose sometimes scaled or an antisymmetrizer under a complex scale, and
# sometimes accumulated into a block of a larger matrix through a view. The
# default corpus above never draws one, and the all-passes shard draws it
# among ten other motifs; here every program is made of them, over complex128,
# so the folded permute's operators and complex scalars meet every pass order.
# ──────────────────────────────────────────────────────────────────────────

from _fuzz_diff_common import _motif_symacc  # noqa: E402


def _symacc_trial(seed):
    rng = np.random.default_rng(280_000 + seed)
    prog = []
    for _ in range(int(rng.integers(1, 4))):
        site = _motif_symacc(rng)
        site = [site] if isinstance(site, tuple) else site  # a pool too small for the site falls back to one statement
        prog += [("loop", 2, site)] if rng.random() < 0.2 else site
    arrays = allpass_seed_arrays(rng, "complex128")
    return check_program_all_passes(prog, arrays, f"csymacc{seed}", rng, dtype="complex128")


@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_symmetrized_accumulation_sites_complex(seed):
    _symacc_trial(seed)


def test_symmetrized_accumulation_sites_complex_fold():
    """The site corpus is not vacuous: SymmetrizedAccumulation folds a fair share of it.

    A site folds only when the shuffled order runs the pass before ElementWiseFusion
    composes the two accumulates, and not when the draw made the transpose a caller's
    tensor or put the halves in a view, so the floor is well under one.
    """
    fired = usable = 0
    for seed in range(60):
        try:
            got = _symacc_trial(seed)
        except pytest.skip.Exception:
            continue
        usable += 1
        fired += "SymmetrizedAccumulation" in got
    assert fired >= 0.2 * usable, f"SymmetrizedAccumulation folded in {fired} of {usable} trials"
