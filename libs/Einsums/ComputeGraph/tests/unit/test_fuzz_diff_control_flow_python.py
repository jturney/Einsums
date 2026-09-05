# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: random programs with generated loops / conditionals.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(200))
def test_fuzz_with_control_flow(seed, dtype):
    rng = np.random.default_rng(10_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    check_program(prog, *_seed_arrays(rng, dtype), f"cf{seed}", dtype=dtype)


@pytest.mark.parametrize("seed", fuzz_seeds(120))
def test_fuzz_control_flow_under_the_region_pipeline(seed):
    """The structural phase over programs whose statements are inside bodies.

    Loops and conditionals stopped being barriers to the region framework, so
    these passes now raise a loop body as a region of its own. That is the
    change this shard exists for: a rewrite that is right on a flat run and
    wrong inside a body is a wrong number here, where before it was unreachable.
    Float64 only, because what is under test is the descent and not the dtype
    dispatch that the shard above already covers in four.
    """
    rng = np.random.default_rng(70_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    check_program_region_pipeline(prog, *_seed_arrays(rng, "float64"), f"cfreg{seed}")
