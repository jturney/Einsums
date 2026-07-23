# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Cross-executor differential fuzz: control-flow and deeply nested programs.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", range(120))
def test_fuzz_cross_executor_control_flow(seed, dtype):
    rng = np.random.default_rng(80_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    check_program_cross_executor(prog, *_seed_arrays(rng, dtype), f"xcf{seed}", dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", range(80))
def test_fuzz_cross_executor_deep_nesting(seed, dtype):
    rng = np.random.default_rng(90_000 + seed)
    prog = _gen_block(rng, depth=4, max_stmts=4)
    check_program_cross_executor(prog, *_seed_arrays(rng, dtype), f"xdeep{seed}", dtype=dtype)
