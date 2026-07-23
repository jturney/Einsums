# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: deeply nested control-flow random programs.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(120))
def test_fuzz_deep_nesting(seed, dtype):
    rng = np.random.default_rng(50_000 + seed)
    prog = _gen_block(rng, depth=4, max_stmts=4)
    check_program(prog, *_seed_arrays(rng, dtype), f"deep{seed}", dtype=dtype)
