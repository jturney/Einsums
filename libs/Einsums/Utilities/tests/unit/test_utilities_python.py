# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for the Utilities bindings.

Two free functions are exposed via Apiary:

  * ``seed_random``: seeds the process-global C++ engine behind every
                     ``create_random_tensor`` draw *and* the randomized
                     projections inside ``truncated_svd`` / ``truncated_syev``.
                     Seeding a numpy generator cannot reach those.
  * ``make_temp_path``: a unique scratch path, returned as ``pathlib.Path``
                     (needs pybind11's stl/filesystem caster, which is easy to
                     drop from the generated TU - hence the type assertion).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import einsums
from einsums.testing import REAL_DTYPES


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_seed_random_makes_draws_reproducible(dtype):
    """Same seed, same tensor."""
    einsums.seed_random(42)
    first = np.asarray(einsums.create_random_tensor("A", [4, 4], dtype=dtype)).copy()
    einsums.seed_random(42)
    second = np.asarray(einsums.create_random_tensor("A", [4, 4], dtype=dtype)).copy()
    assert np.array_equal(first, second)


def test_seed_random_different_seeds_differ():
    """Guards against the binding silently being a no-op, which a
    reproducibility-only assertion would happily pass."""
    einsums.seed_random(42)
    first = np.asarray(einsums.create_random_tensor("A", [8, 8])).copy()
    einsums.seed_random(7)
    other = np.asarray(einsums.create_random_tensor("A", [8, 8])).copy()
    assert not np.array_equal(first, other)


def test_seed_random_pins_the_internal_randomized_projection():
    """The reason this binding exists.

    ``truncated_svd`` draws its over-sampled projection from the C++ engine, so
    identical inputs still give different results run to run. Seeding makes the
    whole call deterministic - not just the input.
    """
    A = einsums.create_zero_tensor("A", [40, 40])
    rng = np.random.default_rng(1)
    np.copyto(np.asarray(A), rng.standard_normal((40, 40)))

    einsums.seed_random(1234)
    _, s_first, _ = einsums.linalg.truncated_svd(A, 3)
    first = np.asarray(s_first).copy()

    einsums.seed_random(1234)
    _, s_second, _ = einsums.linalg.truncated_svd(A, 3)
    second = np.asarray(s_second).copy()

    assert np.array_equal(first, second)


def test_make_temp_path_returns_a_path():
    p = einsums.make_temp_path()
    # The caster is the fragile part: without pybind11/stl/filesystem.h this
    # raises "Unregistered type ... filesystem::path" at call time.
    assert isinstance(p, pathlib.PurePath)


def test_make_temp_path_is_unique():
    """Distinct paths per call - the property that makes it usable for scratch
    files under parallel ctest."""
    paths = {einsums.make_temp_path() for _ in range(32)}
    assert len(paths) == 32


def test_make_temp_path_does_not_create_the_file():
    """Only the name is reserved; the caller decides whether to create it."""
    assert not einsums.make_temp_path().exists()
