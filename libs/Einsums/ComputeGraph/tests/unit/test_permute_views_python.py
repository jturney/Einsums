# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""View/owning combination coverage for einsums.permute.

permute computes C = c_pf * C + a_pf * permute(A). Its C++ template always took
any tensor for A and C, but only owning RuntimeTensor pairs were bound, so a
view in either place was a TypeError from Python. All 2**2 = 4 cells of the
(A, C) x (owning, view) matrix are now bound per dtype.

A view reaches the kernel in two shapes: a slice, whose strides step past its
parent's other elements, and a permute_view, whose axes are out of storage
order. The second, with an axis of extent one, is how a Python caller reaches
the HPTT outer sizes an extent-1 stride used to corrupt.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close

_MODES = ["eager", "graph", "graph+passes"]
_ctr = itertools.count()


def _is_complex(dtype):
    return np.dtype(dtype).kind == "c"


def _values(shape, dtype, seed):
    """Order-one values in ``dtype``, genuinely complex for a complex dtype."""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(shape)
    if _is_complex(dtype):
        vals = vals + 1j * rng.standard_normal(shape)
    return vals.astype(dtype)


def _owning(vals, dtype):
    t = einsums.create_zero_tensor(f"t{next(_ctr)}", list(vals.shape), dtype=dtype)
    if vals.size:
        np.asarray(t)[...] = vals
    return t


def _permuted_view(vals, dtype, perm):
    """A view whose logical data is ``vals``, stored with its axes in ``perm`` order."""
    stored = _owning(np.ascontiguousarray(np.transpose(vals, perm)), dtype)
    return stored.permute_view(list(np.argsort(perm)))


def _slice(P, bounds, mode):
    """A view of ``P``: a slice eagerly (``cg.view`` is capture-only), ``cg.view`` in a graph."""
    if mode == "eager":
        return P[tuple(slice(*b) for b in bounds)]
    return cg.view(P, list(bounds))


def _run(mode, body):
    if mode == "eager":
        body()
        return
    g = cg.Graph(f"permute-views-{next(_ctr)}")
    with cg.capture(g):
        body()
    if mode == "graph+passes":
        g.apply(cg.default_pass_manager())
    g.execute()


def _prefactors(dtype):
    if _is_complex(dtype):
        return 0.5 + 0.25j, 2.0 - 1.0j
    return 0.5, 2.0


# ──────────────────────────────────────────────────────────────────────────
# The (A, C) x (owning, view) matrix, views as slices of a larger parent
# ──────────────────────────────────────────────────────────────────────────

# A is a (3, 4) block; C holds its transpose. A view sits at an offset inside a
# (5, 6) parent, so its strides step past elements the permute must not touch.
_A_BOUNDS = ((1, 4), (2, 6))
_C_BOUNDS = ((0, 4), (3, 6))


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("cell", ["OO", "OV", "VO", "VV"])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_permute_owning_view_cells(dtype, cell, mode):
    a_view, c_view = cell[0] == "V", cell[1] == "V"
    c_pf, a_pf = _prefactors(dtype)

    a_vals = _values((5, 6), dtype, 1)
    c_vals = _values((5, 6), dtype, 2)
    a_block = a_vals[1:4, 2:6]
    PA = _owning(a_vals, dtype) if a_view else None
    A = None if a_view else _owning(a_block, dtype)
    PC = _owning(c_vals, dtype) if c_view else None
    C = None if c_view else _owning(c_vals[0:4, 3:6], dtype)

    def body():
        a = _slice(PA, _A_BOUNDS, mode) if a_view else A
        c = _slice(PC, _C_BOUNDS, mode) if c_view else C
        einsums.permute("ji <- ij", c, a, c_pf, a_pf)

    _run(mode, body)

    expected_block = c_pf * c_vals[0:4, 3:6] + a_pf * a_block.T
    if c_view:
        # The rest of the parent is untouched.
        expected = c_vals.copy()
        expected[0:4, 3:6] = expected_block
        assert_close(np.asarray(PC), expected, dtype=dtype)
    else:
        assert_close(np.asarray(C), expected_block, dtype=dtype)
    if a_view:
        np.testing.assert_array_equal(np.asarray(PA), a_vals)


# ──────────────────────────────────────────────────────────────────────────
# permute_view operands: axes out of storage order
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("storage", [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_permute_of_a_permuted_view(dtype, storage, mode):
    vals = _values((2, 3, 4), dtype, 3)
    A = _permuted_view(vals, dtype, list(storage))
    C = _owning(np.zeros((4, 2, 3), dtype=dtype), dtype)

    _run(mode, lambda: einsums.permute("kij <- ijk", C, A))

    assert_close(np.asarray(C), np.transpose(vals, (2, 0, 1)), dtype=dtype)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("storage", [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_permute_into_a_permuted_view(dtype, storage, mode):
    vals = _values((2, 3, 4), dtype, 4)
    A = _owning(vals, dtype)
    C = _permuted_view(np.zeros((4, 2, 3), dtype=dtype), dtype, list(storage))

    _run(mode, lambda: einsums.permute("kij <- ijk", C, A))

    assert_close(np.asarray(C), np.transpose(vals, (2, 0, 1)), dtype=dtype)


# An axis of extent one is never stepped along, so a permute_view leaves it
# whatever stride it had in storage. HPTT's outer sizes were read off every
# stride, so such a stride could make one smaller than its extent and the plan
# was rejected ("HPTT: outerSizeA invalid"); einsum reached it through
# "ikj <- ijk ; ijk" with a permuted B. Every storage order puts the extent-1
# axis somewhere different.
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("storage", [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)])
@pytest.mark.parametrize("extents", [(2, 1, 3), (1, 2, 3), (2, 3, 1), (1, 6, 1)])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_permute_view_with_an_extent_one_axis(dtype, extents, storage, mode):
    vals = _values(extents, dtype, 5)
    A = _permuted_view(vals, dtype, list(storage))
    C = _owning(np.zeros((extents[0], extents[2], extents[1]), dtype=dtype), dtype)

    _run(mode, lambda: einsums.permute("ikj <- ijk", C, A))

    assert_close(np.asarray(C), np.transpose(vals, (0, 2, 1)), dtype=dtype)
