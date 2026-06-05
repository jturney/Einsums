# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for cg::block_copy.

``einsums.linalg.block_copy(dst, src, dst_offsets, src_offsets, extents)``
copies a contiguous N-D sub-region of src into dst. Same rank and dtype on
both sides; per-axis stride math handles arbitrary layouts.

Unblocks two SCF/MP2 patterns that previously required numpy-style slicing:
  * Density build: extract C[:, :nocc] into a freshly-allocated C_occ
  * MP2 (ia|jb) block: extract eri_mo[:nocc, nocc:, :nocc, nocc:]
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, REAL_DTYPES


# ──────────────────────────────────────────────────────────────────────────
# Eager mode: rank-2 column-slab extract (the SCF C_occ case)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_block_copy_extract_column_slab(dtype):
    nbf, nocc = 6, 4
    C = einsums.create_random_tensor("C", [nbf, nbf], dtype=dtype)
    C_occ = einsums.create_zero_tensor("C_occ", [nbf, nocc], dtype=dtype)

    einsums.linalg.block_copy(C_occ, C, [0, 0], [0, 0], [nbf, nocc])

    np.testing.assert_array_equal(np.asarray(C_occ), np.asarray(C)[:, :nocc])


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_block_copy_extract_with_src_offset(dtype):
    """Extract a virtual block C[:, nocc:] into a fresh tensor."""
    nbf, nocc = 6, 4
    nvirt = nbf - nocc
    C = einsums.create_random_tensor("C", [nbf, nbf], dtype=dtype)
    C_virt = einsums.create_zero_tensor("C_virt", [nbf, nvirt], dtype=dtype)

    einsums.linalg.block_copy(C_virt, C, [0, 0], [0, nocc], [nbf, nvirt])

    np.testing.assert_array_equal(np.asarray(C_virt), np.asarray(C)[:, nocc:])


def test_block_copy_with_dst_offset_writes_into_subregion():
    """Write src into a sub-region of a larger dst."""
    big = einsums.create_zero_tensor("big", [5, 5])
    small = einsums.create_random_tensor("small", [2, 3])

    einsums.linalg.block_copy(big, small, [1, 1], [0, 0], [2, 3])

    big_np = np.asarray(big)
    np.testing.assert_array_equal(big_np[1:3, 1:4], np.asarray(small))
    # Surrounding region untouched.
    assert big_np[0, :].sum() == 0
    assert big_np[4, :].sum() == 0


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_block_copy_rank4_mp2_iajb_block(dtype):
    """Extract eri_mo[:nocc, nocc:, :nocc, nocc:] — the MP2 (ia|jb) block."""
    nocc, nvirt = 3, 4
    nbf = nocc + nvirt
    eri_mo = einsums.create_random_tensor("eri_mo", [nbf, nbf, nbf, nbf], dtype=dtype)
    iajb = einsums.create_zero_tensor("iajb", [nocc, nvirt, nocc, nvirt], dtype=dtype)

    einsums.linalg.block_copy(iajb, eri_mo, [0, 0, 0, 0], [0, nocc, 0, nocc], [nocc, nvirt, nocc, nvirt])

    expected = np.asarray(eri_mo)[:nocc, nocc:, :nocc, nocc:]
    np.testing.assert_array_equal(np.asarray(iajb), expected)


# ──────────────────────────────────────────────────────────────────────────
# Captured mode
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_block_copy_captured_matches_eager(dtype):
    C = einsums.create_random_tensor("C", [6, 6], dtype=dtype)

    eager_occ = einsums.create_zero_tensor("eager_occ", [6, 4], dtype=dtype)
    capt_occ = einsums.create_zero_tensor("capt_occ", [6, 4], dtype=dtype)

    einsums.linalg.block_copy(eager_occ, C, [0, 0], [0, 0], [6, 4])

    g = cg.Graph("bc")
    with cg.capture(g):
        einsums.linalg.block_copy(capt_occ, C, [0, 0], [0, 0], [6, 4])
    g.execute()

    np.testing.assert_array_equal(np.asarray(capt_occ), np.asarray(eager_occ))


def test_block_copy_captured_deferred_until_execute():
    C = einsums.create_random_tensor("C", [4, 4])
    C_occ = einsums.create_zero_tensor("C_occ", [4, 2])

    g = cg.Graph("deferred")
    with cg.capture(g):
        einsums.linalg.block_copy(C_occ, C, [0, 0], [0, 0], [4, 2])

    # Recorded but not executed — destination still zero.
    np.testing.assert_array_equal(np.asarray(C_occ), 0.0)

    g.execute()
    np.testing.assert_array_equal(np.asarray(C_occ), np.asarray(C)[:, :2])


def test_block_copy_captures_dependency_for_optimization():
    """The recorded node carries src as input + dst as output, so passes that
    look at the dataflow graph see the dependency."""
    C = einsums.create_random_tensor("C", [4, 4])
    C_occ = einsums.create_zero_tensor("C_occ", [4, 2])

    g = cg.Graph("depcheck")
    with cg.capture(g):
        einsums.linalg.block_copy(C_occ, C, [0, 0], [0, 0], [4, 2])

    assert g.num_nodes() == 1
    assert g.num_tensors() == 2  # C, C_occ


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────


def test_block_copy_empty_extents_raises():
    src = einsums.create_random_tensor("src", [3])
    dst = einsums.create_zero_tensor("dst", [3])
    with pytest.raises(Exception):
        einsums.linalg.block_copy(dst, src, [], [], [])


def test_block_copy_rank_mismatch_raises():
    src = einsums.create_random_tensor("src", [3, 4])
    dst = einsums.create_zero_tensor("dst", [3, 4, 5])  # rank-3 dst, rank-2 src
    with pytest.raises(Exception):
        einsums.linalg.block_copy(dst, src, [0, 0, 0], [0, 0], [1, 1])


def test_block_copy_dst_offsets_length_mismatch_raises():
    src = einsums.create_random_tensor("src", [3, 4])
    dst = einsums.create_zero_tensor("dst", [3, 4])
    with pytest.raises(Exception):
        einsums.linalg.block_copy(dst, src, [0], [0, 0], [3, 4])  # dst_offsets too short


def test_block_copy_extent_overflow_src_raises():
    src = einsums.create_random_tensor("src", [3, 4])
    dst = einsums.create_zero_tensor("dst", [3, 4])
    with pytest.raises(Exception):
        einsums.linalg.block_copy(dst, src, [0, 0], [1, 0], [3, 4])  # 1+3>3 in axis 0


def test_block_copy_extent_overflow_dst_raises():
    src = einsums.create_random_tensor("src", [3, 4])
    dst = einsums.create_zero_tensor("dst", [3, 4])
    with pytest.raises(Exception):
        einsums.linalg.block_copy(dst, src, [0, 2], [0, 0], [3, 4])  # 2+4>4 in axis 1


# ──────────────────────────────────────────────────────────────────────────
# Compositional SCF density build via occupation-weighted gemm
# ──────────────────────────────────────────────────────────────────────────


def test_scf_density_build_via_block_copy_and_gemm():
    """D = 2 * C_occ @ C_occ^T, where C_occ is extracted via block_copy."""
    nbf, nocc = 5, 3
    C = einsums.create_random_tensor("C", [nbf, nbf])
    C_occ = einsums.create_zero_tensor("C_occ", [nbf, nocc])
    D = einsums.create_zero_tensor("D", [nbf, nbf])

    g = cg.Graph("density")
    with cg.capture(g):
        einsums.linalg.block_copy(C_occ, C, [0, 0], [0, 0], [nbf, nocc])
        einsums.linalg.gemm(2.0, C_occ, C_occ, 0.0, D, trans_b=True)
    g.execute()

    C_np = np.asarray(C)
    expected = 2.0 * C_np[:, :nocc] @ C_np[:, :nocc].T
    np.testing.assert_allclose(np.asarray(D), expected, rtol=1e-5)
