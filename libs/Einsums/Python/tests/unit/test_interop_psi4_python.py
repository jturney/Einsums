#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""einsums.interop.psi4 assembles einsums tensors from psi4-shaped buffers.

The adapter is duck-typed, so these tests run without psi4: FakeMatrix mimics
exactly the surface the adapter documents (nirrep()/symmetry()/.nph/.name),
which doubles as a regression test that the adapter never grows a dependency
on real psi4 types.
"""

import numpy as np
import pytest

import einsums as ein
import einsums.graph as cg
from einsums.interop import psi4 as interop


class FakeMatrix:
    """The slice of psi4.core.Matrix's surface that the adapter documents."""

    def __init__(self, blocks, symmetry=0, name="fake"):
        self._blocks = tuple(np.ascontiguousarray(b, dtype=np.float64) for b in blocks)
        self._symmetry = symmetry
        self.name = name

    def nirrep(self):
        return len(self._blocks)

    def symmetry(self):
        return self._symmetry

    @property
    def nph(self):
        return self._blocks


def tile_array(T, coord):
    return np.array(T.tile_view(list(coord)))


def test_from_matrix_c1_round_trip():
    rng = np.random.default_rng(7)
    block = rng.standard_normal((4, 3))
    T = interop.from_matrix(FakeMatrix([block], name="S"))
    assert T.name == "S"
    assert list(T.shape) == [4, 3]
    np.testing.assert_allclose(tile_array(T, (0, 0)), block)


def test_from_matrix_symmetry_blocked_diagonal():
    rng = np.random.default_rng(11)
    blocks = [rng.standard_normal((d, d)) for d in (3, 0, 2, 1)]
    T = interop.from_matrix(FakeMatrix(blocks))
    assert list(T.shape) == [6, 6]
    assert T.grid_size() == 16  # 4x4 tile grid
    for h, block in enumerate(blocks):
        if block.size == 0:
            assert not T.has_tile([h, h])
        else:
            np.testing.assert_allclose(tile_array(T, (h, h)), block)
    # off-diagonal tiles are structural zeros
    assert not T.has_tile([0, 2])


def test_from_matrix_nontotally_symmetric_operator():
    # symmetry=1: block h has shape (rowspi[h], colspi[h^1]) and lands at tile (h, h^1)
    rowspi = [2, 3]
    colspi = [2, 3]
    blocks = [np.full((rowspi[0], colspi[1]), 1.5), np.full((rowspi[1], colspi[0]), -2.5)]
    T = interop.from_matrix(FakeMatrix(blocks, symmetry=1))
    assert list(T.shape) == [5, 5]
    np.testing.assert_allclose(tile_array(T, (0, 1)), blocks[0])
    np.testing.assert_allclose(tile_array(T, (1, 0)), blocks[1])
    assert not T.has_tile([0, 0])
    assert not T.has_tile([1, 1])


def test_from_matrix_default_name_falls_back():
    T = interop.from_matrix(FakeMatrix([np.eye(2)], name=""))
    assert T.name == "psi4 matrix"
    T = interop.from_matrix(FakeMatrix([np.eye(2)], name="ignored"), name="override")
    assert T.name == "override"


def test_so_eri_assembles_blocks():
    rng = np.random.default_rng(13)
    sopi = [2, 1]
    quads = [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0), (0, 1, 0, 1)]
    blocked = []
    dense = {}
    for quad in quads:
        d = tuple(sopi[h] for h in quad)
        block = rng.standard_normal((d[0] * d[1], d[2] * d[3]))
        blocked.append((list(quad), FakeMatrix([block], name="SO ERI block")))
        dense[quad] = block.reshape(d)

    T = interop.so_eri(blocked, sopi)
    assert list(T.shape) == [3, 3, 3, 3]
    assert T.grid_size() == 16  # 2^4 tile grid
    for quad, expected in dense.items():
        np.testing.assert_allclose(tile_array(T, quad), expected)
    assert not T.has_tile([1, 0, 0, 0])  # symmetry-forbidden block never supplied


def test_so_eri_rejects_wrong_block_shape():
    bad = [([0, 0, 0, 0], FakeMatrix([np.zeros((2, 3))]))]
    with pytest.raises(ValueError, match="expected shape"):
        interop.so_eri(bad, [2, 1])


def test_mo_bra_half_transform_reshape():
    n1, n2, nbf = 2, 3, 4
    rng = np.random.default_rng(17)
    dense = rng.standard_normal((n1, n2, nbf, nbf))
    flat = FakeMatrix([dense.reshape(n1 * n2, nbf * nbf)])
    H = interop.mo_bra_half_transform(flat, n1, n2)
    assert list(H.shape) == [n1, n2, nbf, nbf]
    np.testing.assert_allclose(np.array(H), dense)


def test_mo_bra_half_transform_validates_dims():
    with pytest.raises(ValueError, match="rows"):
        interop.mo_bra_half_transform(FakeMatrix([np.zeros((5, 16))]), 2, 3)
    with pytest.raises(ValueError, match="square"):
        interop.mo_bra_half_transform(FakeMatrix([np.zeros((6, 15))]), 2, 3)


def test_df_tensor_reshape():
    naux, d2, d3 = 5, 2, 3
    rng = np.random.default_rng(19)
    dense = rng.standard_normal((naux, d2, d3))
    Q = interop.df_tensor(FakeMatrix([dense.reshape(naux, d2 * d3)]), d2, d3)
    assert list(Q.shape) == [naux, d2, d3]
    np.testing.assert_allclose(np.array(Q), dense)


def test_shaped_arrays_cross_too():
    # psi4 Matrices with a numpy-shape hint view as already-shaped arrays
    # (Qov() as rank 3, mo_bra_half_transform() as rank 4); the adapters must
    # accept those as well as the flat 2D form.
    rng = np.random.default_rng(31)
    dense3 = rng.standard_normal((4, 2, 3))
    Q = interop.df_tensor(dense3, 2, 3)
    np.testing.assert_allclose(np.array(Q), dense3)
    dense4 = rng.standard_normal((2, 3, 4, 4))
    H = interop.mo_bra_half_transform(dense4, 2, 3)
    np.testing.assert_allclose(np.array(H), dense4)


def test_dense_reshape_and_validation():
    nbf = 3
    rng = np.random.default_rng(29)
    eri = rng.standard_normal((nbf, nbf, nbf, nbf))
    T = interop.dense(FakeMatrix([eri.reshape(nbf * nbf, nbf * nbf)]), (nbf, nbf, nbf, nbf), name="AO ERI")
    assert list(T.shape) == [nbf] * 4
    np.testing.assert_allclose(np.array(T), eri)
    with pytest.raises(ValueError, match="elements"):
        interop.dense(FakeMatrix([np.zeros((2, 2))]), (5,))


def test_block_array_rejects_multi_irrep():
    multi = FakeMatrix([np.zeros((1, 1)), np.zeros((2, 2))])
    with pytest.raises(ValueError, match="single-block"):
        interop.df_tensor(multi, 1, 1)


def test_interop_attribute_on_package():
    assert ein.interop.psi4 is interop


# ---- opportunistic real-psi4 cross-checks ----------------------------------
# Skipped whenever psi4 is not importable (the normal ctest environment, CI).
# On machines where psi4 IS on PYTHONPATH these catch drift in the documented
# psi4-side contract (block layouts, flat shapes, numpy-shape hints) that the
# FakeMatrix tests cannot see.

def _psi4_water_mints(symmetry):
    psi4 = pytest.importorskip("psi4")
    import os

    psi4.core.set_output_file(os.devnull, False)
    mol = psi4.geometry(f"O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry {symmetry}\n")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "STO-3G")
    return psi4, psi4.core.MintsHelper(basis), basis.nbf()


def test_psi4_so_overlap_round_trip():
    _, mints, _ = _psi4_water_mints("c2v")
    S = mints.so_overlap()
    T = interop.from_matrix(S)
    for h in range(S.nirrep()):
        blk = np.asarray(S.nph[h])
        if blk.size == 0:
            assert not T.has_tile([h, h])
            continue
        np.testing.assert_allclose(tile_array(T, (h, h)), blk)


def test_psi4_so_eri_blocked_matches_ao_eri():
    _, mints, nbf = _psi4_water_mints("c1")
    ao = np.asarray(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)
    sopi = [b.shape[0] for b in mints.so_overlap().nph]
    T = interop.so_eri(mints.so_eri_blocked(), sopi)
    np.testing.assert_allclose(tile_array(T, (0, 0, 0, 0)), ao, atol=1e-12)


def test_psi4_mo_bra_half_transform_matches_einsum():
    psi4, mints, nbf = _psi4_water_mints("c1")
    rng = np.random.default_rng(37)
    C1 = rng.standard_normal((nbf, 2))
    C2 = rng.standard_normal((nbf, 3))
    flat = mints.mo_bra_half_transform(psi4.core.Matrix.from_array(C1), psi4.core.Matrix.from_array(C2))
    H = interop.mo_bra_half_transform(flat, 2, 3)
    ao = np.asarray(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)
    ref = np.einsum("mp,nq,mnls->pqls", C1, C2, ao)
    # atol absorbs the engine's Schwarz screening of negligible quartets
    np.testing.assert_allclose(np.array(H), ref, atol=1e-10)


def test_assembled_tensors_work_in_compute_graph():
    # The adapter's outputs are ordinary einsums tensors: capture a contraction
    # over an assembled tiled matrix to prove there is no psi4 residue.
    rng = np.random.default_rng(23)
    blocks = [rng.standard_normal((2, 2)), rng.standard_normal((3, 3))]
    S = interop.from_matrix(FakeMatrix(blocks, name="S"))
    s = ein.zeros((1,), dtype="float64")
    g = cg.Graph("interop-smoke")
    with cg.capture(g):
        ein.linalg.dot(s, S, S)
    g.execute()
    expected = sum(float(np.sum(b * b)) for b in blocks)
    assert np.isclose(float(np.asarray(s).reshape(1)[0]), expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
