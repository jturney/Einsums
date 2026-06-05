#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Validate MintsHelper.so_eri_tiled() (einsums rank-4 TiledRuntimeTensor of SO
ERIs) against psi4's own integrals.

Two checks:
  1. C1 symmetry: SO basis == AO basis, so the single tile (0,0,0,0) must equal
     ao_eri() reshaped to (nbf,nbf,nbf,nbf) — exact value/notation/index check.
  2. C2v (water): every materialized tile must be a symmetry-allowed irrep
     quadruple (hp^hq^hr^hs == 0), and the 8-fold permutational symmetry of
     (pq|rs) must hold within the totally-symmetric (0,0,0,0) block.
"""
import numpy as np
import einsums          # registers the pybind types; runtime inits lazily
import psi4

psi4.core.set_output_file("/tmp/psi4_so_eri.out", False)


def build(symmetry):
    mol = psi4.geometry(f"O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry {symmetry}\n")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "STO-3G")
    return psi4.core.MintsHelper(basis), basis


# ---- 1) C1: exact match to ao_eri ---------------------------------------
mints, basis = build("c1")
nbf = basis.nbf()
ao = np.asarray(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)  # (pq|rs), chemists'
T = mints.so_eri_tiled()
print(f"C1: rank={T.rank()} dims={list(T.dims())} filled_tiles={T.num_filled_tiles()}")
assert T.has_tile([0, 0, 0, 0]), "C1 must have the single (0,0,0,0) tile"
so = np.asarray(T.tile_view([0, 0, 0, 0]))
assert so.shape == ao.shape, f"shape {so.shape} != {ao.shape}"
assert np.allclose(so, ao), f"C1 SO ERI != AO ERI (max diff {np.abs(so-ao).max():.3e})"
print(f"  C1 so_eri_tiled matches ao_eri  (max|Δ|={np.abs(so-ao).max():.2e})")

# ---- 2) C2v: structure + permutational symmetry -------------------------
mints, basis = build("c2v")
T = mints.so_eri_tiled()
nirrep = 4
print(f"C2v: dims={list(T.dims())} filled_tiles={T.num_filled_tiles()}")

# every present tile is a symmetry-allowed quadruple
n_present = 0
for hp in range(nirrep):
    for hq in range(nirrep):
        for hr in range(nirrep):
            for hs in range(nirrep):
                if T.has_tile([hp, hq, hr, hs]):
                    n_present += 1
                    assert (hp ^ hq ^ hr ^ hs) == 0, \
                        f"disallowed tile ({hp},{hq},{hr},{hs}) present"
print(f"  all {n_present} present tiles are symmetry-allowed (hp^hq^hr^hs==0)")

# 8-fold permutational symmetry within the totally-symmetric (A1^4) block
b = np.asarray(T.tile_view([0, 0, 0, 0]))
for perm, name in [((1, 0, 2, 3), "qp|rs"), ((0, 1, 3, 2), "pq|sr"),
                   ((2, 3, 0, 1), "rs|pq"), ((3, 2, 1, 0), "sr|qp")]:
    assert np.allclose(b, b.transpose(perm)), f"(0,0,0,0) block fails {name} symmetry"
print("  (0,0,0,0) block obeys 8-fold permutational symmetry")

# physical sanity: diagonal Coulomb self-repulsion (pp|pp) > 0
diag = np.einsum("pppp->p", b)
assert np.all(diag > 0), "diagonal (pp|pp) must be positive"
print(f"  diagonal (pp|pp) all positive  (min={diag.min():.4f})")

print("ALL SO ERI TILED CHECKS PASS")
